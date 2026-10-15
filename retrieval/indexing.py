import gzip
import json
import os
import time
from typing import Optional

import chromadb
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from chromadb.config import Settings

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from shared.contracts import Chunk
from shared.storage import ObjectStorage, get_storage

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = None


def _get_client():
    global client
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client


def _metadata(chunk: Chunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "chunk_type": chunk.chunk_type,
        "page_number": chunk.page_number,
        "source_ref": chunk.source_ref or "",
        "image_key": chunk.image_key or "",
        "image_gcs_url": chunk.image_gcs_url or "",
        "bounding_box": json.dumps(chunk.bounding_box) if chunk.bounding_box else "",
    }


def _snapshot_key(doc_id: str) -> str:
    return f"vector_index/{doc_id}.json.gz"


def restore_vector_index(doc_id: str, storage: Optional[ObjectStorage] = None) -> bool:
    """Restore saved embeddings into the local Chroma collection when available."""
    storage = storage or get_storage()
    key = _snapshot_key(doc_id)
    if collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
        if not storage.exists(key):
            save_vector_index(doc_id, storage=storage)
        return True

    if not storage.exists(key):
        return False

    snapshot = json.loads(gzip.decompress(storage.get_bytes(key)))
    if not snapshot.get("ids"):
        return False
    collection.upsert(
        ids=snapshot["ids"],
        embeddings=snapshot["embeddings"],
        documents=snapshot["documents"],
        metadatas=snapshot["metadatas"],
    )
    return True


def save_vector_index(doc_id: str, storage: Optional[ObjectStorage] = None) -> None:
    """Save document vectors and metadata as a portable GCS/local object."""
    storage = storage or get_storage()
    records = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "embeddings"],
    )
    if not records["ids"]:
        return

    embeddings = records["embeddings"]
    snapshot = {
        "ids": records["ids"],
        "documents": records["documents"],
        "metadatas": records["metadatas"],
        "embeddings": embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
    }
    packed = gzip.compress(json.dumps(snapshot, separators=(",", ":")).encode("utf-8"))
    storage.put_bytes(_snapshot_key(doc_id), packed, content_type="application/gzip")

chroma_client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection("paper_chunks")


def embed_text(
    text: str,
    max_retries: int = 3,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    for attempt in range(max_retries):
        try:
            response = _get_client().models.embed_content(
                model="gemini-embedding-001",
                contents=[text],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return response.embeddings[0].values
        except ClientError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def index_chunks(
    chunks: list[Chunk],
    doc_id: str,
    replace: bool = False,
    storage: Optional[ObjectStorage] = None,
):
    if any(chunk.doc_id != doc_id for chunk in chunks):
        raise ValueError("All chunks must belong to the requested doc_id")
    if replace:
        collection.delete(where={"doc_id": doc_id})

    existing_ids = set(collection.get(where={"doc_id": doc_id})["ids"])
    remaining = [c for c in chunks if c.chunk_id not in existing_ids]
    print(f"  {len(chunks) - len(remaining)} already indexed, {len(remaining)} remaining")

    for i, chunk in enumerate(remaining):
        embedding = embed_text(chunk.content, task_type="RETRIEVAL_DOCUMENT")
        collection.add(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[_metadata(chunk)],
        )
        print(f"  Indexed {i+1}/{len(remaining)}")
        time.sleep(1)

    save_vector_index(doc_id, storage=storage)
