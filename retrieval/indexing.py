import gzip
import json
import os
import time
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from shared.contracts import Chunk
from shared.storage import ObjectStorage, get_storage

os.environ["ANONYMIZED_TELEMETRY"] = "False"
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


chroma_client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection("paper_chunks")


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int = 4,
) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = _get_client().models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [embedding.values for embedding in response.embeddings]
        except ClientError as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def embed_text(
    text: str,
    max_retries: int = 4,
    task_type: str = "RETRIEVAL_QUERY",
) -> list[float]:
    return embed_texts([text], task_type=task_type, max_retries=max_retries)[0]


def save_vector_index(doc_id: str, storage: Optional[ObjectStorage] = None) -> None:
    """Save this document's vectors and metadata as a portable object."""
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


def index_chunks(
    chunks: list[Chunk],
    doc_id: str,
    replace: bool = False,
    storage: Optional[ObjectStorage] = None,
    batch_size: int = 8,
) -> None:
    if any(chunk.doc_id != doc_id for chunk in chunks):
        raise ValueError("All chunks must belong to the requested doc_id")
    if replace:
        collection.delete(where={"doc_id": doc_id})

    existing_ids = set(collection.get(where={"doc_id": doc_id})["ids"])
    remaining = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    print(f"  {len(chunks) - len(remaining)} already indexed, {len(remaining)} remaining")

    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        embeddings = embed_texts(
            [chunk.content for chunk in batch],
            task_type="RETRIEVAL_DOCUMENT",
        )
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            embeddings=embeddings,
            documents=[chunk.content for chunk in batch],
            metadatas=[_metadata(chunk) for chunk in batch],
        )
        print(f"  Indexed {min(start + batch_size, len(remaining))}/{len(remaining)}")
        time.sleep(2)

    save_vector_index(doc_id, storage=storage)
