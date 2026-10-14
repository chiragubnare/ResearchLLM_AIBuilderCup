import os
import sys
import time
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.contracts import Chunk

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection("paper_chunks")


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", max_retries: int = 4) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in response.embeddings]
        except ClientError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    return embed_texts([text], task_type)[0]


def index_chunks(chunks: list[Chunk], batch_size: int = 8):
    existing_ids = set(collection.get()["ids"])
    remaining = [c for c in chunks if c.id not in existing_ids]
    print(f"  {len(chunks) - len(remaining)} already indexed, {len(remaining)} remaining")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        embeddings = embed_texts([c.content for c in batch])
        collection.add(
            ids=[c.id for c in batch],
            embeddings=embeddings,
            documents=[c.content for c in batch],
            metadatas=[{
                "type": c.type,
                "page_number": c.page_number,
                "source_ref": c.source_ref or "",
            } for c in batch],
        )
        print(f"  Indexed {min(i + batch_size, len(remaining))}/{len(remaining)}")
        time.sleep(2)