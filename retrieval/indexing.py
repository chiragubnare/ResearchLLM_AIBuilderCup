import os
import sys
import time
import chromadb
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from chromadb.config import Settings

os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.contracts import Chunk

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection("paper_chunks")


def embed_text(text: str, max_retries: int = 3) -> list[float]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return response.embeddings[0].values
        except ClientError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def index_chunks(chunks: list[Chunk]):
    existing_ids = set(collection.get()["ids"])
    remaining = [c for c in chunks if c.id not in existing_ids]
    print(f"  {len(chunks) - len(remaining)} already indexed, {len(remaining)} remaining")

    for i, chunk in enumerate(remaining):
        embedding = embed_text(chunk.content)
        collection.add(
            ids=[chunk.id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[{
                "type": chunk.type,
                "page_number": chunk.page_number,
                "source_ref": chunk.source_ref or "",
            }],
        )
        print(f"  Indexed {i+1}/{len(remaining)}")
        time.sleep(1)