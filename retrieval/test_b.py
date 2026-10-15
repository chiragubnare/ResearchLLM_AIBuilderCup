"""Manually index and query the bundled ResNet PDF (requires GEMINI_API_KEY)."""

import hashlib
from pathlib import Path

from ingestion.pipeline import build_chunks
from retrieval.indexing import index_chunks
from retrieval.retriever import retrieve


PDF_PATH = Path(__file__).resolve().parents[1] / "shared" / "sample_data" / "resnet.pdf"

with PDF_PATH.open("rb") as pdf_file:
    doc_id = hashlib.file_digest(pdf_file, "sha256").hexdigest()

print(f"Building chunks from {PDF_PATH}...")
chunks = build_chunks(str(PDF_PATH), doc_id=doc_id)
print(f"Built {len(chunks)} chunks; indexing...")
index_chunks(chunks, doc_id=doc_id)

for question in (
    "What is a residual connection?",
    "Why does degradation happen in deep plain networks?",
    "What top-5 error did the ensemble achieve on ImageNet?",
):
    print(f"\n=== {question}")
    for result in retrieve(question, doc_id=doc_id, top_k=3):
        print(
            f"[{result.chunk.chunk_type}] score={result.score:.3f} "
            f"page={result.chunk.page_number}"
        )
        print(result.chunk.content[:200])
        print("---")
