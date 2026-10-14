import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))

from pipeline import build_chunks
from indexing import index_chunks
from retriever import retrieve

CACHE_PATH = "chunks_cache.pkl"

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from cache")
else:
    print("Building chunks from resnet.pdf...")
    chunks = build_chunks("../shared/sample_data/resnet.pdf")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Built and cached {len(chunks)} chunks")

print(f"\nIndexing {len(chunks)} chunks...")
index_chunks(chunks)

print("\nQuerying...")
results = retrieve("What is a residual connection?")
for q in ["What is a residual connection?",
          "Why does degradation happen in deep plain networks?",
          "What top-5 error did the ensemble achieve on ImageNet?"]:
    print(f"\n=== {q}")
    for r in retrieve(q, top_k=3):
        print(f"[{r.chunk.type}] score={r.score:.3f} page={r.chunk.page_number}")
        print(r.chunk.content[:200])
        print("---")