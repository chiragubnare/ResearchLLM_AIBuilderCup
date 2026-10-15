import os

from app.generation import answer_query

doc_id = os.getenv("PAPER_DOC_ID")
if not doc_id:
    raise SystemExit("Set PAPER_DOC_ID to the SHA-256 doc_id of an indexed paper.")

questions = [
    "What is a residual connection?",
    "What top-5 error did the ensemble achieve on ImageNet?",
    "Who won the 2022 FIFA World Cup?",  # off-topic: should return not-found
]

for q in questions:
    print(f"\n=== {q}")
    resp = answer_query(q, doc_id=doc_id)
    print(resp.answer)
    for c in resp.citations:
        print(f"  -> page {c.chunk.page_number} [{c.chunk.chunk_type}] score={c.score:.3f}")
