# app/calibrate.py
import os

from retrieval.retriever import retrieve

doc_id = os.getenv("PAPER_DOC_ID")
if not doc_id:
    raise SystemExit("Set PAPER_DOC_ID to the SHA-256 doc_id of an indexed paper.")

on_topic = [
    "What is a residual connection?",
    "What top-5 error did the ensemble achieve on ImageNet?",
    "Why does degradation happen in deep plain networks?",
    "How many layers does the deepest ResNet have?",
]
off_topic = [
    "Who won the 2022 FIFA World Cup?",
    "What is a good recipe for pasta carbonara?",
    "How do I fix a flat bicycle tire?",
]

for label, questions in (("ON-TOPIC", on_topic), ("OFF-TOPIC", off_topic)):
    print(f"\n{label}")
    for q in questions:
        results = retrieve(q, doc_id=doc_id, top_k=1)
        if not results:
            print(f"  no indexed results  {q}")
            continue
        top = results[0]
        print(f"  {top.score:.3f}  {q}")
