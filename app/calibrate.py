# app/calibrate.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from retriever import retrieve

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
        top = retrieve(q, top_k=1)[0]
        print(f"  {top.score:.3f}  {q}")