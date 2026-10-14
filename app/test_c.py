from generation import answer_query

questions = [
    "What is a residual connection?",
    "What top-5 error did the ensemble achieve on ImageNet?",
    "Who won the 2022 FIFA World Cup?",  # off-topic: should return not-found
]

for q in questions:
    print(f"\n=== {q}")
    resp = answer_query(q)
    print(resp.answer)
    for c in resp.citations:
        print(f"  -> page {c.chunk.page_number} [{c.chunk.type}] score={c.score:.3f}")