import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.contracts import Chunk, RetrievedChunk
from indexing import embed_text, collection


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunk = Chunk(
            id=results["ids"][0][i],
            type=meta["type"],
            content=results["documents"][0][i],
            page_number=meta["page_number"],
            source_ref=meta["source_ref"] or None,
        )
        score = 1 - results["distances"][0][i]
        retrieved.append(RetrievedChunk(chunk=chunk, score=score))
    return retrieved