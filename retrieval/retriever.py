import json

from shared.contracts import Chunk, RetrievedChunk
from retrieval.indexing import embed_text, collection


def retrieve(query: str, doc_id: str, top_k: int = 5) -> list[RetrievedChunk]:
    query_embedding = embed_text(query, task_type="RETRIEVAL_QUERY")
    count = len(collection.get(where={"doc_id": doc_id})["ids"])
    if not count:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        where={"doc_id": doc_id},
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunk = Chunk(
            chunk_id=results["ids"][0][i],
            doc_id=meta["doc_id"],
            chunk_type=meta["chunk_type"],
            content=results["documents"][0][i],
            page_number=meta["page_number"],
            source_ref=meta["source_ref"] or None,
            image_gcs_url=meta.get("image_gcs_url") or None,
            image_key=meta.get("image_key") or None,
            bounding_box=json.loads(meta["bounding_box"]) if meta.get("bounding_box") else None,
        )
        score = 1 - results["distances"][0][i]
        retrieved.append(RetrievedChunk(chunk=chunk, score=score))
    return retrieved
