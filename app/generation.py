from shared.contracts import ChatResponse, RetrievedChunk
from retrieval.answering import stream_answer, validate_citations
from retrieval.retriever import retrieve


def answer_query(question: str, doc_id: str, top_k: int = 5) -> ChatResponse:
    evidence: list[RetrievedChunk] = retrieve(question, doc_id=doc_id, top_k=top_k)
    if not evidence:
        return ChatResponse(answer="I couldn't find relevant information in the indexed paper.")
    answer = "".join(stream_answer(question, evidence, []))
    validated_answer, citations = validate_citations(answer, evidence)
    return ChatResponse(answer=validated_answer, citations=citations)
