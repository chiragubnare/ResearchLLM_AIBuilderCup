"""Grounded response generation and citation-reference validation."""

import os
import re
from typing import Iterable

from google import genai

from shared.contracts import RetrievedChunk


def stream_answer(question: str, evidence: list[RetrievedChunk], history: list[dict]):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    evidence_text = "\n\n".join(
        f"[S{i}; page {item.chunk.page_number}; {item.chunk.chunk_type}]\n{item.chunk.content}"
        for i, item in enumerate(evidence, start=1)
    )
    conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history[-6:]
    )
    prompt = f"""You are a research paper assistant. Answer using only the supplied evidence.
If the evidence does not answer the question, say that the paper does not provide enough information.
Every factual sentence must end with one or more source markers like [S1]. Use only source
markers provided below. Do not write page numbers yourself; the app converts valid source
markers to page citations. Do not cite a source that does not support the sentence.
Be clear and concise. You may use conversation history to understand follow-up questions, but
the evidence below is the only source of facts.

Conversation history:
{conversation or '(none)'}

Evidence:
{evidence_text}

Question: {question}
Answer:"""
    response_stream: Iterable = client.models.generate_content_stream(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=prompt,
    )
    for response in response_stream:
        if response.text:
            yield response.text


def validate_citations(answer: str, evidence: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """Require each answer sentence to reference a valid retrieved source marker."""
    answer = answer.strip()
    if not answer:
        return "I couldn't generate an answer. Please try again.", []

    abstention = re.fullmatch(
        r"(?:the paper (?:does not|doesn't) provide enough information"
        r"(?: to answer(?: this question)?)?|not enough information in the paper)[.!?]?",
        answer.lower(),
    )
    if abstention:
        return answer, []

    valid_sources = {f"S{i}": item for i, item in enumerate(evidence, start=1)}
    cited_ids = []
    sentences = [
        part
        for part in re.split(r"(?<=[.!?])\s+(?!\[S\d+\])|\n+", answer)
        if part.strip()
    ]
    for sentence in sentences:
        markers = re.findall(r"\[S(\d+)\]", sentence)
        if not markers:
            return (
                "I couldn't verify the answer's citations against the retrieved passages. "
                "Try asking a narrower question.",
                [],
            )
        for marker in markers:
            source_id = f"S{marker}"
            if source_id not in valid_sources:
                return (
                    "I couldn't verify the answer's citations against the retrieved passages. "
                    "Try asking a narrower question.",
                    [],
                )
            cited_ids.append(source_id)

    rendered = re.sub(
        r"\[S(\d+)\]",
        lambda match: f"[p. {valid_sources[f'S{match.group(1)}'].chunk.page_number}]",
        answer,
    )
    cited_sources = [valid_sources[source_id] for source_id in dict.fromkeys(cited_ids)]
    return rendered, cited_sources
