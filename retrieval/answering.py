"""Grounded response generation and citation-reference validation."""

import os
import re
from typing import Iterable, Optional

from google import genai

from shared.contracts import RetrievedChunk

_GREETING = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|yo|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+there)?[!.,?\s]*$",
    re.IGNORECASE,
)
_ABSTENTION = re.compile(
    r"(?:the paper (?:does not|doesn't) provide enough information"
    r"|not enough information in the paper)",
    re.IGNORECASE,
)
_MARKER_GROUP = re.compile(
    r"[\[(]\s*((?:S\s*)?\d+(?:\s*[,;]\s*(?:S\s*)?\d+)*)\s*[\])]",
    re.IGNORECASE,
)
_MARKER = re.compile(r"\[S(\d+)\]", re.IGNORECASE)


def greeting_reply(question: str) -> Optional[str]:
    """Reply to a greeting without searching the paper."""
    if _GREETING.fullmatch(question.strip()):
        return "Hello. Ask about this paper's method, results, or conclusion."
    return None


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
If the evidence does not answer the question, reply with only this sentence:
The paper does not provide enough information.
Every sentence must end with one or more source markers like [S1]. Do not add a title,
introduction, or closing sentence that has no marker. Use only source markers provided
below. Do not write page numbers yourself; the app converts valid source markers to page
citations. Do not cite a source that does not support the sentence.
For a question about the result, contribution, or conclusion, state the finding from the
evidence in a few sentences, each ending with its source marker.
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


def _normalize_markers(answer: str) -> str:
    """Turn [s1], (S1), and [S1, S2] into separate [S1] markers."""

    def replace(match: re.Match) -> str:
        body = match.group(1)
        if not re.search(r"S\s*\d+", body, re.IGNORECASE):
            return match.group(0)
        return "".join(f"[S{number}]" for number in re.findall(r"\d+", body))

    return _MARKER_GROUP.sub(replace, answer)


def _sentences(answer: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?!\[S\d+\])|\n+", answer, flags=re.IGNORECASE)
        if part.strip()
    ]
    merged: list[str] = []
    for part in parts:
        if re.fullmatch(r"(?:\[S\d+\]\s*)+", part, re.IGNORECASE) and merged:
            merged[-1] = f"{merged[-1].rstrip()} {part}"
        else:
            merged.append(part)
    return merged


def validate_citations(answer: str, evidence: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """Keep sentences that cite a retrieved passage and show their page numbers."""
    answer = _normalize_markers(answer.strip())
    if not answer:
        return "I couldn't generate an answer. Please try again.", []

    valid_sources = {f"S{i}": item for i, item in enumerate(evidence, start=1)}
    cited_ids = []
    kept = []
    for sentence in _sentences(answer):
        markers = [f"S{number}" for number in _MARKER.findall(sentence)]
        valid_markers = [source_id for source_id in markers if source_id in valid_sources]
        if not valid_markers:
            continue
        sentence = _MARKER.sub(
            lambda match, sources=valid_sources: (
                f"[S{match.group(1)}]" if f"S{match.group(1)}" in sources else ""
            ),
            sentence,
        )
        sentence = re.sub(r"\s{2,}", " ", sentence).strip()
        if sentence:
            kept.append(sentence)
            cited_ids.extend(valid_markers)

    if not kept:
        if _ABSTENTION.search(answer):
            return "The paper does not provide enough information.", []
        return (
            "I couldn't verify the answer's citations against the retrieved passages. "
            "Try asking a narrower question.",
            [],
        )

    rendered = _MARKER.sub(
        lambda match: f"[p. {valid_sources[f'S{match.group(1)}'].chunk.page_number}]",
        " ".join(kept),
    )
    rendered = re.sub(r"\]\[", "] [", rendered)
    cited_sources = [valid_sources[source_id] for source_id in dict.fromkeys(cited_ids)]
    return rendered, cited_sources
