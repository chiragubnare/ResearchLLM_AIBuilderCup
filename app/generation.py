import os
import re
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from google import genai
from dotenv import load_dotenv
from shared.contracts import ChatResponse, RetrievedChunk
from retriever import retrieve

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEN_FALLBACK_MODEL")  # optional second model, see below

# Placeholder until you run calibrate.py. Low on purpose: wrongly rejecting a real
# question is worse than sending an off-topic one to the LLM, where the prompt refuses it.
MIN_SCORE = 0.30
NOT_FOUND = "I couldn't find this in the uploaded paper."

TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "high demand", "disconnected",
    "connection", "timeout", "timed out", "unavailable", "resource_exhausted",
)

PROMPT = """You answer questions about ONE research paper using only the numbered excerpts below.

Rules:
- Use only information in the excerpts. Do not use outside knowledge.
- Cite the excerpts you rely on inline, like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{not_found}"
- Table excerpts are markdown and may be messy. Quote numbers exactly as shown.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


class LLMUnavailable(Exception):
    pass


def _is_transient(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    return any(m in msg for m in TRANSIENT_MARKERS)


def _generate(model: str, prompt: str) -> str:
    interaction = client.interactions.create(
        model=model, input=[{"type": "text", "text": prompt}]
    )
    return interaction.output_text


def _call_llm(prompt: str, waits=(4, 8, 16, 24)) -> str:
    models = [GEN_MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL else [])
    for model in models:
        for attempt in range(len(waits) + 1):
            try:
                return _generate(model, prompt)
            except Exception as e:
                if not _is_transient(e):
                    raise  # auth, billing, bad request: retrying won't help
                if attempt < len(waits):
                    print(f"  {model} busy ({type(e).__name__}), retrying in {waits[attempt]}s...")
                    time.sleep(waits[attempt])
        print(f"  {model} still unavailable.")
    raise LLMUnavailable("All configured models are unavailable.")


def _build_context(retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for i, r in enumerate(retrieved, start=1):
        label = r.chunk.source_ref or r.chunk.type
        parts.append(f"[{i}] (page {r.chunk.page_number}, {label})\n{r.chunk.content}")
    return "\n\n".join(parts)


def answer_query(question: str, top_k: int = 5) -> ChatResponse:
    retrieved = [r for r in retrieve(question, top_k=top_k) if r.score >= MIN_SCORE]
    if not retrieved:
        return ChatResponse(answer=NOT_FOUND, citations=[])

    prompt = PROMPT.format(
        not_found=NOT_FOUND, context=_build_context(retrieved), question=question
    )
    try:
        answer = _call_llm(prompt).strip()
    except LLMUnavailable:
        return ChatResponse(
            answer="The language model is busy right now. Here are the most relevant "
                   "passages I found; please try again in a moment.",
            citations=retrieved,
        )

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations = [retrieved[n - 1] for n in cited if 1 <= n <= len(retrieved)]
    return ChatResponse(answer=answer, citations=citations)