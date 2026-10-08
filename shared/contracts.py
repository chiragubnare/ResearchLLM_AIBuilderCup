from pydantic import BaseModel
from typing import Literal, Optional


class Chunk(BaseModel):
    id: str
    type: Literal["text", "table", "figure"]
    content: str                      # raw text / table as markdown / figure caption
    page_number: int
    source_ref: Optional[str] = None  # e.g. "Table 2", "Figure 3"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[RetrievedChunk]