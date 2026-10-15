from typing import Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_type: Literal["text", "table_json", "image_caption"]
    content: str
    image_gcs_url: Optional[str] = None
    image_key: Optional[str] = None
    bounding_box: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    source_ref: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
