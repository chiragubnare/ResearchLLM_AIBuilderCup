import sys
import os
import re
import hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shared.contracts import Chunk
from text_extraction import extract_text_blocks_by_page
from table_extraction import extract_tables
from image_extraction import extract_images
from image_captioning import caption_image

MAX_CHARS = 1000    # target chunk size
MIN_CHARS = 250     # smaller than this gets merged into a neighbour
MAX_MERGED = 1400   # ceiling after merging


def _make_id(pdf_path: str, chunk_type: str, page: int, content: str) -> str:
    raw = f"{os.path.basename(pdf_path)}|{chunk_type}|{page}|{content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _is_real_block(text: str, min_chars: int = 40, min_words: int = 6) -> bool:
    return len(text) >= min_chars and len(text.split()) >= min_words


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    packed, current = [], ""
    for p in pieces:
        if current and len(current) + 1 + len(p) > max_chars:
            packed.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        packed.append(current)
    return packed


def _page_text_chunks(blocks: list[str]) -> list[str]:
    pieces = []
    for b in blocks:
        if not _is_real_block(b):
            continue
        if len(b) > MAX_CHARS:
            pieces.extend(_split_sentences(b))
        else:
            pieces.append(b)

    chunks = _pack(pieces, MAX_CHARS)

    merged = []
    for c in chunks:
        if merged and (len(c) < MIN_CHARS or len(merged[-1]) < MIN_CHARS) \
                and len(merged[-1]) + 1 + len(c) <= MAX_MERGED:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    return merged


def build_chunks(pdf_path: str) -> list[Chunk]:
    chunks = []

    for page_num, blocks in extract_text_blocks_by_page(pdf_path).items():
        for text in _page_text_chunks(blocks):
            chunks.append(Chunk(
                id=_make_id(pdf_path, "text", page_num, text),
                type="text", content=text, page_number=page_num,
            ))

    for table in extract_tables(pdf_path):
        chunks.append(Chunk(
            id=_make_id(pdf_path, "table", table["page_number"], table["markdown"]),
            type="table", content=table["markdown"],
            page_number=table["page_number"], source_ref=table["source_ref"],
        ))

    for img in extract_images(pdf_path):
        caption = caption_image(img["image_bytes"], img["ext"])
        chunks.append(Chunk(
            id=_make_id(pdf_path, "figure", img["page_number"], caption),
            type="figure", content=caption,
            page_number=img["page_number"], source_ref=img["source_ref"],
        ))

    return chunks