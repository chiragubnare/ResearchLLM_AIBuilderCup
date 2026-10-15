import hashlib
import json
import os
import re
from typing import Optional

from shared.contracts import Chunk
from shared.storage import ObjectStorage, get_storage
from ingestion.text_extraction import extract_text_blocks_by_page
from ingestion.table_extraction import extract_tables
from ingestion.image_extraction import extract_images
from ingestion.image_captioning import caption_image

MAX_CHARS = 1000
MIN_CHARS = 250
MAX_MERGED = 1400


def _make_id(doc_id: str, chunk_type: str, page: int, content: str) -> str:
    raw = f"{doc_id}|{chunk_type}|{page}|{content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _is_real_block(text: str, min_chars: int = 40, min_words: int = 6) -> bool:
    return len(text) >= min_chars and len(text.split()) >= min_words


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    packed, current = [], ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            packed.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        packed.append(current)
    return packed


def _page_text_chunks(blocks: list[str]) -> list[str]:
    pieces = []
    for block in blocks:
        if not _is_real_block(block):
            continue
        pieces.extend(_split_sentences(block) if len(block) > MAX_CHARS else [block])

    chunks = _pack(pieces, MAX_CHARS)
    merged = []
    for chunk in chunks:
        if (
            merged
            and (len(chunk) < MIN_CHARS or len(merged[-1]) < MIN_CHARS)
            and len(merged[-1]) + 1 + len(chunk) <= MAX_MERGED
        ):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def build_chunks(
    pdf_path: str,
    doc_id: Optional[str] = None,
    storage: Optional[ObjectStorage] = None,
) -> list[Chunk]:
    if doc_id is None:
        digest = hashlib.sha256()
        with open(pdf_path, "rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(block)
        doc_id = digest.hexdigest()
    storage = storage or get_storage()
    chunks = []

    for page_num, blocks in extract_text_blocks_by_page(pdf_path).items():
        for text in _page_text_chunks(blocks):
            chunks.append(Chunk(
                chunk_id=_make_id(doc_id, "text", page_num, text),
                doc_id=doc_id,
                chunk_type="text",
                content=text,
                page_number=page_num,
            ))

    for table in extract_tables(pdf_path):
        content = json.dumps({"rows": table["rows"]}, ensure_ascii=False)
        chunks.append(Chunk(
            chunk_id=_make_id(doc_id, "table_json", table["page_number"], content),
            doc_id=doc_id,
            chunk_type="table_json",
            content=content,
            page_number=table["page_number"],
            source_ref=table["source_ref"],
        ))

    for img in extract_images(pdf_path):
        caption = caption_image(img["image_bytes"], img["ext"])
        extension = re.sub(r"[^a-z0-9]", "", img["ext"].lower()) or "bin"
        image_key = (
            f"extracted_images/{doc_id}/p{img['page_number']}"
            f"_fig{img['image_number']}.{extension}"
        )
        mime_extension = {"jpg": "jpeg"}.get(extension, extension)
        storage.put_bytes(image_key, img["image_bytes"], content_type=f"image/{mime_extension}")
        chunks.append(Chunk(
            chunk_id=_make_id(doc_id, "image_caption", img["page_number"], img["source_ref"]),
            doc_id=doc_id,
            chunk_type="image_caption",
            content=caption,
            page_number=img["page_number"],
            image_gcs_url=storage.uri(image_key),
            image_key=image_key,
            source_ref=img["source_ref"],
        ))

    return chunks
