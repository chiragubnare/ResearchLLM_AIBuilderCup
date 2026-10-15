import hashlib
import json
import os
import re
from typing import Optional

from shared.contracts import Chunk
from shared.storage import ObjectStorage, get_storage
from ingestion.text_extraction import extract_text_by_page
from ingestion.table_extraction import extract_tables
from ingestion.image_extraction import extract_images
from ingestion.image_captioning import caption_image


def _looks_like_prose(text: str, min_avg_line_length: int = 15) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    avg_length = sum(len(l) for l in lines) / len(lines)
    return avg_length >= min_avg_line_length


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

    text_by_page = extract_text_by_page(pdf_path)
    for page_num, text in text_by_page.items():
        if text.strip() and _looks_like_prose(text):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page_num}_text",
                doc_id=doc_id,
                chunk_type="text",
                content=text,
                page_number=page_num,
            ))

    for table_index, table in enumerate(extract_tables(pdf_path), start=1):
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_p{table['page_number']}_table{table_index}",
            doc_id=doc_id,
            chunk_type="table_json",
            content=json.dumps({"rows": table["rows"]}, ensure_ascii=False),
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
        storage.put_bytes(
            image_key,
            img["image_bytes"],
            content_type=f"image/{mime_extension}",
        )
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_p{img['page_number']}_fig{img['image_number']}",
            doc_id=doc_id,
            chunk_type="image_caption",
            content=caption,
            page_number=img["page_number"],
            image_gcs_url=storage.uri(image_key),
            image_key=image_key,
            source_ref=img["source_ref"],
        ))

    return chunks
