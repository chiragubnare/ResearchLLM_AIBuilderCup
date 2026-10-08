import sys
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shared.contracts import Chunk
from text_extraction import extract_text_by_page
from table_extraction import extract_tables
from image_extraction import extract_images
from image_captioning import caption_image


def _looks_like_prose(text: str, min_avg_line_length: int = 15) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    avg_length = sum(len(l) for l in lines) / len(lines)
    return avg_length >= min_avg_line_length


def build_chunks(pdf_path: str) -> list[Chunk]:
    chunks = []

    text_by_page = extract_text_by_page(pdf_path)
    for page_num, text in text_by_page.items():
        if text.strip() and _looks_like_prose(text):
            chunks.append(Chunk(
                id=str(uuid.uuid4()), type="text", content=text, page_number=page_num
            ))

    for table in extract_tables(pdf_path):
        chunks.append(Chunk(
            id=str(uuid.uuid4()), type="table", content=table["markdown"],
            page_number=table["page_number"], source_ref=table["source_ref"]
        ))

    for img in extract_images(pdf_path):
        caption = caption_image(img["image_bytes"], img["ext"])
        chunks.append(Chunk(
            id=str(uuid.uuid4()), type="figure", content=caption,
            page_number=img["page_number"], source_ref=img["source_ref"]
        ))

    return chunks