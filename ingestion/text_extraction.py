import pymupdf


def extract_text_by_page(pdf_path: str) -> dict[int, str]:
    doc = pymupdf.open(pdf_path)
    return {page.number + 1: page.get_text() for page in doc}


def extract_text_blocks_by_page(pdf_path: str) -> dict[int, list[str]]:
    """Returns {page_number: [block_text, ...]} in the PDF's own reading order."""
    doc = pymupdf.open(pdf_path)
    pages = {}
    for page in doc:
        blocks = page.get_text("blocks")  # no sort=True: sorting interleaves two-column layouts
        texts = []
        for b in blocks:
            if b[6] != 0:  # 0 = text block, skip image blocks
                continue
            text = " ".join(b[4].split())
            if text:
                texts.append(text)
        pages[page.number + 1] = texts
    return pages