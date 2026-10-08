import pymupdf


def extract_text_by_page(pdf_path: str) -> dict[int, str]:
    doc = pymupdf.open(pdf_path)
    return {page.number + 1: page.get_text() for page in doc}

