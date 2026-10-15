import pymupdf


def extract_images(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    results = []
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            results.append({
                "page_number": page_num,
                "image_bytes": base_image["image"],
                "ext": base_image["ext"],
                "image_number": img_index,
                "source_ref": f"Figure (p{page_num}-{img_index})",
            })
    return results
