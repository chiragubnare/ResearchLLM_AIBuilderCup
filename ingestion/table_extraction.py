import camelot
import re


def _is_caption(cell: str) -> bool:
    return bool(re.match(r"^(Table|Figure)\s+\d+\.", cell.strip()))


def _is_meaningful_table(markdown: str, max_words_per_cell: int = 10, min_non_empty_cells: int = 4) -> bool:
    rows = [r for r in markdown.split("\n") if r.strip().startswith("|")]
    non_empty_cells = []
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        non_empty_cells.extend(c for c in cells if c and not set(c) <= {"-"})

    if len(non_empty_cells) < min_non_empty_cells:
        return False

    data_cells = [c for c in non_empty_cells if not _is_caption(c)]
    return all(len(c.split()) <= max_words_per_cell for c in data_cells)


def extract_tables(pdf_path: str) -> list[dict]:
    tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    results = []
    for i, t in enumerate(tables):
        markdown = t.df.to_markdown(index=False)
        if _is_meaningful_table(markdown):
            results.append({
                "page_number": t.page,
                "markdown": markdown,
                "rows": t.df.fillna("").astype(str).values.tolist(),
                "source_ref": f"Table {i+1}",
            })
    return results
