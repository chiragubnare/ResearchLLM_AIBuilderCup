# ingestion/inspect_tables.py
import sys
from table_extraction import extract_tables

pdf = sys.argv[1] if len(sys.argv) > 1 else "resnet"
tables = extract_tables(f"../shared/sample_data/{pdf}.pdf")
print(f"Found {len(tables)} tables after filtering")
for t in tables:
    print(t["source_ref"], "- page", t["page_number"], f"({len(t['markdown'])} chars)")