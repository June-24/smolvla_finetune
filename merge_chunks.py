"""
merge_chunks.py
===============
Merges chunk parquet files into data/libero_full/data.parquet
using row-group streaming to avoid loading everything into RAM at once.
"""
import json
from pathlib import Path
import pyarrow.parquet as pq

BASE  = Path(__file__).parent
CHUNKS = [
    "data/chunk_0to9",
    "data/chunk_10to19",
    "data/chunk_20to29",
    "data/chunk_30to39",
]
OUT_DIR = BASE / "data/libero_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_path = OUT_DIR / "data.parquet"
writer   = None
total_rows = 0

for chunk in CHUNKS:
    pq_path = BASE / chunk / "data.parquet"
    if not pq_path.exists():
        print(f"SKIP (missing): {pq_path}")
        continue

    pf = pq.ParquetFile(str(pq_path))
    n_groups = pf.metadata.num_row_groups
    print(f"[{chunk}]  {pf.metadata.num_rows} rows, {n_groups} row-groups")

    for rg in range(n_groups):
        table = pf.read_row_group(rg)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
        total_rows += len(table)
        print(f"  row-group {rg+1}/{n_groups}  ({len(table)} rows)  total so far: {total_rows}", flush=True)

if writer:
    writer.close()

# Merge task_names.json
task_names = {}
for chunk in CHUNKS:
    tf = BASE / chunk / "task_names.json"
    if tf.exists():
        task_names.update(json.load(open(tf)))

json.dump(task_names, open(OUT_DIR / "task_names.json", "w"), indent=2)

print(f"\nDone — {total_rows} total rows, {len(task_names)} tasks")
print(f"Output: {out_path}")
