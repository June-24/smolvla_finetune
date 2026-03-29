"""
download_libero_full.py
=======================
Downloads the full LIBERO dataset (all 40 tasks) from HuggingFace Hub and
saves it as a local Parquet file. Unlike download_libero.py, this version
writes to disk in chunks so it never holds more than CHUNK_SIZE frames in RAM.

Usage:
    python download_libero_full.py [--episodes 2000] [--tasks 0-39] [--out data/libero_full]

Options:
    --episodes   Max episodes to download per run (default: 2000)
    --tasks      Task range, e.g. '0-39' (default: 0-39)
    --out        Output directory (default: data/libero_full)
    --chunk_size Number of frames to buffer before flushing to disk (default: 5000)
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


LIBERO_TASK_NAMES = {
    0:  "pick up the alphabet soup and place it in the basket",
    1:  "pick up the cream cheese and place it in the basket",
    2:  "pick up the salad dressing and place it in the basket",
    3:  "pick up the tomato sauce and place it in the basket",
    4:  "pick up the butter and place it in the basket",
    5:  "pick up the bbq sauce and place it in the basket",
    6:  "pick up the milk and place it in the basket",
    7:  "pick up the ketchup and place it in the basket",
    8:  "pick up the orange juice and place it in the basket",
    9:  "pick up the coffee and place it in the basket",
    10: "put the black bowl on the plate",
    11: "put the chocolate pudding on the plate",
    12: "put the butter on the plate",
    13: "put the red wine on the plate",
    14: "put the salad dressing on the plate",
    15: "put the cream cheese on the plate",
    16: "put the bbq sauce on the plate",
    17: "put the alphabet soup on the plate",
    18: "put the ketchup on the plate",
    19: "put the tomato sauce on the plate",
    20: "stack the alphabet soup on the cream cheese",
    21: "stack the cream cheese on the alphabet soup",
    22: "stack the tomato sauce on the cream cheese",
    23: "stack the cream cheese on the tomato sauce",
    24: "stack the alphabet soup on the tomato sauce",
    25: "stack the tomato sauce on the alphabet soup",
    26: "stack the ketchup on the tomato sauce",
    27: "stack the tomato sauce on the ketchup",
    28: "stack the cream cheese on the ketchup",
    29: "stack the ketchup on the cream cheese",
    30: "put the alphabet soup in the top drawer and close it",
    31: "put the cream cheese in the top drawer and close it",
    32: "put the butter in the top drawer and close it",
    33: "put the black bowl in the top drawer and close it",
    34: "put the chocolate pudding in the top drawer and close it",
    35: "put the salad dressing in the top drawer and close it",
    36: "put the bbq sauce in the top drawer and close it",
    37: "put the ketchup in the top drawer and close it",
    38: "put the tomato sauce in the top drawer and close it",
    39: "put the milk in the top drawer and close it",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",   type=int, default=2000)
    p.add_argument("--tasks",      type=str, default="0-39")
    p.add_argument("--out",        type=str, default="data/libero_full")
    p.add_argument("--chunk_size", type=int, default=5000,
                   help="Frames to buffer in RAM before flushing to disk")
    return p.parse_args()


def parse_task_range(s: str):
    parts = s.split("-")
    if len(parts) == 1:
        v = int(parts[0])
        return v, v
    return int(parts[0]), int(parts[1])


def pil_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def row_to_record(row: dict) -> dict:
    """Convert a streaming HF row to a flat dict with PNG bytes for images."""
    r = {}
    for k, v in row.items():
        if hasattr(v, "save"):          # PIL Image
            r[k] = pil_to_bytes(v)
        elif isinstance(v, np.ndarray):
            r[k] = v.tolist()
        else:
            r[k] = v
    return r


def flush_chunk(records: list, writer, schema_ref: list):
    """Append a list of record dicts to the ParquetWriter, creating it if needed."""
    df = pd.DataFrame(records)

    # Ensure image columns are bytes
    for col in ["observation.images.image", "observation.images.image2"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, bytes) else bytes(x)
            )

    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer[0] is None:
        schema_ref[0] = table.schema
        writer[0] = pq.ParquetWriter(writer[1], schema_ref[0])

    # Cast to consistent schema in case dtypes drift
    table = table.cast(schema_ref[0])
    writer[0].write_table(table)


def main():
    args = parse_args()
    task_lo, task_hi = parse_task_range(args.tasks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "data.parquet"

    print(f"Streaming LIBERO from HuggingFace Hub (chunked write — low RAM)...")
    print(f"  task range  : {task_lo}-{task_hi}")
    print(f"  max episodes: {args.episodes}")
    print(f"  chunk size  : {args.chunk_size} frames")
    print(f"  output      : {out_dir}")
    print()

    ds = load_dataset("HuggingFaceVLA/libero", split="train", streaming=True)

    episodes_seen: set = set()
    chunk: list = []
    total_frames = 0

    # writer[0] = ParquetWriter instance (None until first flush)
    # writer[1] = path string
    writer = [None, str(parquet_path)]
    schema_ref = [None]

    pbar = tqdm(desc="Downloading frames", unit="frames")

    try:
        for sample in ds:
            task_idx = int(sample["task_index"])
            ep_idx   = int(sample["episode_index"])

            if not (task_lo <= task_idx <= task_hi):
                if len(episodes_seen) >= args.episodes:
                    break
                continue

            if ep_idx not in episodes_seen:
                if len(episodes_seen) >= args.episodes:
                    break
                episodes_seen.add(ep_idx)

            chunk.append(row_to_record(sample))
            total_frames += 1
            pbar.update(1)

            if len(chunk) >= args.chunk_size:
                flush_chunk(chunk, writer, schema_ref)
                chunk.clear()

    finally:
        # Flush remaining
        if chunk:
            flush_chunk(chunk, writer, schema_ref)
            chunk.clear()
        if writer[0] is not None:
            writer[0].close()

    pbar.close()

    print(f"\nCollected {total_frames} frames from {len(episodes_seen)} episodes.")

    # --- task name mapping ---
    task_names = {
        str(i): LIBERO_TASK_NAMES.get(i, f"task_{i}")
        for i in range(task_lo, task_hi + 1)
    }
    with open(out_dir / "task_names.json", "w") as f:
        json.dump(task_names, f, indent=2)

    # --- summary ---
    print(f"Parquet → {parquet_path}")
    print(f"Task names → {out_dir}/task_names.json")

    # count episodes per task from the records we wrote
    # (we need to re-scan the parquet for this — cheap, just reads metadata)
    tbl = pq.read_table(str(parquet_path), columns=["task_index", "episode_index"])
    df_summary = tbl.to_pandas()
    ep_task = df_summary.groupby("task_index")["episode_index"].nunique()
    print("\nEpisodes per task:")
    for t, n in ep_task.items():
        print(f"  task {int(t):2d}: {n:3d} ep  — {task_names.get(str(int(t)), '?')}")


if __name__ == "__main__":
    main()
