"""
download_libero.py
==================
Downloads a small subset of LIBERO from HuggingFace Hub and saves it
as a local Parquet file for training.

Usage:
    python download_libero.py [--episodes 20] [--tasks 0-9] [--out data/libero_subset]

The dataset is streamed — only the requested episodes are downloaded.
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
    p.add_argument("--episodes", type=int, default=20,
                   help="Max number of episodes to download (default: 20)")
    p.add_argument("--tasks", type=str, default="0-9",
                   help="Task index range, e.g. '0-9' (default: 0-9)")
    p.add_argument("--out", type=str, default="data/libero_subset",
                   help="Output directory (default: data/libero_subset)")
    return p.parse_args()


def parse_task_range(s: str) -> tuple:
    parts = s.split("-")
    if len(parts) == 1:
        v = int(parts[0])
        return v, v
    return int(parts[0]), int(parts[1])


def pil_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    args = parse_args()
    task_lo, task_hi = parse_task_range(args.tasks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming LIBERO from HuggingFace Hub...")
    print(f"  task range : {task_lo}-{task_hi}")
    print(f"  max episodes: {args.episodes}")
    print(f"  output      : {out_dir}")
    print()

    ds = load_dataset("HuggingFaceVLA/libero", split="train", streaming=True)

    episodes_seen: set = set()
    rows = []

    pbar = tqdm(desc="Collecting frames", unit="frames")

    for sample in ds:
        task_idx = int(sample["task_index"])
        ep_idx   = int(sample["episode_index"])

        # skip tasks outside requested range
        if not (task_lo <= task_idx <= task_hi):
            # if we have enough episodes and we're past them, stop
            if len(episodes_seen) >= args.episodes:
                break
            continue

        # if this is a new episode and we already have enough, stop
        if ep_idx not in episodes_seen:
            if len(episodes_seen) >= args.episodes:
                break
            episodes_seen.add(ep_idx)

        rows.append(sample)
        pbar.update(1)

    pbar.close()
    print(f"\nCollected {len(rows)} frames from {len(episodes_seen)} episodes.")

    # --- convert PIL images to PNG bytes -----------------------------------
    print("Converting images...")
    processed = []
    for row in tqdm(rows):
        r = dict(row)
        img1 = r.get("observation.images.image")
        img2 = r.get("observation.images.image2")
        if hasattr(img1, "save"):
            r["observation.images.image"]  = pil_to_bytes(img1)
            r["observation.images.image2"] = pil_to_bytes(img2)
        processed.append(r)

    # --- save task name mapping --------------------------------------------
    task_names = {
        str(i): LIBERO_TASK_NAMES.get(i, f"task_{i}")
        for i in range(task_lo, task_hi + 1)
    }
    with open(out_dir / "task_names.json", "w") as f:
        json.dump(task_names, f, indent=2)

    # --- write parquet -----------------------------------------------------
    print("Writing Parquet...")
    df = pd.DataFrame(processed)

    # pyarrow can't auto-infer binary from mixed types; cast explicitly
    for col in ["observation.images.image", "observation.images.image2"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, bytes) else bytes(x))

    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_dir / "data.parquet")

    print(f"\nDone! Saved to {out_dir}/data.parquet")
    print(f"Task names  → {out_dir}/task_names.json")
    print()

    # summary
    ep_task: dict = {}
    for row in processed:
        t = row["task_index"]
        ep_task.setdefault(t, set()).add(row["episode_index"])
    print("Episodes per task:")
    for t in sorted(ep_task):
        print(f"  task {t:2d}: {len(ep_task[t]):3d} ep  — {task_names[str(t)]}")


if __name__ == "__main__":
    main()
