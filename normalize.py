"""
normalize.py
============
Computes mean/std normalization statistics for actions and states
from the downloaded LIBERO Parquet file.

Run this ONCE before training:
    python normalize.py --data data/libero_subset --out data/libero_subset/norm_stats.json

The output JSON is loaded by train.py and passed to LiberoDataset.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_stats(data_dir: str, out_path: Optional[str] = None):
    data_dir = Path(data_dir)
    out_path = Path(out_path) if out_path else data_dir / "norm_stats.json"

    pq_path = data_dir / "data.parquet"
    print(f"Loading {pq_path} (columns: action, observation.state only) ...")

    # Only read the two small numeric columns — avoids loading image bytes into RAM
    import pyarrow.parquet as pq
    table = pq.read_table(str(pq_path), columns=["action", "observation.state"])
    df = table.to_pandas()

    # ---- actions -------------------------------------------------------
    actions = np.array(df["action"].tolist(), dtype=np.float32)   # (N, 7)
    action_mean = actions.mean(axis=0).tolist()
    action_std  = actions.std(axis=0).tolist()

    # ---- states --------------------------------------------------------
    states = np.array(df["observation.state"].tolist(), dtype=np.float32)  # (N, 8)
    state_mean = states.mean(axis=0).tolist()
    state_std  = states.std(axis=0).tolist()

    stats = {
        "action": {"mean": action_mean, "std": action_std},
        "state":  {"mean": state_mean,  "std": state_std},
    }

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNormalization stats saved to {out_path}")
    print(f"  action mean: {[f'{v:.4f}' for v in action_mean]}")
    print(f"  action std:  {[f'{v:.4f}' for v in action_std]}")
    print(f"  state  mean: {[f'{v:.4f}' for v in state_mean]}")
    print(f"  state  std:  {[f'{v:.4f}' for v in state_std]}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/libero_subset")
    parser.add_argument("--out",  default=None)
    args = parser.parse_args()
    compute_stats(args.data, args.out)


if __name__ == "__main__":
    main()
