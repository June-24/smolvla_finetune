"""
download.py
===========
Download LIBERO dataset chunks from HuggingFace Hub.

The LIBERO 40-task benchmark is split into 4 suites of 10 tasks each.
Each suite corresponds to one chunk directory in this codebase.

Suite mapping:
    spatial → data/chunk_0to9    (10 spatial-relationship tasks)
    object  → data/chunk_10to19  (10 object-manipulation tasks)
    goal    → data/chunk_20to29  (10 goal-condition tasks)
    long    → data/chunk_30to39  (10 long-horizon tasks)

Usage:
    # Download all 4 chunks
    python download.py

    # Download a single chunk (fastest way to get started)
    python download.py --chunks spatial

    # Download to a custom root
    python download.py --output /path/to/data

    # Verify what's already downloaded without re-downloading
    python download.py --check
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── HuggingFace dataset IDs (verified against lerobot Hub repos) ───────────
CHUNK_META = {
    "spatial": {
        "dataset_id": "lerobot/libero_spatial",
        "local_dir":  "chunk_0to9",
        "description": "LIBERO-Spatial: 10 tasks about spatial relationships",
    },
    "object": {
        "dataset_id": "lerobot/libero_object",
        "local_dir":  "chunk_10to19",
        "description": "LIBERO-Object: 10 tasks about object properties",
    },
    "goal": {
        "dataset_id": "lerobot/libero_goal",
        "local_dir":  "chunk_20to29",
        "description": "LIBERO-Goal: 10 tasks about goal conditions",
    },
    "long": {
        "dataset_id": "lerobot/libero_100",
        "local_dir":  "chunk_30to39",
        "description": "LIBERO-Long: 10 long-horizon manipulation tasks",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Download LIBERO dataset chunks from HuggingFace"
    )
    p.add_argument(
        "--chunks",
        nargs="+",
        choices=list(CHUNK_META.keys()),
        default=list(CHUNK_META.keys()),
        help="Which chunks to download (default: all 4)",
    )
    p.add_argument(
        "--output",
        default="data",
        help="Root directory to save chunks into (default: data/)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Just check which chunks are already downloaded, don't download",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var). "
             "Required only if the dataset is gated.",
    )
    return p.parse_args()


def check_installed():
    """Ensure required libraries are available."""
    missing = []
    for lib in ("huggingface_hub", "pyarrow", "pandas"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Install them with:  pip install huggingface_hub pyarrow pandas")
        sys.exit(1)


def download_chunk(dataset_id: str, out_dir: Path, token=None) -> bool:
    """
    Download a lerobot dataset repo from HuggingFace Hub into out_dir.

    The dataset is stored as:
        out_dir/
            data.parquet      ← merged from all train parquet shards
            task_names.json   ← task index → description mapping
    """
    from huggingface_hub import snapshot_download
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    import tempfile

    print(f"\n{'─'*60}")
    print(f"  Dataset : {dataset_id}")
    print(f"  Output  : {out_dir}")
    print(f"{'─'*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "data.parquet"
    tasks_out   = out_dir / "task_names.json"

    if parquet_out.exists() and tasks_out.exists():
        print("  [skip] Already downloaded.")
        return True

    # ── 1. Download the full dataset repo ─────────────────────────────────
    print("  Downloading from HuggingFace Hub …")
    try:
        cache_dir = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            token=token or os.environ.get("HF_TOKEN"),
            ignore_patterns=["*.mp4", "*.avi", "videos/*"],
        )
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Make sure '{dataset_id}' exists on HuggingFace Hub.")
        return False

    cache_path = Path(cache_dir)
    print(f"  Cached at: {cache_path}")

    # ── 2. Find all parquet shards ─────────────────────────────────────────
    # lerobot datasets store shards at  data/train-NNNNN-of-NNNNN.parquet
    parquet_files = sorted(cache_path.rglob("*.parquet"))
    if not parquet_files:
        print("  [ERROR] No parquet files found in downloaded repo.")
        return False

    print(f"  Found {len(parquet_files)} parquet shard(s).")

    # ── 3. Merge shards → single data.parquet ─────────────────────────────
    if len(parquet_files) == 1 and not parquet_out.exists():
        # Single shard — just copy
        import shutil
        shutil.copy2(parquet_files[0], parquet_out)
        print(f"  Copied shard → {parquet_out}")
    elif not parquet_out.exists():
        print(f"  Merging {len(parquet_files)} shards …")
        writer = None
        total_rows = 0
        for shard in parquet_files:
            pf = pq.ParquetFile(shard)
            for rg_i in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_i)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_out, table.schema)
                writer.write_table(table)
                total_rows += len(table)
        if writer:
            writer.close()
        print(f"  Merged {total_rows:,} rows → {parquet_out}")

    # ── 4. Build task_names.json ───────────────────────────────────────────
    if not tasks_out.exists():
        print("  Building task_names.json …")
        # Read only the task columns
        try:
            table = pq.read_table(str(parquet_out),
                                  columns=["task_index", "task"])
            df = table.to_pandas()
            task_map = (
                df.drop_duplicates("task_index")
                  .set_index("task_index")["task"]
                  .to_dict()
            )
            # Keys must be strings for JSON
            task_map = {str(int(k)): str(v) for k, v in task_map.items()}
        except Exception:
            # Fallback: try "language_instruction" column (older lerobot schema)
            try:
                table = pq.read_table(
                    str(parquet_out),
                    columns=["task_index", "language_instruction"],
                )
                df = table.to_pandas()
                task_map = (
                    df.drop_duplicates("task_index")
                      .set_index("task_index")["language_instruction"]
                      .to_dict()
                )
                task_map = {str(int(k)): str(v) for k, v in task_map.items()}
            except Exception as e:
                print(f"  [WARN] Could not build task_names.json: {e}")
                print("  Creating empty task_names.json — edit it manually.")
                task_map = {}

        with open(tasks_out, "w") as f:
            json.dump(task_map, f, indent=2)
        print(f"  Saved {len(task_map)} task names → {tasks_out}")

    # ── 5. Quick schema validation ─────────────────────────────────────────
    print("  Validating schema …")
    required_cols = {
        "episode_index", "frame_index", "action",
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }
    meta = pq.read_metadata(str(parquet_out))
    schema_cols = {meta.row_group(0).column(i).path_in_schema
                   for i in range(meta.row_group(0).num_columns)}
    # Also read schema columns
    schema = pq.read_schema(str(parquet_out))
    schema_names = set(schema.names)

    missing_cols = required_cols - schema_names
    if missing_cols:
        print(f"  [WARN] Missing expected columns: {missing_cols}")
        print("  The dataset schema may differ from what LiberoDataset expects.")
        print("  You may need to rename columns — see dataset.py for expected names.")
    else:
        print("  Schema OK.")

    print(f"\n  Done! Chunk saved to: {out_dir}")
    return True


def print_status(output_root: Path):
    """Print download status for all chunks."""
    print("\nDownload status:")
    print(f"  Root: {output_root}\n")
    for name, meta in CHUNK_META.items():
        out = output_root / meta["local_dir"]
        parquet = out / "data.parquet"
        tasks   = out / "task_names.json"
        if parquet.exists() and tasks.exists():
            size_mb = parquet.stat().st_size / 1e6
            status = f"✓  {size_mb:,.0f} MB"
        elif parquet.exists():
            status = "⚠  parquet present but missing task_names.json"
        else:
            status = "✗  not downloaded"
        print(f"  {name:10s} ({meta['local_dir']:15s})  {status}")
    print()


def main():
    check_installed()
    args = parse_args()
    output_root = Path(args.output)

    if args.check:
        print_status(output_root)
        return

    print("SmolVLA LIBERO Downloader")
    print(f"Chunks to download: {args.chunks}")
    print(f"Output root: {output_root}\n")

    results = {}
    for chunk_name in args.chunks:
        meta = CHUNK_META[chunk_name]
        out_dir = output_root / meta["local_dir"]
        print(f"[{chunk_name}] {meta['description']}")
        ok = download_chunk(meta["dataset_id"], out_dir, token=args.token)
        results[chunk_name] = ok

    print("\n" + "="*60)
    print("Summary:")
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        local = CHUNK_META[name]["local_dir"]
        print(f"  {icon}  {name:10s} → {output_root / local}")

    all_ok = all(results.values())
    if all_ok:
        print("\nAll chunks downloaded successfully.")
        print("\nNext steps:")
        print("  1. Compute normalization stats:")
        for name, ok in results.items():
            if ok:
                d = output_root / CHUNK_META[name]["local_dir"]
                print(f"       python normalize.py --data {d}")
        print("\n  2. Start training:")
        first = next(k for k, v in results.items() if v)
        d = output_root / CHUNK_META[first]["local_dir"]
        print(f"       python train.py --data {d} --output checkpoints/run1 \\")
        print(f"           --from_pretrained lerobot/smolvla_base --steps 5000 --bf16")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n[ERROR] Failed to download: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
