"""
download_so100.py
=================
Download the lerobot/svla_so100_pickplace dataset from HuggingFace Hub.

This script:
  1. Downloads the dataset via snapshot_download
  2. Reads meta/info.json to detect action/state dims and camera names
  3. Reads meta/tasks.jsonl for task descriptions
  4. Merges all parquet shards into a single data.parquet
  5. Saves task_names.json  (task_index → description)
  6. Saves metadata.json    (dims, column names — used by dataset_so100.py)

Usage:
    python download_so100.py
    python download_so100.py --output data/so100_pickplace
    python download_so100.py --token YOUR_HF_TOKEN
    python download_so100.py --check
"""

import argparse
import json
import os
import sys
from pathlib import Path

DATASET_ID  = "lerobot/svla_so100_pickplace"
DEFAULT_OUT = "data/so100_pickplace"


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download lerobot/svla_so100_pickplace from HuggingFace"
    )
    p.add_argument("--output", default=DEFAULT_OUT,
                   help=f"Directory to save dataset (default: {DEFAULT_OUT})")
    p.add_argument("--token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--check", action="store_true",
                   help="Check download status without downloading")
    return p.parse_args()


def check_installed():
    missing = [lib for lib in ("huggingface_hub", "pyarrow", "pandas")
               if not _can_import(lib)]
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Install:  pip install huggingface_hub pyarrow pandas")
        sys.exit(1)


def _can_import(lib):
    try:
        __import__(lib)
        return True
    except ImportError:
        return False


# ── schema detection ───────────────────────────────────────────────────────

def detect_schema_from_info(cache_path: Path) -> dict:
    """Parse meta/info.json to get feature shapes and camera names."""
    info_path = cache_path / "meta" / "info.json"
    if not info_path.exists():
        print("  [WARN] meta/info.json not found — will infer schema from parquet")
        return {}

    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})
    schema = {}

    # Action dim
    if "action" in features:
        shape = features["action"].get("shape", [])
        if shape:
            schema["action_dim"] = shape[-1] if isinstance(shape, list) else shape
            print(f"  action_dim  = {schema['action_dim']}  (from info.json)")

    # State dim
    state_key = next(
        (k for k in features if k in ("observation.state", "state")), None
    )
    if state_key:
        shape = features[state_key].get("shape", [])
        if shape:
            schema["state_dim"] = shape[-1] if isinstance(shape, list) else shape
            schema["state_col"] = state_key
            print(f"  state_dim   = {schema['state_dim']}  (from info.json)")

    # Image columns
    img_cols = sorted(k for k in features if k.startswith("observation.images."))
    if img_cols:
        schema["image_cols"] = img_cols
        print(f"  image_cols  = {img_cols}")

    return schema


def detect_schema_from_parquet(parquet_path: Path) -> dict:
    """Fallback: infer dims from the parquet schema + first row."""
    import pyarrow.parquet as pq
    import pandas as pd

    print("  Inferring schema from parquet ...")
    pf_schema = pq.read_schema(str(parquet_path))
    col_names = pf_schema.names

    # Find image columns
    img_cols = sorted(c for c in col_names if "images" in c)

    # Read first row to get dims
    table = pq.read_table(str(parquet_path), filters=[("episode_index", "=", 0)])
    if len(table) == 0:
        table = pq.read_table(str(parquet_path)).slice(0, 1)
    df = table.to_pandas()

    schema = {}

    if "action" in df.columns:
        first_action = df["action"].iloc[0]
        if hasattr(first_action, "__len__"):
            schema["action_dim"] = len(first_action)
            print(f"  action_dim  = {schema['action_dim']}  (inferred)")

    state_key = next(
        (c for c in ("observation.state", "state") if c in df.columns), None
    )
    if state_key:
        first_state = df[state_key].iloc[0]
        if hasattr(first_state, "__len__"):
            schema["state_dim"]  = len(first_state)
            schema["state_col"]  = state_key
            print(f"  state_dim   = {schema['state_dim']}  (inferred)")

    if img_cols:
        schema["image_cols"] = img_cols
        print(f"  image_cols  = {img_cols}")

    return schema


# ── task names ─────────────────────────────────────────────────────────────

def build_task_names(cache_path: Path, parquet_path: Path) -> dict:
    """Build task_index → task description mapping.

    Tries (in order):
      1. meta/tasks.jsonl  (lerobot v2 format)
      2. meta/episodes.jsonl  (older format, episode → task mapping)
      3. 'task' column in parquet
      4. 'language_instruction' column in parquet
    """
    import pyarrow.parquet as pq

    # ── 1. tasks.jsonl ────────────────────────────────────────────────
    tasks_jsonl = cache_path / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        task_map = {}
        with open(tasks_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                task_map[str(obj["task_index"])] = obj["task"]
        if task_map:
            print(f"  Found {len(task_map)} tasks in meta/tasks.jsonl")
            return task_map

    # ── 2. Parquet task / language_instruction column ─────────────────
    schema = pq.read_schema(str(parquet_path))
    for col in ("task", "language_instruction"):
        if col in schema.names:
            table = pq.read_table(str(parquet_path),
                                  columns=["task_index", col])
            df = table.to_pandas()
            task_map = (
                df.drop_duplicates("task_index")
                  .set_index("task_index")[col]
                  .to_dict()
            )
            task_map = {str(int(k)): str(v) for k, v in task_map.items()}
            print(f"  Found {len(task_map)} tasks from parquet column '{col}'")
            return task_map

    # ── 3. Single pick-and-place task ─────────────────────────────────
    # The svla_so100_pickplace dataset has one task
    print("  [WARN] No task descriptions found — using generic description")
    return {"0": "pick up the object and place it in the target location"}


# ── main download ──────────────────────────────────────────────────────────

def download(out_dir: Path, token=None) -> bool:
    from huggingface_hub import snapshot_download
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out  = out_dir / "data.parquet"
    tasks_out    = out_dir / "task_names.json"
    metadata_out = out_dir / "metadata.json"

    print(f"\n{'─'*60}")
    print(f"  Dataset : {DATASET_ID}")
    print(f"  Output  : {out_dir}")
    print(f"{'─'*60}")

    if parquet_out.exists() and tasks_out.exists() and metadata_out.exists():
        print("  [skip] Already downloaded.")
        with open(metadata_out) as f:
            meta = json.load(f)
        print(f"  action_dim={meta.get('action_dim')}  "
              f"state_dim={meta.get('state_dim')}  "
              f"image_cols={meta.get('image_cols')}")
        return True

    # ── 1. Download ───────────────────────────────────────────────────
    print("  Downloading from HuggingFace Hub ...")
    try:
        cache_dir = snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=token or os.environ.get("HF_TOKEN"),
            ignore_patterns=["*.mp4", "*.avi", "videos/*"],
        )
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False

    cache_path = Path(cache_dir)
    print(f"  Cached at: {cache_path}")

    # ── 2. Detect schema from info.json ───────────────────────────────
    print("  Detecting schema ...")
    schema_info = detect_schema_from_info(cache_path)

    # ── 3. Merge parquet shards ───────────────────────────────────────
    if not parquet_out.exists():
        parquet_files = sorted(cache_path.rglob("*.parquet"))
        if not parquet_files:
            print("  [ERROR] No parquet files found.")
            return False

        print(f"  Found {len(parquet_files)} parquet shard(s).")

        if len(parquet_files) == 1:
            shutil.copy2(parquet_files[0], parquet_out)
            print(f"  Copied → {parquet_out}")
        else:
            print(f"  Merging {len(parquet_files)} shards ...")
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

    # ── 4. Fallback schema detection from parquet ─────────────────────
    if "action_dim" not in schema_info:
        schema_info.update(detect_schema_from_parquet(parquet_path=parquet_out))

    # ── 5. Task names ─────────────────────────────────────────────────
    if not tasks_out.exists():
        print("  Building task_names.json ...")
        task_map = build_task_names(cache_path, parquet_out)
        with open(tasks_out, "w") as f:
            json.dump(task_map, f, indent=2)
        print(f"  Saved {len(task_map)} task(s) → {tasks_out}")

    # ── 6. Validate and complete metadata ─────────────────────────────
    pq_meta = pq.read_metadata(str(parquet_out))
    pq_schema = pq.read_schema(str(parquet_out))

    n_rows   = pq_meta.num_rows
    n_rg     = pq_meta.num_row_groups
    all_cols = pq_schema.names

    # Count episodes
    try:
        t = pq.read_table(str(parquet_out), columns=["episode_index"])
        import pandas as pd
        n_episodes = int(t.to_pandas()["episode_index"].nunique())
    except Exception:
        n_episodes = -1

    # Determine image columns (in decreasing priority)
    if "image_cols" not in schema_info:
        schema_info["image_cols"] = sorted(
            c for c in all_cols if "images" in c
        )

    image_cols = schema_info.get("image_cols", [])

    # Determine state column
    state_col = schema_info.get("state_col", "observation.state")
    if state_col not in all_cols:
        # try alternate
        state_col = next((c for c in all_cols if "state" in c), "observation.state")
        schema_info["state_col"] = state_col

    # ── 7. Save metadata.json ─────────────────────────────────────────
    metadata = {
        "dataset_id":   DATASET_ID,
        "n_frames":     n_rows,
        "n_row_groups": n_rg,
        "n_episodes":   n_episodes,
        "action_dim":   schema_info.get("action_dim", 6),
        "state_dim":    schema_info.get("state_dim",  6),
        "state_col":    schema_info.get("state_col",  "observation.state"),
        "image_cols":   image_cols,
        "all_columns":  all_cols,
    }

    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata → {metadata_out}")

    # ── 8. Schema validation ──────────────────────────────────────────
    print("  Validating schema ...")
    required = {"episode_index", "frame_index", "action"}
    missing  = required - set(all_cols)
    if missing:
        print(f"  [WARN] Missing columns: {missing}")
    else:
        print("  Schema OK.")

    if len(image_cols) == 0:
        print("  [WARN] No image columns found — check column names manually.")
    elif len(image_cols) == 1:
        print(f"  [INFO] Only 1 camera found: {image_cols[0]}")
        print(f"         Both image slots will use this camera during training.")
    else:
        print(f"  [INFO] {len(image_cols)} cameras: {image_cols}")
        print(f"         Training will use the first two: {image_cols[:2]}")

    print(f"\n  Done!")
    print(f"  Frames     : {n_rows:,}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  action_dim : {metadata['action_dim']}")
    print(f"  state_dim  : {metadata['state_dim']}")
    print(f"  image_cols : {image_cols}")
    print(f"\nNext steps:")
    print(f"  1. Compute normalization stats:")
    print(f"       python normalize.py --data {out_dir}")
    print(f"  2. Evaluate pretrained baseline:")
    print(f"       python evaluate_so100.py --checkpoint lerobot/smolvla_base \\")
    print(f"           --data {out_dir} --out results/so100_pretrained.json")
    print(f"  3. Fine-tune:")
    print(f"       python train_so100.py --data {out_dir} \\")
    print(f"           --output checkpoints/so100_run --steps 5000 --bf16")
    print(f"  4. Evaluate finetuned model:")
    print(f"       python evaluate_so100.py --checkpoint checkpoints/so100_run/final \\")
    print(f"           --data {out_dir} --out results/so100_finetuned.json")
    print(f"  5. Compare:")
    print(f"       python compare_so100.py results/so100_pretrained.json results/so100_finetuned.json")

    return True


# ── check status ───────────────────────────────────────────────────────────

def check_status(out_dir: Path):
    parquet = out_dir / "data.parquet"
    tasks   = out_dir / "task_names.json"
    meta    = out_dir / "metadata.json"

    print(f"\nDownload status for {out_dir}:")
    if parquet.exists():
        size_mb = parquet.stat().st_size / 1e6
        print(f"  data.parquet    : {size_mb:,.0f} MB")
    else:
        print(f"  data.parquet    : NOT FOUND")

    if tasks.exists():
        with open(tasks) as f:
            t = json.load(f)
        print(f"  task_names.json : {len(t)} task(s)")
    else:
        print(f"  task_names.json : NOT FOUND")

    if meta.exists():
        with open(meta) as f:
            m = json.load(f)
        print(f"  metadata.json   : action_dim={m.get('action_dim')}  "
              f"state_dim={m.get('state_dim')}  "
              f"image_cols={m.get('image_cols')}")
    else:
        print(f"  metadata.json   : NOT FOUND")

    norm = out_dir / "norm_stats.json"
    if norm.exists():
        print(f"  norm_stats.json : present")
    else:
        print(f"  norm_stats.json : NOT FOUND  (run normalize.py)")
    print()


# ── entry point ────────────────────────────────────────────────────────────

def main():
    check_installed()
    args = parse_args()
    out_dir = Path(args.output)

    if args.check:
        check_status(out_dir)
        return

    print(f"SmolVLA SO-100 Pick-and-Place Downloader")
    print(f"Dataset : {DATASET_ID}")
    print(f"Output  : {out_dir}\n")

    ok = download(out_dir, token=args.token)
    if not ok:
        print("[ERROR] Download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
