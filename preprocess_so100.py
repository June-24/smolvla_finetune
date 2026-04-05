"""
preprocess_so100.py
===================
Download the SO-100 video files and extract frames into the parquet.

lerobot v2 stores images as MP4 files, NOT embedded in the parquet.
Video structure (actual format):
    videos/{camera_key}/chunk-{N:03d}/file-{N:03d}.mp4
    e.g. videos/observation.images.top/chunk-000/file-000.mp4

All 50 episodes are concatenated into a single video per camera.
The parquet's `index` column (0-based global frame index) maps directly
to the frame position within the concatenated video.

This script:
  1. Downloads the video files (re-uses the already-downloaded parquet)
  2. Reads each video sequentially (no seeking — efficient)
  3. Writes a new data.parquet with PNG-byte image columns embedded
     (same format as LIBERO, compatible with dataset_so100.py)

Run ONCE after download_so100.py:
    python preprocess_so100.py --data data/so100_pickplace

After this, run:
    python evaluate_so100.py --checkpoint lerobot/smolvla_base ...
    python train_so100.py ...
"""

import argparse
import io
import json
import os
from pathlib import Path

from tqdm import tqdm

DATASET_ID = "lerobot/svla_so100_pickplace"


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract SO-100 video frames into the parquet"
    )
    p.add_argument("--data",   default="data/so100_pickplace")
    p.add_argument("--token",  default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--resize", type=int, default=256,
                   help="Resize frames to this square size (default: 256)")
    return p.parse_args()


# ── Video discovery ────────────────────────────────────────────────────────

def find_video_files(cache_path: Path, camera_key: str) -> list:
    """
    Find and sort all MP4 chunk files for a camera.

    Actual lerobot v2 structure:
        videos/{camera_key}/chunk-{N:03d}/file-{N:03d}.mp4

    The camera_key directory uses the EXACT feature name (with dots).
    Returns files sorted by chunk index so frames are in global order.
    """
    # Try direct dotted name first (observed in actual dataset)
    video_root = cache_path / "videos" / camera_key
    if not video_root.exists():
        # Fallback: underscore variant
        video_root = cache_path / "videos" / camera_key.replace(".", "_")
    if not video_root.exists():
        return []

    mp4s = sorted(video_root.rglob("*.mp4"))
    return mp4s


# ── Frame extraction ───────────────────────────────────────────────────────

def extract_frames_sequential(video_path: Path, resize: int,
                               desc: str = "") -> list:
    """
    Read ALL frames from a video sequentially using PyAV.
    Returns a list of PNG bytes, one per frame, in display order.
    PyAV streams frames without loading the full video into RAM.
    """
    import av
    from PIL import Image

    frames = []
    label = desc or video_path.name

    container = av.open(str(video_path))
    stream    = container.streams.video[0]
    n_frames  = stream.frames   # may be 0 if not in container metadata

    pbar = tqdm(total=n_frames if n_frames > 0 else None,
                desc=label, leave=False, unit="fr")

    for packet in container.demux(stream):
        for frame in packet.decode():
            # frame.to_image() → PIL Image (RGB)
            img = frame.to_image().convert("RGB")
            if resize:
                img = img.resize((resize, resize), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            frames.append(buf.getvalue())
            pbar.update(1)

    pbar.close()
    container.close()
    return frames


def _pil_to_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    data_dir = Path(args.data)

    # ── Validate ───────────────────────────────────────────────────────
    parquet_path = data_dir / "data.parquet"
    meta_path    = data_dir / "metadata.json"
    for p in (parquet_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run download_so100.py first."
            )

    with open(meta_path) as f:
        meta = json.load(f)

    image_cols = meta.get("image_cols", [])
    if not image_cols:
        raise ValueError("No image_cols in metadata.json.")

    import pyarrow.parquet as pq
    existing_cols = set(pq.read_schema(str(parquet_path)).names)
    if all(col in existing_cols for col in image_cols):
        print("Images already embedded in parquet — nothing to do.")
        print("Run evaluate_so100.py or train_so100.py directly.")
        return

    print(f"Image columns to embed: {image_cols}\n")

    # ── Download videos ────────────────────────────────────────────────
    from huggingface_hub import snapshot_download
    print("Downloading video files ...")
    print("(The numerical parquet is already present — only downloading MP4s.)\n")

    try:
        cache_dir = snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=args.token or os.environ.get("HF_TOKEN"),
            allow_patterns=["videos/**", "meta/**"],
            ignore_patterns=["*.git*"],
        )
    except Exception as e:
        raise RuntimeError(f"Video download failed: {e}") from e

    cache_path = Path(cache_dir)

    # ── Discover video files ────────────────────────────────────────────
    print("Discovering video files ...")
    cam_videos = {}
    for cam_key in image_cols:
        files = find_video_files(cache_path, cam_key)
        cam_videos[cam_key] = files
        print(f"  {cam_key}: {len(files)} file(s)")
        for f in files:
            print(f"    {f.relative_to(cache_path)}")

    if not all(cam_videos[k] for k in image_cols):
        missing = [k for k in image_cols if not cam_videos[k]]
        raise FileNotFoundError(
            f"No MP4 files found for cameras: {missing}\n"
            f"Searched under: {cache_path / 'videos'}\n"
            f"Available MP4s: {sorted(cache_path.rglob('*.mp4'))}"
        )

    # ── Load numerical parquet ─────────────────────────────────────────
    import pyarrow as pa
    import pandas as pd

    print("\nLoading numerical parquet ...")
    num_df = pq.read_table(str(parquet_path)).to_pandas()

    # Sort by global frame index so rows align with video frame order
    num_df = num_df.sort_values("index").reset_index(drop=True)
    n_frames  = len(num_df)
    n_episodes = num_df["episode_index"].nunique()
    print(f"  {n_frames} frames, {n_episodes} episodes")
    print(f"  index range: {num_df['index'].min()} – {num_df['index'].max()}")

    # ── Extract all frames for each camera ─────────────────────────────
    # The `index` column is 0-based and matches the frame position in the
    # concatenated video. We read all video chunks in order and build
    # a flat array of PNG bytes indexed by global frame index.

    print(f"\nExtracting frames (resize to {args.resize}×{args.resize}) ...")

    cam_frame_bytes: dict[str, list] = {}

    for cam_key in image_cols:
        print(f"\n  Camera: {cam_key}")
        all_bytes = []
        for vid_file in cam_videos[cam_key]:
            chunk_bytes = extract_frames_sequential(
                vid_file, args.resize,
                desc=f"  {cam_key} / {vid_file.name}"
            )
            all_bytes.extend(chunk_bytes)
            print(f"    {vid_file.name}: {len(chunk_bytes)} frames")

        print(f"    Total: {len(all_bytes)} frames (expected {n_frames})")
        if len(all_bytes) < n_frames:
            print(f"    [WARN] Video has fewer frames than parquet rows. "
                  f"Last frame will be repeated.")
            # Pad with last frame
            all_bytes.extend([all_bytes[-1]] * (n_frames - len(all_bytes)))
        elif len(all_bytes) > n_frames:
            print(f"    [WARN] Video has more frames than parquet rows. Truncating.")
            all_bytes = all_bytes[:n_frames]

        cam_frame_bytes[cam_key] = all_bytes

    # ── Write new parquet (episode-by-episode row groups) ──────────────
    out_path = data_dir / "_data_with_images.parquet"
    print(f"\nWriting parquet with embedded images ...")

    # Build schema: original numerical fields + image binary fields
    schema_fields = list(pq.read_schema(str(parquet_path)))
    for cam_key in image_cols:
        schema_fields.append(pa.field(cam_key, pa.large_binary()))
    out_schema = pa.schema(schema_fields)

    writer = pq.ParquetWriter(str(out_path), out_schema)

    episodes = sorted(num_df["episode_index"].unique())
    for ep_idx in tqdm(episodes, desc="Writing episodes"):
        ep_mask  = num_df["episode_index"] == ep_idx
        ep_df    = num_df[ep_mask].reset_index(drop=True)
        ep_rows  = ep_df["index"].tolist()   # global frame indices

        arrays = []
        for field in out_schema:
            col_name = field.name
            if col_name in image_cols:
                png_list = [cam_frame_bytes[col_name][gi] for gi in ep_rows]
                arrays.append(pa.array(png_list, type=pa.large_binary()))
            else:
                col_data = ep_df[col_name].tolist()
                arrays.append(pa.array(col_data, type=field.type))

        ep_table = pa.table(arrays, schema=out_schema)
        writer.write_table(ep_table)

    writer.close()
    print(f"Wrote {len(episodes)} row groups → {out_path}")

    # ── Swap parquets ──────────────────────────────────────────────────
    import shutil
    backup_path = data_dir / "data_no_images.parquet"
    shutil.move(str(parquet_path), str(backup_path))
    shutil.move(str(out_path), str(parquet_path))
    print(f"\nBacked up original → {backup_path.name}")
    print(f"New parquet (with images) → {parquet_path.name}")

    # ── Update metadata.json ───────────────────────────────────────────
    meta["all_columns"] = list(existing_cols) + [
        c for c in image_cols if c not in existing_cols
    ]
    meta["n_row_groups"] = len(episodes)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Validate ──────────────────────────────────────────────────────
    print("\nValidating new parquet ...")
    new_schema = pq.read_schema(str(parquet_path))
    all_ok = True
    for col in image_cols:
        ok = col in new_schema.names
        print(f"  {col}: {'OK' if ok else 'MISSING'}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll done! Run:")
        print(f"  python evaluate_so100.py \\")
        print(f"      --checkpoint lerobot/smolvla_base \\")
        print(f"      --data {data_dir} \\")
        print(f"      --out results/so100_pretrained.json")
    else:
        print("\n[ERROR] Some image columns are missing from the new parquet.")


if __name__ == "__main__":
    main()
