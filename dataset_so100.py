"""
dataset_so100.py
================
PyTorch Dataset for the lerobot/svla_so100_pickplace dataset.

Reads metadata.json (written by download_so100.py) to auto-configure:
  - action_dim  (e.g. 6 for SO-100)
  - state_dim   (e.g. 6 for SO-100)
  - image_cols  (actual camera column names in the parquet)

Batch keys produced (same contract as LiberoDataset):
    observation.images.image              (B, 3, H, W)  float32 [0,1]
    observation.images.image2             (B, 3, H, W)  float32 [0,1]
    observation.state                     (B, state_dim) float32
    observation.language_tokens           (B, 48)        int64
    observation.language_attention_mask   (B, 48)        int64
    action                                (B, chunk_size, action_dim) float32
    action_is_pad                         (B, chunk_size)             bool
"""

import bisect
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


TOKENIZER_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MAX_TOKEN_LEN  = 48
IMAGE_SIZE     = 256


def _get_col_from_table(table, dotted_name: str):
    """
    Access a column from a pyarrow Table by dotted name.

    After preprocess_so100.py runs, image columns are flat (literal dotted
    names like 'observation.images.top'). This function handles both:
      - Flat column: table has a column literally named 'observation.images.top'
      - Nested struct: table has 'observation' struct → navigate with .field()
    """
    import pyarrow as pa

    # Fast path: exact name match
    if dotted_name in table.schema.names:
        col = table.column(dotted_name)
        return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col

    # Slow path: navigate nested struct (observation → images → top)
    parts = dotted_name.split(".")
    if parts[0] not in table.schema.names:
        available = table.schema.names
        raise KeyError(
            f"Column '{dotted_name}' not found in parquet.\n"
            f"Top-level columns: {available}\n"
            "Did you run preprocess_so100.py to embed images into the parquet?"
        )
    arr = table.column(parts[0])
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    for part in parts[1:]:
        arr = arr.field(part)   # navigate StructArray
    return arr


class SO100Dataset(Dataset):
    """
    Dataset for the SO-100 pick-and-place data produced by download_so100.py.

    Reads metadata.json to find:
      - action_dim, state_dim
      - image_cols: which parquet columns hold camera images
      - state_col: name of the joint-state column

    If only one camera column exists, both 'image' and 'image2' slots are
    filled from the same camera (the model can handle duplicate views).

    Images are loaded lazily (row-group caching) to keep RAM low.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 50,
        norm_stats: Optional[dict] = None,
        tokenizer_name: str = TOKENIZER_NAME,
    ):
        self.data_dir   = Path(data_dir)
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self._pq_path   = str(self.data_dir / "data.parquet")

        # ── Load metadata (written by download_so100.py) ──────────────
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {data_dir}. "
                "Run download_so100.py first."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self.action_dim = meta["action_dim"]
        self.state_dim  = meta["state_dim"]
        self.state_col  = meta.get("state_col", "observation.state")

        image_cols = meta.get("image_cols", [])
        if len(image_cols) == 0:
            raise ValueError(
                "No image columns found in metadata.json. "
                "Re-run download_so100.py to regenerate metadata."
            )
        # Map to the two slots we expose (duplicate if only 1 camera)
        self._img_col1 = image_cols[0]
        self._img_col2 = image_cols[1] if len(image_cols) > 1 else image_cols[0]

        # ── Small columns → pandas ────────────────────────────────────
        small_cols = ["episode_index", "frame_index", "task_index",
                      "action", self.state_col]
        # task_index may not always be present; gracefully omit if missing
        schema_cols = set(pq.read_schema(self._pq_path).names)
        small_cols  = [c for c in small_cols if c in schema_cols]

        self.df = pq.read_table(self._pq_path, columns=small_cols).to_pandas()

        # Ensure task_index column exists (default 0 if absent)
        if "task_index" not in self.df.columns:
            self.df["task_index"] = 0

        # ── Row-group offsets for lazy image loading ──────────────────
        rq_meta = pq.read_metadata(self._pq_path)
        self._rg_starts: list[int] = []
        offset = 0
        for rg in range(rq_meta.num_row_groups):
            self._rg_starts.append(offset)
            offset += rq_meta.row_group(rg).num_rows
        self._n_rows = offset

        self._pf: Optional[pq.ParquetFile] = None
        self._cached_rg_idx: int = -1
        self._cached_arr1 = None   # pre-extracted image arrays for current row group
        self._cached_arr2 = None

        # ── Task names ────────────────────────────────────────────────
        task_path = self.data_dir / "task_names.json"
        if task_path.exists():
            with open(task_path) as f:
                self.task_names: dict[str, str] = json.load(f)
        else:
            self.task_names = {"0": "pick up the object and place it"}

        # ── Episode index ─────────────────────────────────────────────
        self._build_episode_index()

        # ── Tokenizer ─────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(
            f"SO100Dataset: {len(self.df)} frames, "
            f"{len(self._episodes)} episodes, "
            f"action_dim={self.action_dim}, state_dim={self.state_dim}, "
            f"chunk_size={chunk_size}"
        )
        print(f"  camera1: {self._img_col1}")
        print(f"  camera2: {self._img_col2}")

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def _build_episode_index(self):
        self.df = self.df.sort_values(
            ["episode_index", "frame_index"]
        ).reset_index(drop=True)

        self._episodes: dict[int, list[int]] = {}
        for row_i, row in self.df.iterrows():
            ep = int(row["episode_index"])
            self._episodes.setdefault(ep, []).append(row_i)

        self._index: list[tuple[int, int]] = []
        for ep, rows in self._episodes.items():
            for local_i in range(len(rows)):
                self._index.append((ep, local_i))

    def __len__(self):
        return len(self._index)

    # ------------------------------------------------------------------
    # Lazy image loading
    # ------------------------------------------------------------------

    def _open_pf(self):
        if self._pf is None:
            self._pf = pq.ParquetFile(self._pq_path, memory_map=True)

    def _get_images_for_row(self, global_row: int):
        """Return (img1_bytes, img2_bytes) for a global row index."""
        self._open_pf()

        rg_idx    = bisect.bisect_right(self._rg_starts, global_row) - 1
        local_row = global_row - self._rg_starts[rg_idx]

        if rg_idx != self._cached_rg_idx:
            raw = self._pf.read_row_group(rg_idx)
            # Pre-extract image arrays so per-row access is O(1)
            self._cached_arr1 = _get_col_from_table(raw, self._img_col1)
            self._cached_arr2 = (self._cached_arr1 if self._img_col2 == self._img_col1
                                  else _get_col_from_table(raw, self._img_col2))
            self._cached_rg_idx = rg_idx

        img1 = self._cached_arr1[local_row].as_py()
        img2 = self._cached_arr2[local_row].as_py()
        return img1, img2

    # ------------------------------------------------------------------
    # Core sample loading
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        ep_id, local_i = self._index[idx]
        ep_rows        = self._episodes[ep_id]
        global_row     = ep_rows[local_i]
        row            = self.df.iloc[global_row]

        # ---- images ---------------------------------------------------
        img1_raw, img2_raw = self._get_images_for_row(global_row)
        img1 = self._decode_image(img1_raw)
        img2 = self._decode_image(img2_raw)

        # ---- state ----------------------------------------------------
        state_val = row[self.state_col]
        state     = torch.tensor(state_val, dtype=torch.float32)

        # ---- language tokens ------------------------------------------
        task_idx  = str(int(row["task_index"]))
        task_desc = self.task_names.get(task_idx, "perform the task") + "\n"
        tokens    = self.tokenizer(
            task_desc,
            padding="max_length",
            max_length=MAX_TOKEN_LEN,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokens["input_ids"].squeeze(0)
        lang_mask   = tokens["attention_mask"].squeeze(0)

        # ---- action chunk ---------------------------------------------
        actions, is_pad = self._load_action_chunk(ep_rows, local_i)

        # ---- normalise ------------------------------------------------
        if self.norm_stats is not None:
            state   = self._normalize(state,   self.norm_stats["state"])
            actions = self._normalize(actions, self.norm_stats["action"])

        return {
            "observation.images.image":            img1,
            "observation.images.image2":           img2,
            "observation.state":                   state,
            "observation.language_tokens":         lang_tokens,
            "observation.language_attention_mask": lang_mask,
            "action":                              actions,
            "action_is_pad":                       is_pad,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_image(self, raw) -> torch.Tensor:
        """PNG/JPEG bytes (or PIL Image) → float32 (3, H, W) in [0, 1]."""
        if isinstance(raw, bytes):
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        elif hasattr(raw, "convert"):
            img = raw.convert("RGB")
        else:
            raise ValueError(f"Unknown image type: {type(raw)}")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_action_chunk(
        self, ep_rows: list[int], local_i: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.zeros(self.chunk_size, self.action_dim, dtype=torch.float32)
        is_pad  = torch.ones(self.chunk_size, dtype=torch.bool)
        for t in range(self.chunk_size):
            src_i = local_i + t
            if src_i < len(ep_rows):
                row_vals = self.df.iloc[ep_rows[src_i]]["action"]
                actions[t] = torch.tensor(row_vals, dtype=torch.float32)
                is_pad[t]  = False
        return actions, is_pad

    @staticmethod
    def _normalize(x: torch.Tensor, stats: dict) -> torch.Tensor:
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std  = torch.tensor(stats["std"],  dtype=torch.float32).clamp(min=1e-8)
        return (x - mean) / std


# ── Episode-level train/val splits ────────────────────────────────────────

def make_splits(
    data_dir: str,
    chunk_size: int = 50,
    norm_stats: Optional[dict] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[SO100Dataset, SO100Dataset]:
    """Return (train_dataset, val_dataset) with episode-level splits."""
    full = SO100Dataset(data_dir, chunk_size=chunk_size, norm_stats=norm_stats)

    episodes = list(full._episodes.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val   = max(1, int(len(episodes) * val_fraction))
    val_eps = set(episodes[:n_val])
    trn_eps = set(episodes[n_val:])

    train_ds = _filtered_copy(full, trn_eps)
    val_ds   = _filtered_copy(full, val_eps)
    return train_ds, val_ds


def _filtered_copy(ds: SO100Dataset, episode_set: set) -> SO100Dataset:
    import copy
    new_ds = copy.copy(ds)
    new_ds._index = [
        (ep, local_i)
        for ep, local_i in ds._index
        if ep in episode_set
    ]
    return new_ds
