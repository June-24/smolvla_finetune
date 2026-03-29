"""
dataset.py
==========
PyTorch Dataset that reads the downloaded LIBERO Parquet file and builds
batches in exactly the format SmolVLAPolicy.forward() expects.

Batch keys produced:
    observation.images.image          (B, 3, H, W)  float32 in [0,1]
    observation.images.image2         (B, 3, H, W)  float32 in [0,1]
    observation.state                 (B, 8)         float32 (raw)
    observation.language_tokens       (B, 48)        int64
    observation.language_attention_mask (B, 48)      int64
    action                            (B, chunk_size, 7)  float32 (normalized)
    action_is_pad                     (B, chunk_size)     bool
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
IMAGE_SIZE     = 256   # LIBERO images are 256×256

IMG_COLS = ["observation.images.image", "observation.images.image2"]


class LiberoDataset(Dataset):
    """
    One sample = one timestep.
    The dataset loads 'chunk_size' future actions for each timestep.
    If fewer actions remain in the episode, the remainder is zero-padded and
    action_is_pad is set to True for those positions.

    Images are loaded lazily: only the parquet row group that contains the
    requested row is read, and the last-used row group is cached per worker.
    This keeps RAM usage low even for 30+ GB parquet files.
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

        # ── small columns only → pandas (episode/frame/task/action/state)
        small_cols = ["episode_index", "frame_index", "task_index",
                      "action", "observation.state"]
        self.df = pq.read_table(self._pq_path, columns=small_cols).to_pandas()

        # ── row-group index for lazy image loading
        # Stores cumulative row offsets so we can binary-search for any row's group.
        meta = pq.read_metadata(self._pq_path)
        self._rg_starts: list[int] = []   # first global row of each row-group
        offset = 0
        for rg in range(meta.num_row_groups):
            self._rg_starts.append(offset)
            offset += meta.row_group(rg).num_rows
        self._n_rows = offset

        # Per-worker lazy state (not serialised across workers):
        self._pf: Optional[pq.ParquetFile] = None   # opened lazily
        self._cached_rg_idx: int = -1
        self._cached_rg = None                       # pyarrow Table slice

        # ── task names
        with open(self.data_dir / "task_names.json") as f:
            self.task_names: dict[str, str] = json.load(f)

        # ── episode index
        self._build_episode_index()

        # ── tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"LiberoDataset: {len(self.df)} frames, "
              f"{len(self._episodes)} episodes, chunk_size={chunk_size}")

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
        """Open the ParquetFile once per worker process."""
        if self._pf is None:
            self._pf = pq.ParquetFile(self._pq_path, memory_map=True)

    def _get_images_for_row(self, global_row: int):
        """Return (img1_bytes, img2_bytes) for a global row index.

        Reads the row's row-group from disk (or returns cached copy if the
        same row-group was used for the previous call — common in sequential
        DataLoader access patterns).
        """
        self._open_pf()

        # Which row group contains global_row?
        rg_idx = bisect.bisect_right(self._rg_starts, global_row) - 1
        local_row = global_row - self._rg_starts[rg_idx]

        if rg_idx != self._cached_rg_idx:
            self._cached_rg = self._pf.read_row_group(rg_idx, columns=IMG_COLS)
            self._cached_rg_idx = rg_idx

        img1 = self._cached_rg.column(IMG_COLS[0])[local_row].as_py()
        img2 = self._cached_rg.column(IMG_COLS[1])[local_row].as_py()
        return img1, img2

    # ------------------------------------------------------------------
    # Core sample loading
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        ep_id, local_i = self._index[idx]
        ep_rows = self._episodes[ep_id]
        global_row = ep_rows[local_i]

        row = self.df.iloc[global_row]

        # ---- images -------------------------------------------------------
        img1_raw, img2_raw = self._get_images_for_row(global_row)
        img1 = self._decode_image(img1_raw)
        img2 = self._decode_image(img2_raw)

        # ---- state --------------------------------------------------------
        state = torch.tensor(row["observation.state"], dtype=torch.float32)

        # ---- language tokens ----------------------------------------------
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

        # ---- action chunk -------------------------------------------------
        actions, is_pad = self._load_action_chunk(ep_rows, local_i)

        # ---- normalise ----------------------------------------------------
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
        """PNG bytes → float32 tensor (3, H, W) in [0, 1]."""
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
        actions = torch.zeros(self.chunk_size, 7, dtype=torch.float32)
        is_pad  = torch.ones(self.chunk_size,     dtype=torch.bool)
        for t in range(self.chunk_size):
            src_i = local_i + t
            if src_i < len(ep_rows):
                row = self.df.iloc[ep_rows[src_i]]
                actions[t] = torch.tensor(row["action"], dtype=torch.float32)
                is_pad[t]  = False
        return actions, is_pad

    @staticmethod
    def _normalize(x: torch.Tensor, stats: dict) -> torch.Tensor:
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std  = torch.tensor(stats["std"],  dtype=torch.float32).clamp(min=1e-8)
        return (x - mean) / std


# ----------------------------------------------------------------------
# Convenience: build train/val splits from a single data directory
# ----------------------------------------------------------------------

def make_splits(
    data_dir: str,
    chunk_size: int = 50,
    norm_stats: Optional[dict] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple["LiberoDataset", "LiberoDataset"]:
    """Return (train_dataset, val_dataset) with episode-level splits."""
    full = LiberoDataset(data_dir, chunk_size=chunk_size, norm_stats=norm_stats)

    episodes = list(full._episodes.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val    = max(1, int(len(episodes) * val_fraction))
    val_eps  = set(episodes[:n_val])
    trn_eps  = set(episodes[n_val:])

    train_ds = _filtered_copy(full, trn_eps)
    val_ds   = _filtered_copy(full, val_eps)
    return train_ds, val_ds


def _filtered_copy(ds: LiberoDataset, episode_set: set) -> LiberoDataset:
    import copy
    new_ds = copy.copy(ds)
    new_ds._index = [
        (ep, local_i)
        for ep, local_i in ds._index
        if ep in episode_set
    ]
    return new_ds
