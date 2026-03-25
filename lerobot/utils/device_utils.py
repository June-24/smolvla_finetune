"""Stub for lerobot.utils.device_utils."""

import torch


def get_safe_dtype(dtype: torch.dtype, device: str) -> torch.dtype:
    """Return a dtype safe for the given device (e.g., no float64 on MPS)."""
    if str(device) == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype
