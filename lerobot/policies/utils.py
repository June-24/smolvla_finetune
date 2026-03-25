"""Minimal policy utilities used by SmolVLAPolicy."""

import logging
from collections import deque

import torch

logger = logging.getLogger(__name__)


def populate_queues(queues: dict, batch: dict) -> dict:
    """
    Push the latest observations from batch into each deque in queues.
    Returns a dict where each key maps to a tensor stacked along the
    n_obs_steps dimension.

    In training we don't use temporal queues (we sample full chunks from
    the dataset), so this is only exercised during online rollouts.
    """
    for key in queues:
        if key in batch:
            obs = batch[key]
            queues[key].append(obs)

    # stack into (n_obs_steps, *original_shape)
    stacked = {}
    for key, q in queues.items():
        stacked[key] = torch.stack(list(q), dim=1)  # (B, T, ...)

    return stacked


def log_model_loading_keys(missing_keys: list, unexpected_keys: list):
    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]} ...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]} ...")
