"""
Minimal PreTrainedPolicy base class.

SmolVLAPolicy inherits from this.  The two key things we provide:
  1. nn.Module base so all PyTorch machinery works.
  2. from_pretrained() that downloads safetensors weights from HF Hub.
"""

import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


class ActionSelectKwargs:
    """Placeholder — SmolVLAPolicy.select_action accepts **ActionSelectKwargs."""
    pass


class PreTrainedPolicy(nn.Module):
    """Base class for SmolVLA (and other lerobot policies)."""

    name: str = ""  # set by subclass

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str):
        import safetensors.torch as st

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # save weights
        st.save_file(self.state_dict(), save_dir / "model.safetensors")

        # save config
        if self.config is not None and hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(save_dir)

        logger.info(f"Saved policy to {save_dir}")

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, config=None, **kwargs):
        """
        Load a pretrained SmolVLAPolicy from a local dir or HuggingFace Hub repo.

        Args:
            pretrained_name_or_path: HF repo id (e.g. "lerobot/smolvla_base") or
                                     a local directory containing model.safetensors
                                     and config.json.
            config: optional SmolVLAConfig override; if None the config.json from
                    the repo is used.
        """
        import safetensors.torch as st
        from lerobot.configs.policies import PreTrainedConfig

        local_path = Path(pretrained_name_or_path)

        # --- resolve files -----------------------------------------------
        if local_path.is_dir():
            model_file = local_path / "model.safetensors"
            config_file = local_path / "config.json"
        else:
            # download from Hub
            model_file = Path(
                hf_hub_download(pretrained_name_or_path, "model.safetensors")
            )
            try:
                config_file = Path(
                    hf_hub_download(pretrained_name_or_path, "config.json")
                )
            except Exception:
                config_file = None

        # --- load config -------------------------------------------------
        if config is None and config_file is not None and config_file.exists():
            config = PreTrainedConfig.from_pretrained(str(config_file.parent))

        # --- instantiate model -------------------------------------------
        policy = cls(config=config, **kwargs)

        # --- load weights ------------------------------------------------
        state_dict = st.load_file(str(model_file))
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading pretrained weights:\n{missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading pretrained weights:\n{unexpected[:10]}")

        logger.info(f"Loaded pretrained policy from {pretrained_name_or_path}")
        return policy

    # ------------------------------------------------------------------
    # PEFT helpers (used by SmolVLAPolicy)
    # ------------------------------------------------------------------

    def _get_default_peft_targets(self):
        return []
