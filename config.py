"""
config.py
=========
Standalone configuration for SmolVLA fine-tuning.
No lerobot dependencies — pure Python stdlib + dataclasses.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Feature / normalization types ─────────────────────────────────────────

class FeatureType(str, Enum):
    STATE   = "state"
    VISUAL  = "visual"
    ACTION  = "action"
    ENV_STATE = "env_state"


class NormalizationMode(str, Enum):
    MEAN_STD = "mean_std"
    MIN_MAX  = "min_max"
    IDENTITY = "identity"


@dataclass
class PolicyFeature:
    type:  FeatureType
    shape: tuple


# ── SmolVLA configuration ─────────────────────────────────────────────────

@dataclass
class SmolVLAConfig:
    # Feature specs — set by the caller to describe the robot / dataset
    input_features:  dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)

    # Temporal structure
    n_obs_steps:    int = 1
    chunk_size:     int = 50
    n_action_steps: int = 50

    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE":  NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Padding dimensions — short vectors are zero-padded to these sizes
    max_state_dim:  int = 32
    max_action_dim: int = 32

    # Image pre-processing: resize to (W, H) with aspect-ratio padding
    resize_imgs_with_padding: tuple = (512, 512)

    # Extra empty cameras to add (set 0 for LIBERO)
    empty_cameras: int = 0

    # ── VLM backbone ──────────────────────────────────────────────────────
    vlm_model_name:  str  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False   # True when starting from SmolVLA pretrained

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer_max_length:    int  = 48
    add_image_special_tokens: bool = False
    pad_language_to:          str  = "longest"

    # ── Action expert architecture ─────────────────────────────────────────
    num_expert_layers:        int   = -1    # -1 = same depth as VLM
    num_vlm_layers:           int   = 16
    self_attn_every_n_layers: int   = 2
    expert_width_multiplier:  float = 0.75

    # ── Attention ─────────────────────────────────────────────────────────
    attention_mode: str  = "cross_attn"
    prefix_length:  int  = -1
    use_cache:      bool = True

    # ── Flow matching ─────────────────────────────────────────────────────
    num_steps:  int   = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    # ── Fine-tuning flags ─────────────────────────────────────────────────
    freeze_vision_encoder: bool = True
    train_expert_only:     bool = True
    train_state_proj:      bool = True

    # ── Misc ──────────────────────────────────────────────────────────────
    device:       Optional[str] = None
    compile_model: bool = False
    compile_mode:  str  = "max-autotune"

    # Aloha-specific (unused for LIBERO, keep False)
    adapt_to_pi_aloha:            bool = False
    use_delta_joint_actions_aloha: bool = False

    # ── Validation ────────────────────────────────────────────────────────

    def __post_init__(self):
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be >= "
                f"n_action_steps ({self.n_action_steps})"
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError("`use_delta_joint_actions_aloha` is not supported.")

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def image_features(self) -> dict:
        return {k: v for k, v in self.input_features.items()
                if v.type == FeatureType.VISUAL}

    @property
    def action_feature(self) -> Optional[PolicyFeature]:
        for v in self.output_features.values():
            if v.type == FeatureType.ACTION:
                return v
        return None

    @property
    def state_feature(self) -> Optional[PolicyFeature]:
        for v in self.input_features.values():
            if v.type == FeatureType.STATE:
                return v
        return None

    def validate_features(self) -> None:
        """Add empty-camera placeholders if requested."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            )

    # ── Serialisation ─────────────────────────────────────────────────────

    def save_pretrained(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        data: dict = {}

        # Serialise feature dicts explicitly
        for feat_key in ("input_features", "output_features"):
            mapping = getattr(self, feat_key)
            data[feat_key] = {
                k: {"type": v.type.value, "shape": list(v.shape)}
                for k, v in mapping.items()
            }

        # All remaining scalar / simple fields
        skip = {"input_features", "output_features"}
        for name in self.__dataclass_fields__:
            if name in skip:
                continue
            val = getattr(self, name)
            if isinstance(val, Enum):
                data[name] = val.value
            elif isinstance(val, tuple):
                data[name] = list(val)
            elif isinstance(val, dict):
                # normalization_mapping — convert Enum values
                data[name] = {
                    k: (v.value if isinstance(v, Enum) else v)
                    for k, v in val.items()
                }
            else:
                data[name] = val

        with open(save_dir / "config.json", "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "SmolVLAConfig":
        config_path = Path(save_dir) / "config.json"
        with open(config_path) as f:
            data = json.load(f)

        # Reconstruct PolicyFeature objects
        for feat_key in ("input_features", "output_features"):
            if feat_key in data and isinstance(data[feat_key], dict):
                data[feat_key] = {
                    k: PolicyFeature(
                        type=FeatureType(v["type"].lower()),
                        shape=tuple(v["shape"]),
                    )
                    for k, v in data[feat_key].items()
                    if isinstance(v, dict)
                }

        # Lists → tuples for known tuple fields
        for tuple_field in ("resize_imgs_with_padding",):
            if tuple_field in data and isinstance(data[tuple_field], list):
                data[tuple_field] = tuple(data[tuple_field])

        # Filter to known dataclass fields only
        known = set(cls.__dataclass_fields__.keys())
        data = {k: v for k, v in data.items() if k in known}

        return cls(**data)
