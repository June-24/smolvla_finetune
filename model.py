"""
model.py
========
SmolVLAPolicy and VLAFlowMatching — standalone, no lerobot dependencies.

Implements:
  * Conditional Flow Matching training forward pass
  * Action-chunk inference via iterative denoising
  * save_pretrained / from_pretrained for checkpointing
"""

import logging
import math
from collections import deque
from pathlib import Path
from typing import TypedDict

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import SmolVLAConfig
from expert import SmolVLMWithExpertModel

logger = logging.getLogger(__name__)

# ── String keys used in batch dicts ───────────────────────────────────────
OBS_STATE                   = "observation.state"
OBS_LANGUAGE_TOKENS         = "observation.language_tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language_attention_mask"
ACTION                      = "action"


# ── Inline utilities (previously from lerobot stubs) ──────────────────────

def _get_safe_dtype(dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Return dtype safe for the device (no float64 on MPS)."""
    if device_type == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


# ── Positional encoding ───────────────────────────────────────────────────

def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Sine-cosine positional embeddings for scalar timesteps."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must be shape (batch_size,)")

    dtype    = _get_safe_dtype(torch.float64, str(device.type))
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period   = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input      = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# ── Attention mask helpers ─────────────────────────────────────────────────

def make_att_2d_masks(pad_masks, att_masks):
    """Build 2-D boolean attention masks from padding + causal masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum        = torch.cumsum(att_masks, dim=1)
    att_2d_masks  = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks  = (pad_masks[:, None, :] * pad_masks[:, :, None]).bool()
    return att_2d_masks & pad_2d_masks


# ── Tensor helpers ─────────────────────────────────────────────────────────

def resize_with_pad(img, width, height, pad_value=-1):
    """Resize (B,C,H,W) image preserving aspect ratio, pad to (width, height)."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, got {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio           = max(cur_width / width, cur_height / height)
    resized_height  = int(cur_height / ratio)
    resized_width   = int(cur_width  / ratio)
    resized_img     = F.interpolate(
        img, size=(resized_height, resized_width),
        mode="bilinear", align_corners=False,
    )
    pad_height = max(0, int(height - resized_height))
    pad_width  = max(0, int(width  - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def pad_vector(vector, new_dim):
    """Zero-pad the last dimension to new_dim. Works for 2-D and 3-D tensors."""
    if vector.shape[-1] == new_dim:
        return vector
    shape         = list(vector.shape)
    current_dim   = shape[-1]
    shape[-1]     = new_dim
    new_vector    = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def pad_tensor(tensor, max_len, pad_value=0):
    """Pad sequence dimension (dim=1) to max_len."""
    b, d = tensor.shape[:2]
    padded = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value,
        dtype=tensor.dtype, device=tensor.device,
    )
    padded[:, :d] = tensor
    return padded


# ── Aloha gripper conversion helpers (unused for LIBERO) ──────────────────

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def safe_arcsin(value):
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))

def aloha_gripper_to_angular(value):
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)
    def linear_to_radian(lp, arm_length, horn_radius):
        v = (horn_radius**2 + lp**2 - arm_length**2) / (2 * horn_radius * lp)
        return safe_arcsin(v)
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)

def aloha_gripper_from_angular(value):
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)

def aloha_gripper_from_angular_inv(value):
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


# ── ActionSelectKwargs ────────────────────────────────────────────────────

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay:      int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon:    int | None


# ══════════════════════════════════════════════════════════════════════════
# VLAFlowMatching
# ══════════════════════════════════════════════════════════════════════════

class VLAFlowMatching(nn.Module):
    """
    Core neural network for SmolVLA:
      VLM (frozen) encodes images + language → KV cache
      Action expert (trained) denoises action tokens via cross-attention to KV cache
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id              = config.vlm_model_name,
            freeze_vision_encoder = config.freeze_vision_encoder,
            train_expert_only     = config.train_expert_only,
            load_vlm_weights      = config.load_vlm_weights,
            attention_mode        = config.attention_mode,
            num_expert_layers     = config.num_expert_layers,
            num_vlm_layers        = config.num_vlm_layers,
            self_attn_every_n_layers = config.self_attn_every_n_layers,
            expert_width_multiplier  = config.expert_width_multiplier,
            device                = config.device if config.device is not None else "auto",
        )

        vlm_hidden_size    = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size

        self.state_proj       = nn.Linear(config.max_state_dim,   vlm_hidden_size)
        self.action_in_proj   = nn.Linear(config.max_action_dim,  expert_hidden_size)
        self.action_out_proj  = nn.Linear(expert_hidden_size,     config.max_action_dim)
        self.action_time_mlp_in  = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size,     expert_hidden_size)

        self.set_requires_grad()

        # Image / language special tokens
        tok = self.vlm_with_expert.processor.tokenizer
        self.fake_image_token   = tok.fake_image_token_id
        self.global_image_token = tok.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length   = config.prefix_length

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward        = torch.compile(self.forward,        mode=config.compile_mode)

    def set_requires_grad(self):
        for p in self.state_proj.parameters():
            p.requires_grad = self.config.train_state_proj

    # ── Noise / time sampling ─────────────────────────────────────────────

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape,
                            dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        t = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return t * 0.999 + 0.001

    # ── Embedding helpers ─────────────────────────────────────────────────

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        """Embed images + language + state into prefix tokens."""
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks):
            if self.add_image_special_tokens:
                start_tok = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(
                            device=self.vlm_with_expert.vlm.device
                        )
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                start_mask = torch.ones_like(start_tok[:, :, 0], dtype=torch.bool)
                att_masks += [0] * start_mask.shape[-1]
                embs.append(start_tok)
                pad_masks.append(start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * (img_emb_dim ** 0.5)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                end_tok = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                end_mask = torch.ones_like(end_tok[:, :, 0], dtype=torch.bool)
                embs.append(end_tok)
                pad_masks.append(end_mask)
                att_masks += [0] * end_mask.shape[1]

        # Language
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        # State
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize, device_ref = state_emb.shape[0], state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device_ref)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        embs      = torch.cat(embs,      dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs      = pad_tensor(embs,      self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy actions + timestep into suffix tokens for the expert."""
        action_emb = self.action_in_proj(noisy_actions)
        device     = action_emb.device
        bsize      = action_emb.shape[0]
        dtype      = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(dtype=dtype)

        time_emb        = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs      = [action_time_emb]
        pad_masks = [torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=device)]
        att_masks = [1] * self.config.chunk_size

        embs      = torch.cat(embs,      dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    # ── Training forward ──────────────────────────────────────────────────

    def forward(
        self, images, img_masks, lang_tokens, lang_masks,
        state, actions, noise=None, time=None,
    ) -> Tensor:
        """Compute per-element flow-matching MSE losses (B, T, D)."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks   = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks   = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    # ── Inference ─────────────────────────────────────────────────────────

    def sample_actions(
        self,
        images, img_masks, lang_tokens, lang_masks, state,
        noise=None, **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Iterative denoising to produce an action chunk (B, chunk_size, max_action_dim)."""
        bsize  = state.shape[0]
        device = state.device

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Build KV cache once
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt        = -1.0 / num_steps
        x_t       = noise

        for step in range(num_steps):
            time_val    = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)
            v_t         = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """Single denoising step given cached VLM prefix."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len   = suffix_pad_masks.shape[1]
        batch_size   = prefix_pad_masks.shape[0]
        prefix_len   = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids   = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


# ══════════════════════════════════════════════════════════════════════════
# SmolVLAPolicy  (top-level API)
# ══════════════════════════════════════════════════════════════════════════

class SmolVLAPolicy(nn.Module):
    """
    Wraps VLAFlowMatching for training and inference.

    Training:
        loss, loss_dict = policy(batch)

    Inference:
        actions = policy.predict_action_chunk(batch)   # (B, chunk_size, action_dim)
        action  = policy.select_action(batch)          # (B, action_dim)
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        config.validate_features()
        self.config = config
        self.model  = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """Call whenever the environment resets."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    # ── Training ──────────────────────────────────────────────────────────

    def forward(
        self, batch: dict[str, Tensor],
        noise=None, time=None, reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Training forward — returns (loss, loss_dict)."""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION]    = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state             = self.prepare_state(batch)
        lang_tokens       = batch[OBS_LANGUAGE_TOKENS]
        lang_masks        = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions           = self.prepare_action(batch)
        actions_is_pad    = batch.get("action_is_pad")

        loss_dict = {}
        losses    = self.model.forward(
            images, img_masks, lang_tokens, lang_masks,
            state, actions, noise, time,
        )

        # Trim to original action dim
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        losses = losses[:, :, :self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    # ── Inference ─────────────────────────────────────────────────────────

    def _get_action_chunk(
        self, batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        state             = self.prepare_state(batch)
        lang_tokens       = batch[OBS_LANGUAGE_TOKENS]
        lang_masks        = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Trim padding
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Return full action chunk (B, n_action_steps, action_dim)."""
        self.eval()
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        # Push latest observations into queues
        for k in batch:
            if k in self._queues and k != ACTION:
                self._queues[k].append(batch[k])
        return self._get_action_chunk(batch, noise, **kwargs)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Return single action for environment stepping."""
        self.eval()
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        for k in batch:
            if k in self._queues and k != ACTION:
                self._queues[k].append(batch[k])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            # Transpose (B, T, D) → (T, B, D) and push each timestep
            self._queues[ACTION].extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )

        return self._queues[ACTION].popleft()

    # ── Batch preparation ─────────────────────────────────────────────────

    def prepare_images(self, batch):
        images, img_masks = [], []
        present_keys = [k for k in self.config.image_features if k in batch]
        missing_keys = [k for k in self.config.image_features if k not in batch]

        if not present_keys:
            raise ValueError(
                f"No image features found in batch. "
                f"Expected: {list(self.config.image_features.keys())}"
            )

        for key in present_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0  # [0,1] → [-1,1] for SigLIP

            bsize  = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)

            images.append(img)
            img_masks.append(mask)

        # Fill in empty cameras (if any)
        for _ in range(min(len(missing_keys), self.config.empty_cameras)):
            images.append(torch.ones_like(img) * -1)
            img_masks.append(torch.zeros_like(mask))

        return images, img_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    # ── Aloha helpers ─────────────────────────────────────────────────────

    def _pi_aloha_decode_state(self, state):
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    # ── Serialization ─────────────────────────────────────────────────────

    def save_pretrained(self, save_dir: str) -> None:
        """Save weights (safetensors) + config.json to save_dir."""
        import safetensors.torch as st

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        st.save_file(self.state_dict(), save_dir / "model.safetensors")
        self.config.save_pretrained(save_dir)
        logger.info(f"Saved policy to {save_dir}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        config: SmolVLAConfig | None = None,
        **kwargs,
    ) -> "SmolVLAPolicy":
        """
        Load from a local directory or HuggingFace Hub repo.

        Args:
            pretrained_name_or_path: local dir containing model.safetensors + config.json,
                                     or an HF Hub repo id (e.g. "lerobot/smolvla_base").
            config: optional config override; if None the saved config.json is used.
        """
        import safetensors.torch as st
        from huggingface_hub import hf_hub_download

        local_path = Path(pretrained_name_or_path)

        if local_path.is_dir():
            model_file  = local_path / "model.safetensors"
            config_file = local_path / "config.json"
        else:
            model_file = Path(hf_hub_download(pretrained_name_or_path, "model.safetensors"))
            try:
                config_file = Path(hf_hub_download(pretrained_name_or_path, "config.json"))
            except Exception:
                config_file = None

        if config is None and config_file is not None and config_file.exists():
            config = SmolVLAConfig.from_pretrained(str(config_file.parent))

        policy = cls(config=config, **kwargs)

        state_dict       = st.load_file(str(model_file))
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:10]}")

        logger.info(f"Loaded policy from {pretrained_name_or_path}")
        return policy
