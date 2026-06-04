"""
SmolVLA cross-attention probe.

Installs a lightweight hook on SmolVLMWithExpertModel._attn_weight_hook
to capture attention weight tensors during inference without modifying the
main inference logic.

Usage:
    probe = SmolVLAAttentionProbe(policy)
    with probe:
        actions = policy.predict_action_chunk(observation)
    capture = probe.last_capture  # AttentionCapture
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass  # avoid circular imports


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TokenOffsets:
    """Describes the position slices of each semantic group in the prefix KV sequence.

    All indices are with respect to the key/value sequence seen by the action expert
    during cross-attention (i.e. the VLM prefix output, length = prefix_len).
    """

    camera_slices: list[tuple[int, int]]  # [(start, end), ...] one per camera
    lang_slice: tuple[int, int]           # (start, end) for language tokens
    state_slice: tuple[int, int]          # (start, end) for state token(s)
    prefix_len: int                       # total prefix length (may include padding)

    @classmethod
    def from_counts(
        cls,
        n_patches_per_camera: list[int],
        n_lang_tokens: int,
        n_state_tokens: int = 1,
        add_image_special_tokens: bool = False,
    ) -> TokenOffsets:
        """Build offsets from semantic token counts."""
        camera_slices: list[tuple[int, int]] = []
        pos = 0
        for n_patches in n_patches_per_camera:
            if add_image_special_tokens:
                pos += 1  # image_start token
            start = pos
            pos += n_patches
            camera_slices.append((start, pos))
            if add_image_special_tokens:
                pos += 1  # image_end token
        lang_slice = (pos, pos + n_lang_tokens)
        pos += n_lang_tokens
        state_slice = (pos, pos + n_state_tokens)
        pos += n_state_tokens
        return cls(
            camera_slices=camera_slices,
            lang_slice=lang_slice,
            state_slice=state_slice,
            prefix_len=pos,
        )

    @classmethod
    def auto_detect(
        cls,
        n_patches_per_camera: list[int],
        prefix_len: int,
        n_state_tokens: int = 1,
        add_image_special_tokens: bool = False,
    ) -> TokenOffsets:
        """Infer n_lang_tokens from prefix_len and known counts."""
        img_toks = sum(n_patches_per_camera)
        if add_image_special_tokens:
            img_toks += 2 * len(n_patches_per_camera)
        n_lang_tokens = prefix_len - img_toks - n_state_tokens
        return cls.from_counts(
            n_patches_per_camera=n_patches_per_camera,
            n_lang_tokens=max(0, n_lang_tokens),
            n_state_tokens=n_state_tokens,
            add_image_special_tokens=add_image_special_tokens,
        )

    def camera_patch_probs(
        self, probs: torch.Tensor, cam_idx: int = 0
    ) -> torch.Tensor:
        """Slice patch-level attention for one camera.

        Args:
            probs: [B, H, q_len, kv_len] attention probabilities
            cam_idx: camera index

        Returns:
            Tensor [B, H, q_len, n_patches]
        """
        s, e = self.camera_slices[cam_idx]
        return probs[..., s:e]

    def lang_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Slice language-token attention."""
        s, e = self.lang_slice
        return probs[..., s:e]

    def state_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Slice state-token attention."""
        s, e = self.state_slice
        return probs[..., s:e]


@dataclass
class AttentionCapture:
    """All attention weights captured during one forward-pass (sample_actions call).

    Layout:
        cross_attn[step][layer_idx] = probs tensor [B, H, chunk_size, prefix_len]
        self_attn[step][layer_idx]  = probs tensor [B, H, chunk_size, prefix_len + chunk_size]

    cross_attn holds the maps from action tokens → VLM prefix (image + lang + state).
    These are the semantically meaningful maps for visualization.
    """

    cross_attn: list[dict[int, torch.Tensor]] = field(default_factory=list)
    self_attn: list[dict[int, torch.Tensor]] = field(default_factory=list)
    token_offsets: TokenOffsets | None = None

    @property
    def num_steps(self) -> int:
        return len(self.cross_attn)

    def mean_cross_attn(self, step: int | None = None) -> torch.Tensor | None:
        """Average cross-attn over heads and action steps.

        Args:
            step: if None, averages over all denoising steps; otherwise a single step.

        Returns:
            Tensor [B, chunk_size, prefix_len] or None if no data.
        """
        if not self.cross_attn:
            return None
        steps = [step] if step is not None else list(range(len(self.cross_attn)))
        layers_list = [
            torch.stack(list(self.cross_attn[s].values()), dim=0)  # [L, B, H, q, kv]
            for s in steps
            if self.cross_attn[s]
        ]
        if not layers_list:
            return None
        stacked = torch.stack(layers_list, dim=0)  # [S, L, B, H, q, kv]
        return stacked.mean(dim=(0, 1, 3))  # → [B, q, kv]


# ---------------------------------------------------------------------------
# Probe context manager
# ---------------------------------------------------------------------------


class SmolVLAAttentionProbe:
    """Context manager that captures SmolVLA attention weights non-intrusively.

    Only works with smolvla policies (attention_mode='cross_attn').

    Example::

        probe = SmolVLAAttentionProbe(policy)
        with probe:
            actions = policy.predict_action_chunk(obs)
        capture = probe.last_capture  # AttentionCapture

        # Or as a per-call helper:
        with probe as p:
            actions = policy.predict_action_chunk(obs)
            token_offsets = p.build_token_offsets(obs)
    """

    def __init__(
        self,
        policy,
        keep_cpu_copy: bool = True,
    ):
        """
        Args:
            policy: SmolVLAPolicy instance.
            keep_cpu_copy: if True, immediately move captured tensors to CPU
                (avoids GPU memory accumulation during multi-step inference).
        """
        self._policy = policy
        cfg = policy.config
        self._vlm_with_expert = policy.model.vlm_with_expert

        self._chunk_size: int = cfg.chunk_size
        # Use the actual layer count from the loaded model, not the config value.
        # cfg.num_vlm_layers can be 0 or -1 meaning "use all layers", in which case
        # vlm_with_expert.num_vlm_layers holds the true count set during __init__.
        self._num_vlm_layers: int = self._vlm_with_expert.num_vlm_layers
        self._self_attn_every_n: int = self._vlm_with_expert.self_attn_every_n_layers
        self._add_image_special_tokens: bool = cfg.add_image_special_tokens
        self._keep_cpu = keep_cpu_copy

        self._call_count = 0
        self._current_capture: AttentionCapture | None = None
        self.last_capture: AttentionCapture | None = None

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _hook(self, probs: torch.Tensor) -> None:
        """Receives probs [B, H, q_len, kv_len] after every attention computation."""
        c = self._call_count
        self._call_count += 1

        # First num_vlm_layers calls are the prefix phase (fill_kv_cache=True) → skip
        if c < self._num_vlm_layers:
            return

        denoising_call = c - self._num_vlm_layers
        step_idx = denoising_call // self._num_vlm_layers
        layer_idx = denoising_call % self._num_vlm_layers

        # self_attn_every_n <= 0 means no self-attn interleaving → all layers are cross-attn
        if self._self_attn_every_n > 0:
            is_cross = layer_idx % self._self_attn_every_n != 0
        else:
            is_cross = True

        cap = self._current_capture
        if cap is None:
            return

        while len(cap.cross_attn) <= step_idx:
            cap.cross_attn.append({})
        while len(cap.self_attn) <= step_idx:
            cap.self_attn.append({})

        stored = probs.detach().cpu() if self._keep_cpu else probs.detach()
        if is_cross:
            cap.cross_attn[step_idx][layer_idx] = stored
        else:
            cap.self_attn[step_idx][layer_idx] = stored

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SmolVLAAttentionProbe:
        self._call_count = 0
        self._current_capture = AttentionCapture()
        self._vlm_with_expert._attn_weight_hook = self._hook
        return self

    def __exit__(self, *args) -> None:
        self._vlm_with_expert._attn_weight_hook = None
        self.last_capture = self._current_capture
        self._current_capture = None

    # ------------------------------------------------------------------
    # Token offset helpers
    # ------------------------------------------------------------------

    def build_token_offsets(
        self,
        observation: dict[str, torch.Tensor],
        n_state_tokens: int = 1,
    ) -> TokenOffsets | None:
        """Compute TokenOffsets from a preprocessed observation dict.

        Reads n_lang_tokens directly from the language-token tensor in the
        observation dict (observation.language.tokens), then derives n_patches
        per camera as:
            n_patches = (prefix_len - n_lang_tokens - n_state_tokens) // n_cameras

        We intentionally do NOT call embed_image here: after SmolVLA's
        preprocessor the image tensors are in SmolVLM2-Video format which is
        incompatible with embed_image's expected [B, C, H, W] input, and even
        a correctly-shaped dummy image can fail if the connector's hardcoded
        reshape does not match the model's configured patch count.

        Args:
            observation: preprocessed observation dict (after policy preprocessor).
            n_state_tokens: number of state tokens (default 1).

        Returns:
            TokenOffsets or None if no images found or no capture available.
        """
        from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_TOKENS

        img_keys = sorted(k for k in observation if OBS_IMAGES in k and "empty" not in k)
        if not img_keys:
            return None

        n_cameras = len(img_keys)

        # Get n_lang_tokens from the language token tensor (always present after
        # SmolVLA's preprocessor, which tokenizes the task string).
        lang_key = OBS_LANGUAGE_TOKENS
        if lang_key not in observation:
            # fallback: use tokenizer_max_length from config
            n_lang_tokens = getattr(self._policy.config, "tokenizer_max_length", 48)
        else:
            lang_tensor = observation[lang_key]
            # shape: [B, seq_len] — take seq_len
            n_lang_tokens = lang_tensor.shape[-1]

        # Determine prefix_len from cross-attn probs kv_len
        prefix_len: int | None = None
        cap = self.last_capture or self._current_capture
        if cap and cap.cross_attn:
            for step_maps in cap.cross_attn:
                if step_maps:
                    first_probs = next(iter(step_maps.values()))
                    prefix_len = first_probs.shape[-1]
                    break

        if prefix_len is None:
            return None

        # Derive per-camera patch count from prefix composition
        n_patches_total = prefix_len - n_lang_tokens - n_state_tokens
        if n_patches_total <= 0 or n_patches_total % n_cameras != 0:
            # Prefix padding (pad_language_to="max_length") can inflate n_lang_tokens
            # beyond the actual task tokens. Back-solve for n_patches using the
            # actual attention mask if available.
            attn_mask_key = OBS_LANGUAGE_TOKENS.replace(".tokens", ".attention_mask")
            if attn_mask_key in observation:
                n_lang_active = int(observation[attn_mask_key].sum(dim=-1).max().item())
                n_patches_total = prefix_len - n_lang_active - n_state_tokens
            # If still not divisible, round down per camera
            n_patches_per_camera_count = max(1, n_patches_total // n_cameras)
        else:
            n_patches_per_camera_count = n_patches_total // n_cameras

        n_patches_per_camera = [n_patches_per_camera_count] * n_cameras

        offsets = TokenOffsets.auto_detect(
            n_patches_per_camera=n_patches_per_camera,
            prefix_len=prefix_len,
            n_state_tokens=n_state_tokens,
            add_image_special_tokens=self._add_image_special_tokens,
        )
        if cap is not None:
            cap.token_offsets = offsets
        return offsets
