"""PI05 task-conditioned image feature probe.

Captures prefix hidden states (image patch + language token representations)
after PI05's PaliGemmaWithExpertModel joint-attention forward pass, and
action→image attention weights from gemma_expert denoising steps.

Both are captured using PyTorch instance-dict monkey-patching (not
register_forward_hook) because PI05 calls sub-models via .forward() directly,
which bypasses nn.Module.__call__ and therefore standard forward hooks.

Architecture note
-----------------
PI05's embed_image() calls PaliGemmaModel.get_image_features(), which runs
SigLIP vision_tower → multi_modal_projector → stores the projected patch
sequence in image_outputs.pooler_output. For a 224×224 image with patch
size 14 this yields 256 patch tokens per camera. These spatial tokens are
concatenated with language tokens to form the prefix:

    prefix = [cam0_patches | cam1_patches | … | lang_tokens]

During denoising, gemma_expert.model attends to prefix KV cache + action
tokens. The action→image sub-block (first n_img_toks KV positions) captures
which image patches the action denoiser focuses on each step.

Implementation note — why output_attentions=True is NOT used on PiGemmaModel
----------------------------------------------------------------------------
_PiGemmaDecoderLayerBase.forward returns a plain tensor (hidden_states), not
a tuple. PiGemmaModel.forward has a bug on the output_attentions path: it
does all_self_attns += (layer_outputs[1],) where layer_outputs is a tensor,
so layer_outputs[1] is tensor indexing at dim 0 index 1 — fails for batch=1.

Fix: each self_attn module's forward is patched via instance __dict__ to
inject output_attentions=True at the attention level, which forces eager
attention and real weight computation. A register_forward_hook on each
self_attn captures the (attn_output, attn_weights) tuple. A wrapper around
gemma_expert.model.forward groups per-layer captures into denoising steps.

Usage::

    probe = PI05FeatureProbe(policy)
    with probe:
        actions = policy.predict_action_chunk(observation)
        probe.set_token_layout(observation)
    capture = probe.last_capture
    img_feats = capture.img_features          # [B, n_cameras, N_patches, D]
    attn_map  = capture.action_to_img_attn()  # [B, n_cameras, N_patches]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass  # avoid circular imports


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TokenLayout:
    """Prefix token positions for one PI05 forward pass.

    Layout: [cam0_patches | cam1_patches | … | camN_patches | lang_tokens]
    All counts refer to the token axis of prefix_hidden (dim=1).
    """

    n_cameras: int               # number of camera inputs
    n_patches_per_camera: int    # patch tokens per camera (256 for 224×224/patch14)
    n_lang_tokens: int           # total language sequence length (including padding)
    prefix_len: int              # total prefix length (= n_cameras*n_patches + n_lang)

    @property
    def camera_slices(self) -> list[tuple[int, int]]:
        """(start, end) index pairs for each camera's patch tokens."""
        slices = []
        for i in range(self.n_cameras):
            s = i * self.n_patches_per_camera
            slices.append((s, s + self.n_patches_per_camera))
        return slices

    @property
    def lang_slice(self) -> tuple[int, int]:
        """(start, end) for the language token block."""
        s = self.n_cameras * self.n_patches_per_camera
        return (s, s + self.n_lang_tokens)

    @property
    def patch_grid_shape(self) -> tuple[int, int]:
        """(H, W) spatial grid shape for each camera's patches."""
        N = self.n_patches_per_camera
        side = int(math.isqrt(N))
        if side * side == N:
            return side, side
        for h in range(side, 0, -1):
            if N % h == 0:
                return h, N // h
        return 1, N


@dataclass
class PI05FeatureCapture:
    """Prefix hidden states and denoising attention captured during one predict_action_chunk.

    prefix_hidden:   [B, prefix_len, D] — task-conditioned token representations
                     from the PaliGemma prefix forward pass.
    token_layout:    set by probe.set_token_layout() after inference.
    denoising_attn:  list of per-denoising-step attention dicts.
                     Each entry maps layer_idx → [B, total_kv_len] float32
                     (self-attention averaged over heads and action tokens).
    prefix_attn:     dict mapping layer_idx → [B, prefix_len, prefix_len] float32
                     (PaliGemma prefix self-attention, averaged over heads).
                     Extract the language→image sub-block via lang_to_img_attn().
    """

    prefix_hidden: torch.Tensor | None = None
    token_layout: TokenLayout | None = None
    denoising_attn: list[dict[int, torch.Tensor]] = field(default_factory=list)
    prefix_attn: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def img_features(self) -> torch.Tensor | None:
        """Per-patch task-conditioned features, shape [B, n_cameras, N_patches, D]."""
        if self.prefix_hidden is None or self.token_layout is None:
            return None
        cams = [self.prefix_hidden[:, s:e, :] for s, e in self.token_layout.camera_slices]
        return torch.stack(cams, dim=1)

    @property
    def img_features_mean(self) -> torch.Tensor | None:
        """Mean-pooled image feature per camera, shape [B, n_cameras, D]."""
        feats = self.img_features
        return feats.mean(dim=2) if feats is not None else None

    @property
    def lang_features(self) -> torch.Tensor | None:
        """Language token hidden states, shape [B, n_lang, D]."""
        if self.prefix_hidden is None or self.token_layout is None:
            return None
        s, e = self.token_layout.lang_slice
        return self.prefix_hidden[:, s:e, :]

    def action_to_img_attn(
        self, token_layout: TokenLayout | None = None
    ) -> torch.Tensor | None:
        """Action→image attention averaged over denoising steps, heads, action tokens.

        During denoising, gemma_expert attends to prefix KV cache + action tokens.
        Extracts the action→image sub-block (first n_img_toks KV positions) and
        averages over all denoising steps and transformer layers.

        Args:
            token_layout: override capture.token_layout if provided.

        Returns:
            [B, n_cameras, N_patches] float32, or None if no data captured.
        """
        layout = token_layout or self.token_layout
        if not self.denoising_attn or layout is None:
            return None

        n_img_toks = layout.n_cameras * layout.n_patches_per_camera

        step_means: list[torch.Tensor] = []
        for step_attn in self.denoising_attn:
            layer_tensors: list[torch.Tensor] = []
            for layer_idx in sorted(step_attn.keys()):
                attn_kv = step_attn[layer_idx]  # [B, total_kv_len]
                if attn_kv.shape[-1] < n_img_toks:
                    continue
                layer_tensors.append(attn_kv[:, :n_img_toks].float())
            if layer_tensors:
                step_means.append(torch.stack(layer_tensors, dim=0).mean(dim=0))

        if not step_means:
            return None

        result = torch.stack(step_means, dim=0).mean(dim=0)  # [B, n_img_toks]
        B = result.shape[0]
        return result.reshape(B, layout.n_cameras, layout.n_patches_per_camera)

    def lang_to_img_attn(
        self,
        token_layout: TokenLayout | None = None,
        lang_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Language→image attention from PaliGemma prefix phase.

        During the prefix forward pass, PaliGemma's Gemma LM processes
        [image_patches | lang_tokens] with full self-attention (image tokens
        first, then causal lang tokens that can attend back to image tokens).

        Extracts the lang→image sub-block from each layer's attention matrix
        and averages over layers and active language tokens.

        Args:
            token_layout: override capture.token_layout if provided.
            lang_mask: [B, n_lang] bool, True = active token (not padding).
                       When provided, only active lang tokens are averaged over.

        Returns:
            [B, n_cameras, N_patches] float32, or None if prefix_attn is empty.
        """
        layout = token_layout or self.token_layout
        if not self.prefix_attn or layout is None:
            return None

        n_img_toks = layout.n_cameras * layout.n_patches_per_camera
        lang_start = n_img_toks  # image tokens occupy positions 0:n_img_toks

        layer_results: list[torch.Tensor] = []
        for layer_idx in sorted(self.prefix_attn.keys()):
            full_attn = self.prefix_attn[layer_idx].float()  # [B, prefix_len, prefix_len]
            if full_attn.shape[-1] < n_img_toks or full_attn.shape[-2] <= lang_start:
                continue
            # lang→img block: query = lang positions, key = img positions
            lang_img = full_attn[:, lang_start:, :n_img_toks]  # [B, n_lang, n_img_toks]

            if lang_mask is not None:
                # Average only over active (non-padding) language tokens
                mask = lang_mask.float()  # [B, n_lang] (or broadcastable)
                n_lang_cap = lang_img.shape[1]
                mask = mask[:, :n_lang_cap].unsqueeze(-1)  # [B, n_lang, 1]
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                averaged = (lang_img * mask).sum(dim=1) / denom.squeeze(1)  # [B, n_img_toks]
            else:
                averaged = lang_img.mean(dim=1)  # [B, n_img_toks]

            layer_results.append(averaged)

        if not layer_results:
            return None

        result = torch.stack(layer_results, dim=0).mean(dim=0)  # [B, n_img_toks]
        B = result.shape[0]
        return result.reshape(B, layout.n_cameras, layout.n_patches_per_camera)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class PI05FeatureProbe:
    """Context manager that captures PI05 prefix hidden states and denoising attention.

    Two quantities are captured per inference call:

    1. Prefix hidden states — output of PaliGemmaWithExpertModel prefix phase
       (image patch + language token representations, shape [B, prefix_len, D]).

    2. Action→image attention — per-layer self-attention weights from
       gemma_expert denoising steps, averaged over heads and action tokens.
       Stored as [B, total_kv_len] per layer per step in denoising_attn.
       Retrieve as [B, n_cameras, N_patches] via capture.action_to_img_attn().

    Three interception points (all via instance __dict__ patching or hooks):

    A. PaliGemmaWithExpertModel.__dict__["forward"] — captures prefix hidden
       states; PI05 calls this directly (bypasses __call__).

    B. Each gemma_expert layer's self_attn.__dict__["forward"] — injects
       output_attentions=True so GemmaAttention computes real weights; called
       via __call__ so this takes effect normally.

    C. register_forward_hook on each self_attn — captures the returned
       (attn_output, attn_weights) tuple after (B).

    D. gemma_expert.model.__dict__["forward"] — groups per-layer hook captures
       into one denoising step; PI05 calls this directly (bypasses __call__).

    E. paligemma.model.language_model.__dict__["forward"] — wraps the prefix
       Gemma LM forward with output_attentions=True and stores per-layer
       prefix self-attention; PI05 calls this directly (bypasses __call__).
       PaliGemma's standard GemmaModel handles output_attentions correctly
       (no PiGemmaModel bug). PI05 already sets eager attention for PaliGemma
       before inference, so real weights are guaranteed.

    NOTE: output_attentions=True is NOT passed to PiGemmaModel.forward (D)
    because PI05's _PiGemmaDecoderLayerBase returns a plain tensor and
    PiGemmaModel has a bug (layer_outputs[1] does tensor indexing for batch
    dim, fails when B=1).

    Example::

        probe = PI05FeatureProbe(policy)
        with probe:
            actions = policy.predict_action_chunk(obs)
            probe.set_token_layout(obs)
        capture = probe.last_capture
        attn  = capture.action_to_img_attn()   # [B, n_cameras, N_patches]
        lattn = capture.lang_to_img_attn()     # [B, n_cameras, N_patches]
    """

    def __init__(self, policy, keep_cpu_copy: bool = True):
        """
        Args:
            policy: PI05Policy instance.
            keep_cpu_copy: move captured tensors to CPU immediately to avoid
                GPU memory accumulation across steps.
        """
        self._policy = policy
        self._pwe = policy.model.paligemma_with_expert
        self._keep_cpu = keep_cpu_copy
        self._current_capture: PI05FeatureCapture | None = None
        self.last_capture: PI05FeatureCapture | None = None
        self._gem_model_patched = None
        self._palilm_patched = None
        self._attn_hooks: list = []
        self._patched_attn_modules: list = []
        self._palilm_attn_hooks: list = []
        self._patched_palilm_attn_modules: list = []

    # ------------------------------------------------------------------
    # Internal callback — prefix phase only
    # ------------------------------------------------------------------

    def _on_forward(self, output) -> None:
        """Called after every PaliGemmaWithExpertModel.forward invocation.

        output = ([prefix_hidden, suffix_hidden], prefix_past_key_values)
        Prefix phase: prefix_hidden is a tensor, suffix_hidden is None.
        Denoising phase: prefix_hidden is None, suffix_hidden is a tensor.
        """
        outputs_list, _ = output
        prefix_out = outputs_list[0]
        if prefix_out is None:
            return  # denoising phase — handled by hooks
        cap = self._current_capture
        if cap is None:
            return
        stored = prefix_out.detach().cpu() if self._keep_cpu else prefix_out.detach()
        cap.prefix_hidden = stored

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> PI05FeatureProbe:
        self._current_capture = PI05FeatureCapture()

        # === A: Patch PaliGemmaWithExpertModel.forward (prefix hidden states) ===
        # PI05's sample_actions calls paligemma_with_expert.forward(...) directly,
        # bypassing nn.Module.__call__ and register_forward_hook.
        _probe = self
        _pwe_instance = self._pwe
        _orig_cls_forward = type(self._pwe).forward

        def _patched_forward(*args, **kwargs):
            output = _orig_cls_forward(_pwe_instance, *args, **kwargs)
            _probe._on_forward(output)
            return output

        self._pwe.__dict__["forward"] = _patched_forward

        # === B+C: Patch self_attn.forward + hooks for denoising attention ===
        #
        # PiGemmaModel.forward has a bug when output_attentions=True:
        #   layer_outputs[1]  — but layer_outputs is a plain tensor, not tuple.
        #   With B=1, tensor[1] hits IndexError (dim 0 size 1).
        #
        # Workaround: inject output_attentions=True at the attention-module level
        # (not at PiGemmaModel level). self_attn is called via __call__, so both
        # the instance-dict patch and register_forward_hook take effect normally.
        _step_buffer: dict[int, torch.Tensor] = {}
        keep_cpu = self._keep_cpu

        layers = self._pwe.gemma_expert.model.layers
        _hooks: list = []
        _patched_attn: list = []

        for layer_idx, decoder_layer in enumerate(layers):
            attn_module = decoder_layer.self_attn
            _orig_attn_cls_fwd = type(attn_module).forward
            _attn_instance = attn_module

            def _make_attn_patch(orig_fwd, instance):
                def _patched_attn_forward(*args, **kwargs):
                    # Force eager attention so real weights are returned.
                    kwargs["output_attentions"] = True
                    return orig_fwd(instance, *args, **kwargs)
                return _patched_attn_forward

            attn_module.__dict__["forward"] = _make_attn_patch(
                _orig_attn_cls_fwd, _attn_instance
            )
            _patched_attn.append(attn_module)

            def _make_hook(lidx):
                def _hook(module, inp, output):
                    # output = (attn_output, attn_weights) from GemmaAttention
                    if not (isinstance(output, (tuple, list)) and len(output) >= 2):
                        return
                    attn_weights = output[1]
                    if attn_weights is None:
                        return
                    # attn_weights: [B, H, action_len, kv_len]
                    # Average over heads and action tokens → [B, kv_len]
                    avg = attn_weights.float().mean(dim=(1, 2))
                    _step_buffer[lidx] = avg.detach().cpu() if keep_cpu else avg.detach()
                return _hook

            h = attn_module.register_forward_hook(_make_hook(layer_idx))
            _hooks.append(h)

        self._attn_hooks = _hooks
        self._patched_attn_modules = _patched_attn

        # === D: Patch gemma_expert.model.forward (group by denoising step) ===
        # Wraps the original forward (without output_attentions=True) so that
        # after each denoising step all hook-captured layer attentions are
        # collected into one dict and appended to capture.denoising_attn.
        _gem_model = self._pwe.gemma_expert.model
        _orig_gem_cls_forward = type(_gem_model).forward

        def _patched_gem_forward(*args, **kwargs):
            _step_buffer.clear()
            output = _orig_gem_cls_forward(_gem_model, *args, **kwargs)
            cap = _probe._current_capture
            if cap is not None and _step_buffer:
                cap.denoising_attn.append(dict(_step_buffer))
            return output

        _gem_model.__dict__["forward"] = _patched_gem_forward
        self._gem_model_patched = _gem_model

        # === E: Patch paligemma.model.language_model for prefix lang→img attention ===
        #
        # paligemma.model.language_model is ALSO PiGemmaModel (confirmed: same bug
        # at pi_gemma.py:307 when output_attentions=True hits layer_outputs[1] on
        # a plain tensor). Apply the identical hook-based workaround as gemma_expert:
        # inject output_attentions=True at the self_attn level, capture via hooks,
        # group into cap.prefix_attn via a language_model.forward wrapper.
        #
        # Unlike gemma_expert (multiple denoising steps), the prefix forward runs
        # exactly once per inference call — no step counter needed.
        _prefix_buf: dict[int, torch.Tensor] = {}

        _palilm = self._pwe.paligemma.model.language_model
        _orig_palilm_cls_fwd = type(_palilm).forward
        _palilm_instance = _palilm

        palilm_layers = _palilm.layers
        _palilm_hooks: list = []
        _patched_palilm_attn: list = []

        for layer_idx, decoder_layer in enumerate(palilm_layers):
            attn_module = decoder_layer.self_attn
            _orig_pattn_cls_fwd = type(attn_module).forward
            _pattn_instance = attn_module

            def _make_palilm_attn_patch(orig_fwd, instance):
                def _patched(*args, **kwargs):
                    kwargs["output_attentions"] = True
                    return orig_fwd(instance, *args, **kwargs)
                return _patched

            attn_module.__dict__["forward"] = _make_palilm_attn_patch(
                _orig_pattn_cls_fwd, _pattn_instance
            )
            _patched_palilm_attn.append(attn_module)

            def _make_palilm_hook(lidx):
                def _hook(module, inp, output):
                    if not (isinstance(output, (tuple, list)) and len(output) >= 2):
                        return
                    attn_weights = output[1]
                    if attn_weights is None:
                        return
                    # [B, H, prefix_len, prefix_len] → avg heads → [B, prefix_len, prefix_len]
                    avg = attn_weights.float().mean(dim=1)
                    _prefix_buf[lidx] = avg.detach().cpu() if keep_cpu else avg.detach()
                return _hook

            h = attn_module.register_forward_hook(_make_palilm_hook(layer_idx))
            _palilm_hooks.append(h)

        self._palilm_attn_hooks = _palilm_hooks
        self._patched_palilm_attn_modules = _patched_palilm_attn

        def _patched_palilm_forward(*args, **kwargs):
            _prefix_buf.clear()
            output = _orig_palilm_cls_fwd(_palilm_instance, *args, **kwargs)
            cap = _probe._current_capture
            if cap is not None and _prefix_buf:
                cap.prefix_attn.update(_prefix_buf)
            return output

        _palilm.__dict__["forward"] = _patched_palilm_forward
        self._palilm_patched = _palilm

        return self

    def __exit__(self, *args) -> None:
        # Restore PaliGemmaWithExpertModel.forward.
        self._pwe.__dict__.pop("forward", None)

        # Restore gemma_expert self_attn patches.
        for attn_module in self._patched_attn_modules:
            attn_module.__dict__.pop("forward", None)
        self._patched_attn_modules = []

        # Remove gemma_expert forward hooks.
        for h in self._attn_hooks:
            h.remove()
        self._attn_hooks = []

        # Restore gemma_expert.model.forward.
        if self._gem_model_patched is not None:
            self._gem_model_patched.__dict__.pop("forward", None)
            self._gem_model_patched = None

        # Restore paligemma language_model self_attn patches.
        for attn_module in self._patched_palilm_attn_modules:
            attn_module.__dict__.pop("forward", None)
        self._patched_palilm_attn_modules = []

        # Remove paligemma language_model forward hooks.
        for h in self._palilm_attn_hooks:
            h.remove()
        self._palilm_attn_hooks = []

        # Restore paligemma.model.language_model.forward.
        if self._palilm_patched is not None:
            self._palilm_patched.__dict__.pop("forward", None)
            self._palilm_patched = None

        self.last_capture = self._current_capture
        self._current_capture = None

    # ------------------------------------------------------------------
    # Token layout helper
    # ------------------------------------------------------------------

    def set_token_layout(
        self,
        observation: dict[str, torch.Tensor],
        n_state_tokens: int = 0,
    ) -> TokenLayout | None:
        """Derive and attach TokenLayout from a preprocessed observation dict.

        Call this inside the with-block (after inference) or just after __exit__.
        Must be called once per inference step because each call produces a new
        PI05FeatureCapture with token_layout=None.

        Args:
            observation: preprocessed observation dict (after policy preprocessor).
            n_state_tokens: additional prefix tokens beyond image + language (0
                for standard PI05 — state is embedded via time MLP, not prefix).

        Returns:
            TokenLayout or None if derivation fails.
        """
        from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_TOKENS

        img_keys = sorted(k for k in observation if OBS_IMAGES in k and "empty" not in k)
        n_cameras = len(img_keys)
        if n_cameras == 0:
            return None

        lang_key = OBS_LANGUAGE_TOKENS
        if lang_key in observation:
            n_lang_tokens = observation[lang_key].shape[-1]
        else:
            n_lang_tokens = getattr(self._policy.config, "tokenizer_max_length", 200)

        cap = self.last_capture or self._current_capture
        if cap is None or cap.prefix_hidden is None:
            return None

        prefix_len = cap.prefix_hidden.shape[1]
        n_img_total = prefix_len - n_lang_tokens - n_state_tokens

        if n_img_total <= 0 or n_img_total % n_cameras != 0:
            # Padding can inflate n_lang_tokens — try attention mask
            attn_mask_key = lang_key.replace(".tokens", ".attention_mask")
            if attn_mask_key in observation:
                n_lang_active = int(observation[attn_mask_key].sum(dim=-1).max().item())
                n_img_total = prefix_len - n_lang_active - n_state_tokens
                n_lang_tokens = n_lang_active

        if n_img_total <= 0 or n_cameras == 0:
            return None

        n_patches_per_camera = max(1, n_img_total // n_cameras)

        layout = TokenLayout(
            n_cameras=n_cameras,
            n_patches_per_camera=n_patches_per_camera,
            n_lang_tokens=n_lang_tokens,
            prefix_len=prefix_len,
        )
        cap.token_layout = layout
        return layout
