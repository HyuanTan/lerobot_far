"""
SmolVLA attention-weight visualizer.

Provides three visualization modes:
  1. spatial_heatmap   — per-camera patch attention overlaid on the original image
  2. language_heatmap  — per-language-token attention bar chart
  3. temporal_grid     — how attention evolves across denoising steps

All functions accept an AttentionCapture produced by SmolVLAAttentionProbe
and return matplotlib Figure objects (or save them to disk).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from .attn_probe import AttentionCapture, TokenOffsets


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for attention visualization. "
            "Install it with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _aggregate_cross_attn(
    capture: AttentionCapture,
    steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    action_token_agg: str = "max",
) -> torch.Tensor:
    """Aggregate cross-attn maps to shape [B, prefix_len].

    Args:
        capture: AttentionCapture from SmolVLAAttentionProbe.
        steps: denoising steps to include; None = all.
        layers: layer indices to include; None = all.
        action_token_agg: how to aggregate over action token dim ('max' or 'mean').

    Returns:
        Tensor [B, prefix_len], float32.
    """
    maps = []
    step_range = steps if steps is not None else range(len(capture.cross_attn))
    for s in step_range:
        if s >= len(capture.cross_attn):
            continue
        step_maps = capture.cross_attn[s]
        layer_range = layers if layers is not None else list(step_maps.keys())
        for l in layer_range:
            if l not in step_maps:
                continue
            p = step_maps[l].float()  # [B, H, chunk_size, prefix_len]
            p = p.mean(dim=1)         # [B, chunk_size, prefix_len]
            if action_token_agg == "max":
                p, _ = p.max(dim=1)   # [B, prefix_len]
            else:
                p = p.mean(dim=1)     # [B, prefix_len]
            maps.append(p)

    if not maps:
        raise ValueError("No cross-attention maps found in capture.")
    return torch.stack(maps, dim=0).mean(dim=0)  # [B, prefix_len]


def _patch_grid_shape(n_patches: int, img_hw: tuple[int, int] | None) -> tuple[int, int]:
    """Return (H_grid, W_grid) for n_patches."""
    if img_hw is not None:
        h, w = img_hw
        # Vision model patch size is typically 14 for SigLIP
        for patch_size in [14, 16, 32]:
            ph, pw = h // patch_size, w // patch_size
            if ph * pw == n_patches:
                return ph, pw
    # Fallback: assume square or near-square
    side = int(math.isqrt(n_patches))
    if side * side == n_patches:
        return side, side
    # Non-square: find closest factorisation
    for h in range(side, 0, -1):
        if n_patches % h == 0:
            return h, n_patches // h
    return 1, n_patches


# ---------------------------------------------------------------------------
# 1. Spatial image attention heatmap
# ---------------------------------------------------------------------------


def spatial_heatmap(
    capture: AttentionCapture,
    images: list[np.ndarray] | None = None,
    cam_idx: int = 0,
    steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    batch_idx: int = 0,
    img_hw: tuple[int, int] | None = None,
    alpha: float = 0.45,
    colormap: str = "hot",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Overlay spatial attention heatmap on the corresponding camera image.

    Args:
        capture: AttentionCapture from SmolVLAAttentionProbe.
        images: list of RGB uint8 numpy arrays [H, W, 3], one per camera.
            If None, only the heatmap is shown without background.
        cam_idx: which camera to visualize.
        steps: denoising steps to include (None = all).
        layers: cross-attn layer indices to include (None = all).
        batch_idx: batch element to visualize.
        img_hw: (height, width) of the original image for patch grid inference.
        alpha: opacity of the attention overlay.
        colormap: matplotlib colormap name.
        title: figure title.
        save_path: if given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    offsets = capture.token_offsets
    if offsets is None:
        raise ValueError("capture.token_offsets is None — call probe.build_token_offsets() first.")
    if cam_idx >= len(offsets.camera_slices):
        raise ValueError(f"cam_idx={cam_idx} out of range (n_cameras={len(offsets.camera_slices)})")

    agg = _aggregate_cross_attn(capture, steps=steps, layers=layers)  # [B, prefix_len]
    attn = agg[batch_idx]  # [prefix_len]

    s, e = offsets.camera_slices[cam_idx]
    patch_attn = attn[s:e].numpy()  # [n_patches]
    n_patches = e - s
    ph, pw = _patch_grid_shape(n_patches, img_hw)

    # Crop/pad to ph*pw if needed
    patch_attn = patch_attn[:ph * pw].reshape(ph, pw)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if images is not None and cam_idx < len(images):
        img = images[cam_idx]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        ax.imshow(img)

    # Upsample patch attn to image size
    from PIL import Image as PILImage
    import io
    attn_min, attn_max = patch_attn.min(), patch_attn.max()
    if attn_max > attn_min:
        patch_norm = (patch_attn - attn_min) / (attn_max - attn_min)
    else:
        patch_norm = patch_attn

    target_hw = img_hw if img_hw is not None else (ph * 14, pw * 14)
    pil_attn = PILImage.fromarray((patch_norm * 255).astype(np.uint8), mode="L")
    pil_attn = pil_attn.resize((target_hw[1], target_hw[0]), PILImage.BILINEAR)
    attn_up = np.array(pil_attn) / 255.0

    ax.imshow(attn_up, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=colormap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title or f"Camera {cam_idx} — spatial attention (mean over selected steps/layers)")
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 2. Language token attention heatmap
# ---------------------------------------------------------------------------


def language_heatmap(
    capture: AttentionCapture,
    token_labels: list[str] | None = None,
    steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    batch_idx: int = 0,
    max_tokens: int = 48,
    colormap: str = "Blues",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Bar chart of attention per language token.

    Args:
        capture: AttentionCapture from SmolVLAAttentionProbe.
        token_labels: list of token strings for x-axis labels. If None, token indices are used.
        steps: denoising steps to include (None = all).
        layers: cross-attn layer indices to include (None = all).
        batch_idx: batch element to visualize.
        max_tokens: truncate to this many tokens.
        colormap: matplotlib colormap name.
        title: figure title.
        save_path: if given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    offsets = capture.token_offsets
    if offsets is None:
        raise ValueError("capture.token_offsets is None — call probe.build_token_offsets() first.")

    agg = _aggregate_cross_attn(capture, steps=steps, layers=layers)  # [B, prefix_len]
    attn = agg[batch_idx]  # [prefix_len]

    s, e = offsets.lang_slice
    lang_attn = attn[s:e].numpy()  # [n_lang_tokens]
    n_toks = min(len(lang_attn), max_tokens)
    lang_attn = lang_attn[:n_toks]

    labels = (token_labels or [str(i) for i in range(n_toks)])[:n_toks]

    cmap = plt.get_cmap(colormap)
    norm_vals = lang_attn / (lang_attn.max() + 1e-9)
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(1, 1, figsize=(max(8, n_toks * 0.35), 4))
    bars = ax.bar(range(n_toks), lang_attn, color=colors)
    ax.set_xticks(range(n_toks))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Attention probability")
    ax.set_title(title or "Language token attention (mean over selected steps/layers)")
    ax.set_xlim(-0.5, n_toks - 0.5)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. Temporal attention evolution grid
# ---------------------------------------------------------------------------


def temporal_grid(
    capture: AttentionCapture,
    images: list[np.ndarray] | None = None,
    cam_idx: int = 0,
    steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    batch_idx: int = 0,
    img_hw: tuple[int, int] | None = None,
    alpha: float = 0.5,
    colormap: str = "hot",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Grid showing spatial attention for each denoising step.

    Rows = selected steps, single column with image overlay.

    Args:
        capture: AttentionCapture from SmolVLAAttentionProbe.
        images: list of RGB numpy arrays, one per camera.
        cam_idx: camera to visualize.
        steps: which denoising steps to show (None = all).
        layers: which layers to aggregate over per step (None = all).
        batch_idx: batch element to visualize.
        img_hw: (H, W) of original image.
        alpha: heatmap opacity.
        colormap: matplotlib colormap name.
        title: overall figure title.
        save_path: if given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    offsets = capture.token_offsets
    if offsets is None:
        raise ValueError("capture.token_offsets is None — call probe.build_token_offsets() first.")
    if cam_idx >= len(offsets.camera_slices):
        raise ValueError(f"cam_idx={cam_idx} out of range.")

    n_steps = len(capture.cross_attn)
    step_range = list(steps) if steps is not None else list(range(n_steps))
    n_show = len(step_range)

    s_cam, e_cam = offsets.camera_slices[cam_idx]
    n_patches = e_cam - s_cam
    ph, pw = _patch_grid_shape(n_patches, img_hw)

    bg_image = None
    if images is not None and cam_idx < len(images):
        bg_image = images[cam_idx]
        if bg_image.dtype != np.uint8:
            bg_image = (np.clip(bg_image, 0.0, 1.0) * 255).astype(np.uint8)

    ncols = min(5, n_show)
    nrows = math.ceil(n_show / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if n_show == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    from PIL import Image as PILImage

    for plot_idx, step_idx in enumerate(step_range):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]

        if step_idx >= len(capture.cross_attn) or not capture.cross_attn[step_idx]:
            ax.axis("off")
            continue

        # Aggregate over layers for this step
        agg = _aggregate_cross_attn(capture, steps=[step_idx], layers=layers)
        patch_attn = agg[batch_idx, s_cam:e_cam].numpy()
        patch_attn = patch_attn[:ph * pw].reshape(ph, pw)

        attn_min, attn_max = patch_attn.min(), patch_attn.max()
        patch_norm = (patch_attn - attn_min) / (attn_max - attn_min + 1e-9)

        if bg_image is not None:
            ax.imshow(bg_image)
            target_hw = (bg_image.shape[0], bg_image.shape[1])
        else:
            target_hw = img_hw or (ph * 14, pw * 14)

        pil_attn = PILImage.fromarray((patch_norm * 255).astype(np.uint8), mode="L")
        pil_attn = pil_attn.resize((target_hw[1], target_hw[0]), PILImage.BILINEAR)
        attn_up = np.array(pil_attn) / 255.0
        ax.imshow(attn_up, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
        ax.set_title(f"Step {step_idx}", fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for plot_idx in range(n_show, nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row][col].axis("off")

    fig.suptitle(title or f"Temporal attention evolution — camera {cam_idx}", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Convenience: save all standard plots for one inference call
# ---------------------------------------------------------------------------


def save_inference_attention(
    capture: AttentionCapture,
    output_dir: str | Path,
    images: list[np.ndarray] | None = None,
    token_labels: list[str] | None = None,
    img_hw: tuple[int, int] | None = None,
    episode: int = 0,
    timestep: int = 0,
) -> list[Path]:
    """Save the standard set of attention visualizations for one inference call.

    Saves:
      - spatial_cam{i}_ep{episode}_t{timestep}.png  per camera
      - language_ep{episode}_t{timestep}.png
      - temporal_cam0_ep{episode}_t{timestep}.png

    Args:
        capture: AttentionCapture with token_offsets set.
        output_dir: directory to write PNG files into.
        images: list of RGB numpy arrays [H, W, 3], one per camera.
        token_labels: decoded language token strings for the language heatmap.
        img_hw: (H, W) of original image for patch-grid inference.
        episode: episode number for file naming.
        timestep: observation timestep for file naming.

    Returns:
        List of saved file paths.
    """
    _require_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    offsets = capture.token_offsets

    # Spatial heatmaps — one per camera
    n_cameras = len(offsets.camera_slices) if offsets else (len(images) if images else 1)
    for cam_idx in range(n_cameras):
        path = output_dir / f"spatial_cam{cam_idx}_ep{episode}_t{timestep:04d}.png"
        try:
            fig = spatial_heatmap(
                capture,
                images=images,
                cam_idx=cam_idx,
                img_hw=img_hw,
                title=f"Ep {episode} T {timestep} — Camera {cam_idx} attention",
                save_path=path,
            )
            plt.close(fig)
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"spatial_heatmap cam{cam_idx} failed: {exc}")

    # Language heatmap
    if offsets is not None:
        path = output_dir / f"language_ep{episode}_t{timestep:04d}.png"
        try:
            fig = language_heatmap(
                capture,
                token_labels=token_labels,
                title=f"Ep {episode} T {timestep} — language attention",
                save_path=path,
            )
            plt.close(fig)
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"language_heatmap failed: {exc}")

    # Temporal grid (first camera only)
    path = output_dir / f"temporal_cam0_ep{episode}_t{timestep:04d}.png"
    try:
        fig = temporal_grid(
            capture,
            images=images,
            cam_idx=0,
            img_hw=img_hw,
            title=f"Ep {episode} T {timestep} — temporal attention evolution",
            save_path=path,
        )
        plt.close(fig)
        saved.append(path)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"temporal_grid failed: {exc}")

    return saved
