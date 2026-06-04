"""PI05 task-conditioned image feature visualizer.

Five visualization types built on PI05FeatureCapture:

  1. lang_img_attn_map       — per-camera spatial heatmap of language→image
                               attention from PaliGemma prefix self-attention
                               (averaged over layers, heads, active lang tokens)
                               ← semantically meaningful: visual grounding site
  2. action_img_attn_map     — per-camera spatial heatmap of action→image
                               attention from gemma_expert denoising forward
                               (averaged over steps, heads, and action tokens)
  3. lang_similarity_bar     — per-language-token bar chart of mean image
                               patch cosine similarity (prefix hidden states)
  4. temporal_drift_line     — L2 distance between consecutive-step image
                               features, per camera (episode-level)
  5. episode_feature_pca     — 2-D PCA of mean image features across steps
                               (episode-level)

Convenience entry points:
  save_step_features()       — saves plots 1, 2 & 3 for one inference step
  save_episode_features()    — saves plots 4 & 5 for a whole episode
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from .pi05_feature_probe import PI05FeatureCapture, TokenLayout


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required: pip install matplotlib")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between rows of a [M, D] and b [N, D] → [M, N]."""
    a_n = torch.nn.functional.normalize(a.float(), dim=-1)
    b_n = torch.nn.functional.normalize(b.float(), dim=-1)
    return a_n @ b_n.T


def _patch_grid(n_patches: int, img_hw: tuple[int, int] | None) -> tuple[int, int]:
    """Infer (H_grid, W_grid) for n_patches, trying common patch sizes first."""
    if img_hw is not None:
        h, w = img_hw
        for ps in [14, 16, 32, 8]:
            ph, pw = h // ps, w // ps
            if ph * pw == n_patches:
                return ph, pw
    side = int(math.isqrt(n_patches))
    if side * side == n_patches:
        return side, side
    for h in range(side, 0, -1):
        if n_patches % h == 0:
            return h, n_patches // h
    return 1, n_patches


# ---------------------------------------------------------------------------
# Shared overlay helper
# ---------------------------------------------------------------------------


def _overlay_heatmap(
    ax,
    patch_attn: "np.ndarray",
    img_hw: "tuple[int, int]",
    images: "list[np.ndarray] | None",
    cam_idx: int,
    alpha: float,
    colormap: str,
) -> None:
    """Normalize patch_attn, resize to image HW, and overlay on ax."""
    from PIL import Image as PILImage

    if images is not None and cam_idx < len(images):
        img = images[cam_idx]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        ax.imshow(img)
        target_hw = (img.shape[0], img.shape[1])
    else:
        target_hw = img_hw

    vmin, vmax = patch_attn.min(), patch_attn.max()
    patch_norm = (patch_attn - vmin) / (vmax - vmin + 1e-9)

    pil_attn = PILImage.fromarray((patch_norm * 255).astype(np.uint8), mode="L")
    pil_attn = pil_attn.resize((target_hw[1], target_hw[0]), PILImage.BILINEAR)
    heat = np.array(pil_attn) / 255.0

    ax.imshow(heat, cmap=colormap, alpha=alpha, vmin=0, vmax=1)

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=colormap)
    sm.set_array([])
    ax.get_figure().colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 1. Language→image attention map — PaliGemma prefix self-attention
# ---------------------------------------------------------------------------


def lang_img_attn_map(
    capture: PI05FeatureCapture,
    images: list[np.ndarray] | None = None,
    cam_idx: int = 0,
    batch_idx: int = 0,
    img_hw: tuple[int, int] | None = None,
    token_layout: TokenLayout | None = None,
    lang_mask: torch.Tensor | None = None,
    alpha: float = 0.45,
    colormap: str = "jet",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Spatial heatmap of language→image attention from PaliGemma prefix phase.

    During the prefix forward, PaliGemma's Gemma LM processes
    [image_patches | lang_tokens] jointly. Language tokens can attend back to
    image patches (image tokens appear earlier in the causal sequence).

    This sub-block (lang→img) is where VISUAL GROUNDING actually happens:
    the model decides which image regions are relevant to the task description.
    Averaged over all transformer layers, attention heads, and active language
    tokens to produce a single spatial map per camera.

    High intensity → that image region was referenced by the task description.

    Args:
        capture: PI05FeatureCapture with token_layout and prefix_attn set.
        images: list of HWC uint8 numpy arrays, one per camera.
        cam_idx: which camera to visualize.
        batch_idx: which batch element.
        img_hw: (H, W) of the original image for patch grid inference.
        token_layout: override capture.token_layout if provided.
        lang_mask: [B, n_lang] bool, True = active token (exclude padding).
        alpha: heatmap overlay opacity.
        colormap: matplotlib colormap (default "jet" for lang attention).
        title: figure title.
        save_path: if given, save to this path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    layout = token_layout or capture.token_layout
    if layout is None:
        raise ValueError("token_layout is None — call probe.set_token_layout() first.")
    if cam_idx >= layout.n_cameras:
        raise ValueError(f"cam_idx={cam_idx} out of range (n_cameras={layout.n_cameras})")

    attn = capture.lang_to_img_attn(layout, lang_mask=lang_mask)  # [B, n_cameras, N_patches]
    if attn is None:
        raise ValueError(
            "No prefix_attn data in capture. Ensure PI05FeatureProbe patched "
            "paligemma.model.language_model and the model uses eager attention."
        )

    patch_attn = attn[batch_idx, cam_idx].float().numpy()  # [N_patches]
    N = layout.n_patches_per_camera

    if img_hw is None and images is not None and cam_idx < len(images):
        img_hw = (images[cam_idx].shape[0], images[cam_idx].shape[1])

    ph, pw = _patch_grid(N, img_hw)
    patch_grid_arr = patch_attn[: ph * pw].reshape(ph, pw)

    fallback_hw = img_hw or (ph * 14, pw * 14)
    n_layers = len(capture.prefix_attn)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _overlay_heatmap(ax, patch_grid_arr, fallback_hw, images, cam_idx, alpha, colormap)
    ax.set_title(title or f"Camera {cam_idx} — lang→image attn ({n_layers} PaliGemma layers)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Action→image attention map — real attention from gemma_expert denoising
# ---------------------------------------------------------------------------


def action_img_attn_map(
    capture: PI05FeatureCapture,
    images: list[np.ndarray] | None = None,
    cam_idx: int = 0,
    batch_idx: int = 0,
    img_hw: tuple[int, int] | None = None,
    token_layout: TokenLayout | None = None,
    alpha: float = 0.45,
    colormap: str = "hot",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Spatial heatmap of action→image attention from gemma_expert denoising.

    During each denoising step, gemma_expert attends to the prefix KV cache
    (image patches + language tokens) and its own action tokens.  This
    function extracts the action→image sub-block, averaged over all denoising
    steps, transformer layers, attention heads, and action tokens.

    High intensity → the action denoiser pays attention to that image region.
    This is semantically equivalent to SmolVLA cross-attention maps.

    Requires PI05FeatureProbe to have patched gemma_expert.model.forward with
    output_attentions=True, and eager (non-Flash) attention at inference time.

    Args:
        capture: PI05FeatureCapture with token_layout and denoising_attn set.
        images: list of HWC uint8 numpy arrays, one per camera.
        cam_idx: which camera to visualize.
        batch_idx: which batch element.
        img_hw: (H, W) of the original image for patch grid inference.
        token_layout: override capture.token_layout if provided.
        alpha: heatmap overlay opacity.
        colormap: matplotlib colormap.
        title: figure title.
        save_path: if given, save to this path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    layout = token_layout or capture.token_layout
    if layout is None:
        raise ValueError("token_layout is None — call probe.set_token_layout() first.")

    if cam_idx >= layout.n_cameras:
        raise ValueError(f"cam_idx={cam_idx} out of range (n_cameras={layout.n_cameras})")

    attn = capture.action_to_img_attn(layout)  # [B, n_cameras, N_patches]
    if attn is None:
        raise ValueError(
            "No denoising attention data in capture. "
            "Ensure PI05FeatureProbe patched gemma_expert.model and the model "
            "uses eager (not Flash) attention."
        )

    patch_attn = attn[batch_idx, cam_idx].float().numpy()  # [N_patches]
    N = layout.n_patches_per_camera

    if img_hw is None and images is not None and cam_idx < len(images):
        img_hw = (images[cam_idx].shape[0], images[cam_idx].shape[1])

    ph, pw = _patch_grid(N, img_hw)
    patch_grid_arr = patch_attn[: ph * pw].reshape(ph, pw)
    fallback_hw = img_hw or (ph * 14, pw * 14)
    n_steps = len(capture.denoising_attn)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _overlay_heatmap(ax, patch_grid_arr, fallback_hw, images, cam_idx, alpha, colormap)
    ax.set_title(title or f"Camera {cam_idx} — action→image attn ({n_steps} denoising steps)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Language similarity bar — per-token mean patch cosine-sim
# ---------------------------------------------------------------------------


def lang_similarity_bar(
    capture: PI05FeatureCapture,
    token_labels: list[str] | None = None,
    cam_idx: int = 0,
    batch_idx: int = 0,
    lang_mask: torch.Tensor | None = None,
    max_tokens: int = 60,
    colormap: str = "Blues",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Bar chart: mean cosine-similarity between image patches and each language token.

    Shows which words in the task description are most aligned with the visual
    representation after joint self-attention processing.

    Args:
        capture: PI05FeatureCapture with token_layout set.
        token_labels: decoded token strings for x-axis labels.
        cam_idx: camera index.
        batch_idx: batch element.
        lang_mask: [B, n_lang] bool mask (1 = active/non-padding token).
        max_tokens: truncate display to this many tokens.
        colormap: matplotlib colormap for bar colors.
        title: figure title.
        save_path: save path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    if capture.token_layout is None:
        raise ValueError("capture.token_layout is None — call probe.set_token_layout() first.")

    img_feats = capture.img_features
    lang_feats = capture.lang_features
    if img_feats is None or lang_feats is None:
        raise ValueError("capture has no feature data.")

    patches = img_feats[batch_idx, cam_idx].float()  # [N_patches, D]
    lang = lang_feats[batch_idx].float()             # [n_lang, D]

    # Compute [N_patches, n_lang] → mean over patches → [n_lang]
    sim_matrix = _cosine_sim(patches, lang)          # [N_patches, n_lang]
    per_token_sim = sim_matrix.mean(dim=0).numpy()   # [n_lang]

    # Apply active mask
    if lang_mask is not None:
        active = lang_mask[batch_idx].bool().numpy()
        per_token_sim = per_token_sim[active]
        if token_labels is not None:
            active_idxs = np.where(active)[0]
            token_labels = [token_labels[i] for i in active_idxs if i < len(token_labels)]

    n_toks = min(len(per_token_sim), max_tokens)
    per_token_sim = per_token_sim[:n_toks]
    labels = (token_labels or [str(i) for i in range(n_toks)])[:n_toks]

    cmap = plt.get_cmap(colormap)
    vmin, vmax = per_token_sim.min(), per_token_sim.max()
    norm_vals = (per_token_sim - vmin) / (vmax - vmin + 1e-9)
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(max(8, n_toks * 0.35), 4))
    ax.bar(range(n_toks), per_token_sim, color=colors)
    ax.set_xticks(range(n_toks))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean patch cosine-similarity")
    ax.set_title(title or f"Camera {cam_idx} — image-language token alignment")
    ax.set_xlim(-0.5, n_toks - 0.5)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Temporal feature drift — L2 distance between consecutive steps
# ---------------------------------------------------------------------------


def temporal_drift_line(
    captures: list[PI05FeatureCapture],
    batch_idx: int = 0,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Line chart of per-camera feature drift (L2) across episode timesteps.

    Each point t represents ||mean_img_feat[t] - mean_img_feat[t-1]||₂.

    Args:
        captures: list of PI05FeatureCapture, one per inference step, in order.
        batch_idx: batch element.
        title: figure title.
        save_path: save path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    # Collect [T, n_cameras, D] mean image features
    valid = [c for c in captures if c.img_features_mean is not None]
    if len(valid) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data (<2 steps)", ha="center", va="center")
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    feats = torch.stack([c.img_features_mean[batch_idx] for c in valid], dim=0).float()  # [T, n_cam, D]
    n_cameras = feats.shape[1]
    diffs = (feats[1:] - feats[:-1]).norm(dim=-1)  # [T-1, n_cam]

    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 0.3), 4))
    steps = list(range(1, len(valid)))
    for cam in range(n_cameras):
        ax.plot(steps, diffs[:, cam].numpy(), marker="o", markersize=3, label=f"Camera {cam}")

    ax.set_xlabel("Inference step")
    ax.set_ylabel("||feat[t] − feat[t−1]||₂")
    ax.set_title(title or "Temporal feature drift (per camera)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Episode feature PCA — 2-D scatter of mean image features over time
# ---------------------------------------------------------------------------


def episode_feature_pca(
    captures: list[PI05FeatureCapture],
    cam_idx: int = 0,
    batch_idx: int = 0,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """2-D PCA scatter of mean image features across episode inference steps.

    Colored by step index; a clear trajectory indicates the scene evolves
    smoothly in feature space (task progression).

    Args:
        captures: list of PI05FeatureCapture in temporal order.
        cam_idx: camera to visualize.
        batch_idx: batch element.
        title: figure title.
        save_path: save path.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    valid = [c for c in captures if c.img_features_mean is not None]
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data (<3 steps)", ha="center", va="center")
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    feats = torch.stack(
        [c.img_features_mean[batch_idx, cam_idx] for c in valid], dim=0
    ).float()  # [T, D]

    # Simple PCA via SVD
    mean = feats.mean(dim=0, keepdim=True)
    centered = feats - mean
    try:
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T  # [T, 2]
    except Exception:
        # Fallback: first two raw dimensions
        proj = centered[:, :2]

    proj_np = proj.numpy()
    T = len(valid)
    colors = plt.cm.viridis(np.linspace(0, 1, T))

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(T - 1):
        ax.plot(proj_np[i:i+2, 0], proj_np[i:i+2, 1], color=colors[i], linewidth=1)
    sc = ax.scatter(proj_np[:, 0], proj_np[:, 1], c=np.arange(T), cmap="viridis", s=30, zorder=3)
    plt.colorbar(sc, ax=ax, label="Step")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title or f"Camera {cam_idx} — image feature PCA trajectory")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: save step-level and episode-level plots
# ---------------------------------------------------------------------------


def save_step_features(
    capture: PI05FeatureCapture,
    output_dir: str | Path,
    images: list[np.ndarray] | None = None,
    token_labels: list[str] | None = None,
    img_hw: tuple[int, int] | None = None,
    lang_mask: torch.Tensor | None = None,
    episode: int = 0,
    timestep: int = 0,
) -> list[Path]:
    """Save per-step feature visualization plots for one inference call.

    Saves per camera:
      lang_img_attn_cam{i}_ep{episode}_t{timestep:04d}.png  — language→image attn
                                                               (PaliGemma prefix, semantic)
      action_attn_cam{i}_ep{episode}_t{timestep:04d}.png    — action→image attn
                                                               (gemma_expert denoising)
      lang_similarity_cam{i}_ep{episode}_t{timestep:04d}.png — per-token cosine-sim bar

    Returns list of saved paths.
    """
    _require_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if capture.token_layout is None:
        import logging
        logging.getLogger(__name__).warning(
            "save_step_features: token_layout is None — call probe.set_token_layout() first."
        )
        return []

    saved: list[Path] = []
    n_cameras = capture.token_layout.n_cameras

    for cam_idx in range(n_cameras):
        # 1. Language→image attention heatmap (PaliGemma prefix — semantic grounding)
        path = output_dir / f"lang_img_attn_cam{cam_idx}_ep{episode}_t{timestep:04d}.png"
        try:
            fig = lang_img_attn_map(
                capture,
                images=images,
                cam_idx=cam_idx,
                img_hw=img_hw,
                lang_mask=lang_mask,
                title=f"Ep {episode} T {timestep} — Camera {cam_idx} lang→image attn",
                save_path=path,
            )
            plt.close(fig)
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"lang_img_attn_map cam{cam_idx} failed: {exc}")

        # 2. Action→image attention heatmap (gemma_expert denoising)
        path = output_dir / f"action_attn_cam{cam_idx}_ep{episode}_t{timestep:04d}.png"
        try:
            fig = action_img_attn_map(
                capture,
                images=images,
                cam_idx=cam_idx,
                img_hw=img_hw,
                title=f"Ep {episode} T {timestep} — Camera {cam_idx} action→image attn",
                save_path=path,
            )
            plt.close(fig)
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"action_img_attn_map cam{cam_idx} failed: {exc}")

        # 3. Language token cosine-similarity bar
        path = output_dir / f"lang_similarity_cam{cam_idx}_ep{episode}_t{timestep:04d}.png"
        try:
            fig = lang_similarity_bar(
                capture,
                token_labels=token_labels,
                cam_idx=cam_idx,
                lang_mask=lang_mask,
                title=f"Ep {episode} T {timestep} — Camera {cam_idx} lang alignment",
                save_path=path,
            )
            plt.close(fig)
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"lang_similarity_bar cam{cam_idx} failed: {exc}")

    return saved


def save_episode_features(
    captures: list[PI05FeatureCapture],
    output_dir: str | Path,
    episode: int = 0,
) -> list[Path]:
    """Save temporal drift and PCA plots for a completed episode.

    Saves:
      temporal_drift_ep{episode}.png
      feature_pca_cam{i}_ep{episode}.png  (per camera)

    Returns list of saved paths.
    """
    _require_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # Temporal drift (all cameras combined)
    path = output_dir / f"temporal_drift_ep{episode:04d}.png"
    try:
        fig = temporal_drift_line(
            captures,
            title=f"Episode {episode} — temporal feature drift",
            save_path=path,
        )
        plt.close(fig)
        saved.append(path)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"temporal_drift_line failed: {exc}")

    # PCA per camera
    n_cameras = 0
    for c in captures:
        if c.token_layout is not None:
            n_cameras = c.token_layout.n_cameras
            break

    for cam_idx in range(n_cameras):
        path = output_dir / f"feature_pca_cam{cam_idx}_ep{episode:04d}.png"
        try:
            fig = episode_feature_pca(
                captures,
                cam_idx=cam_idx,
                title=f"Episode {episode} — Camera {cam_idx} feature PCA",
                save_path=path,
            )
            plt.close(fig)
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"episode_feature_pca cam{cam_idx} failed: {exc}")

    return saved
