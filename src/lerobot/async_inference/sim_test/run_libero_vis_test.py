"""Unified in-process LIBERO evaluation with attention/feature visualization.

Supports SmolVLA (cross-attention maps) and PI05 (lang→image + action→image
attention heatmaps) from a single script.  Switch models via --policy_type and
enable visualization via --enable_attn_vis.

This is the in-process counterpart of running attn_policy_server.py +
run_libero_test.py together, without any gRPC overhead.

Usage::

    # SmolVLA — cross-attention heatmaps
    python -m lerobot.async_inference.sim_test.run_libero_vis_test \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=lerobot/smolvla_base \\
        --env_task=libero_10 \\
        --enable_attn_vis=true \\
        --attn_output_dir=./smolvla_attn \\
        --attn_save_every_n=3

    # PI05 — lang→image + action→image heatmaps
    python -m lerobot.async_inference.sim_test.run_libero_vis_test \\
        --policy_type=pi05 \\
        --pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044 \\
        --env_task=libero_spatial \\
        --enable_attn_vis=true \\
        --attn_output_dir=./pi05_attn \\
        --attn_save_every_n=5 \\
        --save_episode_plots=true

    # Either model — evaluation only (no visualization overhead)
    python -m lerobot.async_inference.sim_test.run_libero_vis_test \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=lerobot/smolvla_base \\
        --env_task=libero_10 \\
        --enable_attn_vis=false

Output files (when enable_attn_vis=true):
  SmolVLA:  <attn_output_dir>/ep{N}_t{T:04d}_cam{C}_*.png
  PI05:     <attn_output_dir>/lang_img_attn_cam{C}_ep{N}_t{T:04d}.png
             <attn_output_dir>/action_attn_cam{C}_ep{N}_t{T:04d}.png
             <attn_output_dir>/lang_similarity_cam{C}_ep{N}_t{T:04d}.png
             <attn_output_dir>/temporal_drift_ep{N}.png   (if save_episode_plots)
             <attn_output_dir>/pca_ep{N}.png              (if save_episode_plots)
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import torch

from .sim_client import EpisodeResult, _extract_success, _get_task_description


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LiberoVisConfig:
    """Unified in-process LIBERO evaluation with optional attention visualization."""

    # ── LIBERO environment ───────────────────────────────────────────────────
    env_task: str = field(
        default="libero_10",
        metadata={"help": "LIBERO suite name (libero_10, libero_spatial, libero_object, …)"},
    )
    obs_type: str = field(
        default="pixels_agent_pos",
        metadata={"help": "Observation type: 'pixels' or 'pixels_agent_pos'"},
    )
    camera_name: str = field(
        default="agentview_image,robot0_eye_in_hand_image",
        metadata={"help": "Comma-separated LIBERO camera names"},
    )
    task_ids: list[int] | None = field(
        default=None,
        metadata={"help": "Specific task IDs to evaluate (None = all tasks in suite)"},
    )
    max_episode_steps: int = field(
        default=500,
        metadata={"help": "Max steps per episode"},
    )
    episodes_per_task: int = field(
        default=1,
        metadata={"help": "Number of evaluation episodes per task"},
    )

    # ── Policy ───────────────────────────────────────────────────────────────
    policy_type: str = field(
        default="smolvla",
        metadata={"help": "Policy type: 'smolvla' or 'pi05'"},
    )
    pretrained_name_or_path: str = field(
        default="",
        metadata={"help": "HF Hub repo or local path to the pretrained checkpoint"},
    )
    policy_device: str = field(
        default="cuda",
        metadata={"help": "Device for policy inference ('cuda', 'cpu', 'cuda:1', …)"},
    )

    # ── Visualization master switch ──────────────────────────────────────────
    enable_attn_vis: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable attention/feature visualization. "
                "False = zero probe overhead, plain evaluation only."
            )
        },
    )
    attn_output_dir: str = field(
        default="./attn_vis",
        metadata={"help": "Directory to write visualization PNGs"},
    )
    attn_save_every_n: int = field(
        default=1,
        metadata={"help": "Save visualizations every N inference calls (1 = every call)"},
    )
    keep_cpu_copy: bool = field(
        default=True,
        metadata={"help": "Move captured attention tensors to CPU immediately to save GPU memory"},
    )
    save_episode_plots: bool = field(
        default=True,
        metadata={
            "help": (
                "Save episode-level temporal drift + PCA trajectory plots at episode end. "
                "Applies to PI05 only (requires ≥ 2 captures per episode)."
            )
        },
    )

    # ── Video recording ──────────────────────────────────────────────────────
    save_video: bool = field(
        default=False,
        metadata={"help": "Save a per-episode mp4 video of env observations"},
    )
    video_dir: str = field(
        default="./libero_vis_videos",
        metadata={"help": "Directory for episode video files (ep{N}_{success|failed}.mp4)"},
    )
    video_camera: str = field(
        default="agentview_image",
        metadata={"help": "Camera key in obs['pixels'] to use for video recording"},
    )
    video_fps: int = field(
        default=30,
        metadata={"help": "Frames per second for saved videos"},
    )

    # ── Results ──────────────────────────────────────────────────────────────
    results_dir: str = field(
        default="./libero_vis_results",
        metadata={"help": "Directory for per-episode and aggregate JSON result files"},
    )
    save_results: bool = field(
        default=True,
        metadata={"help": "Persist results to JSON"},
    )


# ---------------------------------------------------------------------------
# Image extraction helper (shared between SmolVLA and PI05 paths)
# ---------------------------------------------------------------------------


def _extract_images(observation: dict) -> list[np.ndarray]:
    """Return list of [H, W, C] uint8 arrays for each camera in `observation`."""
    from lerobot.utils.constants import OBS_IMAGES

    images: list[np.ndarray] = []
    for k in sorted(k for k in observation if OBS_IMAGES in k and "empty" not in k):
        t = observation[k]
        if t.ndim == 4:
            t = t[0]  # first batch element
        arr = t.detach().cpu().float().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))  # [C,H,W] → [H,W,C]
        images.append(arr)
    return images


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _extract_video_frame(obs_np: dict, camera: str) -> np.ndarray | None:
    """Extract a single (H, W, 3) uint8 frame from a SyncVectorEnv observation.

    Tries `camera` first, then falls back to the first available key in
    obs['pixels'].  Applies the same 180° flip that LiberoProcessorStep uses
    so the saved video is visually correct.
    """
    pixels = obs_np.get("pixels", {})
    if not isinstance(pixels, dict) or not pixels:
        return None
    img = pixels.get(camera)
    if img is None:
        img = next(iter(pixels.values()), None)
    if not isinstance(img, np.ndarray):
        return None
    # SyncVectorEnv batches as (n_envs, H, W, C); unbatch first env
    frame = img[0] if img.ndim == 4 else img
    # LIBERO raw camera output is rotated 180° — flip to match rendering
    return np.ascontiguousarray(frame[::-1, ::-1])


def _save_episode_video(
    frames: list[np.ndarray],
    episode_id: int,
    success: bool,
    video_dir: str,
    video_fps: int,
) -> None:
    """Write collected frames to an mp4 file using imageio."""
    if not frames:
        return
    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if success else "failed"
    video_path = out_dir / f"ep{episode_id:04d}_{status}.mp4"
    try:
        import imageio.v3 as iio

        iio.imwrite(
            str(video_path),
            np.stack(frames),
            fps=video_fps,
            codec="libx264",
            macro_block_size=1,
        )
        logging.info(f"[LiberoVisTest] Video saved → {video_path}  ({len(frames)} frames)")
    except Exception as exc:
        logging.warning(f"[LiberoVisTest] Video save failed: {exc}")


# ---------------------------------------------------------------------------
# Unified in-process runner
# ---------------------------------------------------------------------------


class InProcessVisRunner:
    """Loads a policy locally and runs inference with optional attention probing.

    Dispatches to SmolVLAAttentionProbe or PI05FeatureProbe based on policy_type.
    Mirrors the server-side preprocessing pipeline of policy_server.py so that
    observations from LIBERO envs are processed identically.
    """

    def __init__(
        self,
        cfg: LiberoVisConfig,
        env_preprocessor,
        lerobot_features: dict,
    ):
        self.cfg = cfg
        self.logger = logging.getLogger("InProcessVisRunner")

        # Load policy
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self.logger.info(f"Loading '{cfg.policy_type}' from '{cfg.pretrained_name_or_path}' ...")
        policy_class = get_policy_class(cfg.policy_type)
        self.policy = policy_class.from_pretrained(cfg.pretrained_name_or_path)
        self.policy.to(cfg.policy_device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=cfg.pretrained_name_or_path,
            preprocessor_overrides={"device_processor": {"device": cfg.policy_device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.env_preprocessor = env_preprocessor
        self.lerobot_features = lerobot_features
        self.actions_per_chunk = self.policy.config.chunk_size

        # Create probe (or None when disabled)
        self._probe: Any | None = None
        if cfg.enable_attn_vis:
            self._probe = self._create_probe()

        self._infer_call_count: int = 0
        self._smolvla_offsets_built: bool = False  # log token-offset info once

    # ------------------------------------------------------------------

    def _create_probe(self) -> Any | None:
        """Instantiate the appropriate probe for the current policy type."""
        try:
            if self.cfg.policy_type == "smolvla":
                from ..attn_probe import SmolVLAAttentionProbe

                probe = SmolVLAAttentionProbe(self.policy, keep_cpu_copy=self.cfg.keep_cpu_copy)
                self.logger.info("SmolVLAAttentionProbe ready.")
                return probe
            elif self.cfg.policy_type == "pi05":
                from ..pi05_feature_probe import PI05FeatureProbe

                probe = PI05FeatureProbe(self.policy, keep_cpu_copy=self.cfg.keep_cpu_copy)
                self.logger.info("PI05FeatureProbe ready.")
                return probe
            else:
                self.logger.warning(
                    f"Policy type '{self.cfg.policy_type}' has no attention probe — "
                    "running without visualization."
                )
                return None
        except Exception as exc:
            self.logger.warning(f"Could not create probe for '{self.cfg.policy_type}': {exc}")
            return None

    # ------------------------------------------------------------------

    def predict_chunk(
        self,
        obs_np: dict,
        task_description: str,
        episode: int = 0,
        timestep: int = 0,
    ) -> tuple[torch.Tensor, Any | None]:
        """Run one inference call, optionally capturing attention.

        Args:
            obs_np: Raw batched gym observation dict (numpy, first dim = n_envs).
            task_description: Task instruction string.
            episode: Episode index used for output file naming.
            timestep: Step index within episode used for output file naming.

        Returns:
            Tuple of:
                action_tensor: [chunk_size, action_dim] CPU float tensor.
                capture: AttentionCapture (SmolVLA) or PI05FeatureCapture (PI05), or None.
        """
        from lerobot.envs.utils import preprocess_observation

        # 1. gym obs → lerobot tensors (handles SyncVectorEnv batch dim)
        lerobot_obs = preprocess_observation(obs_np)
        # 2. env preprocessor (flip images, flatten robot_state)
        lerobot_obs = self.env_preprocessor(lerobot_obs)
        # 3. inject task string
        lerobot_obs["task"] = task_description
        # 4. policy preprocessor (tokenize, resize, normalize, device placement)
        lerobot_obs = self.preprocessor(lerobot_obs)

        should_capture = (
            self._probe is not None
            and self._infer_call_count % self.cfg.attn_save_every_n == 0
        )
        capture: Any | None = None

        if should_capture:
            with self._probe as probe:
                action_tensor = self.policy.predict_action_chunk(lerobot_obs)
            capture = self._post_probe(probe, lerobot_obs, episode, timestep)
        else:
            action_tensor = self.policy.predict_action_chunk(lerobot_obs)

        self._infer_call_count += 1

        # 5. postprocess — mirror server-side pipeline
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        _, chunk_size, _ = action_tensor.shape
        processed = [self.postprocessor(action_tensor[:, i, :]) for i in range(chunk_size)]
        action_tensor = torch.stack(processed, dim=1).squeeze(0)  # [chunk_size, action_dim]

        return action_tensor.detach().cpu(), capture

    # ------------------------------------------------------------------
    # Probe post-processing — dispatch on policy type
    # ------------------------------------------------------------------

    def _post_probe(
        self,
        probe: Any,
        observation: dict,
        episode: int,
        timestep: int,
    ) -> Any | None:
        """Finalize capture (build layout/offsets) and save visualizations."""
        if self.cfg.policy_type == "smolvla":
            return self._post_smolvla(probe, observation, episode, timestep)
        elif self.cfg.policy_type == "pi05":
            return self._post_pi05(probe, observation, episode, timestep)
        return None

    def _post_smolvla(
        self,
        probe,
        observation: dict,
        episode: int,
        timestep: int,
    ) -> Any | None:
        """SmolVLA: build token offsets then save cross-attention heatmaps."""
        from ..attn_visualizer import save_inference_attention

        capture = probe.last_capture
        if capture is None or not capture.cross_attn:
            return None

        try:
            probe.build_token_offsets(observation)
            if capture.token_offsets is not None and not self._smolvla_offsets_built:
                self._smolvla_offsets_built = True
                self.logger.info(
                    f"[smolvla] TokenOffsets: {len(capture.token_offsets.camera_slices)} cams, "
                    f"lang={capture.token_offsets.lang_slice}, "
                    f"prefix_len={capture.token_offsets.prefix_len}"
                )
        except Exception as exc:
            self.logger.warning(f"[smolvla] build_token_offsets failed: {exc}")

        images = _extract_images(observation)
        img_hw = (images[0].shape[0], images[0].shape[1]) if images else None
        out_dir = Path(self.cfg.attn_output_dir)
        try:
            saved = save_inference_attention(
                capture=capture,
                output_dir=out_dir,
                images=images,
                token_labels=None,
                img_hw=img_hw,
                episode=episode,
                timestep=timestep,
            )
            self.logger.info(
                f"[smolvla] ep{episode} t{timestep:04d}: saved {len(saved)} plots → {out_dir}"
            )
        except Exception as exc:
            self.logger.warning(f"[smolvla] save_inference_attention failed: {exc}")
        return capture

    def _post_pi05(
        self,
        probe,
        observation: dict,
        episode: int,
        timestep: int,
    ) -> Any | None:
        """PI05: build token layout then save lang→image + action→image heatmaps."""
        from ..pi05_feature_visualizer import save_step_features

        # set_token_layout must be called after the probe context exits
        probe.set_token_layout(observation)
        capture = probe.last_capture

        if capture is None or capture.token_layout is None:
            self.logger.warning(
                f"[pi05] ep{episode} t{timestep}: token_layout not built — skipping plots."
            )
            return None

        images = _extract_images(observation)
        img_hw = (images[0].shape[0], images[0].shape[1]) if images else None

        lang_mask = None
        attn_key = next(
            (k for k in observation if "attention_mask" in k and "language" in k), None
        )
        if attn_key is not None:
            lang_mask = observation[attn_key].bool().cpu()

        out_dir = Path(self.cfg.attn_output_dir)
        try:
            saved = save_step_features(
                capture=capture,
                output_dir=out_dir,
                images=images,
                token_labels=None,
                img_hw=img_hw,
                lang_mask=lang_mask,
                episode=episode,
                timestep=timestep,
            )
            self.logger.info(
                f"[pi05] ep{episode} t{timestep:04d}: saved {len(saved)} plots → {out_dir}"
            )
        except Exception as exc:
            self.logger.warning(f"[pi05] save_step_features failed: {exc}")
        return capture


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _run_episode_inprocess(
    runner: InProcessVisRunner,
    env,
    episode_id: int,
    task_description: str,
    max_steps: int,
    record_video: bool = False,
    video_camera: str = "agentview_image",
) -> tuple[EpisodeResult, list[Any], list[np.ndarray]]:
    """Run one episode in-process.

    Returns:
        (EpisodeResult, episode_captures, video_frames)

    episode_captures: non-None only when enable_attn_vis=True and the chunk
        falls on an attn_save_every_n boundary.  Used for PI05 episode plots.
    video_frames: list of (H, W, 3) uint8 arrays, one per env step (empty
        when record_video=False).
    """
    t_start = time.perf_counter()

    obs_np, _ = env.reset()

    action_chunk: torch.Tensor | None = None
    chunk_pos = 0
    step = 0
    success = False
    episode_captures: list[Any] = []
    video_frames: list[np.ndarray] = []

    while step < max_steps:
        if action_chunk is None or chunk_pos >= len(action_chunk):
            action_chunk, capture = runner.predict_chunk(
                obs_np, task_description, episode=episode_id, timestep=step
            )
            if capture is not None:
                episode_captures.append(capture)
            chunk_pos = 0

        # Collect video frame before stepping (captures initial state too)
        if record_video:
            frame = _extract_video_frame(obs_np, video_camera)
            if frame is not None:
                video_frames.append(frame)

        action = action_chunk[chunk_pos].numpy()  # [action_dim]
        chunk_pos += 1

        obs_np, _, terminated, truncated, info = env.step(action[np.newaxis])

        done = bool(terminated[0] if hasattr(terminated, "__len__") else terminated)
        trunc = bool(truncated[0] if hasattr(truncated, "__len__") else truncated)
        success = _extract_success(info)
        step += 1

        if done or trunc or success:
            # Capture the final frame after the last step
            if record_video:
                frame = _extract_video_frame(obs_np, video_camera)
                if frame is not None:
                    video_frames.append(frame)
            break

    duration = time.perf_counter() - t_start
    return (
        EpisodeResult(
            episode_id=episode_id,
            task_description=task_description,
            success=success,
            steps=step,
            duration_s=duration,
        ),
        episode_captures,
        video_frames,
    )


# ---------------------------------------------------------------------------
# Results saver
# ---------------------------------------------------------------------------


def _save_results(results: list[EpisodeResult], cfg: LiberoVisConfig) -> None:
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list] = defaultdict(list)
    for r in results:
        by_task[r.task_description].append(r)
    task_stats = [
        {
            "task_description": desc,
            "episodes": len(eps),
            "success_rate": sum(r.success for r in eps) / len(eps),
            "avg_steps": sum(r.steps for r in eps) / len(eps),
            "avg_duration_s": sum(r.duration_s for r in eps) / len(eps),
        }
        for desc, eps in sorted(by_task.items())
    ]
    aggregate = {
        "total_episodes": len(results),
        "overall_success_rate": sum(r.success for r in results) / len(results),
        "per_task": task_stats,
        "config": {
            "policy_type": cfg.policy_type,
            "pretrained_name_or_path": cfg.pretrained_name_or_path,
            "env_task": cfg.env_task,
            "episodes_per_task": cfg.episodes_per_task,
            "enable_attn_vis": cfg.enable_attn_vis,
            "attn_output_dir": cfg.attn_output_dir if cfg.enable_attn_vis else None,
            "attn_save_every_n": cfg.attn_save_every_n,
        },
    }
    (out_dir / "episodes.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    logging.info(f"[LiberoVisTest] Results saved to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def run_libero_vis_test(cfg: LiberoVisConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("[LiberoVisTest] Config:\n" + pformat(asdict(cfg)))

    if cfg.enable_attn_vis:
        logging.info(
            f"[LiberoVisTest] Visualization enabled for '{cfg.policy_type}' → {cfg.attn_output_dir} "
            f"(every {cfg.attn_save_every_n} inference call(s))"
        )
    else:
        logging.info("[LiberoVisTest] Visualization disabled — plain evaluation mode")

    # ── Build LIBERO envs ────────────────────────────────────────────────────
    from lerobot.envs.factory import make_env, make_env_config
    from lerobot.envs.utils import env_to_policy_features

    env_cfg = make_env_config(
        "libero",
        task=cfg.env_task,
        obs_type=cfg.obs_type,
        camera_name=cfg.camera_name,
    )
    if cfg.task_ids is not None:
        env_cfg.task_ids = cfg.task_ids
    if hasattr(env_cfg, "episode_length"):
        env_cfg.episode_length = cfg.max_episode_steps

    envs_dict = make_env(env_cfg, n_envs=1)
    env_preprocessor, _ = env_cfg.get_env_processors()

    try:
        lerobot_features = env_to_policy_features(env_cfg)
    except Exception as exc:
        logging.warning(f"[LiberoVisTest] Could not build lerobot features: {exc}. Using {{}}.")
        lerobot_features = {}

    task_list = [
        (suite, tid, env)
        for suite, task_envs in envs_dict.items()
        for tid, env in sorted(task_envs.items())
    ]
    logging.info(f"[LiberoVisTest] Built {len(task_list)} task env(s)")

    if not task_list:
        logging.error("[LiberoVisTest] No task environments created. Aborting.")
        return

    # ── Load policy and create runner ────────────────────────────────────────
    runner = InProcessVisRunner(cfg, env_preprocessor, lerobot_features)

    # ── Episode loop ─────────────────────────────────────────────────────────
    all_results: list[EpisodeResult] = []
    global_ep = 0
    t_all_start = time.perf_counter()

    try:
        for task_idx, (suite_name, task_id, task_env) in enumerate(task_list):
            task_desc = _get_task_description(task_env)
            logging.info(
                f"[LiberoVisTest] ══ Task {task_idx + 1}/{len(task_list)} "
                f"| suite={suite_name} | task_id={task_id} | '{task_desc}' ══"
            )

            task_results: list[EpisodeResult] = []
            for ep_local in range(cfg.episodes_per_task):
                result, episode_captures, video_frames = _run_episode_inprocess(
                    runner=runner,
                    env=task_env,
                    episode_id=global_ep,
                    task_description=task_desc,
                    max_steps=cfg.max_episode_steps,
                    record_video=cfg.save_video,
                    video_camera=cfg.video_camera,
                )
                task_results.append(result)
                all_results.append(result)

                logging.info(
                    f"[LiberoVisTest] task={task_id} ep={ep_local}/{cfg.episodes_per_task - 1} "
                    f"success={result.success}  steps={result.steps}  "
                    f"duration={result.duration_s:.2f}s  captures={len(episode_captures)}"
                )

                # Save episode video
                if cfg.save_video and video_frames:
                    _save_episode_video(
                        frames=video_frames,
                        episode_id=global_ep,
                        success=result.success,
                        video_dir=cfg.video_dir,
                        video_fps=cfg.video_fps,
                    )

                # PI05 episode-level plots (temporal drift + PCA)
                if (
                    cfg.policy_type == "pi05"
                    and cfg.enable_attn_vis
                    and cfg.save_episode_plots
                    and len(episode_captures) >= 2
                ):
                    from ..pi05_feature_visualizer import save_episode_features

                    out_dir = Path(cfg.attn_output_dir)
                    try:
                        saved = save_episode_features(
                            captures=episode_captures,
                            output_dir=out_dir,
                            episode=global_ep,
                        )
                        logging.info(
                            f"[LiberoVisTest] ep{global_ep}: saved {len(saved)} episode plots → {out_dir}"
                        )
                    except Exception as exc:
                        logging.warning(f"[LiberoVisTest] save_episode_features failed: {exc}")

                global_ep += 1

            sr = sum(r.success for r in task_results) / len(task_results) if task_results else 0.0
            logging.info(f"[LiberoVisTest] Task {task_id} success_rate={sr:.1%}")

    finally:
        for _, _, env in task_list:
            try:
                env.close()
            except Exception:
                pass

    # ── Summary ──────────────────────────────────────────────────────────────
    total_t = time.perf_counter() - t_all_start
    if all_results:
        overall_sr = sum(r.success for r in all_results) / len(all_results)
        logging.info(
            f"[LiberoVisTest] ═══ Final summary ═══\n"
            f"  policy_type  : {cfg.policy_type}\n"
            f"  suite        : {cfg.env_task}\n"
            f"  episodes     : {len(all_results)}\n"
            f"  overall_sr   : {overall_sr:.1%}\n"
            f"  total_time   : {total_t:.2f}s\n"
            f"  vis_enabled  : {cfg.enable_attn_vis}\n"
            + (f"  attn_plots   : {cfg.attn_output_dir}" if cfg.enable_attn_vis else "")
        )

    if cfg.save_results and all_results:
        _save_results(all_results, cfg)


if __name__ == "__main__":
    run_libero_vis_test()
