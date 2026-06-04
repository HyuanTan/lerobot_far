"""In-process LIBERO evaluation with PI05 task-conditioned feature analysis.

Loads the PI05 policy and LIBERO environments in the same process (no gRPC)
so that PI05FeatureProbe can hook into the model's forward pass directly.

Per inference step: saves action→image attention heatmaps and language-alignment
bar charts to feat_output_dir.
Per episode: saves temporal feature drift and PCA trajectory plots.

Usage::

    python -m lerobot.async_inference.sim_test.run_libero_pi05_feat_test \\
        --env_task=libero_spatial \\
        --pretrained_name_or_path=/path/to/pi05_checkpoint \\
        --policy_device=cuda \\
        --episodes_per_task=1 \\
        --feat_output_dir=./pi05_features \\
        --feat_save_every_n=5
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import torch

from .configs import LiberoSimConfig
from .sim_client import EpisodeResult, _extract_success, _get_task_description
from ..pi05_feature_probe import PI05FeatureCapture, PI05FeatureProbe
from ..pi05_feature_visualizer import save_episode_features, save_step_features


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LiberoPI05FeatConfig:
    """Configuration for in-process PI05 LIBERO evaluation with feature analysis."""

    # ── LIBERO environment ──────────────────────────────────────────────────
    env_task: str = field(
        default="libero_10",
        metadata={"help": "LIBERO suite name (libero_spatial, libero_object, libero_10, …)"},
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

    # ── Policy ─────────────────────────────────────────────────────────────
    pretrained_name_or_path: str = field(
        default="",
        metadata={"help": "HF Hub repo or local path to pretrained PI05 checkpoint"},
    )
    policy_device: str = field(
        default="cuda",
        metadata={"help": "Device for policy inference ('cuda', 'cpu', …)"},
    )
    policy_type: str = field(
        default="pi05",
        metadata={"help": "Policy type (only pi05 is probed; others run plain)"},
    )

    # ── Feature capture ─────────────────────────────────────────────────────
    feat_output_dir: str = field(
        default="./pi05_features",
        metadata={"help": "Directory to write feature visualization PNGs"},
    )
    feat_save_every_n: int = field(
        default=1,
        metadata={"help": "Save step-level plots every N inference calls (1 = every call)"},
    )
    feat_keep_cpu_copy: bool = field(
        default=True,
        metadata={"help": "Move captured tensors to CPU immediately to save GPU memory"},
    )
    feat_save_episode_plots: bool = field(
        default=True,
        metadata={"help": "Save temporal drift and PCA plots at the end of each episode"},
    )

    # ── Results ─────────────────────────────────────────────────────────────
    results_dir: str = field(
        default="./libero_pi05_results",
        metadata={"help": "Directory for JSON result files"},
    )
    save_results: bool = field(
        default=True,
        metadata={"help": "Persist per-episode and aggregate results to JSON"},
    )


# ---------------------------------------------------------------------------
# In-process inference runner
# ---------------------------------------------------------------------------


class PI05InProcessRunner:
    """Loads PI05 locally and runs inference with feature probing.

    Mirrors the server-side preprocessing pipeline of policy_server.py so
    that observations from LIBERO envs are processed identically.
    """

    def __init__(
        self,
        cfg: LiberoPI05FeatConfig,
        env_preprocessor,
        lerobot_features: dict,
    ):
        self.cfg = cfg
        self.logger = logging.getLogger("PI05InProcessRunner")

        # Load policy
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self.logger.info(f"Loading '{cfg.policy_type}' from '{cfg.pretrained_name_or_path}' …")
        policy_class = get_policy_class(cfg.policy_type)
        self.policy = policy_class.from_pretrained(cfg.pretrained_name_or_path)
        self.policy.to(cfg.policy_device)
        self.policy.eval()

        # Build preprocessor / postprocessor (same as server side)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=cfg.pretrained_name_or_path,
            preprocessor_overrides={"device_processor": {"device": cfg.policy_device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self.env_preprocessor = env_preprocessor
        self.lerobot_features = lerobot_features
        self.actions_per_chunk = self.policy.config.chunk_size

        # Feature probe (only for pi05)
        self._probe: PI05FeatureProbe | None = None
        if cfg.policy_type == "pi05":
            try:
                self._probe = PI05FeatureProbe(
                    self.policy,
                    keep_cpu_copy=cfg.feat_keep_cpu_copy,
                )
                self.logger.info("PI05FeatureProbe ready.")
            except Exception as exc:
                self.logger.warning(f"Could not create probe: {exc}")

        self._infer_call_count = 0

    def predict_chunk(
        self,
        obs_np: dict,
        task_description: str,
        episode: int = 0,
        timestep: int = 0,
    ) -> tuple[torch.Tensor, PI05FeatureCapture | None]:
        """Run one inference call, optionally capturing features.

        Args:
            obs_np: Raw batched gym observation dict (numpy, first dim = n_envs).
            task_description: Task instruction string.
            episode: Episode index for file naming.
            timestep: Step index within episode for file naming.

        Returns:
            (action_tensor [chunk_size, action_dim] CPU, capture or None)
        """
        from lerobot.envs.utils import preprocess_observation

        # 1. gym obs → lerobot tensors
        lerobot_obs = preprocess_observation(obs_np)
        # 2. env preprocessor (flip images, flatten state)
        lerobot_obs = self.env_preprocessor(lerobot_obs)
        # 3. inject task string
        lerobot_obs["task"] = task_description
        # 4. policy preprocessor (tokenise, normalise, device placement)
        lerobot_obs = self.preprocessor(lerobot_obs)

        # 5. run inference
        should_capture = (
            self._probe is not None
            and self._infer_call_count % self.cfg.feat_save_every_n == 0
        )
        capture: PI05FeatureCapture | None = None

        if should_capture:
            with self._probe as probe:
                action_tensor = self.policy.predict_action_chunk(lerobot_obs)
            # Build token layout immediately after exit (uses probe.last_capture)
            probe.set_token_layout(lerobot_obs)
            capture = probe.last_capture
            # Save step-level plots asynchronously (we do it inline — fast enough)
            self._save_step(capture, lerobot_obs, episode, timestep)
        else:
            action_tensor = self.policy.predict_action_chunk(lerobot_obs)

        self._infer_call_count += 1

        # 6. postprocess — mirror server pipeline
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        _, chunk_size, _ = action_tensor.shape
        processed = [self.postprocessor(action_tensor[:, i, :]) for i in range(chunk_size)]
        action_tensor = torch.stack(processed, dim=1).squeeze(0)  # [chunk_size, action_dim]

        return action_tensor.detach().cpu(), capture

    def _save_step(
        self,
        capture: PI05FeatureCapture | None,
        observation: dict,
        episode: int,
        timestep: int,
    ) -> None:
        """Extract images from obs and save step-level feature plots."""
        if capture is None or capture.token_layout is None:
            self.logger.warning(
                f"[feat] ep{episode} t{timestep}: token_layout not built — skipping step plots."
            )
            return

        from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK

        # Extract images in [0, 1] float (before PI05's internal [-1, 1] normalisation)
        images: list[np.ndarray] = []
        img_hw = None
        for k in sorted(k for k in observation if OBS_IMAGES in k and "empty" not in k):
            t = observation[k]
            if t.ndim == 4:
                t = t[0]  # take first batch element
            arr = t.detach().cpu().float().numpy()
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, C]
            images.append(arr)
            if img_hw is None:
                img_hw = (arr.shape[0], arr.shape[1])

        # Language attention mask for active-token filtering
        lang_mask = None
        attn_key = next((k for k in observation if "attention_mask" in k and "language" in k), None)
        if attn_key is not None:
            lang_mask = observation[attn_key].bool().cpu()

        out_dir = Path(self.cfg.feat_output_dir)
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
                f"[feat] ep{episode} t{timestep:04d}: saved {len(saved)} plots → {out_dir}"
            )
        except Exception as exc:
            self.logger.warning(f"[feat] save_step_features failed: {exc}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _run_episode_inprocess(
    runner: PI05InProcessRunner,
    env,
    episode_id: int,
    task_description: str,
    max_steps: int,
) -> tuple[EpisodeResult, list[PI05FeatureCapture]]:
    """Run one episode in-process; returns EpisodeResult and list of captures."""
    t_start = time.perf_counter()

    obs_np, _ = env.reset()

    action_chunk: torch.Tensor | None = None
    chunk_pos = 0
    step = 0
    success = False
    episode_captures: list[PI05FeatureCapture] = []

    while step < max_steps:
        if action_chunk is None or chunk_pos >= len(action_chunk):
            action_chunk, capture = runner.predict_chunk(
                obs_np, task_description, episode=episode_id, timestep=step
            )
            if capture is not None:
                episode_captures.append(capture)
            chunk_pos = 0

        action = action_chunk[chunk_pos].numpy()
        chunk_pos += 1

        obs_np, reward, terminated, truncated, info = env.step(action[np.newaxis])

        done = bool(terminated[0] if hasattr(terminated, "__len__") else terminated)
        trunc = bool(truncated[0] if hasattr(truncated, "__len__") else truncated)
        success = _extract_success(info)
        step += 1

        if done or trunc or success:
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
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def run_libero_pi05_feat_test(cfg: LiberoPI05FeatConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("[PI05FeatTest] Config:\n" + pformat(asdict(cfg)))

    # ── Build LIBERO envs ──────────────────────────────────────────────────
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
        logging.warning(f"[PI05FeatTest] Could not build lerobot features: {exc}. Using {{}}.")
        lerobot_features = {}

    task_list = [
        (suite, tid, env)
        for suite, task_envs in envs_dict.items()
        for tid, env in sorted(task_envs.items())
    ]
    logging.info(f"[PI05FeatTest] Built {len(task_list)} task env(s)")

    # ── Load policy and create runner ──────────────────────────────────────
    runner = PI05InProcessRunner(cfg, env_preprocessor, lerobot_features)

    # ── Episode loop ───────────────────────────────────────────────────────
    all_results: list[EpisodeResult] = []
    global_ep = 0
    t_all_start = time.perf_counter()

    try:
        for task_idx, (suite_name, task_id, task_env) in enumerate(task_list):
            task_desc = _get_task_description(task_env)
            logging.info(
                f"[PI05FeatTest] ══ Task {task_idx + 1}/{len(task_list)} "
                f"| suite={suite_name} | task_id={task_id} | '{task_desc}' ══"
            )

            for ep_local in range(cfg.episodes_per_task):
                result, episode_captures = _run_episode_inprocess(
                    runner=runner,
                    env=task_env,
                    episode_id=global_ep,
                    task_description=task_desc,
                    max_steps=cfg.max_episode_steps,
                )
                all_results.append(result)

                logging.info(
                    f"[PI05FeatTest] task={task_id} ep={ep_local} "
                    f"success={result.success}  steps={result.steps}  "
                    f"duration={result.duration_s:.2f}s  "
                    f"captures={len(episode_captures)}"
                )

                # Episode-level plots (drift + PCA)
                if cfg.feat_save_episode_plots and len(episode_captures) >= 2:
                    out_dir = Path(cfg.feat_output_dir)
                    try:
                        saved = save_episode_features(
                            captures=episode_captures,
                            output_dir=out_dir,
                            episode=global_ep,
                        )
                        logging.info(
                            f"[PI05FeatTest] Episode {global_ep}: saved {len(saved)} "
                            f"episode plots → {out_dir}"
                        )
                    except Exception as exc:
                        logging.warning(f"[PI05FeatTest] save_episode_features failed: {exc}")

                global_ep += 1

            task_results = [r for r in all_results if r.task_description == task_desc]
            sr = sum(r.success for r in task_results) / len(task_results)
            logging.info(f"[PI05FeatTest] Task {task_id} success_rate={sr:.1%}")

    finally:
        for _, _, env in task_list:
            try:
                env.close()
            except Exception:
                pass

    # ── Summary ────────────────────────────────────────────────────────────
    total_t = time.perf_counter() - t_all_start
    if all_results:
        overall_sr = sum(r.success for r in all_results) / len(all_results)
        logging.info(
            f"[PI05FeatTest] ═══ Final summary ═══\n"
            f"  suite        : {cfg.env_task}\n"
            f"  episodes     : {len(all_results)}\n"
            f"  overall_sr   : {overall_sr:.1%}\n"
            f"  total_time   : {total_t:.2f}s\n"
            f"  feat plots   : {cfg.feat_output_dir}"
        )

    if cfg.save_results and all_results:
        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        by_task: dict[str, list] = defaultdict(list)
        for r in all_results:
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
            "total_episodes": len(all_results),
            "overall_success_rate": sum(r.success for r in all_results) / len(all_results),
            "per_task": task_stats,
            "config": {
                "policy_type": cfg.policy_type,
                "pretrained_name_or_path": cfg.pretrained_name_or_path,
                "env_task": cfg.env_task,
                "episodes_per_task": cfg.episodes_per_task,
                "feat_output_dir": cfg.feat_output_dir,
                "feat_save_every_n": cfg.feat_save_every_n,
            },
        }
        (out_dir / "episodes.json").write_text(
            json.dumps([asdict(r) for r in all_results], indent=2), encoding="utf-8"
        )
        (out_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        logging.info(f"[PI05FeatTest] Results saved to {out_dir}")


if __name__ == "__main__":
    run_libero_pi05_feat_test()
