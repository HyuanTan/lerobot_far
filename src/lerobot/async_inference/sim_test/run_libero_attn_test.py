"""
In-process LIBERO evaluation with SmolVLA cross-attention visualization.

Runs the LIBERO environment and policy in the SAME process (no gRPC) so that
SmolVLAAttentionProbe can hook directly into the policy's attention computation.
Saves attention visualizations for each inference call to `attn_output_dir`.

This is complementary to the client-server approach (attn_policy_server.py):
  - Use THIS script when you want attention analysis without network overhead.
  - Use attn_policy_server.py when you want attention capture inside the
    existing client-server test flow.

Usage:
```shell
python -m lerobot.async_inference.sim_test.run_libero_attn_test \\
    --env_task=libero_10 \\
    --pretrained_name_or_path=/path/to/smolvla \\
    --policy_device=cuda \\
    --episodes_per_task=1 \\
    --attn_output_dir=./smolvla_attention \\
    --attn_save_every_n=3
```
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
from ..attn_probe import SmolVLAAttentionProbe
from ..attn_visualizer import save_inference_attention


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LiberoAttnConfig:
    """Configuration for in-process LIBERO evaluation with attention visualization."""

    # ── LIBERO environment (mirrors LiberoSimConfig) ─────────────────────────
    env_task: str = field(
        default="libero_10",
        metadata={"help": "LIBERO suite name"},
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
    pretrained_name_or_path: str = field(
        default="",
        metadata={"help": "HF Hub repo or local path to pretrained SmolVLA checkpoint"},
    )
    policy_device: str = field(
        default="cuda",
        metadata={"help": "Device for policy inference"},
    )
    policy_type: str = field(
        default="smolvla",
        metadata={"help": "Policy type (only smolvla is probed; others run without probe)"},
    )

    # ── Attention capture ────────────────────────────────────────────────────
    attn_output_dir: str = field(
        default="./smolvla_attention",
        metadata={"help": "Directory to write attention visualization PNGs"},
    )
    attn_save_every_n: int = field(
        default=1,
        metadata={"help": "Save attention visualizations every N inference calls (1 = every call)"},
    )
    attn_keep_cpu_copy: bool = field(
        default=True,
        metadata={"help": "Move captured attention tensors to CPU immediately"},
    )

    # ── Results ──────────────────────────────────────────────────────────────
    results_dir: str = field(
        default="./libero_attn_results",
        metadata={"help": "Directory for per-episode and aggregate JSON result files"},
    )
    save_results: bool = field(
        default=True,
        metadata={"help": "Persist results to JSON"},
    )


# ---------------------------------------------------------------------------
# In-process inference loop
# ---------------------------------------------------------------------------


class InProcessInferenceRunner:
    """Loads a SmolVLA policy locally and runs inference with attention probing.

    Mirrors the server-side preprocessing pipeline of policy_server.py so that
    observations from the LIBERO env are processed identically.
    """

    def __init__(
        self,
        cfg: LiberoAttnConfig,
        env_preprocessor,
        lerobot_features: dict,
    ):
        self.cfg = cfg
        self.logger = logging.getLogger("InProcessInferenceRunner")

        # Load policy
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self.logger.info(f"Loading policy '{cfg.policy_type}' from '{cfg.pretrained_name_or_path}' ...")
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

        # Attention probe (only for smolvla)
        self._probe: SmolVLAAttentionProbe | None = None
        if cfg.policy_type == "smolvla":
            try:
                self._probe = SmolVLAAttentionProbe(
                    self.policy,
                    keep_cpu_copy=cfg.attn_keep_cpu_copy,
                )
                self.logger.info("SmolVLAAttentionProbe ready.")
            except Exception as exc:
                self.logger.warning(f"Could not create probe: {exc}")

        self._infer_call_count = 0
        self._token_offsets_built = False

    def predict_chunk(
        self,
        obs_np: dict,
        task_description: str,
        episode: int = 0,
        timestep: int = 0,
    ) -> torch.Tensor:
        """Run one inference call and optionally save attention visualizations.

        Args:
            obs_np: Raw batched gym observation dict (numpy arrays, first dim = n_envs).
            task_description: Task instruction string.
            episode: Episode index for file naming.
            timestep: Step index within episode for file naming.

        Returns:
            action_tensor: [chunk_size, action_dim] CPU tensor.
        """
        from lerobot.envs.utils import preprocess_observation

        # 1. gym obs → lerobot tensors (handles batched SyncVectorEnv obs)
        lerobot_obs = preprocess_observation(obs_np)

        # 2. env preprocessor (flip images, flatten state)
        lerobot_obs = self.env_preprocessor(lerobot_obs)

        # 3. inject task string
        lerobot_obs["task"] = task_description

        # 4. policy preprocessor (resize, normalize, device placement)
        lerobot_obs = self.preprocessor(lerobot_obs)

        # 5. run inference (with probe if available)
        should_vis = (
            self._probe is not None
            and self._infer_call_count % self.cfg.attn_save_every_n == 0
        )

        if should_vis:
            with self._probe as probe:
                action_tensor = self.policy.predict_action_chunk(lerobot_obs)
            self._maybe_save_attention(probe, lerobot_obs, episode, timestep)
        else:
            action_tensor = self.policy.predict_action_chunk(lerobot_obs)

        self._infer_call_count += 1

        # 6. postprocess — mirror server-side pipeline
        # Ensure 3D: [B, chunk_size, action_dim]
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)

        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            single = action_tensor[:, i, :]  # [B, action_dim]
            processed_actions.append(self.postprocessor(single))
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)  # [chunk_size, action_dim]

        return action_tensor.detach().cpu()

    def _maybe_save_attention(
        self,
        probe: SmolVLAAttentionProbe,
        observation: dict,
        episode: int,
        timestep: int,
    ) -> None:
        """Build token offsets (once) and save attention visualizations."""
        capture = probe.last_capture
        if capture is None or not capture.cross_attn:
            return

        # Build token offsets for this capture (must be called every inference call
        # because each call produces a new AttentionCapture with token_offsets=None)
        try:
            probe.build_token_offsets(observation)
            if capture.token_offsets is not None and not self._token_offsets_built:
                self._token_offsets_built = True
                self.logger.info(
                    f"[attn] TokenOffsets: {len(capture.token_offsets.camera_slices)} cams, "
                    f"lang={capture.token_offsets.lang_slice}, "
                    f"state={capture.token_offsets.state_slice}, "
                    f"prefix_len={capture.token_offsets.prefix_len}"
                )
        except Exception as exc:
            self.logger.warning(f"[attn] build_token_offsets failed: {exc}")

        # Extract images for visualization (normalized float → uint8)
        from lerobot.utils.constants import OBS_IMAGES
        images = []
        img_hw = None
        for k in sorted(k for k in observation if OBS_IMAGES in k and "empty" not in k):
            t = observation[k]
            if t.ndim == 4:
                t = t[0]
            arr = t.detach().cpu().float().numpy()
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, C]
            images.append(arr)
            if img_hw is None:
                img_hw = (arr.shape[0], arr.shape[1])

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
                f"[attn] ep{episode} t{timestep:04d}: saved {len(saved)} attention plots → {out_dir}"
            )
        except Exception as exc:
            self.logger.warning(f"[attn] save_inference_attention failed: {exc}")


# ---------------------------------------------------------------------------
# Episode runner (replaces SimRobotClient for in-process use)
# ---------------------------------------------------------------------------


def _run_episode_inprocess(
    runner: InProcessInferenceRunner,
    env,
    episode_id: int,
    task_description: str,
    max_steps: int,
) -> EpisodeResult:
    """Run one episode in-process: reset env, iterate steps with chunked actions."""
    t_start = time.perf_counter()

    obs_np, _ = env.reset()
    # obs_np is from SyncVectorEnv: each key has shape [n_envs, ...].
    # Pass as-is to predict_chunk; preprocess_observation handles the batch dim.

    action_chunk: torch.Tensor | None = None
    chunk_pos = 0
    step = 0
    success = False

    while step < max_steps:
        # Request a new chunk when the current one is exhausted
        if action_chunk is None or chunk_pos >= len(action_chunk):
            action_chunk = runner.predict_chunk(
                obs_np, task_description, episode=episode_id, timestep=step
            )
            chunk_pos = 0

        action = action_chunk[chunk_pos].numpy()  # [action_dim]
        chunk_pos += 1

        # Step the environment (SyncVectorEnv expects [n_envs, action_dim])
        obs_np, reward, terminated, truncated, info = env.step(action[np.newaxis])

        done = bool(terminated[0] if hasattr(terminated, "__len__") else terminated)
        trunc = bool(truncated[0] if hasattr(truncated, "__len__") else truncated)

        success = _extract_success(info)
        step += 1

        if done or trunc or success:
            break

    duration = time.perf_counter() - t_start
    return EpisodeResult(
        episode_id=episode_id,
        task_description=task_description,
        success=success,
        steps=step,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def run_libero_attn_test(cfg: LiberoAttnConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("[LiberoAttnTest] Config:\n" + pformat(asdict(cfg)))

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
        logging.warning(f"[LiberoAttnTest] Could not build lerobot features: {exc}. Using empty dict.")
        lerobot_features = {}

    task_list = [
        (suite, tid, env)
        for suite, task_envs in envs_dict.items()
        for tid, env in sorted(task_envs.items())
    ]
    logging.info(f"[LiberoAttnTest] Built {len(task_list)} task env(s)")

    # ── Load policy and create runner ────────────────────────────────────────
    runner = InProcessInferenceRunner(cfg, env_preprocessor, lerobot_features)

    # ── Episode loop ─────────────────────────────────────────────────────────
    all_results: list[EpisodeResult] = []
    global_ep = 0

    t_all_start = time.perf_counter()
    try:
        for task_idx, (suite_name, task_id, task_env) in enumerate(task_list):
            task_desc = _get_task_description(task_env)
            logging.info(
                f"[LiberoAttnTest] ══ Task {task_idx + 1}/{len(task_list)} "
                f"| suite={suite_name} | task_id={task_id} | '{task_desc}' ══"
            )

            task_results: list[EpisodeResult] = []
            for ep_local in range(cfg.episodes_per_task):
                result = _run_episode_inprocess(
                    runner=runner,
                    env=task_env,
                    episode_id=global_ep,
                    task_description=task_desc,
                    max_steps=cfg.max_episode_steps,
                )
                task_results.append(result)
                all_results.append(result)
                global_ep += 1

                logging.info(
                    f"[LiberoAttnTest] task={task_id} ep={ep_local}/{cfg.episodes_per_task - 1} "
                    f"success={result.success}  steps={result.steps}  duration={result.duration_s:.2f}s"
                )

            sr = sum(r.success for r in task_results) / len(task_results) if task_results else 0.0
            logging.info(f"[LiberoAttnTest] Task {task_id} success_rate={sr:.1%}")
    finally:
        for _, _, env in task_list:
            try:
                env.close()
            except Exception:
                pass

    # ── Summary and results save ─────────────────────────────────────────────
    total_t = time.perf_counter() - t_all_start
    if all_results:
        overall_sr = sum(r.success for r in all_results) / len(all_results)
        logging.info(
            f"[LiberoAttnTest] ═══ Final summary ═══\n"
            f"  suite        : {cfg.env_task}\n"
            f"  episodes     : {len(all_results)}\n"
            f"  overall_sr   : {overall_sr:.1%}\n"
            f"  total_time   : {total_t:.2f}s\n"
            f"  attn plots   : {cfg.attn_output_dir}"
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
                "attn_output_dir": cfg.attn_output_dir,
                "attn_save_every_n": cfg.attn_save_every_n,
            },
        }
        (out_dir / "episodes.json").write_text(
            json.dumps([asdict(r) for r in all_results], indent=2), encoding="utf-8"
        )
        (out_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        logging.info(f"[LiberoAttnTest] Results saved to {out_dir}")


if __name__ == "__main__":
    run_libero_attn_test()
