# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SimRobotClient — simulation async-inference client for LIBERO environments.

Drives a gym SyncVectorEnv (LIBERO) via a gRPC policy server, mirroring the
two-thread architecture of RobotClient (BaseAsyncClient: receiver + control loop).

Observation preprocessing pipeline (client-side, obs_pre_mapped=True):
  1. preprocess_observation()    numpy gym keys → lerobot tensor keys
                                 pixels.image → observation.images.image  (B,C,H,W) float32
                                 robot_state  → observation.robot_state   nested tensor dict
  2. env_preprocessor            LiberoProcessorStep:
                                 - flip images 180° (dims=[2,3])
                                 - observation.robot_state → observation.state (B,8) float32
  3. task string injection       lerobot_obs["task"] = task_description

obs_pre_mapped=True tells the server to skip raw_observation_to_observation() and only
apply image resize and policy normalisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..base_client import BaseAsyncClient
from ..helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.rtc import RTCConfig


# ── Episode result ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode_id: int
    task_description: str
    success: bool
    steps: int
    duration_s: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_success(info: dict | list | None) -> bool:
    """Extract success flag from VectorEnv info (handles multiple formats).

    Gymnasium VectorEnv can return info as:
      - a list/tuple of per-env dicts
      - a flat dict with is_success/success keys
      - a dict with integer sub-env keys (e.g. {0: {"is_success": ...}})
    """
    if info is None:
        return False
    if isinstance(info, (list, tuple)):
        return any(
            bool(x.get("is_success", x.get("success", False)))
            for x in info if isinstance(x, dict)
        )
    if isinstance(info, dict):
        for key in ("is_success", "success", "_is_success"):
            if key in info:
                val = info[key]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                if isinstance(val, (list, tuple)):
                    return any(bool(v) for v in val)
                return bool(val)
        # Gymnasium VectorEnv stores per-env info under integer keys
        for _k, v in info.items():
            if isinstance(v, dict):
                for sk in ("is_success", "success"):
                    if sk in v:
                        return bool(v[sk])
    return False


def _get_task_description(env) -> str:
    """Extract the natural-language task description from a gym VectorEnv.

    Works for SyncVectorEnv wrapping LiberoEnv (task_description attribute on
    each sub-env).  Falls back to the task name if available.
    """
    if hasattr(env, "envs") and env.envs:
        sub = env.envs[0]
        if hasattr(sub, "task_description"):
            return str(sub.task_description)
        if hasattr(sub, "task"):
            return str(sub.task)
    return ""


# ── SimRobotClient ────────────────────────────────────────────────────────────

class SimRobotClient(BaseAsyncClient):
    """Async-inference client driving a LIBERO SyncVectorEnv via a gRPC policy server.

    Call sequence:
        client = SimRobotClient(config, env, env_preprocessor, lerobot_features)
        assert client.start()
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        for ep in range(N):
            result = client.run_episode(ep, max_steps, first_episode=(ep == 0),
                                        task_description=desc)
        client.stop()
        receiver.join(timeout=5)

    For multi-task evaluation, swap client.env and pass task_description to
    run_episode() between tasks without restarting the client or receiver thread.
    """

    prefix = "sim_client"

    def __init__(
        self,
        config: Any,
        env: Any,
        env_preprocessor: Any,
        lerobot_features: dict,
        task_description: str = "",
    ):
        """
        Args:
            config:           LiberoSimConfig (fields required by BaseAsyncClient +
                              policy_type, pretrained_name_or_path, policy_device).
            env:              gym SyncVectorEnv wrapping LiberoEnv (n_envs=1).
            env_preprocessor: PolicyProcessorPipeline from env_cfg.get_env_processors()[0].
            lerobot_features: lerobot feature spec from env_to_policy_features(env_cfg).
            task_description: Natural-language task string injected into each observation.
                              Update self._task_str before each task when iterating tasks.
        """
        super().__init__(config)
        self.env = env
        self._env_preprocessor = env_preprocessor
        self._lerobot_features = lerobot_features
        self._task_str = task_description
        self._last_obs_raw: dict = {}

        # Video recording (driven by config.save_video / video_dir / video_camera / video_fps)
        self._record_video: bool = getattr(config, "save_video", False)
        self._video_dir: str = getattr(config, "video_dir", "./libero_videos")
        self._video_camera: str = getattr(config, "video_camera", "image")
        self._video_fps: int = getattr(config, "video_fps", config.fps if hasattr(config, "fps") else 30)

        # Payload optimisation: transmit images as uint8 to reduce gRPC payload ~4x.
        # The server detects uint8 tensors in the obs_pre_mapped path and converts
        # them back to float32 [0,1] before running the policy preprocessor.
        self._transmit_images_as_uint8: bool = getattr(config, "transmit_images_as_uint8", False)
        if self._transmit_images_as_uint8:
            self.logger.info(
                "[sim_client] transmit_images_as_uint8=True: "
                "images sent as uint8 [0,255] — server converts to float32 before preprocessor"
            )

    # ── Abstract hook implementations ─────────────────────────────────────────

    def _build_policy_config(self) -> RemotePolicyConfig:
        _rtc_cfg: RTCConfig | None = None
        if self.config.rtc_execution_horizon > 0:
            _rtc_cfg = RTCConfig(enabled=True, execution_horizon=self.config.rtc_execution_horizon)
            self.logger.info(
                f"[sim_client] RTC re-planning enabled | "
                f"execution_horizon={self.config.rtc_execution_horizon}"
            )
        return RemotePolicyConfig(
            policy_type=self.config.policy_type,
            pretrained_name_or_path=self.config.pretrained_name_or_path,
            lerobot_features=self._lerobot_features,
            actions_per_chunk=self.config.actions_per_chunk,
            device=self.config.policy_device,
            rtc_config=_rtc_cfg,
        )

    def _capture_raw_obs(self) -> dict:
        return self._last_obs_raw

    def _preprocess_obs(self, raw_obs: dict) -> dict:
        """Convert raw LIBERO gym obs to lerobot-format tensors and inject task string.

        Step 1 — preprocess_observation():
          pixels.image      → observation.images.image   (B, C, H, W) float32 [0, 1]
          robot_state       → observation.robot_state    nested tensor dict

        Step 2 — env_preprocessor (LiberoProcessorStep):
          observation.images.*  → flipped 180° (dims=[2, 3])
          observation.robot_state → observation.state    (B, 8) float32
                                    [eef_pos(3), axis_angle(3), gripper(2)]

        Step 3 — task string injection.

        Step 4 (optional) — uint8 downcast when transmit_images_as_uint8=True:
          observation.images.*  float32 [0,1] → uint8 [0,255]
          Reduces gRPC payload ~4x.  Server converts back to float32 before preprocessor.

        obs_pre_mapped=True: the server skips raw→lerobot key remapping and only
        applies image resize and policy normalisation.
        """
        lerobot_obs = preprocess_observation(raw_obs)
        lerobot_obs = self._env_preprocessor(lerobot_obs)
        lerobot_obs["task"] = self._task_str

        if self._transmit_images_as_uint8:
            for key, val in lerobot_obs.items():
                if "image" in key and isinstance(val, torch.Tensor) and val.is_floating_point():
                    lerobot_obs[key] = (val * 255).to(torch.uint8)

        return lerobot_obs

    def _build_timed_observation(
        self,
        processed_obs: dict,
        timestep: int,
        infer_delay: int,
        leftover: torch.Tensor | None,
    ) -> TimedObservation:
        return TimedObservation(
            timestamp=time.time(),
            observation=processed_obs,
            timestep=timestep,
            inference_delay=infer_delay,
            leftover_actions=leftover,
            obs_pre_mapped=True,
        )

    def _execute_action(self, timed_action: TimedAction) -> Any:
        """Step the env with the given action; update self._last_obs_raw."""
        action_np = timed_action.get_action().cpu().numpy()
        if action_np.ndim == 1:
            action_np = action_np[np.newaxis, :]  # (1, action_dim) for VectorEnv API

        obs_raw, _reward, terminated, truncated, info = self.env.step(action_np)
        self._last_obs_raw = obs_raw

        done = bool(np.any(terminated) or np.any(truncated))
        return obs_raw, done, info

    # ── Video helpers ─────────────────────────────────────────────────────────

    def _extract_frame(self, obs_raw: dict) -> np.ndarray | None:
        """Extract a single (H, W, 3) uint8 frame from a VectorEnv observation.

        Tries self._video_camera first, then falls back to the first available
        camera key in obs["pixels"].
        """
        pixels = obs_raw.get("pixels", {})
        if not isinstance(pixels, dict) or not pixels:
            return None
        img = pixels.get(self._video_camera)
        if img is None:
            img = next(iter(pixels.values()), None)
        if not isinstance(img, np.ndarray):
            return None
        # SyncVectorEnv batches as (n_envs, H, W, C); unbatch the first env
        frame = img[0] if img.ndim == 4 else img
        # LIBERO raw camera output is rotated 180°; flip both H and W axes so the
        # saved video matches the visually correct orientation (same correction that
        # LiberoEnv.render() and LiberoProcessorStep both apply independently).
        return np.ascontiguousarray(frame[::-1, ::-1])

    def _save_episode_video(
        self,
        frames: list[np.ndarray],
        episode_id: int,
        success: bool,
        extra_tag: str = "",
    ) -> None:
        """Write collected frames to an mp4 file using imageio.

        extra_tag is appended before the extension, e.g. extra_tag="_retry"
        produces ep0012_success_retry.mp4.
        """
        if not frames:
            return
        video_dir = Path(self._video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        status = "success" if success else "failed"
        video_path = video_dir / f"ep{episode_id:04d}_{status}{extra_tag}.mp4"
        try:
            import imageio.v3 as iio
            iio.imwrite(
                str(video_path),
                np.stack(frames),
                fps=self._video_fps,
                codec="libx264",
            )
            self.logger.info(
                f"[sim_client] Video saved → {video_path}  ({len(frames)} frames)"
            )
        except Exception as exc:
            self.logger.warning(f"[sim_client] Video save failed: {exc}")

    # ── Episode helpers ───────────────────────────────────────────────────────

    def _send_initial_obs(self, obs_raw: dict) -> None:
        """Send the first obs of an episode with must_go=True to trigger immediate inference.

        Sets is_episode_start=True so the server increments _episode_generation, enabling
        the cross-episode stale-chunk guard in GetActions.  Also consumes
        _next_obs_is_episode_start so the subsequent control_loop_observation() call does
        not send a second is_episode_start=True on a non-must_go obs (which the server
        ignores anyway, but avoids confusion).
        """
        processed = self._preprocess_obs(obs_raw)
        observation = TimedObservation(
            timestamp=time.time(),
            observation=processed,
            timestep=0,
            inference_delay=0,
            leftover_actions=None,
            must_go=True,
            obs_pre_mapped=True,
            is_episode_start=True,
        )
        self.send_observation(observation)
        self.must_go.clear()
        # Consume the flag so control_loop_observation() doesn't double-fire the
        # episode_generation increment with a subsequent non-must_go obs.
        self._next_obs_is_episode_start = False

    def run_episode(
        self,
        episode_id: int,
        max_steps: int,
        first_episode: bool = False,
        task_description: str | None = None,
    ) -> EpisodeResult:
        """Run one complete episode.

        Args:
            episode_id:       episode index (for logging and results)
            max_steps:        maximum env steps before forced termination
            first_episode:    if True, waits at the start_barrier to sync the
                              receiver thread (only needed for the very first episode)
            task_description: natural-language task description to inject into obs.
                              Updates self._task_str for this and subsequent episodes
                              until changed again.  When None, the current value is kept.
        """
        if task_description is not None:
            self._task_str = task_description

        self._reset_loop_state()
        ep_start = time.perf_counter()
        self.logger.info(
            f"[sim_client] ── Episode {episode_id} start ── task='{self._task_str}'"
        )

        frames: list[np.ndarray] = []

        obs_raw, _info = self.env.reset()
        self._last_obs_raw = obs_raw
        if self._record_video:
            frame = self._extract_frame(obs_raw)
            if frame is not None:
                frames.append(frame)

        if first_episode:
            self.start_barrier.wait()

        self._send_initial_obs(obs_raw)

        done = False
        step = 0
        last_info: dict = {}

        while not done and step < max_steps and self.running:
            t_loop = time.perf_counter()

            if self.actions_available():
                obs_raw, done, last_info = self.control_loop_action()
                step += 1
                if self._record_video and isinstance(obs_raw, dict):
                    frame = self._extract_frame(obs_raw)
                    if frame is not None:
                        frames.append(frame)
                if done:
                    break

            if self._ready_to_send_observation():
                self.control_loop_observation()

            work_t = time.perf_counter() - t_loop
            time.sleep(max(0.0, self.config.environment_dt - work_t))

            if step > 0 and step % 20 == 0:
                with self.action_queue_lock:
                    qsz = self.action_queue.qsize()
                self.logger.info(
                    f"[sim_client] step={step}/{max_steps}  queue={qsz}  running={self.running}"
                )

        success = _extract_success(last_info)
        duration = time.perf_counter() - ep_start

        if self._record_video:
            self._save_episode_video(frames, episode_id, success)

        status_tag = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"[sim_client] ══ Episode {episode_id} [{status_tag}] ══  "
            f"steps={step}  duration={duration:.2f}s  task='{self._task_str}'"
        )

        return EpisodeResult(
            episode_id=episode_id,
            task_description=self._task_str,
            success=success,
            steps=step,
            duration_s=duration,
        )
