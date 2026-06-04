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

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --interpolation_multiplier=1 \
    --rtc_execution_horizon=20 \
    --queue_size_monitor_interval=10 \
    --queue_size_monitor_path=queue_size.png
```
"""

import datetime
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.policies.rtc import ActionInterpolator, RTCConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

import time as _time

from .base_client import BaseAsyncClient
from .configs import RobotClientConfig
from .timing import ControlStepRecord
from .helpers import (
    Action,
    Observation,
    QueueSizeMonitor,
    RemotePolicyConfig,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
    resize_images_in_raw_obs,
    resize_images_with_model_pad,
)


class TrajectoryRecorder:
    """Thread-safe per-episode recorder for action chunks and executed actions.

    Two threads write concurrently:
      - action receiver thread  → record_chunk()
      - control loop thread     → record_executed()

    A single lock protects all mutations.  Each call to start_episode() flushes
    the previous episode's data to a JSON file and opens a fresh record.

    JSON schema per file:
    {
      "task": str,
      "episode": int,
      "start_time": float,          # wall time (seconds since epoch)
      "action_keys": [str, ...],    # ordered feature keys matching action tensor
      "chunks": [
        {
          "obs_id":      int,        # obs timestep that triggered inference
          "received_at": float,      # wall time chunk arrived at client
          "timesteps":   [int, ...], # per-action timestep from server
          "actions":     [[float, ...], ...]  # raw action vectors
        }, ...
      ],
      "executed": [
        {
          "timestep":    int,        # queue slot timestep of the raw policy action
          "executed_at": float,      # wall time robot.send_action() was called
          "action":      [float, ...],
          "interpolated": bool       # True when interpolation_multiplier > 1 sub-step
        }, ...
      ]
    }
    """

    def __init__(self, output_dir: str) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict | None = None
        self._current_ep: int = -1   # tracks episode number across next_episode() calls

    def start_episode(self, task: str, episode: int, action_keys: list[str]) -> None:
        """Flush the previous episode (if any) and start a fresh record."""
        with self._lock:
            self._flush_locked()
            self._current_ep = episode
            self._data = {
                "task": task,
                "episode": episode,
                "start_time": time.time(),
                "action_keys": action_keys,
                "chunks": [],
                "executed": [],
            }

    def next_episode(self, task: str, action_keys: list[str]) -> Path | None:
        """Flush the current episode and open the next one (called on TASK_DONE).

        The episode number is incremented internally so the new file gets a
        distinct index without requiring _reset_loop_state() to be called.
        Returns the path of the flushed file, or None if it was empty.
        """
        with self._lock:
            saved = self._flush_locked()
            self._current_ep += 1
            self._data = {
                "task": task,
                "episode": self._current_ep,
                "start_time": time.time(),
                "action_keys": action_keys,
                "chunks": [],
                "executed": [],
            }
            return saved

    def record_chunk(self, obs_id: int, timed_actions: list) -> None:
        """Record one received action chunk (called from the receiver thread)."""
        with self._lock:
            if self._data is None:
                return
            self._data["chunks"].append({
                "obs_id": obs_id,
                "received_at": time.time(),
                "timesteps": [ta.get_timestep() for ta in timed_actions],
                "actions": [ta.get_action().tolist() for ta in timed_actions],
            })

    def record_executed(
        self,
        timestep: int,
        action: torch.Tensor,
        interpolated: bool,
        feedback_state: dict | None = None,
    ) -> None:
        """Record one action that was actually sent to the robot motors.

        Args:
            timestep:       Queue slot timestep of the originating raw policy action.
            action:         The actual tensor sent (may be interpolated).
            interpolated:   True when this is a sub-step between two policy actions.
            feedback_state: Latest joint positions read from the robot (pose only,
                            keyed as "motor_name.pos"). None when unavailable.
        """
        with self._lock:
            if self._data is None:
                return
            entry: dict = {
                "timestep": timestep,
                "executed_at": time.time(),
                "action": action.tolist(),
                "interpolated": interpolated,
            }
            if feedback_state is not None:
                entry["feedback_state"] = feedback_state
            self._data["executed"].append(entry)

    def flush(self) -> Path | None:
        """Write the current episode to disk and return the file path (or None if empty)."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> Path | None:
        """Must be called with self._lock held."""
        if self._data is None:
            return None
        if not self._data["chunks"] and not self._data["executed"]:
            self._data = None
            return None
        ts = datetime.datetime.fromtimestamp(self._data["start_time"]).strftime("%Y%m%d_%H%M%S")
        ep = self._data["episode"]
        path = self._dir / f"episode_{ep:04d}_{ts}.json"
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)
        self._data = None
        return path


class RobotClient(BaseAsyncClient):
    """Async-inference client for physical robot hardware.

    Extends BaseAsyncClient with robot-specific hooks:
      * _build_policy_config()    — maps robot hardware features to RemotePolicyConfig
      * _capture_raw_obs()        — reads from robot hardware + injects task string
      * _preprocess_obs()         — optional client-side resize (letterbox or model-specific pad)
      * _build_timed_observation() — obs_pre_mapped=False (server does lerobot conversion)
      * control_loop_action()     — overridden for ActionInterpolator sub-step control
      * stop()                    — robot.disconnect() + channel close
    """

    prefix = "robot_client"

    def __init__(self, config: RobotClientConfig):
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        super().__init__(config)

        # Linear interpolator: expands each policy action into (multiplier) sub-steps,
        # running the robot at (fps * multiplier) Hz without extra inference calls.
        self.interpolator = ActionInterpolator(multiplier=config.interpolation_multiplier)

        # Task string injected into each observation; updated by control_loop(task=…).
        self._current_task: str = config.task

        self.logger.info("Robot connected and ready")

        # Trajectory recorder (disabled when record_trajectory=False)
        self._traj_recorder: TrajectoryRecorder | None = None
        self._last_feedback_state: dict | None = None  # latest .pos snapshot from get_observation()
        if config.record_trajectory and config.trajectory_output_dir:
            self._traj_recorder = TrajectoryRecorder(config.trajectory_output_dir)
            self.logger.info(f"Trajectory recording enabled → {config.trajectory_output_dir}")

    # ── Trajectory recording hooks ────────────────────────────────────────────

    def _reset_loop_state(self) -> None:
        """Start a new trajectory file at the beginning of each task episode."""
        super()._reset_loop_state()
        if self._traj_recorder is not None:
            self._traj_recorder.start_episode(
                task=self._current_task,
                episode=self._current_episode,   # already incremented by super()
                action_keys=list(self.robot.action_features.keys()),
            )

    def _on_chunk_received(self, timed_actions: list, obs_id: int) -> None:
        """Record each incoming action chunk (called from the receiver thread)."""
        if self._traj_recorder is not None:
            self._traj_recorder.record_chunk(obs_id, timed_actions)

    def _read_feedback_state(self) -> None:
        """Update _last_feedback_state via a cheap bus read (no cameras, ~2–4 ms).

        Called every control step (guarded by `self._traj_recorder is not None`),
        giving per-step full-arm joint positions aligned with the executed action
        rather than the sparse observation-send cadence.
        """
        try:
            pos_raw = self.robot.bus.sync_read("Present_Position")
            self._last_feedback_state = {
                f"{motor}.pos": float(val) for motor, val in pos_raw.items()
            }
        except Exception:
            pass  # stale state remains; do not interrupt the control loop

    def _on_task_done(self) -> None:
        """Called when SmartRobotClient confirms TASK_DONE (pick-place cycle complete).

        Flushes the current episode's trajectory to disk and opens a new episode
        file for the next pick-place cycle — all within a single control_loop() run.
        """
        if self._traj_recorder is not None:
            saved = self._traj_recorder.next_episode(
                task=self._current_task,
                action_keys=list(self.robot.action_features.keys()),
            )
            if saved is not None:
                self.logger.info(f"Trajectory saved (TASK_DONE) → {saved}")

    # ── Abstract hook implementations ─────────────────────────────────────────

    def _build_policy_config(self) -> RemotePolicyConfig:
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        _rtc_cfg: RTCConfig | None = None
        if self.config.rtc_execution_horizon > 0:
            _rtc_cfg = RTCConfig(enabled=True, execution_horizon=self.config.rtc_execution_horizon)
            self.logger.info(
                f"RTC re-planning enabled | execution_horizon={self.config.rtc_execution_horizon}"
            )
        return RemotePolicyConfig(
            self.config.policy_type,
            self.config.pretrained_name_or_path,
            lerobot_features,
            self.config.actions_per_chunk,
            self.config.policy_device,
            rtc_config=_rtc_cfg,
        )

    def _capture_raw_obs(self) -> dict:
        raw = self.robot.get_observation()
        raw["task"] = self._current_task
        return raw

    def _preprocess_obs(self, raw_obs: dict) -> dict:
        if self.config.obs_image_resize_hw is not None:
            if self.config.obs_image_use_model_resize:
                return resize_images_with_model_pad(
                    raw_obs, self.config.policy_type, self.config.obs_image_resize_hw
                )
            return resize_images_in_raw_obs(raw_obs, self.config.obs_image_resize_hw)
        return raw_obs  # server converts via raw_observation_to_observation()

    def _build_timed_observation(
        self,
        processed_obs: dict,
        timestep: int,
        infer_delay: int,
        leftover,
    ) -> TimedObservation:
        return TimedObservation(
            timestamp=time.time(),
            observation=processed_obs,
            timestep=timestep,
            inference_delay=infer_delay,
            leftover_actions=leftover,
            obs_pre_mapped=False,
            skip_server_resize=self.config.obs_image_use_model_resize,
        )

    # ── Override: action execution with interpolator ──────────────────────────

    def control_loop_action(self, verbose: bool = False) -> Any:
        """Dequeue one raw action and feed the ActionInterpolator.

        When interpolation_multiplier > 1 this is called at (fps * multiplier) Hz;
        the interpolator's sub-step buffer is exhausted before a new raw action is
        dequeued, keeping the robot running smoothly between policy updates.
        """
        if self._control_step_recorder is not None:
            with self.latest_action_lock:
                _ts_now = self.latest_action
            self._control_step_recorder.add(ControlStepRecord(
                wall_time=_time.time(),
                episode=self._current_episode,
                timestep=_ts_now,
            ))

        _dequeued_new = False
        if self.interpolator.needs_new_action():
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                timed_action = self.action_queue.get_nowait()
                # Step 3: remove consumed timestep so it is not sent back as leftover.
                self._orig_buf.pop(timed_action.get_timestep(), None)

            self.interpolator.add(timed_action.get_action().cpu())
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
            _dequeued_new = True

        action = self.interpolator.get()
        if action is None:
            return None

        if self._traj_recorder is not None:
            # interpolated=True only when the multiplier expands one policy action into
            # multiple sub-steps; the first sub-step (_dequeued_new=True) is the raw
            # policy action; subsequent sub-steps are linearly interpolated values.
            _interpolated = (self.config.interpolation_multiplier > 1) and (not _dequeued_new)
            with self.latest_action_lock:
                _ts = self.latest_action
            self._traj_recorder.record_executed(
                _ts, action, _interpolated, feedback_state=self._last_feedback_state
            )

        return self.robot.send_action(self._action_tensor_to_action_dict(action))

    # ── Override: stop with robot disconnect ──────────────────────────────────

    def stop(self) -> None:
        if self._traj_recorder is not None:
            try:
                saved = self._traj_recorder.flush()
                if saved is not None:
                    self.logger.info(f"Trajectory saved → {saved}")
            except BaseException as exc:
                self.logger.warning(f"trajectory_recorder.flush() raised: {exc}")

        self.shutdown_event.set()
        # Run robot.disconnect() in a daemon thread so the main thread is never
        # blocked inside serial C code (select.select) when a second Ctrl+C arrives.
        # A KeyboardInterrupt raised inside select.select() causes CPython to crash
        # with "FATAL: exception not rethrown" before any Python finally-block runs,
        # making it impossible to save timing logs or the queue-size image.
        _disc = threading.Thread(
            target=self.robot.disconnect, daemon=True, name="robot-disconnect"
        )
        _disc.start()
        try:
            _disc.join(timeout=5.0)
        except BaseException:
            pass  # second Ctrl+C during join — daemon thread is reaped on process exit
        if _disc.is_alive():
            self.logger.warning("robot.disconnect() did not finish within 5 s — proceeding")
        else:
            self.logger.debug("Robot disconnected")
        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    # ── Robot-specific helper ─────────────────────────────────────────────────

    def _action_tensor_to_action_dict(self, action: torch.Tensor) -> dict[str, float]:
        return {key: action[i].item() for i, key in enumerate(self.robot.action_features)}

    # ── Main control loop ─────────────────────────────────────────────────────

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations."""
        self._current_task = task
        # Reset per-run state so the first control_loop_observation() sends
        # is_episode_start=True, giving the server the same cross-episode guard that
        # sim_client gets via _send_initial_obs.
        self._reset_loop_state()
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        # When interpolation_multiplier > 1 this is shorter than environment_dt,
        # running the robot at (fps * multiplier) Hz without extra inference calls.
        control_interval = self.interpolator.get_control_interval(self.config.fps)

        while self.running:
            t_loop = time.perf_counter()

            if not self.interpolator.needs_new_action() or self.actions_available():
                self.control_loop_action(verbose)

            if self._traj_recorder is not None:
                self._read_feedback_state()

            if self._ready_to_send_observation():
                self.control_loop_observation()

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - t_loop) * 1000:.2f}")
            time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))

        return None, None


def _write_timing_summary_txt(timing_output_dir: str, logger=None) -> None:
    """Write a human-readable timing summary from *_summary.json files."""
    import json as _json
    from pathlib import Path as _Path
    import logging as _logging

    _log = logger or _logging.getLogger(__name__)
    output_dir = _Path(timing_output_dir)

    def _fmt_table(json_path: "_Path") -> list[str]:
        if not json_path.exists():
            return []
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        n = data.get("n_records", "?")
        prefix = json_path.stem.replace("_summary", "")
        rows = [
            f"[TimingRecorder] {prefix}  ({n} records)",
            f"  {'field':<30} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}",
            "  " + "-" * 72,
        ]
        for key, stats in data.items():
            if key == "n_records" or not isinstance(stats, dict):
                continue
            if "mean" not in stats or stats["mean"] is None:
                continue
            nan_tag = f"  [{stats['n_nan']} NaN]" if stats.get("n_nan", 0) > 0 else ""
            rows.append(
                f"  {key:<30} {stats['mean']:>7.2f}  "
                f"{stats['p50']:>7.2f}  "
                f"{stats['p95']:>7.2f}  "
                f"{stats['p99']:>7.2f}  "
                f"{stats['max']:>7.2f}{nan_tag}"
            )
        return rows

    txt_lines: list[str] = []
    for prefix in ("client_obs_sent", "client_chunk_recv", "client_chunk_action", "client_aggregate"):
        tbl = _fmt_table(output_dir / f"{prefix}_summary.json")
        if tbl:
            txt_lines.extend(tbl)
            txt_lines.append("")

    if not txt_lines:
        return

    txt_path = output_dir / "timing_summary.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    _log.info(f"[save_timing] Consolidated timing summary → {txt_path}")


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    init_logging(console_level=cfg.log_level.upper())
    logging.info(pformat(asdict(cfg)))

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        if cfg.timing_output_dir:
            client.enable_timing(cfg.timing_output_dir)

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        queue_monitor: QueueSizeMonitor | None = None
        if cfg.queue_size_monitor_interval > 0:
            queue_monitor = QueueSizeMonitor(
                data=client.action_queue_size,
                interval=cfg.queue_size_monitor_interval,
                path=cfg.queue_size_monitor_path,
            )
            queue_monitor.start()
            client.logger.info(
                f"Queue size monitor started — saving PNG every "
                f"{cfg.queue_size_monitor_interval}s to {cfg.queue_size_monitor_path}"
            )

        try:
            client.control_loop(task=cfg.task)
        finally:
            if queue_monitor is not None:
                try:
                    queue_monitor.stop()  # final render runs synchronously in this thread
                except BaseException as exc:
                    client.logger.warning(f"queue_monitor.stop() raised: {exc}")
            try:
                client.stop()
            except BaseException as exc:
                client.logger.warning(f"client.stop() raised: {exc}")
            try:
                action_receiver_thread.join(timeout=5.0)
            except BaseException:
                pass
            try:
                client.save_timing()
            except BaseException as exc:
                client.logger.warning(f"save_timing() raised: {exc}")
            if cfg.timing_output_dir:
                try:
                    _write_timing_summary_txt(cfg.timing_output_dir, client.logger)
                except BaseException as exc:
                    client.logger.warning(f"_write_timing_summary_txt() raised: {exc}")
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
