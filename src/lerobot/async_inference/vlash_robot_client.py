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
VLASHRobotClient — RobotClient extended with VLASH future-state injection.

VLASH async inference differs from LeRobot's RTC in how it handles inference
latency: instead of conditioning the denoiser on `inference_delay` and
`prev_chunk_left_over`, it replaces `observation.state` with the **last action
of the most recently received chunk**.  This tells the policy "by the time you
finish inferring, the robot will be approximately at this position", allowing
the new chunk to begin smoothly from the predicted future pose.

Usage:
    python -m lerobot.async_inference.vlash_robot_client \\
        --robot.type=so100_follower \\
        --robot.port=/dev/tty.usbmodem58760431541 \\
        --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="pick the apple" \\
        --server_address=127.0.0.1:8080 \\
        --policy_type=vlash_pi05 \\
        --pretrained_name_or_path=/path/to/checkpoint \\
        --policy_device=cuda \\
        --client_device=cpu \\
        --actions_per_chunk=16 \\
        --fps=30
"""

import logging
import threading
import time
from dataclasses import asdict
from pprint import pformat

import draccus

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
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import QueueSizeMonitor, TimedAction
from .robot_client import RobotClient

class VLASHRobotClient(RobotClient):
    """RobotClient with VLASH future-state injection.

    Extends RobotClient with two overrides that implement VLASH's async strategy:

    1. ``_aggregate_action_queues()`` — after merging an incoming chunk into the
       queue, captures the chunk's final action as ``_vlash_future_state``.

    2. ``_capture_raw_obs()`` — replaces each motor-joint value in the raw
       observation with the corresponding value from ``_vlash_future_state``.
       When ``raw_observation_to_observation()`` runs on the server, it reads
       these overridden joint values and builds an ``observation.state`` tensor
       that represents the robot's *predicted* future pose rather than its
       current pose.

    Set ``policy_type=vlash_pi05`` (or ``vlash_pi0``) so the paired
    ``VLASHPolicyServer`` loads the correct VLASH policy.
    """

    prefix = "vlash_robot_client"

    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        # Numpy array [action_dim] — last action of the most recent incoming chunk.
        # Written by receive_actions() thread, read by control_loop() thread.
        self._vlash_future_state = None
        self._vlash_future_state_lock = threading.Lock()

    # ── VLASH future-state injection ─────────────────────────────────────────

    def _aggregate_action_queues(
        self,
        incoming: list[TimedAction],
        aggregate_fn=None,
    ) -> dict:
        """Merge incoming chunk and capture its last action as the future state.

        Called from ``receive_actions()`` (background thread) after each chunk
        arrives from the server.  The last ``TimedAction`` in ``incoming`` (which
        is ordered by timestep) becomes the future state injected into the next
        observation.
        """
        if incoming:
            # incoming is ordered by timestep; [-1] is the final action of the chunk
            last_action = incoming[-1].get_action().cpu()
            with self._vlash_future_state_lock:
                self._vlash_future_state = last_action.numpy()
        return super()._aggregate_action_queues(incoming, aggregate_fn)

    def _capture_raw_obs(self) -> dict:
        """Read robot obs and replace joint values with the predicted future pose.

        Overrides the motor-joint scalars in the raw observation dict with values
        from the last received action chunk.  The server's
        ``raw_observation_to_observation()`` will then assemble an
        ``observation.state`` tensor from these overridden values instead of the
        robot's actual current joint positions.

        Falls back to real joint positions before the first chunk is received.
        """
        raw = super()._capture_raw_obs()  # reads robot + injects task string
        with self._vlash_future_state_lock:
            future_state = self._vlash_future_state
        if future_state is not None:
            for i, key in enumerate(self.robot.action_features):
                if i < len(future_state):
                    raw[key] = float(future_state[i])
        return raw


# ── Entry point ───────────────────────────────────────────────────────────────


@draccus.wrap()
def async_vlash_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = VLASHRobotClient(cfg)

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
                    queue_monitor.stop()
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
            client.logger.info("VLASH client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_vlash_client()
