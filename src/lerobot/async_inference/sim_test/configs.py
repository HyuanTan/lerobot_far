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

"""LiberoSimConfig — configuration for async-inference LIBERO evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..configs import AGGREGATE_FUNCTIONS, get_aggregate_function


@dataclass
class LiberoSimConfig:
    """Configuration for running async-inference evaluation on real LIBERO environments.

    Connects to a running policy_server.py (started separately) and drives one or
    more LIBERO task suites via SimRobotClient.

    Example:
        python -m lerobot.async_inference.sim_test.run_libero_test \\
            --env_task=libero_10 \\
            --policy_type=smolvla \\
            --pretrained_name_or_path=user/smolvla_base \\
            --server_address=localhost:8080 \\
            --actions_per_chunk=16 \\
            --episodes_per_task=10 \\
            --results_dir=./libero_results
    """

    # ── LIBERO environment ───────────────────────────────────────────────────
    env_task: str = field(
        default="libero_10",
        metadata={"help": "LIBERO suite name (libero_10, libero_spatial, libero_object, libero_goal, libero_90)"},
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
    max_episode_steps: int | None = field(
        default=None,
        metadata={"help": "Max steps per episode (None = use suite default)"},
    )

    # ── Episode control ──────────────────────────────────────────────────────
    episodes_per_task: int = field(
        default=10,
        metadata={"help": "Number of evaluation episodes per task"},
    )

    # ── Server connection ────────────────────────────────────────────────────
    server_address: str = field(
        default="localhost:8080",
        metadata={"help": "policy_server address (host:port)"},
    )

    # ── Policy ───────────────────────────────────────────────────────────────
    policy_type: str = field(
        default="smolvla",
        metadata={"help": "Policy type identifier (e.g. 'smolvla', 'pi0', 'pi05')"},
    )
    pretrained_name_or_path: str = field(
        default="",
        metadata={"help": "HF Hub repo or local path to the pretrained model"},
    )
    policy_device: str = field(
        default="cuda",
        metadata={"help": "Device for policy inference on the server"},
    )
    client_device: str = field(
        default="cpu",
        metadata={"help": "Device for action tensors on the client side"},
    )

    # ── Client / control loop ────────────────────────────────────────────────
    fps: int = field(default=30, metadata={"help": "Target control-loop frequency (Hz)"})
    actions_per_chunk: int = field(
        default=16,
        metadata={"help": "Number of actions per chunk requested from the server"},
    )
    chunk_size_threshold: float = field(
        default=0.5,
        metadata={"help": "Send observation when queue drops below this fraction of chunk size"},
    )
    aggregate_fn_name: str = field(
        default="latest_only",
        metadata={"help": f"Action merge strategy. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )
    rtc_execution_horizon: int = field(
        default=0,
        metadata={"help": "RTC execution horizon (0 = disabled)"},
    )
    interpolation_multiplier: int = field(
        default=1,
        metadata={"help": "Control rate multiplier via linear interpolation (1 = off)"},
    )

    # ── gRPC payload optimisation ────────────────────────────────────────────
    transmit_images_as_uint8: bool = field(
        default=False,
        metadata={
            "help": (
                "Transmit camera images as uint8 [0,255] instead of float32 [0,1]. "
                "Reduces gRPC observation payload ~4x (e.g. 3 MB → 0.75 MB for two 256×256 cameras). "
                "The server auto-converts uint8 → float32 before the policy preprocessor. "
                "Enable for latency comparison: --transmit_images_as_uint8=true"
            )
        },
    )

    # ── Video recording ──────────────────────────────────────────────────────
    save_video: bool = field(
        default=False,
        metadata={"help": "Save a per-episode mp4 video of env observations"},
    )
    video_dir: str = field(
        default="./libero_videos",
        metadata={"help": "Directory for episode video files (ep{N}_{success|failed}.mp4)"},
    )
    video_camera: str = field(
        default="image",
        metadata={"help": "Camera key in obs['pixels'] to use for video (e.g. 'image', 'image2')"},
    )
    video_fps: int = field(
        default=30,
        metadata={"help": "Frames per second for saved videos"},
    )

    # ── Results ──────────────────────────────────────────────────────────────
    results_dir: str = field(
        default="./libero_results",
        metadata={"help": "Directory for per-episode and aggregate JSON result files"},
    )
    save_results: bool = field(default=True, metadata={"help": "Persist results to JSON"})
    timing_output_dir: str | None = field(
        default=None,
        metadata={"help": "If set, write timing records (JSONL + summary JSON) to this directory"},
    )
    queue_size_monitor_interval: float = field(
        default=0.0,
        metadata={
            "help": (
                "If > 0, save a queue-size PNG every N seconds during the run (non-blocking, SSH-safe). "
                "Use together with queue_size_monitor_path."
            )
        },
    )
    queue_size_monitor_path: str = field(
        default="queue_size.png",
        metadata={"help": "Output path for the periodic queue-size PNG"},
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = field(
        default="INFO",
        metadata={"help": "Python logging level: DEBUG / INFO / WARNING / ERROR"},
    )

    # ── Internal (set dynamically per task by run_libero_test) ───────────────
    task: str = field(
        default="",
        metadata={"help": "Current task description (updated at runtime per task)"},
    )

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def environment_dt(self) -> float:
        return 1.0 / self.fps

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if not 0.0 <= self.chunk_size_threshold <= 1.0:
            raise ValueError(
                f"chunk_size_threshold must be in [0, 1], got {self.chunk_size_threshold}"
            )
        if self.interpolation_multiplier < 1:
            raise ValueError(
                f"interpolation_multiplier must be >= 1, got {self.interpolation_multiplier}"
            )
        if self.rtc_execution_horizon < 0:
            raise ValueError(
                f"rtc_execution_horizon must be >= 0, got {self.rtc_execution_horizon}"
            )
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)
