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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    log_level: str = field(
        default="INFO",
        metadata={"help": "Python logging level: DEBUG / INFO / WARNING / ERROR. DEBUG enables per-obs similarity values."},
    )

    obs_similarity_atol: float = field(
        default=1.0,
        metadata={
            "help": (
                "Tolerance for the joint-space similarity check used to decide whether an incoming "
                "observation is 'too similar' to the last inferred one and should be skipped. "
                "Two observations are considered similar when "
                "torch.linalg.norm(state_new - state_prev) < obs_similarity_atol. "
                "Units match observation.state (degrees for SO-101/SO-100, radians for other robots). "
                "Lower = more sensitive (re-infer on smaller motion); "
                "higher = less sensitive (skip more re-inferences). "
                "Default 1.0 corresponds to roughly 1° total L2 joint change for SO-101. "
                "Set to 0.0 to disable the check entirely (always re-infer every observation)."
            )
        },
    )

    # Fine-grained timing statistics (disabled when None)
    timing_output_dir: str | None = field(
        default=None,
        metadata={"help": "If set, write per-step timing records (JSONL + summary JSON) to this directory"},
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

        if self.obs_similarity_atol < 0:
            raise ValueError(f"obs_similarity_atol must be non-negative, got {self.obs_similarity_atol}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # RTC re-planning on server: set > 0 to enable leftover-guided chunk generation.
    # Only honoured for RTC-capable policies (smolvla, pi0, pi05); ignored for others.
    # Controls execution_horizon in RTCConfig sent to the server.
    rtc_execution_horizon: int = field(
        default=0,
        metadata={
            "help": (
                "Execution horizon for server-side RTC re-planning (0 = disabled). "
                "When > 0, the client sends leftover actions with each observation and the "
                "server uses them as prev_chunk_left_over in predict_action_chunk. "
                "Only effective for smolvla / pi0 / pi05 policies."
            )
        },
    )

    # Safety margin added on top of the p50 latency estimate for infer_delay.
    # Expressed in seconds so it scales correctly across fps (converted to steps
    # dynamically as ceil(spike_buffer_s / dt)).
    # Larger values → more conservative (safer against latency spikes, but more
    # stale actions).  Smaller values → more aggressive (fresher actions, higher
    # risk of occasional starvation handled by force_must_go).
    # Default 0.15 s = 3 steps @20 Hz, 2 steps @10 Hz, 5 steps @30 Hz.
    spike_buffer_s: float = field(
        default=0.15,
        metadata={
            "help": (
                "Safety buffer (seconds) added to the p50 latency estimate in the FALLBACK "
                "(bootstrap) infer_delay formula used before the split-component trackers have "
                "enough samples.  Scaled to steps as ceil(spike_buffer_s / dt) (fps-invariant). "
                "Fallback infer_delay = min(ceil(p50/dt) + buffer_steps, ceil(p90/dt))."
            )
        },
    )

    # ── Split-component (Tier 2) infer_delay estimation ──────────────────────────
    # complete_s is split into two components with different statistical natures:
    #   server_infer (stable, σ/μ≈15%, single-peaked) → high quantile (cheap to cover)
    #   overhead = complete_s − server_infer (heavy-tailed: gRPC bimodal + queue_wait)
    #                                          → moderate quantile (rare spikes caught by
    #                                            force_must_go, so no need to cover the tail)
    # infer_delay = ceil( (infer_q-quantile(server_infer) + overhead_q-quantile(overhead)) / dt )
    infer_latency_quantile: float = field(
        default=0.90,
        metadata={
            "help": (
                "Quantile of the stable server-inference latency component used for infer_delay. "
                "Higher (e.g. 0.95) = safer coverage of inference jitter; the component is "
                "single-peaked so covering its upper tail is cheap. Range (0, 1)."
            )
        },
    )
    overhead_latency_quantile: float = field(
        default=0.75,
        metadata={
            "help": (
                "Quantile of the heavy-tailed overhead component (gRPC + queue_wait + deser) "
                "used for infer_delay.  Lower (e.g. 0.50) = fresher actions, occasional "
                "starvation caught by force_must_go; higher (e.g. 0.90) = more conservative. "
                "Range (0, 1).  Keep below infer_latency_quantile — the tail here is bimodal/"
                "heavy and covering it inflates infer_delay (the overcorrection failure mode)."
            )
        },
    )

    # Interpolation configuration: multiply the control rate by this factor using linear interpolation
    # between consecutive policy actions (1 = off, 2 = 2x, 3 = 3x, etc.)
    interpolation_multiplier: int = field(
        default=1,
        metadata={
            "help": (
                "Control rate multiplier via linear interpolation between consecutive policy actions. "
                "1 = disabled (raw policy actions), 2 = 2x control rate, 3 = 3x control rate. "
                "Higher values smooth motion at the cost of responsiveness."
            )
        },
    )

    # Fine-grained timing statistics (disabled when None)
    timing_output_dir: str | None = field(
        default=None,
        metadata={"help": "If set, write per-step timing JSONL records to this directory (for analyze_timing / analyze_rtc)"},
    )

    # ── Payload compression (方案1 + 方案2) ─────────────────────────────────────
    # 方案1: Resize camera images before pickling and sending.
    # Accepts a uniform (H, W) applied to all cameras, or a per-camera dict
    # {'cam_key': (H, W), ...} for cameras of different sizes/aspect ratios.
    # The server still resizes to the policy's expected resolution, but starting
    # from a smaller image reduces gRPC payload significantly.
    # Resize uses letterbox (no aspect-ratio distortion).
    obs_image_resize_hw: tuple[int, int] | dict[str, tuple[int, int]] | None = field(
        default=None,
        metadata={
            "help": (
                "Letterbox-resize camera images before sending to server. "
                "Uniform: (H, W) applied to all cameras. "
                "Per-camera: {'top': (480, 640), 'wrist': (480, 640), 'front': (480, 640)}. "
                "None = no resize (send at full camera resolution)."
            )
        },
    )

    # 方案1b: Use model-specific resize+pad instead of letterbox when obs_image_resize_hw is set.
    # When True, the client applies the model's own resize+pad strategy (matching training-time
    # preprocessing) and signals the server to skip its resize step.
    # Requires obs_image_resize_hw to be set to the model's target resolution:
    #   smolvla → (512, 512), pi05 → (224, 224).
    # Only supported for policy_type in {smolvla, pi05, pi0}.
    obs_image_use_model_resize: bool = field(
        default=False,
        metadata={
            "help": (
                "Use model-specific resize+pad (instead of letterbox) when obs_image_resize_hw is set. "
                "Ensures client-side image preprocessing matches training-time behavior: "
                "smolvla uses top+left padding (content at bottom-right); "
                "pi05/pi0 uses centered padding. "
                "Server bypasses its resize step; model's internal resize_with_pad handles the image. "
                "Requires obs_image_resize_hw set to the model target: smolvla=(512,512), pi05=(224,224). "
                "Supported policy_types: smolvla, pi05, pi0."
            )
        },
    )

    # 方案2: JPEG-compress camera images before pickling and sending.
    # When set, the client encodes each HWC uint8 image array as JPEG bytes using
    # jpeg_encode_images_in_raw_obs(), reducing payload by ~10-20× versus raw pixels.
    # The server decodes before running raw_observation_to_observation().
    # Typical recommended value: 85 (good quality / size trade-off).
    obs_image_jpeg_quality: int | None = field(
        default=None,
        metadata={
            "help": (
                "JPEG quality (1–95) for compressing camera images before sending. "
                "None = disabled (send raw pixel arrays). "
                "Typical value: 85. Can be combined with obs_image_resize_hw."
            )
        },
    )

    # Debug configuration
    queue_size_monitor_interval: float = field(
        default=0.0,
        metadata={
            "help": (
                "If > 0, start a background thread that saves a queue-size PNG every N seconds "
                "during the control loop (non-blocking, works over SSH). "
                "Use together with queue_size_monitor_path."
            )
        },
    )
    queue_size_monitor_path: str = field(
        default="queue_size.png",
        metadata={"help": "Output path for the periodic queue-size PNG written by queue_size_monitor_interval"},
    )

    # Trajectory recording: obs_id + full action chunk + actually-executed actions per task
    record_trajectory: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, record received action chunks (obs_id + full chunk) and the actions actually "
                "sent to the robot into per-episode JSON files under trajectory_output_dir. "
                "Each task call writes a new file. Use analyze_trajectory.py to visualise EE paths."
            )
        },
    )
    trajectory_output_dir: str = field(
        default="trajectories",
        metadata={"help": "Directory to write per-episode trajectory JSON files (used when record_trajectory=True)"},
    )

    log_level: str = field(
        default="INFO",
        metadata={"help": "Python logging level: DEBUG / INFO / WARNING / ERROR"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.interpolation_multiplier < 1:
            raise ValueError(f"interpolation_multiplier must be >= 1, got {self.interpolation_multiplier}")

        if self.rtc_execution_horizon < 0:
            raise ValueError(f"rtc_execution_horizon must be >= 0, got {self.rtc_execution_horizon}")

        if self.spike_buffer_s < 0:
            raise ValueError(f"spike_buffer_s must be >= 0, got {self.spike_buffer_s}")

        for _qname, _q in (("infer_latency_quantile", self.infer_latency_quantile),
                           ("overhead_latency_quantile", self.overhead_latency_quantile)):
            if not (0.0 < _q < 1.0):
                raise ValueError(f"{_qname} must be in the open interval (0, 1), got {_q}")

        if self.obs_image_resize_hw is not None:
            def _check_hw(hw, label="obs_image_resize_hw"):
                if len(hw) != 2 or any(d <= 0 for d in hw):
                    raise ValueError(f"{label} must be a (H, W) tuple with positive values, got {hw}")

            if isinstance(self.obs_image_resize_hw, dict):
                for cam_key, hw in self.obs_image_resize_hw.items():
                    _check_hw(hw, label=f"obs_image_resize_hw['{cam_key}']")
            else:
                _check_hw(self.obs_image_resize_hw)

        if self.obs_image_jpeg_quality is not None:
            if not (1 <= self.obs_image_jpeg_quality <= 95):
                raise ValueError(
                    f"obs_image_jpeg_quality must be between 1 and 95, got {self.obs_image_jpeg_quality}"
                )

        if self.obs_image_use_model_resize:
            if self.obs_image_resize_hw is None:
                raise ValueError(
                    "obs_image_use_model_resize=True requires obs_image_resize_hw to be set "
                    "to the model's target resolution (e.g. smolvla=(512,512), pi05=(224,224))."
                )
            _supported = {"smolvla", "pi05", "pi0"}
            if self.policy_type not in _supported:
                raise ValueError(
                    f"obs_image_use_model_resize is not supported for policy_type={self.policy_type!r}. "
                    f"Supported: {_supported}"
                )

        if self.rtc_execution_horizon > 0 and self.aggregate_fn_name != "latest_only":
            raise ValueError(
                f"RTC (rtc_execution_horizon={self.rtc_execution_horizon}) requires "
                f"--aggregate_fn_name=latest_only, got '{self.aggregate_fn_name}'. "
                "Blending RTC-guided chunks via other strategies corrupts the prefix "
                "continuity that RTC depends on."
            )

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "interpolation_multiplier": self.interpolation_multiplier,
            "rtc_execution_horizon": self.rtc_execution_horizon,
            "aggregate_fn_name": self.aggregate_fn_name,
        }
