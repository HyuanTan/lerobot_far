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

import io
import logging
import logging.handlers
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image as PILImage

from lerobot.configs import PolicyFeature

# NOTE: Configs need to be loaded for the client to be able to instantiate the policy config
from lerobot.policies import (  # noqa: F401
    ACTConfig,
    DiffusionConfig,
    PI0Config,
    PI05Config,
    SmolVLAConfig,
    VQBeTConfig,
)
from lerobot.robots.robot import Robot
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.utils import init_logging

Action = torch.Tensor

# observation as received from the robot (can be numpy arrays, floats, etc.)
RawObservation = dict[str, Any]

# observation as those recorded in LeRobot dataset (keys are different)
LeRobotObservation = dict[str, torch.Tensor]

# observation, ready for policy inference (image keys resized)
Observation = dict[str, torch.Tensor]


class QueueSizeMonitor:
    """Background-thread monitor that periodically saves a queue-size PNG.

    Uses the Agg (non-GUI) matplotlib backend so it is thread-safe and works
    over SSH without a display.  The daemon thread exits automatically when the
    main process exits, so no explicit cleanup is required unless you want to
    force a final render via stop().

    Args:
        data:     The live list that control_loop_action() appends to.
        interval: Seconds between PNG refreshes (default 10 s).
        path:     Output file path (default "queue_size.png").
    """

    def __init__(self, data: list[int], interval: float = 10.0, path: str = "queue_size.png"):
        self._data = data
        self._interval = interval
        self._path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="QueueSizeMonitor")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop, then perform the final render in the calling thread.

        The background thread is given up to 5 s to finish any in-progress periodic render.
        The definitive final render (with all accumulated data) is then executed synchronously
        here so it is not subject to daemon-thread lifetime issues or the CPython
        'FATAL: exception not rethrown' crash that can occur when KeyboardInterrupt is
        delivered during thread.join()'s internal lock.acquire().
        """
        self._stop.set()
        try:
            self._thread.join(timeout=5.0)
        except BaseException:
            pass  # KeyboardInterrupt during join — daemon thread will be reaped on process exit

        # Final render in the calling (main) thread with all accumulated data.
        snapshot = list(self._data)
        if snapshot:
            try:
                self._render(snapshot)
            except BaseException as exc:
                import traceback
                logging.getLogger(__name__).warning(
                    "[QueueSizeMonitor] Final render failed: %s\n%s",
                    exc, traceback.format_exc(),
                )
        else:
            logging.getLogger(__name__).warning(
                "[QueueSizeMonitor] No data to render (action_queue_size is empty)."
            )

    def _render(self, snapshot: list[int]) -> None:
        import os
        import tempfile
        import traceback

        # switch_backend("agg") is safe to call at any time — it works even when
        # matplotlib.pyplot was already imported by another part of the codebase
        # with a non-GUI backend (e.g. TkAgg, Qt5Agg).  Using matplotlib.use("Agg")
        # here instead would be a no-op (with a warning) if pyplot was imported first,
        # leaving the active backend as TkAgg, which raises TclError on headless hosts.
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")

        try:
            fig, ax = plt.subplots(figsize=(10, 4))
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[QueueSizeMonitor] plt.subplots() failed: %s\n%s",
                exc, traceback.format_exc(),
            )
            raise

        ax.plot(range(len(snapshot)), snapshot, linewidth=0.8)
        ax.set_title(f"Action Queue Size Over Time  (n={len(snapshot)} steps)")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Queue Size")
        if snapshot:
            ax.set_ylim(0, max(snapshot) * 1.1 + 1)
        ax.grid(True, alpha=0.3)
        zeros = snapshot.count(0)
        stats = (
            f"mean={sum(snapshot)/len(snapshot):.1f}  "
            f"max={max(snapshot)}  "
            f"starved={zeros} ({100*zeros//len(snapshot)}%)"
        )
        ax.set_title(ax.get_title() + f"\n{stats}", fontsize=9)
        fig.tight_layout()

        # Atomic write: save to a sibling .tmp file then rename so that a
        # mid-write kill (second Ctrl+C, SIGKILL, OOM) never corrupts the
        # previous PNG — os.replace() is atomic on POSIX filesystems.
        # Explicitly pass format='png' so matplotlib does not try to infer
        # the format from the '.tmp' extension (which it does not recognise).
        out = Path(self._path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
        try:
            os.close(tmp_fd)
            fig.savefig(tmp_path, format="png", dpi=120, bbox_inches="tight")
            os.replace(tmp_path, str(out))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        finally:
            plt.close(fig)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            snapshot = list(self._data)  # atomic copy under CPython GIL
            if snapshot:
                try:
                    self._render(snapshot)
                except Exception as exc:
                    import traceback
                    logging.getLogger(__name__).warning(
                        "[QueueSizeMonitor] Periodic render failed: %s\n%s",
                        exc, traceback.format_exc(),
                    )
        # Final render is handled by stop() in the calling thread, not here.


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: torch.tensor, resize_dims: tuple[int, int, int]) -> torch.tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


# TODO(Steven): Consider implementing a pipeline step for this
def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    skip_resize: bool = False,
) -> Observation:
    observation = {}

    observation = prepare_raw_observation(
        raw_observation, lerobot_features, policy_image_features, skip_resize=skip_resize
    )
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0)
        else:
            observation[k] = v

    return observation


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> torch.Tensor:
    """Extract the state from a raw observation."""
    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> dict[str, torch.Tensor]:
    """Extract the images from a raw observation."""
    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """Make a lerobot observation from a raw observation."""
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    skip_resize: bool = False,
) -> Observation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. Greps all observation.images.<> keys
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    if skip_resize:
        # Client applied model-specific resize+pad; images are already at the target resolution.
        # Only permute HWC→CHW without any interpolation so the model's internal resize_with_pad
        # remains a no-op and aspect ratio / padding alignment are preserved.
        image_dict = {
            key: torch.tensor(lerobot_obs[key]).permute(2, 0, 1)
            for key in image_keys
        }
    else:
        # Turns the image features to (C, H, W) with H, W matching the policy image features.
        # This reduces the resolution of the images
        image_dict = {
            key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
            for key in image_keys
        }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Get a logger using the standardized logging setup from utils.py.

    Args:
        name: Logger name (e.g., 'policy_server', 'robot_client')
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # Initialize the standardized logging
    init_logging(log_file=log_file, display_pid=False)

    # Return a named logger
    return logging.getLogger(name)


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False
    # Step 2: latency-aware hint passed to policy.predict_action_chunk(inference_delay=...)
    # inference_delay      — conservative estimate (max / p95 of recent latencies).
    # inference_delay_low  — optimistic estimate (p50 of recent latencies).
    # MultiCandidatePolicyServer uses the pair [low, high] as the two delay variants
    # instead of fixed base_delay ± delta, so diversity tracks real latency spread.
    # 0 means "not available" (tracker has fewer than min_samples samples).
    inference_delay: int = 0
    inference_delay_low: int = 0
    # Step 3: remaining pre-postprocessed actions from the previous chunk (shape: T×action_dim).
    # Passed as prev_chunk_left_over to RTC-capable policies.
    leftover_actions: torch.Tensor | None = None
    # When True the observation is already in lerobot format (e.g. sent by sim_client after
    # preprocess_observation + env_preprocessor).  policy_server skips raw_observation_to_observation().
    obs_pre_mapped: bool = False
    # True only for the FIRST obs of a new episode (set by _reset_loop_state).
    # Distinct from must_go=True, which is also set by post-chunk must_go.set() and
    # _force_must_go queue-empty re-triggers.  policy_server only increments
    # _episode_generation for is_episode_start=True, not for every must_go=True.
    is_episode_start: bool = False
    # When True, camera image arrays in observation have been JPEG-encoded to bytes
    # by the client to reduce gRPC payload size.  policy_server decodes them before
    # calling raw_observation_to_observation() or processing the obs_pre_mapped path.
    jpeg_images: bool = False
    # When True, the client has already applied model-specific resize+pad (via
    # obs_image_use_model_resize), so policy_server must skip resize_robot_observation_image
    # and only do the HWC→CHW permute.  Images arrive at the model's target resolution.
    skip_server_resize: bool = False
    # Client-side processing time (ms) between obs.timestamp and pickle.dumps().
    # Currently populated with jpeg_encode_ms only (the dominant variable overhead).
    # Used by the server to compute adj_one_way_ms = one_way_ms - client_send_overhead_ms,
    # a closer approximation to true network one-way latency.  0.0 when JPEG is disabled.
    client_send_overhead_ms: float = 0.0
    # When True, server skips inference AND similarity check and does NOT update
    # last_processed_obs.  Set by _BackgroundObsSender for trajectory-phase obs
    # (RECOVERY / LIFT_RETRY / REWIND_RETRY) that exist only to mark timesteps as
    # seen, not to drive policy decisions.
    # Pickle compat: old servers without this field receive the extra __dict__ key and
    # ignore it (must_go=False drives their path); use getattr(obs,'skip_inference',False).
    skip_inference: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class FPSTracker:
    """Utility class to track FPS metrics over time."""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Calculate average FPS vs target"""
        self.total_obs_count += 1

        # Initialize first observation time
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # Calculate overall average FPS (since start)
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """Reset the FPS tracker state"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class ActionChunk:
    """Wrapper for the action chunk returned by the server.

    Carries postprocessed actions for execution, the pre-postprocessed originals
    needed for RTC leftover tracking, and the server-side inference time for
    latency-aware delay computation.
    """

    timed_actions: list[TimedAction]
    # Pre-postprocessed actions in policy model-space (shape: N × action_dim).
    # None for policies that don't support RTC leftover.
    original_actions: torch.Tensor | None = None
    # Wall-clock time for the full inference pipeline (prepare → preprocess → infer → postprocess).
    inference_time_s: float = 0.0


@dataclass
class CandidateMeta:
    """Per-candidate metadata produced by the server scorer."""

    inference_delay: int       # RTC delay used for this candidate
    noise_idx: int             # index within the same-delay noise batch
    jerk: float                # smoothness score (lower = smoother)
    vel_peak: float            # max joint-velocity magnitude
    server_score: float        # final composite server score (higher = better)


@dataclass
class ActionBundle:
    """Multi-candidate payload returned by MultiCandidatePolicyServer.

    Phase 1: only `selected` is populated; `candidates` is empty.
    Phase 2: `candidates` holds all K shortlisted ActionChunks so the client
             can apply its own selection policy.

    Serialised with pickle in the existing `Actions.data` gRPC field.
    The client detects the type at runtime:
        obj = pickle.loads(data)
        if isinstance(obj, ActionBundle): ...
        elif isinstance(obj, ActionChunk): ...
    """

    # Best chunk selected server-side (always present).
    selected: ActionChunk
    # Top-K chunks for client-side selection (empty list in Phase 1).
    candidates: list[ActionChunk] = field(default_factory=list)
    # Per-candidate metadata aligned with `candidates`.
    candidate_meta: list[CandidateMeta] = field(default_factory=list)
    # Server score of `selected` (for logging).
    selected_score: float = 0.0
    # Index into `candidates` that was pre-selected server-side (Phase 2 hint).
    server_selected_idx: int = 0
    # Wall-clock inference time; set by PolicyServer.GetActions() after _predict_action_chunk().
    # Mirrors ActionChunk.inference_time_s so PolicyServer can treat ActionBundle uniformly.
    inference_time_s: float = 0.0
    # Full N candidates (including non-top_k) — populated only when
    # MultiCandidateServerConfig.record_all_candidates=True.
    all_candidates: list[ActionChunk] = field(default_factory=list)
    all_candidate_meta: list[CandidateMeta] = field(default_factory=list)

    @property
    def timed_actions(self) -> list:
        """Proxy to selected.timed_actions for PolicyServer dispatch logging (lines 408-409).

        PolicyServer.GetActions() accesses action_chunk.timed_actions[0].get_timestep()
        to log the dispatched chunk range.  This property satisfies that access without
        requiring changes to policy_server.py.
        """
        return self.selected.timed_actions


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    # Optional RTCConfig; if provided, server initialises the policy's RTC processor.
    # Only honoured for RTC-capable policies (smolvla, pi0, pi05).
    rtc_config: Any | None = None


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """Check if two observations are similar, under a tolerance threshold. Measures distance between
    observations as the difference in joint-space between the two observations.

    NOTE(fracapuano): This is a very simple check, and it is enough for the current use case.
    An immediate next step is to use (fast) perceptual difference metrics comparing some camera views,
    to surpass this joint-space similarity check.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)


def observations_similarity_norm(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict]
) -> float:
    """Return the joint-space L2 norm between two observations (lower = more similar).

    Extracts observation.state from both obs and returns
    float(torch.linalg.norm(state1 - state2)).  Used for logging alongside
    observations_similar() to avoid a second state-extraction round-trip.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )
    return float(torch.linalg.norm(obs1_state - obs2_state))


# ── Payload compression utilities ─────────────────────────────────────────────
# These operate on the raw observation dict (camera images as HWC uint8 numpy
# arrays or torch.Tensor) and are shared by the client-side encoder and the
# server-side decoder.

def _is_image_array(v: Any) -> bool:
    """Return True if *v* looks like an HWC uint8 image (numpy or torch)."""
    if isinstance(v, np.ndarray):
        return v.ndim == 3 and v.dtype == np.uint8
    if isinstance(v, torch.Tensor):
        return v.ndim == 3 and v.dtype == torch.uint8
    return False


def _letterbox(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Scale HWC uint8 array to fit within (h, w) without distortion, pad remainder with zeros."""
    cur_h, cur_w = arr.shape[:2]
    scale = min(w / cur_w, h / cur_h)
    new_w, new_h = int(cur_w * scale), int(cur_h * scale)
    resized = np.array(PILImage.fromarray(arr).resize((new_w, new_h), PILImage.BILINEAR))
    canvas = np.zeros((h, w, arr.shape[2]), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas


def _resize_with_pad_smolvla_numpy(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """HWC uint8 → HWC uint8, smolvla-style: top+left padding (content at bottom-right), value=0.

    Matches modeling_smolvla.resize_with_pad(pad_value=0):
      F.pad(resized, (pad_width, 0, pad_height, 0)) puts padding on left and top.
    """
    cur_h, cur_w = arr.shape[:2]
    ratio = max(cur_w / w, cur_h / h)
    new_w, new_h = int(cur_w / ratio), int(cur_h / ratio)
    resized = np.array(PILImage.fromarray(arr).resize((new_w, new_h), PILImage.BILINEAR))
    canvas = np.zeros((h, w, arr.shape[2]), dtype=np.uint8)
    canvas[h - new_h:, w - new_w:] = resized
    return canvas


def _resize_with_pad_pi05_numpy(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """HWC uint8 → HWC uint8, pi0.5-style: centered padding, value=0.

    Matches modeling_pi05.resize_with_pad_torch(): divmod-based symmetric padding.
    """
    cur_h, cur_w = arr.shape[:2]
    ratio = max(cur_w / w, cur_h / h)
    new_w, new_h = int(cur_w / ratio), int(cur_h / ratio)
    resized = np.array(PILImage.fromarray(arr).resize((new_w, new_h), PILImage.BILINEAR))
    pad_h0, _ = divmod(h - new_h, 2)
    pad_w0, _ = divmod(w - new_w, 2)
    canvas = np.zeros((h, w, arr.shape[2]), dtype=np.uint8)
    canvas[pad_h0:pad_h0 + new_h, pad_w0:pad_w0 + new_w] = resized
    return canvas


_MODEL_RESIZE_FN: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "smolvla": _resize_with_pad_smolvla_numpy,
    "pi05": _resize_with_pad_pi05_numpy,
    "pi0": _resize_with_pad_pi05_numpy,
}


def resize_images_with_model_pad(
    raw_obs: dict,
    policy_type: str,
    target_hw: tuple[int, int] | dict[str, tuple[int, int]],
) -> dict:
    """Return a shallow copy of *raw_obs* with images resized using the model-specific
    resize+pad strategy so that client-side preprocessing matches training-time preprocessing.

    - smolvla: pad on top and left (content at bottom-right), value=0
    - pi05/pi0: centered padding, value=0

    Requires obs_image_resize_hw to specify the model's target resolution.
    """
    resize_fn = _MODEL_RESIZE_FN.get(policy_type)
    if resize_fn is None:
        raise ValueError(
            f"obs_image_use_model_resize is not supported for policy_type={policy_type!r}. "
            f"Supported policy types: {list(_MODEL_RESIZE_FN)}"
        )

    tasks: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    for k, v in raw_obs.items():
        if _is_image_array(v):
            if isinstance(target_hw, dict):
                hw = target_hw.get(k)
                if hw is None:
                    continue
            else:
                hw = target_hw
            tasks[k] = (v.numpy() if isinstance(v, torch.Tensor) else v, hw)

    if len(tasks) <= 1:
        results = {k: resize_fn(arr, *hw) for k, (arr, hw) in tasks.items()}
    else:
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            fts = {k: ex.submit(resize_fn, arr, *hw) for k, (arr, hw) in tasks.items()}
        results = {k: f.result() for k, f in fts.items()}

    return {k: (results[k] if k in results else v) for k, v in raw_obs.items()}


def resize_images_in_raw_obs(
    raw_obs: dict,
    target_hw: tuple[int, int] | dict[str, tuple[int, int]],
) -> dict:
    """Return a shallow copy of *raw_obs* with HWC uint8 images letterboxed (no distortion).

    target_hw can be:
      - ``(H, W)``                        — same target for every camera
      - ``{'cam_key': (H, W), ...}``      — per-camera targets; cameras not in the dict
                                            are passed through unchanged

    Non-image values always pass through unchanged.  The original dict is never modified.
    When multiple images are present, resizes them in parallel (ThreadPoolExecutor,
    max_workers = number of images).  PIL resize releases the GIL, so thread-level
    parallelism is genuine.  The serial path is used for ≤1 image to avoid
    thread-pool overhead on single-camera setups.
    """
    # Collect image tasks; skip images absent from a per-camera target_hw dict.
    tasks: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    for k, v in raw_obs.items():
        if _is_image_array(v):
            if isinstance(target_hw, dict):
                hw = target_hw.get(k)
                if hw is None:
                    continue  # not in per-camera dict → pass through unchanged
            else:
                hw = target_hw
            tasks[k] = (v.numpy() if isinstance(v, torch.Tensor) else v, hw)

    if len(tasks) <= 1:
        results = {k: _letterbox(arr, *hw) for k, (arr, hw) in tasks.items()}
    else:
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            fts = {k: ex.submit(_letterbox, arr, *hw) for k, (arr, hw) in tasks.items()}
        results = {k: f.result() for k, f in fts.items()}

    return {k: (results[k] if k in results else v) for k, v in raw_obs.items()}


def _encode_jpeg(arr: np.ndarray, quality: int) -> bytes:
    """Encode a single HWC uint8 numpy array as JPEG bytes (used by the parallel path)."""
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_encode_images_in_raw_obs(raw_obs: dict, quality: int) -> dict:
    """Return a shallow copy of *raw_obs* with HWC uint8 image arrays replaced by JPEG bytes.

    Compresses each camera image using PIL JPEG at the given quality (1–95).
    The original dict is never modified.  Only HWC uint8 arrays/tensors are
    encoded; all other values (state, task string, …) are passed through as-is.
    When multiple images are present, encodes them in parallel (ThreadPoolExecutor,
    max_workers = number of images).  libjpeg releases the GIL, so encoding is
    genuinely concurrent.  The serial path is used for ≤1 image.
    """
    tasks: dict[str, np.ndarray] = {
        k: (v.numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in raw_obs.items()
        if _is_image_array(v)
    }

    if len(tasks) <= 1:
        results = {k: _encode_jpeg(arr, quality) for k, arr in tasks.items()}
    else:
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            fts = {k: ex.submit(_encode_jpeg, arr, quality) for k, arr in tasks.items()}
        results = {k: f.result() for k, f in fts.items()}

    return {k: (results[k] if k in results else v) for k, v in raw_obs.items()}


def jpeg_decode_images_in_raw_obs(raw_obs: dict) -> dict:
    """Return a shallow copy of *raw_obs* with JPEG bytes decoded back to HWC uint8 numpy arrays.

    Decodes every value that is a ``bytes`` object (previously encoded by
    :func:`jpeg_encode_images_in_raw_obs`).  All other values pass through unchanged.
    """
    out = {}
    for k, v in raw_obs.items():
        if isinstance(v, bytes):
            out[k] = np.array(PILImage.open(io.BytesIO(v)))
        else:
            out[k] = v
    return out
