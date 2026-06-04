#!/usr/bin/env python

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
Real robot evaluation with RTC continuity and zone-proportion analysis.

Extends eval_with_real_robot.py with per-inference metrics recording:
  - Zone proportions: overleft / transition / freed (actual step counts & fractions)
  - Inference latency and derived inference_delay
  - Action continuity: L2 distance between old and new actions in the overlapping zone

Three RTC zones per inference call (see RTCProcessor.get_prefix_weights):
  overleft   [0 .. inference_delay)          weight=1.0  (fully anchored)
  transition [inference_delay .. exec_horiz) weight 1→0  (soft guidance)
  freed      [exec_horiz .. chunk_size)      weight=0    (unconstrained)

Metrics are written to <output_dir>/rtc_metrics.jsonl in real time and
visualized via matplotlib at the end of the run.

Usage:
    uv run examples/rtc/eval_real_robot_rtc_analysis.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=cuda \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.id=so100_follower \
        --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Pick up the red cube" \
        --duration=120 \
        --output_dir=rtc_analysis_output

    # Dry-run without a robot (mock mode) for testing analysis/plotting:
    uv run examples/rtc/eval_real_robot_rtc_analysis.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=cuda \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --mock_robot=true \
        --task="Pick up the red cube" \
        --duration=30 \
        --output_dir=rtc_analysis_output
"""

import json
import logging
import math
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    gridspec = None

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule, parser
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    to_relative_actions,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zone analysis helpers
# ---------------------------------------------------------------------------

def compute_zone_steps(inference_delay: int, execution_horizon: int, chunk_size: int) -> dict:
    """Return step counts and fractions for each RTC zone.

    Args:
        inference_delay:   Steps anchored to previous chunk (weight=1).
        execution_horizon: End of the transition zone (weight 1→0).
        chunk_size:        Total number of actions per chunk.

    Returns dict with keys:
        overleft_steps, transition_steps, freed_steps,
        overleft_frac,  transition_frac,  freed_frac
    """
    start = min(inference_delay, execution_horizon)
    end = min(execution_horizon, chunk_size)
    total = chunk_size

    overleft = max(0, start)
    transition = max(0, end - start)
    freed = max(0, total - end)

    return {
        "overleft_steps": overleft,
        "transition_steps": transition,
        "freed_steps": freed,
        "overleft_frac": overleft / total if total > 0 else 0.0,
        "transition_frac": transition / total if total > 0 else 0.0,
        "freed_frac": freed / total if total > 0 else 0.0,
    }


def compute_continuity_metrics(prev_actions: Tensor | None, new_actions: Tensor,
                                inference_delay: int) -> dict:
    """Measure action continuity in the overlapping region.

    The overlap zone is the first `inference_delay` steps of the new chunk,
    which correspond to the first `inference_delay` steps of `prev_actions`.

    Args:
        prev_actions:    Leftover actions from previous chunk  (T_prev, A).
        new_actions:     New action chunk                      (T_new, A).
        inference_delay: How many steps overlap.

    Returns dict with:
        overlap_l2_mean:  Mean per-dim L2 norm in overlap region (NaN if no overlap).
        overlap_cosine:   Cosine similarity of flattened overlap vectors (NaN if no overlap).
    """
    result = {"overlap_l2_mean": float("nan"), "overlap_cosine": float("nan")}

    if prev_actions is None or inference_delay <= 0:
        return result

    n_overlap = min(inference_delay, prev_actions.shape[0], new_actions.shape[0])
    if n_overlap <= 0:
        return result

    old = prev_actions[:n_overlap].float()
    new = new_actions[:n_overlap].float()

    diff = (old - new).norm(dim=-1).mean().item()
    result["overlap_l2_mean"] = diff

    old_flat = old.flatten()
    new_flat = new.flatten()
    cos = torch.nn.functional.cosine_similarity(old_flat.unsqueeze(0), new_flat.unsqueeze(0)).item()
    result["overlap_cosine"] = cos

    return result


# ---------------------------------------------------------------------------
# RTCMetricsLogger — thread-safe, writes JSONL line per inference call
# ---------------------------------------------------------------------------

@dataclass
class InferenceRecord:
    """One inference call's worth of metrics."""
    call_index: int
    wall_time: float         # seconds since run start
    inference_latency: float # seconds
    inference_delay: int     # steps
    chunk_size: int
    execution_horizon: int
    overleft_steps: int
    transition_steps: int
    freed_steps: int
    overleft_frac: float
    transition_frac: float
    freed_frac: float
    overlap_l2_mean: float
    overlap_cosine: float


class RTCMetricsLogger:
    """Collects per-inference records and writes them to a JSONL file.

    Args:
        output_path: Path to the JSONL log file.
        chunk_size:  Number of actions in each policy chunk.
    """

    def __init__(self, output_path: Path, chunk_size: int):
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.lock = Lock()
        self.records: list[InferenceRecord] = []
        self._call_index = 0
        self._t0 = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(output_path, "w")  # noqa: SIM115
        logger.info(f"[METRICS] Writing RTC metrics to {output_path}")

    def record(
        self,
        inference_latency: float,
        inference_delay: int,
        execution_horizon: int,
        prev_actions: Tensor | None,
        new_actions: Tensor,
    ) -> InferenceRecord:
        zones = compute_zone_steps(inference_delay, execution_horizon, self.chunk_size)
        continuity = compute_continuity_metrics(prev_actions, new_actions, inference_delay)

        rec = InferenceRecord(
            call_index=self._call_index,
            wall_time=time.perf_counter() - self._t0,
            inference_latency=inference_latency,
            inference_delay=inference_delay,
            chunk_size=self.chunk_size,
            execution_horizon=execution_horizon,
            **zones,
            **continuity,
        )

        with self.lock:
            self.records.append(rec)
            self._call_index += 1
            self._file.write(json.dumps(rec.__dict__) + "\n")
            self._file.flush()

        return rec

    def close(self):
        with self.lock:
            self._file.close()

    @staticmethod
    def load_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_rtc_metrics(records: list[dict], output_dir: Path):
    """Generate and save analysis plots from recorded RTC metrics.

    Produces:
      rtc_zone_proportions.png  — stacked-area chart of overleft/transition/freed over time
      rtc_latency.png           — inference latency and delay over calls
      rtc_continuity.png        — action continuity (L2 / cosine) over calls
      rtc_zone_summary.png      — pie chart + bar chart of average zone proportions
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("[VIZ] matplotlib not available — skipping plots")
        return

    if not records:
        logger.warning("[VIZ] No records to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.array([r["wall_time"] for r in records])
    calls = np.arange(len(records))

    ol = np.array([r["overleft_frac"] for r in records])
    tr = np.array([r["transition_frac"] for r in records])
    fr = np.array([r["freed_frac"] for r in records])

    latencies = np.array([r["inference_latency"] * 1000 for r in records])  # ms
    delays = np.array([r["inference_delay"] for r in records])
    exec_hz = np.array([r["execution_horizon"] for r in records])

    l2s = np.array([r["overlap_l2_mean"] for r in records])
    coss = np.array([r["overlap_cosine"] for r in records])

    # ------------------------------------------------------------------ #
    # 1) Zone proportions stacked area
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        times, ol, tr, fr,
        labels=["overleft (weight=1)", "transition (weight 1→0)", "freed (weight=0)"],
        colors=["#e74c3c", "#f39c12", "#2ecc71"],
        alpha=0.85,
    )
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Fraction of chunk")
    ax.set_title("RTC Zone Proportions Over Time")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "rtc_zone_proportions.png")

    # ------------------------------------------------------------------ #
    # 2) Latency + inference_delay
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(calls, latencies, color="#3498db", linewidth=1.5, label="Inference latency (ms)")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(calls, delays, color="#e74c3c", linewidth=1.5, label="inference_delay (steps)")
    axes[1].plot(calls, exec_hz, color="#2ecc71", linewidth=1.5, linestyle="--", label="execution_horizon")
    axes[1].set_xlabel("Inference call index")
    axes[1].set_ylabel("Steps")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Inference Latency and Delay over Time")
    _save(fig, output_dir / "rtc_latency.png")

    # ------------------------------------------------------------------ #
    # 3) Action continuity
    # ------------------------------------------------------------------ #
    valid_l2 = ~np.isnan(l2s)
    valid_cos = ~np.isnan(coss)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    if valid_l2.any():
        axes[0].plot(calls[valid_l2], l2s[valid_l2], color="#9b59b6", linewidth=1.5)
    axes[0].set_ylabel("L2 distance (overlap zone)")
    axes[0].set_title("Action Continuity in Overlap Zone")
    axes[0].grid(True, alpha=0.3)

    if valid_cos.any():
        axes[1].plot(calls[valid_cos], coss[valid_cos], color="#1abc9c", linewidth=1.5)
        axes[1].set_ylim(-1, 1)
        axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel("Inference call index")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].grid(True, alpha=0.3)
    _save(fig, output_dir / "rtc_continuity.png")

    # ------------------------------------------------------------------ #
    # 4) Summary: pie + bar
    # ------------------------------------------------------------------ #
    avg_ol = float(np.mean(ol))
    avg_tr = float(np.mean(tr))
    avg_fr = float(np.mean(fr))

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax_pie = fig.add_subplot(gs[0, 0])
    labels = ["overleft", "transition", "freed"]
    sizes = [avg_ol, avg_tr, avg_fr]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    ax_pie.set_title("Average Zone Proportions")

    ax_bar = fig.add_subplot(gs[0, 1])
    step_counts = [
        np.mean([r["overleft_steps"] for r in records]),
        np.mean([r["transition_steps"] for r in records]),
        np.mean([r["freed_steps"] for r in records]),
    ]
    bars = ax_bar.bar(labels, step_counts, color=colors, alpha=0.85)
    for bar, count in zip(bars, step_counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{count:.1f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax_bar.set_ylabel("Avg step count")
    ax_bar.set_title(f"Avg Steps per Zone  (chunk_size={records[0]['chunk_size']})")
    ax_bar.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"RTC Zone Summary  ({len(records)} inference calls)\n"
        f"exec_horizon={records[0]['execution_horizon']}  "
        f"avg_delay={np.mean(delays):.1f} ± {np.std(delays):.1f} steps  "
        f"avg_latency={np.mean(latencies):.0f} ms",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, output_dir / "rtc_zone_summary.png")

    logger.info(f"[VIZ] Saved plots to {output_dir}")
    _print_summary(records, latencies, delays, ol, tr, fr, l2s, coss)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"[VIZ]   {path}")
    plt.close(fig)


def _print_summary(records, latencies, delays, ol, tr, fr, l2s, coss):
    logger.info("=" * 70)
    logger.info("RTC Zone Analysis Summary")
    logger.info(f"  Total inference calls  : {len(records)}")
    logger.info(f"  chunk_size             : {records[0]['chunk_size']}")
    logger.info(f"  execution_horizon      : {records[0]['execution_horizon']}")
    logger.info(f"  Latency   mean/std/max : "
                f"{np.mean(latencies):.1f} / {np.std(latencies):.1f} / {np.max(latencies):.1f} ms")
    logger.info(f"  Delay     mean/std/max : "
                f"{np.mean(delays):.1f} / {np.std(delays):.1f} / {np.max(delays):.0f} steps")
    logger.info(f"  overleft  mean fraction: {np.mean(ol)*100:.1f}%  "
                f"({np.mean([r['overleft_steps'] for r in records]):.1f} steps avg)")
    logger.info(f"  transition mean fraction: {np.mean(tr)*100:.1f}%  "
                f"({np.mean([r['transition_steps'] for r in records]):.1f} steps avg)")
    logger.info(f"  freed     mean fraction: {np.mean(fr)*100:.1f}%  "
                f"({np.mean([r['freed_steps'] for r in records]):.1f} steps avg)")
    valid_l2 = l2s[~np.isnan(l2s)]
    if len(valid_l2):
        logger.info(f"  L2 (overlap) mean/std  : {np.mean(valid_l2):.4f} / {np.std(valid_l2):.4f}")
    valid_cos = coss[~np.isnan(coss)]
    if len(valid_cos):
        logger.info(f"  Cosine (overlap) mean  : {np.mean(valid_cos):.4f}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Robot wrapper (same as eval_with_real_robot.py)
# ---------------------------------------------------------------------------

class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> dict:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RTCAnalysisConfig(HubMixin):
    """Config for real-robot RTC analysis script."""

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None

    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    duration: float = 60.0
    fps: float = 10.0
    interpolation_multiplier: int = 1
    device: str | None = None
    action_queue_size_to_get_new_actions: int = 30
    task: str = field(default="", metadata={"help": "Task description"})

    # Analysis output
    output_dir: str = field(
        default="rtc_analysis_output",
        metadata={"help": "Directory for metrics and plots"},
    )

    # Mock mode: skip real robot, used to test analysis/plotting
    mock_robot: bool = field(
        default=False,
        metadata={"help": "If true, simulate a robot (no hardware required)"},
    )

    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_disable_cudagraphs: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (--policy.path)")

        if self.robot is None and not self.mock_robot:
            raise ValueError("Robot configuration must be provided, or set --mock_robot=true")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# relative-action re-anchoring (identical to eval_with_real_robot.py)
# ---------------------------------------------------------------------------

def _reanchor_relative_rtc_prefix(
    prev_actions_absolute: Tensor,
    current_state: Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device,
) -> Tensor:
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


# ---------------------------------------------------------------------------
# get_actions thread (extended with metrics recording)
# ---------------------------------------------------------------------------

def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCAnalysisConfig,
    metrics_logger: RTCMetricsLogger,
):
    try:
        logger.info("[GET_ACTIONS] Starting")

        latency_tracker = LatencyTracker()
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        if cfg.mock_robot:
            # In mock mode we still need feature dicts; use empty ones
            observation_features_hw = {}
            dataset_features = {}
        else:
            observation_features_hw = {
                key: value
                for key, value in robot.observation_features().items()
                if key.endswith(".pos") or isinstance(value, tuple)
            }
            dataset_features = hw_to_dataset_features(observation_features_hw, "observation")

        policy_device = policy.config.device

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
        )

        relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if relative_step is not None:
            if relative_step.action_names is None:
                cfg_names = getattr(cfg.policy, "action_feature_names", None)
                if cfg_names:
                    relative_step.action_names = list(cfg_names)
                elif not cfg.mock_robot:
                    relative_step.action_names = [
                        k for k in robot.robot.action_features if k.endswith(".pos")
                    ]

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                if cfg.mock_robot:
                    # Mock observation: zero tensors matching policy's expectations
                    obs_with_policy_features = _make_mock_observation(policy, policy_device)
                else:
                    obs = robot.get_observation()
                    obs_processed = robot_observation_processor(obs)
                    obs_with_policy_features = build_dataset_frame(
                        dataset_features, obs_processed, prefix="observation"
                    )
                    for name in obs_with_policy_features:
                        obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                        if "image" in name:
                            obs_with_policy_features[name] = (
                                obs_with_policy_features[name].type(torch.float32) / 255
                            )
                            obs_with_policy_features[name] = (
                                obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                            )
                        obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                        obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]
                if not cfg.mock_robot:
                    obs_with_policy_features["robot_type"] = (
                        robot.robot.name if hasattr(robot.robot, "name") else ""
                    )

                preproceseded_obs = preprocessor(obs_with_policy_features)

                if (
                    prev_actions is not None
                    and relative_step is not None
                    and not cfg.mock_robot
                    and OBS_STATE in obs_with_policy_features
                ):
                    with action_queue.lock:
                        if action_queue.queue is not None:
                            prev_actions_abs = action_queue.queue[action_queue.last_index:].clone()
                        else:
                            prev_actions_abs = None
                    if prev_actions_abs is not None and prev_actions_abs.numel() > 0:
                        prev_actions = _reanchor_relative_rtc_prefix(
                            prev_actions_absolute=prev_actions_abs,
                            current_state=obs_with_policy_features[OBS_STATE],
                            relative_step=relative_step,
                            normalizer_step=normalizer_step,
                            policy_device=policy_device,
                        )

                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # Keep raw actions (pre-postprocess) for continuity analysis
                raw_actions = actions.squeeze(0).clone().detach().cpu()

                original_actions = actions.squeeze(0).clone()
                postprocessed_actions = postprocessor(actions).squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions too small — "
                        "should exceed inference_delay + execution_horizon."
                    )

                # --- Record metrics ---
                prev_for_metrics = prev_actions.detach().cpu() if prev_actions is not None else None
                rec = metrics_logger.record(
                    inference_latency=new_latency,
                    inference_delay=new_delay,
                    execution_horizon=cfg.rtc.execution_horizon,
                    prev_actions=prev_for_metrics,
                    new_actions=raw_actions,
                )
                logger.debug(
                    f"[METRICS] call={rec.call_index}  lat={new_latency*1000:.0f}ms  "
                    f"delay={new_delay}  overleft={rec.overleft_steps}  "
                    f"transition={rec.transition_steps}  freed={rec.freed_steps}"
                )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
            else:
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] Shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _make_mock_observation(policy, device) -> dict:
    """Build a zeroed observation dict for mock-mode testing."""
    obs = {}
    chunk_size = getattr(policy.config, "chunk_size", 50)
    action_dim = getattr(policy.config, "max_action_dim", 6)
    obs["observation.state"] = torch.zeros(1, action_dim, device=device)
    return obs


# ---------------------------------------------------------------------------
# actor_control thread (unchanged from eval_with_real_robot.py)
# ---------------------------------------------------------------------------

def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCAnalysisConfig,
):
    try:
        logger.info("[ACTOR] Starting")

        if cfg.mock_robot:
            # In mock mode just drain the queue
            while not shutdown_event.is_set():
                action_queue.get()
                time.sleep(1.0 / cfg.fps / cfg.interpolation_multiplier)
            return

        action_keys = [k for k in robot.action_features() if k.endswith(".pos")]
        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        action_interval = interpolator.get_control_interval(cfg.fps)
        action_count = 0

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            if interpolator.needs_new_action():
                new_action = action_queue.get()
                if new_action is not None:
                    interpolator.add(new_action.cpu())

            action = interpolator.get()
            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(action_keys)}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Shutting down. Total actions: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@parser.wrap()
def main(cfg: RTCAnalysisConfig):
    init_logging()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RTC Real-Robot Zone Analysis")
    logger.info(f"  output_dir        : {output_dir}")
    logger.info(f"  mock_robot        : {cfg.mock_robot}")
    logger.info(f"  rtc.enabled       : {cfg.rtc.enabled}")
    logger.info(f"  rtc.execution_horizon: {cfg.rtc.execution_horizon}")
    logger.info(f"  fps               : {cfg.fps}")
    logger.info(f"  duration          : {cfg.duration}s")
    logger.info("=" * 70)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # --- Load policy ---
    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type in ("pi05", "pi0"):
        config.compile_model = cfg.use_torch_compile

    if getattr(config, "use_peft", False):
        from peft import PeftConfig, PeftModel

        peft_path = cfg.policy.pretrained_path
        peft_cfg = PeftConfig.from_pretrained(peft_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_cfg.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_path, config=peft_cfg)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to(cfg.device)
    policy.eval()

    if cfg.use_torch_compile and hasattr(torch, "compile") and policy.type not in ("pi05", "pi0"):
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}
        policy.predict_action_chunk = torch.compile(policy.predict_action_chunk, **compile_kwargs)
        logger.info("torch.compile applied to predict_action_chunk")

    chunk_size = getattr(policy.config, "chunk_size", getattr(policy.config, "max_action_dim", 50))
    logger.info(f"Policy chunk_size: {chunk_size}")

    # --- Metrics logger ---
    metrics_logger = RTCMetricsLogger(
        output_path=output_dir / "rtc_metrics.jsonl",
        chunk_size=chunk_size,
    )

    # --- Robot ---
    robot = None
    robot_wrapper = None
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    if not cfg.mock_robot:
        logger.info(f"Connecting robot: {cfg.robot.type}")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        robot_wrapper = RobotWrapper(robot)
    else:
        logger.info("Mock robot mode — no hardware required")
        robot_wrapper = None

    action_queue = ActionQueue(cfg.rtc)

    get_actions_thread = Thread(
        target=get_actions,
        args=(
            policy, robot_wrapper, robot_observation_processor,
            action_queue, shutdown_event, cfg, metrics_logger,
        ),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()

    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()

    logger.info(f"Running for {cfg.duration}s ...")
    t0 = time.time()

    try:
        while not shutdown_event.is_set() and (time.time() - t0) < cfg.duration:
            elapsed = time.time() - t0
            n = len(metrics_logger.records)
            if n:
                last = metrics_logger.records[-1]
                logger.info(
                    f"[MAIN] {elapsed:.0f}s  calls={n}  "
                    f"last delay={last.inference_delay}  "
                    f"overleft={last.overleft_frac*100:.0f}%  "
                    f"transition={last.transition_frac*100:.0f}%  "
                    f"freed={last.freed_frac*100:.0f}%"
                )
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping")
    finally:
        shutdown_event.set()

    if get_actions_thread.is_alive():
        get_actions_thread.join(timeout=10)
    if actor_thread.is_alive():
        actor_thread.join(timeout=10)

    metrics_logger.close()

    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    # --- Visualize ---
    records = metrics_logger.records
    if records:
        record_dicts = [r.__dict__ for r in records]
        visualize_rtc_metrics(record_dicts, output_dir)
    else:
        logger.warning("No inference calls recorded — no plots generated")

    logger.info(f"Done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
