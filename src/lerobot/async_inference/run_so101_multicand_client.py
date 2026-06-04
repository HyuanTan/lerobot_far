"""Multi-candidate async-inference client for SO-101 real robot.

Mirrors the LIBERO ``run_libero_multicand_test.MultiCandSimClient`` design but
inherits from ``SmartRobotClient`` instead of ``SimSmartClient``.  The gRPC
server (``multi_candidate_server.py``) is reused unchanged — it is hardware-
agnostic and operates on normalised action tensors only.

Key differences vs LIBERO version
───────────────────────────────────
- Grasp phase source  : GripperStateMonitor._phase  (Feetech load+pos readings)
                        instead of SimSmartClient._grasp_phase (MuJoCo qpos)
- Phase enum mapping  : SO101 APPROACHING/OPENING → NORMAL
                        SO101 CLOSING             → CLOSING
                        SO101 HOLDING/DROPPING    → HOLDING
- Gripper convention  : DEGREES  0°=closed  30°=open
                        LIBERO   normalised −1=open  +1=close
                        → Layer-4/5 threshold checks use so101_gripper_open_deg
                          and so101_gripper_empty_deg config fields
- _last_executed_action: read from _last_feedback_state (bus Present_Position)
                          instead of sim obs robot_state
- Episode lifecycle   : SmartRobotClient's control_loop() handles TASK_DONE and
                        STOP internally; _on_task_done() hook is overridden here

Usage::

    # 1. Start multi-candidate server (separate terminal):
    python -m lerobot.async_inference.multi_candidate_server \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=your/model \\
        --host=localhost --port=8080 \\
        --n_candidates=4 --top_k=2

    # 2. Run SO-101 multi-candidate client:
    python -m lerobot.async_inference.run_so101_multicand_client \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyUSB0 \\
        --robot.cameras='{ front: {type: opencv, index_or_path: 0} }' \\
        --task="pick the red block and place it in the bin" \\
        --server_address=localhost:8080 \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=your/model \\
        --fps=10 --actions_per_chunk=16 \\
        --client_smooth_alpha=0.3 \\
        --record_trajectory=true \\
        --trajectory_dir=./mc_trajectories \\
        --results_dir=./mc_results \\
        --data_collect_dir=./mc_data
"""

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
from lerobot.utils.utils import init_logging
import torch

from .smart_robot_client import GraspPhase, GripperDecision, SmartRobotClient, SmartRobotClientConfig
from .sim_test.trajectory_recorder import TrajectoryRecorder as MCTrajectoryRecorder
from .helpers import ActionBundle, ActionChunk, TimedAction, get_logger


# ── Phase helpers ──────────────────────────────────────────────────────────────


def _map_phase(p: GraspPhase | None) -> str:
    """Map SO101 GraspPhase enum to the 3-state scoring layer name.

    SO101 has 5 phases; LIBERO scoring layers only need 3 semantic states:
      NORMAL  — arm is moving (not grasping): approaching, opening, going home
      CLOSING — gripper is actively closing onto the object
      HOLDING — object confirmed in hand (includes intentional-release ramp)
    """
    if p is None:
        return "NORMAL"
    if p == GraspPhase.CLOSING:
        return "CLOSING"
    if p in (GraspPhase.HOLDING, GraspPhase.DROPPING):
        # DROPPING = policy commanding open from a confirmed hold; object still in hand.
        return "HOLDING"
    # APPROACHING, OPENING
    return "NORMAL"


# ── Data collectors ────────────────────────────────────────────────────────────


class ClientOutcomeCollector:
    """Appends per-episode outcome records to <output_dir>/client_outcomes.jsonl.

    Schema: {episode_id, success, steps, duration_s, sm_retries, success_after_retry}
    Join with server's candidates.jsonl on episode_id for offline analysis.
    """

    def __init__(self, output_dir: str) -> None:
        self._path = Path(output_dir) / "client_outcomes.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logging.getLogger("ClientOutcomeCollector").info(
            f"[MC] Client outcomes → {self._path}"
        )

    def record(
        self,
        episode_id: int,
        success: bool,
        steps: int,
        duration_s: float,
        sm_retries: int = 0,
        success_after_retry: bool = False,
    ) -> None:
        rec = {
            "episode_id": episode_id,
            "success": bool(success),
            "steps": int(steps),
            "duration_s": float(duration_s),
            "sm_retries": int(sm_retries),
            "success_after_retry": bool(success_after_retry),
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


class ClientStepCollector:
    """Appends per-inference-step client-side records to client_steps.jsonl.

    Schema mirrors the LIBERO version for offline analysis compatibility::

        {
          "episode_id":           int,
          "timestep":             int,
          "client_selected_idx":  int,       # -1 = Phase-1 server-only
          "client_override":      bool,
          "n_candidates":         int,
          "server_scores":        [float],   # raw server composite scores
          "continuity_scores":    [float],   # per-candidate -||cand[0]-prev||₂
          "combined_scores":      [float],   # (1−α)*server + α*continuity
          "execution_continuity": float | null,
          "prev_action":          [float] | null,
          "selected_first_action":[float] | null,
          "episode_phase":        float,     # step / max_episode_steps ∈ [0,1]
          "delay_selected":       int | null,
          "robot_state":          [float] | null,
          "grasp_phase":          str,       # NORMAL | CLOSING | HOLDING
          "alpha_effective":      float,
          "uncertainty_mode":     bool       # True when spread slow mode active
        }
    """

    def __init__(self, output_dir: str) -> None:
        self._path = Path(output_dir) / "client_steps.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logging.getLogger("ClientStepCollector").info(
            f"[MC] Client steps → {self._path}"
        )

    def record(
        self,
        episode_id: int,
        timestep: int,
        client_selected_idx: int,
        client_override: bool,
        n_candidates: int,
        server_scores: list[float],
        continuity_scores: list[float],
        combined_scores: list[float],
        execution_continuity: float | None,
        prev_action: list[float] | None,
        selected_first_action: list[float] | None,
        episode_phase: float,
        delay_selected: int | None = None,
        robot_state: list[float] | None = None,
        grasp_phase: str = "NORMAL",
        alpha_effective: float = 0.0,
        uncertainty_mode: bool = False,
    ) -> None:
        rec = {
            "episode_id": episode_id,
            "timestep": timestep,
            "client_selected_idx": client_selected_idx,
            "client_override": client_override,
            "n_candidates": n_candidates,
            "server_scores": [round(s, 6) for s in server_scores],
            "continuity_scores": [round(s, 6) for s in continuity_scores],
            "combined_scores": [round(s, 6) for s in combined_scores],
            "execution_continuity": (
                round(execution_continuity, 6) if execution_continuity is not None else None
            ),
            "prev_action": (
                [round(v, 6) for v in prev_action] if prev_action is not None else None
            ),
            "selected_first_action": (
                [round(v, 6) for v in selected_first_action]
                if selected_first_action is not None
                else None
            ),
            "episode_phase": round(episode_phase, 4),
            "delay_selected": delay_selected,
            "robot_state": (
                [round(v, 6) for v in robot_state] if robot_state is not None else None
            ),
            "grasp_phase": grasp_phase,
            "alpha_effective": round(alpha_effective, 4),
            "uncertainty_mode": uncertainty_mode,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class MultiCandSO101Config(SmartRobotClientConfig):
    """SmartRobotClientConfig extended with multi-candidate client settings."""

    # ── Client-side re-ranking ─────────────────────────────────────────────────
    client_smooth_alpha: float = field(
        default=0.3,
        metadata={
            "help": (
                "Continuity weight in combined_score = (1-α)*server + α*continuity. "
                "Phase-adaptive: CLOSING uses α×0.3, HOLDING uses min(α×1.8, 0.7). "
                "0.0 = pure server ranking; 1.0 = pure continuity."
            )
        },
    )
    action_limit_min: float = field(
        default=-10.0,
        metadata={
            "help": (
                "Min allowed action value (degrees) for joint-limit gate. "
                "Candidates with any action below this are discarded. "
                "SO-101 joints typically operate in 0–270° range; "
                "-10 gives a small buffer for calibration offset."
            )
        },
    )
    action_limit_max: float = field(
        default=310.0,
        metadata={
            "help": (
                "Max allowed action value (degrees) for joint-limit gate. "
                "310 covers the full SO-101 joint range with a small buffer."
            )
        },
    )

    # ── SO-101 gripper convention ──────────────────────────────────────────────
    so101_gripper_open_deg: float = field(
        default=20.0,
        metadata={
            "help": (
                "Gripper position threshold (degrees) above which the gripper is "
                "classified as 'open'. Matches SmartRobotClientConfig.gripper_pos_open_threshold. "
                "RISK: Layer-4 CLOSING penalty and Layer-5 HOLDING slip-gate depend on this. "
                "Set to match gripper_pos_open_threshold for consistency."
            )
        },
    )
    so101_gripper_empty_deg: float = field(
        default=8.0,
        metadata={
            "help": (
                "Gripper position threshold (degrees) below which the gripper is "
                "classified as 'fully closed with no object'. Matches "
                "SmartRobotClientConfig.gripper_pos_empty_threshold. "
                "RISK: Layer-5 slip-gate uses this to detect grip-open candidates. "
                "Set to match gripper_pos_empty_threshold for consistency."
            )
        },
    )

    # ── O1: spread-based uncertainty slow mode ────────────────────────────────
    spread_uncertainty_threshold: float = field(
        default=0.15,
        metadata={
            "help": (
                "Rolling mean spread_l2 above which uncertainty-slow-mode activates "
                "(NORMAL phase only): alpha scaled up by spread_slow_alpha_scale. "
                "Default 0.15 is higher than LIBERO (0.08) to account for real-robot "
                "sensor noise and policy distribution shift. Set 0.0 to disable."
            )
        },
    )
    spread_slow_alpha_scale: float = field(
        default=1.5,
        metadata={
            "help": (
                "Alpha multiplier when rolling spread exceeds spread_uncertainty_threshold "
                "(NORMAL phase only). Capped at 0.9. "
                "Default 1.5 → base_alpha=0.3 becomes 0.45 under high uncertainty."
            )
        },
    )
    spread_slow_mode_window: int = field(
        default=5,
        metadata={"help": "Number of recent chunk spread_l2 values averaged for uncertainty signal."},
    )

    # ── O3: per-bundle score normalisation ────────────────────────────────────
    server_score_normalize: str = field(
        default="softmax",
        metadata={
            "help": (
                "Per-bundle normalisation before combining server and continuity scores. "
                "'none' = raw scores. 'softmax' = scale-invariant (recommended). "
                "'minmax' = rescale to [0,1] per bundle."
            )
        },
    )

    # ── Candidate stats logging ───────────────────────────────────────────────
    log_candidate_stats: bool = field(
        default=True,
        metadata={"help": "Log per-bundle candidate stats at INFO level."},
    )

    # ── Data collection ───────────────────────────────────────────────────────
    data_collect_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "If set, write per-episode outcomes to <data_collect_dir>/client_outcomes.jsonl "
                "and per-step records to client_steps.jsonl."
            )
        },
    )

    # ── Trajectory recording ──────────────────────────────────────────────────
    record_trajectory: bool = field(
        default=False,
        metadata={
            "help": (
                "Write per-episode MC trajectory JSON files to trajectory_dir. "
                "Each file contains chunk data (candidates, scores) and executed steps "
                "with joint feedback state for offline analysis."
            )
        },
    )
    trajectory_dir: str = field(
        default="./mc_trajectories",
        metadata={"help": "Output directory for per-episode trajectory JSON files."},
    )

    # ── Anti-repeat penalty after retry ──────────────────────────────────────
    retry_anti_repeat_steps: int = field(
        default=30,
        metadata={
            "help": (
                "Number of inference bundles after a retry event (RECOVERY / LIFT_RETRY / "
                "REWIND_RETRY) during which the anti-repeat penalty is active. "
                "At policy fps=10 this covers ~3 seconds of re-approach. 0 = disabled."
            )
        },
    )
    retry_anti_min_dist: float = field(
        default=15.0,
        metadata={
            "help": (
                "Arm-joint L2 distance threshold (degrees) below which a candidate is "
                "considered 'too similar' to the failed trajectory and penalised. "
                "Typical approach-phase joint delta is 30–60°; 15° is a conservative threshold "
                "that catches direct replays without rejecting genuinely different approaches."
            )
        },
    )
    retry_anti_penalty: float = field(
        default=0.35,
        metadata={
            "help": (
                "Score penalty subtracted from candidates that replicate the failed approach "
                "(arm-joint distance < retry_anti_min_dist). Applied to both the first-action "
                "check and the full-chunk check; set to 0.0 to disable a specific check."
            )
        },
    )

    # ── Results / summary ─────────────────────────────────────────────────────
    results_dir: str = field(
        default="./mc_results",
        metadata={"help": "Output directory for summary.txt and aggregate.json."},
    )

    # ── Output root ───────────────────────────────────────────────────────────
    save_root_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional root directory prepended to all output paths "
                "(data_collect_dir, trajectory_dir, results_dir, timing_output_dir, "
                "trajectory_output_dir, queue_size_monitor_path). "
                "Empty string = use paths as-is. "
                "Example: --save_root_path=./outputs/eval/so101/run1"
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.save_root_path:
            root = Path(self.save_root_path)
            if self.data_collect_dir is not None:
                self.data_collect_dir = str(root / self.data_collect_dir)
            self.trajectory_dir = str(root / self.trajectory_dir)
            self.results_dir = str(root / self.results_dir)
            if self.timing_output_dir is not None:
                self.timing_output_dir = str(root / self.timing_output_dir)
            self.trajectory_output_dir = str(root / self.trajectory_output_dir)
            self.queue_size_monitor_path = str(root / self.queue_size_monitor_path)

    max_episode_steps: int = field(
        default=500,
        metadata={
            "help": (
                "Expected maximum steps per pick-place cycle. Used to compute "
                "episode_phase = step / max_episode_steps in trajectory data. "
                "Does not impose a hard time limit (SmartRobotClient runs until TASK_DONE/STOP)."
            )
        },
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _chunk_actions(chunk: ActionChunk) -> list[list[float]]:
    """ActionChunk.timed_actions → T × D float list for JSON serialisation."""
    return [ta.get_action().float().tolist() for ta in chunk.timed_actions]


def _build_all_cands_data(bundle: ActionBundle) -> list[dict]:
    """Build all_candidates serialisation from bundle.all_candidates (if present)."""
    if not bundle.all_candidates:
        return []
    top_k_ids = {id(c) for c in bundle.candidates}
    result = []
    for rank, (cand, meta) in enumerate(
        zip(bundle.all_candidates, bundle.all_candidate_meta)
    ):
        result.append({
            "rank": rank,
            "in_top_k": id(cand) in top_k_ids,
            "delay": meta.inference_delay if meta else None,
            "noise_idx": meta.noise_idx if meta else rank,
            "jerk": round(meta.jerk, 6) if meta else 0.0,
            "vel_peak": round(meta.vel_peak, 6) if meta else 0.0,
            "server_score": round(meta.server_score, 6) if meta else 0.0,
            "actions": _chunk_actions(cand),
        })
    return result


# ── Client ─────────────────────────────────────────────────────────────────────


class MultiCandSO101Client(SmartRobotClient):
    """SmartRobotClient with multi-candidate chunk selection.

    Overrides _resolve_raw_chunk() to intercept ActionBundle from a
    MultiCandidatePolicyServer, then applies the same 5-layer scoring as
    MultiCandSimClient — adapted for SO-101's degree-based gripper convention.

    Thread model (identical to LIBERO version):
      receiver thread → _resolve_raw_chunk() / _client_select()
      control thread  → control_loop_action() / _update_mc_state_from_feedback()
      CPython GIL makes _last_executed_action / _current_robot_state reads safe
      without an explicit lock (heuristic use; stale by at most one step).
    """

    prefix = "mc_so101_client"

    def __init__(self, config: MultiCandSO101Config) -> None:
        super().__init__(config)
        self._mc_cfg = config

        # P1: continuity (control thread writes, receiver thread reads — GIL safe)
        self._last_executed_action: torch.Tensor | None = None
        # P2: robot state snapshot (degrees, same order as robot.action_features)
        self._current_robot_state: list[float] | None = None

        # O1: rolling spread for uncertainty detection (per-episode, receiver thread)
        self._recent_spreads: deque = deque(maxlen=max(1, config.spread_slow_mode_window))
        self._spread_slow_activations: int = 0  # cumulative across episodes

        # Anti-repeat: store context at retry trigger; cleared per episode.
        # Written by control thread (_on_retry_triggered), read by receiver thread
        # (_client_select) — GIL-safe (single object reference assignment).
        self._retry_anti_action: torch.Tensor | None = None   # arm state at failure (D,)
        self._retry_anti_chunk: torch.Tensor | None = None    # last selected chunk (T, D)
        self._retry_active_steps: int = 0  # countdown; anti-repeat fires when > 0
        self._last_selected_chunk: "ActionChunk | None" = None  # updated per bundle

        # Telemetry counters (never reset mid-run)
        self._client_override_count: int = 0
        self._bundle_recv_count: int = 0

        # Per-episode MC state
        self._mc_episode_id: int = 0          # own counter — incremented on TASK_DONE/STOP
        self._mc_traj_chunk_count: int = 0    # reset per episode
        self._mc_step_count: int = 0          # reset per episode
        self._mc_ep_start_time: float = time.time()

        # Data collectors
        self._step_collector: ClientStepCollector | None = (
            ClientStepCollector(config.data_collect_dir) if config.data_collect_dir else None
        )
        self._outcome_collector: ClientOutcomeCollector | None = (
            ClientOutcomeCollector(config.data_collect_dir) if config.data_collect_dir else None
        )

        # MC trajectory recorder (sim_test schema: chunks + executed steps)
        self._mc_traj_recorder: MCTrajectoryRecorder | None = (
            MCTrajectoryRecorder(config.trajectory_dir) if config.record_trajectory else None
        )
        if self._mc_traj_recorder is not None:
            self._mc_traj_recorder.reset(episode_id=0, task=config.task)

        # Warn on potential threshold mismatch (RISK: gripper convention)
        _open_mismatch = abs(config.so101_gripper_open_deg - config.gripper_pos_open_threshold) > 1.0
        _empty_mismatch = abs(config.so101_gripper_empty_deg - config.gripper_pos_empty_threshold) > 1.0
        if _open_mismatch or _empty_mismatch:
            self.logger.warning(
                "[mc_so101] RISK: Gripper threshold mismatch between MC scorer and SM:\n"
                f"  so101_gripper_open_deg={config.so101_gripper_open_deg}  "
                f"vs gripper_pos_open_threshold={config.gripper_pos_open_threshold}\n"
                f"  so101_gripper_empty_deg={config.so101_gripper_empty_deg}  "
                f"vs gripper_pos_empty_threshold={config.gripper_pos_empty_threshold}\n"
                "  Set so101_gripper_open_deg = gripper_pos_open_threshold and "
                "so101_gripper_empty_deg = gripper_pos_empty_threshold for consistency."
            )

        self.logger.info(
            f"[mc_so101] ENABLED | "
            f"alpha={config.client_smooth_alpha} | "
            f"normalize={config.server_score_normalize} | "
            f"spread_thr={config.spread_uncertainty_threshold} | "
            f"gripper_open_deg={config.so101_gripper_open_deg} | "
            f"gripper_empty_deg={config.so101_gripper_empty_deg} | "
            f"record_traj={config.record_trajectory} | "
            f"data_collect={config.data_collect_dir}"
        )

    # ── SM phase access (receiver thread) ─────────────────────────────────────

    def _get_current_phase_str(self) -> str:
        """Read SO101 GraspPhase from GripperStateMonitor and map to scoring name.

        GIL-safe: reads a single enum reference; worst case sees the phase from
        the previous SM update (< 1 control step old), which is acceptable for
        heuristic scoring.
        """
        if self._gripper_monitor is None:
            # SM disabled — fall through to NORMAL scoring
            return "NORMAL"
        raw_phase = self._gripper_monitor._phase
        mapped = _map_phase(raw_phase)
        self.logger.debug(
            f"[mc_so101] phase_read: SO101={raw_phase.name} → scoring={mapped}"
        )
        return mapped

    def _derive_scoring_phase(self, bundle: "ActionBundle") -> str:
        """Return scoring phase, correcting for 1-step lag at NORMAL→CLOSING boundary.

        Problem: _gripper_monitor._phase is set by the control thread scanning the
        OLD action queue.  When the server sends a new CLOSING chunk, the receiver
        thread calls _client_select() and reads NORMAL (stale) for up to 1 control
        step, causing two compounding errors:

          1. alpha_phase = base_alpha  (should be base_alpha × 0.3 in CLOSING)
             → continuity weight is 3× too high, over-penalising the correct
               close-gripper candidate.
          2. O2 EE-only continuity is skipped (phase != CLOSING)
             → full 6-DOF continuity penalty fires, including the gripper axis.
               A close-cmd ≈ 0° vs prev ≈ 28° produces a ~28° penalty on the
               best candidate, causing the client to prefer an open-gripper one.

        Fix: inspect bundle.selected's gripper trajectory directly.
          CLOSING signature: chunk starts with gripper open (>= so101_gripper_open_deg)
          and contains at least one closed command (< so101_gripper_empty_deg).
          This matches scan_intended_phase()'s CLOSING branch:
            has_open AND has_closed AND positions[0] >= pos_open.

          HOLDING is NOT inferred from chunk content — it requires load-feedback
          confirmation inside GripperStateMonitor.  We preserve the monitor's
          HOLDING/NORMAL reading for that case.

        Only upgrades NORMAL→CLOSING; never downgrades CLOSING or HOLDING.
        """
        monitor_phase = self._get_current_phase_str()

        # Already CLOSING or HOLDING — trust the monitor (no upgrade needed)
        if monitor_phase in ("CLOSING", "HOLDING"):
            return monitor_phase

        # SM disabled — no phase-adaptive scoring
        if self._gripper_monitor is None:
            return monitor_phase

        # GOINGHOME: _place_occurred means the object was confirmed dropped.
        # The policy is now returning to the home position — a trajectory never
        # seen during normal pick-place training (OOD).  Setting alpha = 0.1
        # lets the server score drive candidate selection instead of continuity,
        # which would otherwise anchor to the drop position and cause oscillation.
        # This is checked BEFORE the CLOSING chunk-content upgrade so that residual
        # close commands in the queue do not falsely re-classify go-home as CLOSING.
        if self._gripper_monitor._place_occurred:
            self.logger.debug(
                "[mc_so101] phase_lag_fix: _place_occurred → GOINGHOME"
            )
            return "GOINGHOME"

        # Inspect server's top-1 chunk for CLOSING signature
        ref_chunk = bundle.selected
        if not ref_chunk or not ref_chunk.timed_actions:
            return monitor_phase

        _open_th = self._mc_cfg.so101_gripper_open_deg
        _empty_th = self._mc_cfg.so101_gripper_empty_deg

        positions = [float(ta.get_action()[-1]) for ta in ref_chunk.timed_actions]
        has_open   = any(p >= _open_th  for p in positions)
        has_closed = any(p <  _empty_th for p in positions)

        if has_open and has_closed and positions[0] >= _open_th:
            self.logger.debug(
                f"[mc_so101] phase_lag_fix: NORMAL→CLOSING corrected "
                f"(bundle.selected gripper: start={positions[0]:.1f}° "
                f"end={positions[-1]:.1f}°; "
                f"open_th={_open_th:.1f}° empty_th={_empty_th:.1f}°)"
            )
            return "CLOSING"

        return monitor_phase

    # ── Feedback-state → MC state update (control thread) ─────────────────────

    def _update_mc_state_from_feedback(self) -> None:
        """Refresh _last_executed_action and _current_robot_state from bus feedback.

        Uses Present_Position readings (degrees) stored in _last_feedback_state
        by _read_feedback_state().  Called every control step from
        control_loop_action() so the receiver thread always has a fresh
        continuity reference.

        RISK: _last_feedback_state is None until the first bus read succeeds
        (typically after the first control_loop_action() call).  The receiver
        thread guards against None via `if _prev_action is not None` in
        _client_select(), so no candidate is penalised for continuity before
        the first action is sent.
        """
        if not self._last_feedback_state:
            self.logger.debug("[mc_so101] _update_mc_state: _last_feedback_state empty — skip")
            return
        try:
            features = list(self.robot.action_features.keys())
            state = [float(self._last_feedback_state.get(k, 0.0)) for k in features]
            self._last_executed_action = torch.tensor(state, dtype=torch.float32)
            self._current_robot_state = state
        except Exception as exc:
            self.logger.warning(f"[mc_so101] _update_mc_state failed: {exc}")

    # ── Action tracking hook ───────────────────────────────────────────────────

    def control_loop_action(self, verbose: bool = False) -> Any:
        """Delegate to SmartRobotClient and track executed state + MC step recording.

        Bus read strategy:
          - When _traj_recorder (robot_client style) is set: SM loop already calls
            _read_feedback_state() at step 1b after this returns; we read the
            previous iteration's value — acceptable for continuity scoring.
          - When _traj_recorder is None (no robot_client traj): we call
            _read_feedback_state() ourselves so _last_executed_action is fresh.
        """
        result = super().control_loop_action(verbose)
        if result is not None:
            # Ensure feedback is fresh when robot_client traj recorder is not set
            if self._traj_recorder is None:
                self._read_feedback_state()
            self._update_mc_state_from_feedback()
            self._mc_step_count += 1

            # MC trajectory: record executed step
            if self._mc_traj_recorder is not None and self._last_executed_action is not None:
                _phase_str = self._get_current_phase_str()
                _ep_phase = min(1.0, self._mc_step_count / max(1, self._mc_cfg.max_episode_steps))
                with self.latest_action_lock:
                    _ts = self.latest_action
                self._mc_traj_recorder.record_step({
                    "timestep": _ts,
                    "action": self._last_executed_action.tolist(),
                    "robot_state": self._current_robot_state,
                    "episode_phase": round(_ep_phase, 4),
                    "grasp_phase": _phase_str,
                })
        return result

    # ── Core hook: ActionBundle → ActionChunk (receiver thread) ───────────────

    def _resolve_raw_chunk(self, raw: Any) -> Any:
        """Intercept ActionBundle from MultiCandidatePolicyServer.

        Falls back to bundle.selected on any internal error so a scoring bug
        never halts the control loop.
        """
        if not isinstance(raw, ActionBundle):
            return raw

        self._bundle_recv_count += 1
        bundle: ActionBundle = raw

        try:
            result = self._resolve_action_bundle(bundle)
        except Exception as exc:
            import traceback as _tb
            self.logger.error(
                f"[mc_so101] _resolve_raw_chunk failed — returning bundle.selected as fallback: "
                f"{exc}\n{_tb.format_exc()}"
            )
            result = bundle.selected

        # Propagate server inference time to ActionChunk for base-class latency tracker
        if isinstance(result, ActionChunk) and bundle.inference_time_s > 0:
            result.inference_time_s = bundle.inference_time_s
        return result

    def _resolve_action_bundle(self, bundle: ActionBundle) -> ActionChunk:
        """Phase 1 (top_k=1) or Phase 2 (top_k>1) dispatch."""
        _ts = (
            bundle.selected.timed_actions[0].get_timestep()
            if bundle.selected and bundle.selected.timed_actions
            else -1
        )
        _ep_phase = (
            min(1.0, _ts / max(1, self._mc_cfg.max_episode_steps)) if _ts >= 0 else 0.0
        )
        chunk_idx = self._mc_traj_chunk_count
        self._mc_traj_chunk_count += 1
        candidates = bundle.candidates

        if not candidates:
            # Phase 1: server already picked best-1; skip re-ranking
            if self._mc_cfg.log_candidate_stats:
                self.logger.info(
                    f"[mc_so101] Bundle #{self._bundle_recv_count} "
                    f"(Phase 1 t={_ts}): server best-1 only"
                )
            if self._mc_traj_recorder is not None:
                _ph_str = self._get_current_phase_str()
                self._mc_traj_recorder.record_chunk({
                    "chunk_idx": chunk_idx,
                    "first_timestep": _ts,
                    "n_candidates": 1,
                    "selected_candidate_idx": -1,
                    "client_override": False,
                    "server_score": round(bundle.selected_score, 6),
                    "spread_l2": 0.0,
                    "grasp_phase": _ph_str,
                    "alpha_effective": self._mc_cfg.client_smooth_alpha,
                    "selected_actions": _chunk_actions(bundle.selected),
                    "candidates": [],
                    "all_candidates": _build_all_cands_data(bundle),
                })
            self._last_selected_chunk = bundle.selected
            return bundle.selected

        # O1: compute spread before _client_select so uncertainty mode can read it
        _spread = self._compute_spread_l2(candidates)
        self._recent_spreads.append(_spread)

        # Phase 2: client re-ranking
        selected, step_data = self._client_select(bundle, _ts, _ep_phase)

        if self._step_collector is not None:
            self._step_collector.record(**step_data)

        if self._mc_traj_recorder is not None:
            server_scores = step_data["server_scores"]
            continuity_scores = step_data["continuity_scores"]
            combined_scores = step_data["combined_scores"]
            best_rank = step_data["client_selected_idx"]
            client_override = step_data["client_override"]
            cands_data = []
            for rank, cand in enumerate(candidates):
                meta = (
                    bundle.candidate_meta[rank]
                    if bundle.candidate_meta and rank < len(bundle.candidate_meta)
                    else None
                )
                cands_data.append({
                    "delay": meta.inference_delay if meta else None,
                    "noise_idx": meta.noise_idx if meta else rank,
                    "jerk": round(meta.jerk, 6) if meta else 0.0,
                    "vel_peak": round(meta.vel_peak, 6) if meta else 0.0,
                    "server_score": round(server_scores[rank], 6) if rank < len(server_scores) else 0.0,
                    "continuity_score": round(continuity_scores[rank], 6) if rank < len(continuity_scores) else 0.0,
                    "combined_score": round(combined_scores[rank], 6) if rank < len(combined_scores) else 0.0,
                    "selected": rank == best_rank,
                    "actions": _chunk_actions(cand),
                })
            self._mc_traj_recorder.record_chunk({
                "chunk_idx": chunk_idx,
                "first_timestep": _ts,
                "n_candidates": len(candidates),
                "selected_candidate_idx": best_rank,
                "client_override": client_override,
                "server_score": round(bundle.selected_score, 6),
                "spread_l2": round(_spread, 6),
                "grasp_phase": step_data.get("grasp_phase", "NORMAL"),
                "alpha_effective": step_data.get("alpha_effective", self._mc_cfg.client_smooth_alpha),
                "selected_actions": _chunk_actions(selected),
                "candidates": cands_data,
                "all_candidates": _build_all_cands_data(bundle),
            })
        self._last_selected_chunk = selected
        return selected

    # ── Scoring core ───────────────────────────────────────────────────────────

    def _client_select(
        self, bundle: ActionBundle, timestep: int, episode_phase: float
    ) -> tuple[ActionChunk, dict]:
        """Re-rank bundle.candidates with phase-aware scoring.

        Identical 5-layer structure to MultiCandSimClient._client_select(), with
        two SO-101-specific adaptations:
          Layer 4 CLOSING penalty: gripper cmd < so101_gripper_open_deg means
            closing (not cmd < 0 as in normalised LIBERO space).
          Layer 5 HOLDING slip gate: any cmd > so101_gripper_empty_deg means
            the candidate opens the gripper mid-chunk (degrees, not normalised).

        RISK logging: all gripper threshold comparisons are logged at DEBUG so
        calibration issues can be spotted from logs without code changes.
        """
        cfg = self._mc_cfg
        candidates = bundle.candidates
        base_alpha = cfg.client_smooth_alpha
        lim_lo, lim_hi = cfg.action_limit_min, cfg.action_limit_max
        chunk_len = len(candidates[0].timed_actions) if candidates else 1
        _open_th = cfg.so101_gripper_open_deg    # RISK: degrees, 0=closed 30=open
        _empty_th = cfg.so101_gripper_empty_deg  # RISK: empty-close threshold

        # SM phase at selection time — uses chunk-content inspection to correct
        # 1-step lag at NORMAL→CLOSING boundary (see _derive_scoring_phase docstring)
        phase_str = self._derive_scoring_phase(bundle)
        sm_on = (self._gripper_monitor is not None)

        # ── Layer 1: phase-adaptive base alpha ─────────────────────────────────
        if sm_on and phase_str == "CLOSING":
            alpha_phase = base_alpha * 0.3   # trust server quality during close
        elif sm_on and phase_str == "HOLDING":
            alpha_phase = min(base_alpha * 1.8, 0.7)  # trust continuity while holding
        elif sm_on and phase_str == "GOINGHOME":
            alpha_phase = base_alpha * 0.1   # server drives go-home; continuity would anchor to drop position
        else:
            alpha_phase = base_alpha

        # ── Anti-repeat: snapshot active flag and decrement counter (once per bundle) ──
        # Decremented here so the countdown tracks inference steps, not per-candidate.
        _retry_active = self._retry_active_steps > 0
        if _retry_active:
            self._retry_active_steps -= 1

        # ── O1: uncertainty slow mode (NORMAL only) ───────────────────────────
        uncertainty_mode = False
        if (
            cfg.spread_uncertainty_threshold > 0.0
            and self._recent_spreads
            and phase_str == "NORMAL"
        ):
            mean_spread = sum(self._recent_spreads) / len(self._recent_spreads)
            if mean_spread > cfg.spread_uncertainty_threshold:
                alpha_phase = min(alpha_phase * cfg.spread_slow_alpha_scale, 0.9)
                uncertainty_mode = True
                self._spread_slow_activations += 1
                self.logger.info(
                    f"[mc_so101] t={timestep} UNCERTAINTY_SLOW: "
                    f"mean_spread={mean_spread:.4f} > thr={cfg.spread_uncertainty_threshold:.3f}"
                    f" → alpha_phase={alpha_phase:.3f}"
                )

        # Snapshot P1/P2 inputs (GIL-safe single reads)
        _prev_action: torch.Tensor | None = self._last_executed_action
        _robot_state: list[float] | None = self._current_robot_state

        if _prev_action is None:
            self.logger.warning(
                f"[mc_so101] t={timestep} RISK: _last_executed_action is None "
                "(no control step yet or feedback unavailable) — continuity scores will be 0"
            )

        # ── Pass 1: raw server + continuity scores ─────────────────────────────
        server_scores_raw: list[float] = []
        cont_scores_raw: list[float] = []
        alpha_effs: list[float] = []

        for rank, cand in enumerate(candidates):
            meta = bundle.candidate_meta[rank] if bundle.candidate_meta else None
            srv = meta.server_score if meta is not None else float(-rank)
            server_scores_raw.append(srv)

            # Layer 2: latency-corrected reference
            delay_steps = meta.inference_delay if meta else 0
            ref_idx = min(delay_steps, len(cand.timed_actions) - 1)

            # Layer 3: high-latency alpha attenuation
            latency_frac = delay_steps / max(chunk_len, 1)
            alpha_eff = alpha_phase * max(0.2, 1.0 - 0.5 * latency_frac)
            alpha_effs.append(alpha_eff)

            # Continuity score
            cont = 0.0
            if alpha_eff > 0.0 and _prev_action is not None and cand.timed_actions:
                ref_a = cand.timed_actions[ref_idx].get_action().float().cpu()
                if sm_on and phase_str == "CLOSING" and ref_a.shape[0] > 1:
                    # O2: EE-only continuity — exclude gripper (last dim).
                    # Without this, closing cmd (e.g. 0°) vs prev open (e.g. 28°)
                    # causes a ~28° continuity penalty, wrongly discouraging the
                    # best close-gripper candidate.
                    cont = -float((ref_a[:-1] - _prev_action.float().cpu()[:-1]).norm())
                    self.logger.debug(
                        f"[mc_so101] t={timestep} rank={rank} O2_EE_cont: "
                        f"ref_gripper={ref_a[-1]:.1f}° prev_gripper={float(_prev_action[-1]):.1f}° "
                        f"EE_cont={cont:.4f}"
                    )
                else:
                    cont = -float((ref_a - _prev_action.float().cpu()).norm())
            cont_scores_raw.append(cont)

        # ── O3: per-bundle normalisation ───────────────────────────────────────
        norm_mode = cfg.server_score_normalize

        def _softmax(vals: list[float]) -> list[float]:
            m = max(vals)
            exps = [math.exp(v - m) for v in vals]
            s = sum(exps)
            return [e / s for e in exps]

        def _minmax(vals: list[float]) -> list[float]:
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            if rng < 1e-9:
                return [0.5] * len(vals)
            return [(v - lo) / rng for v in vals]

        if len(candidates) > 1 and norm_mode != "none":
            _norm = _softmax if norm_mode == "softmax" else _minmax
            server_scores_norm = _norm(server_scores_raw)
            cont_scores_norm   = _norm(cont_scores_raw)
        else:
            server_scores_norm = server_scores_raw
            cont_scores_norm   = cont_scores_raw

        # ── Pass 2: combined scores + penalties + limit gates ──────────────────
        combined_scores: list[float] = []
        passed_limit: list[bool] = []

        for rank, cand in enumerate(candidates):
            meta = bundle.candidate_meta[rank] if bundle.candidate_meta else None
            alpha_eff = alpha_effs[rank]
            srv  = server_scores_norm[rank]
            cont = cont_scores_norm[rank]
            combined = (1.0 - alpha_eff) * srv + alpha_eff * cont

            # ── Layer 4: CLOSING EE-stability penalties ─────────────────────
            # RISK: gripper direction uses degrees (0°=closed, 30°=open)
            # Unlike LIBERO (normalised, action<0 = open), SO-101 action > _open_th = open
            if sm_on and phase_str == "CLOSING" and cand.timed_actions:
                gripper_cmds = [float(ta.get_action()[-1]) for ta in cand.timed_actions]
                # Candidate opens gripper during a close phase — penalise
                has_open_cmd = any(g > _open_th for g in gripper_cmds)
                if has_open_cmd:
                    combined -= 0.5
                    self.logger.debug(
                        f"[mc_so101] t={timestep} rank={rank} CLOSING_PENALTY: "
                        f"gripper_cmds max={max(gripper_cmds):.1f}° > open_th={_open_th}° "
                        f"→ −0.5"
                    )
                if meta and meta.vel_peak > 0:
                    combined -= 0.05 * meta.vel_peak

            combined_scores.append(combined)

            # Joint-limit gate
            ok = self._passes_limit_check(cand, lim_lo, lim_hi)
            if not ok:
                self.logger.debug(
                    f"[mc_so101] t={timestep} rank={rank} out-of-range "
                    f"[{lim_lo:.1f},{lim_hi:.1f}]"
                )

            # ── Layer 5: HOLDING slip gate ──────────────────────────────────
            # RISK: gripper direction uses degrees; any cmd > _empty_th = gripper opening
            # Without this gate a candidate that opens during HOLDING would slip the object.
            if sm_on and phase_str == "HOLDING" and cand.timed_actions:
                gripper_cmds = [float(ta.get_action()[-1]) for ta in cand.timed_actions]
                has_opening_cmd = any(g > _empty_th for g in gripper_cmds)
                if has_opening_cmd:
                    ok = False
                    self.logger.debug(
                        f"[mc_so101] t={timestep} rank={rank} HOLDING_SLIP_GATE: "
                        f"gripper_cmds max={max(gripper_cmds):.1f}° > empty_th={_empty_th}° "
                        f"→ discarded"
                    )

            # ── Layer 5.5: Anti-repeat penalty after retry ──────────────────
            # After empty-grasp/slip triggers RECOVERY / LIFT_RETRY / REWIND_RETRY,
            # penalise candidates that closely replicate the failed approach trajectory.
            # Two independent checks (arm-joint first-action + full-chunk shape);
            # each applies cfg.retry_anti_penalty if similarity exceeds threshold.
            if _retry_active and cfg.retry_anti_penalty > 0.0 and cand.timed_actions:
                _delay = (
                    bundle.candidate_meta[rank].inference_delay
                    if bundle.candidate_meta else 0
                )
                _ref_idx = min(_delay, len(cand.timed_actions) - 1)
                _ref_a = cand.timed_actions[_ref_idx].get_action().float().cpu()

                # Check A: first-action (latency-corrected) vs failed arm state
                if self._retry_anti_action is not None:
                    _anti_arm = self._retry_anti_action.float().cpu()
                    _anti_dist = float((_ref_a[:-1] - _anti_arm[:-1]).norm())
                    if _anti_dist < cfg.retry_anti_min_dist:
                        combined -= cfg.retry_anti_penalty
                        combined_scores[rank] = combined  # keep list in sync
                        self.logger.debug(
                            f"[mc_so101] t={timestep} rank={rank} ANTI_REPEAT_ACTION: "
                            f"arm_dist={_anti_dist:.1f}° < thr={cfg.retry_anti_min_dist:.1f}° "
                            f"→ -{cfg.retry_anti_penalty:.2f}"
                        )

                # Check B: full-chunk trajectory shape vs failed chunk
                if self._retry_anti_chunk is not None:
                    try:
                        _cand_traj = torch.stack(
                            [ta.get_action().float().cpu() for ta in cand.timed_actions]
                        )  # [T_cand, D]
                        _anti_traj = self._retry_anti_chunk  # [T_anti, D]
                        _min_t = min(_cand_traj.shape[0], _anti_traj.shape[0])
                        # Arm-only comparison (exclude last gripper dim)
                        _chunk_dist = float(
                            (_cand_traj[:_min_t, :-1] - _anti_traj[:_min_t, :-1])
                            .norm(dim=-1).mean().item()
                        )
                        if _chunk_dist < cfg.retry_anti_min_dist:
                            combined -= cfg.retry_anti_penalty
                            combined_scores[rank] = combined
                            self.logger.debug(
                                f"[mc_so101] t={timestep} rank={rank} ANTI_REPEAT_CHUNK: "
                                f"chunk_dist={_chunk_dist:.1f}° < thr={cfg.retry_anti_min_dist:.1f}° "
                                f"→ -{cfg.retry_anti_penalty:.2f}"
                            )
                    except Exception:
                        pass

            passed_limit.append(ok)

        # Select best surviving candidate
        best_rank = -1
        best_combined = float("-inf")
        for rank in range(len(candidates)):
            if passed_limit[rank] and combined_scores[rank] > best_combined:
                best_combined = combined_scores[rank]
                best_rank = rank

        if best_rank < 0:
            self.logger.warning(
                f"[mc_so101] t={timestep}: all {len(candidates)} candidates failed checks — "
                "falling back to bundle.selected"
            )
            best_rank = 0
            selected = bundle.selected
        else:
            selected = candidates[best_rank]

        client_override = best_rank != 0
        if client_override:
            self._client_override_count += 1

        alpha_eff_best = (
            alpha_effs[best_rank] if alpha_effs and best_rank < len(alpha_effs) else alpha_phase
        )

        if cfg.log_candidate_stats:
            srv_str  = "  ".join(f"c{i}:{s:.3f}" for i, s in enumerate(server_scores_raw))
            cont_str = "  ".join(f"c{i}:{s:.3f}" for i, s in enumerate(cont_scores_raw))
            slow_tag = " [SLOW]" if uncertainty_mode else ""
            self.logger.info(
                f"[mc_so101] t={timestep} Bundle #{self._bundle_recv_count} | "
                f"n={len(candidates)} chosen={best_rank} "
                f"phase={phase_str} α={alpha_eff_best:.3f}{slow_tag} | "
                f"srv=[{srv_str}] cont=[{cont_str}] | "
                f"overrides={self._client_override_count}"
            )

        # P1: execution continuity of chosen candidate
        exec_cont: float | None = None
        selected_first: list[float] | None = None
        if selected.timed_actions:
            first_a = selected.timed_actions[0].get_action().float()
            selected_first = first_a.tolist()
            if _prev_action is not None:
                exec_cont = float((first_a - _prev_action.float()).norm())

        delay_selected: int | None = None
        if bundle.candidate_meta and best_rank < len(bundle.candidate_meta):
            delay_selected = bundle.candidate_meta[best_rank].inference_delay

        step_record = dict(
            episode_id=self._mc_episode_id,
            timestep=timestep,
            client_selected_idx=best_rank,
            client_override=client_override,
            n_candidates=len(candidates),
            server_scores=server_scores_raw,
            continuity_scores=cont_scores_raw,
            combined_scores=combined_scores,
            execution_continuity=exec_cont,
            prev_action=_prev_action.tolist() if _prev_action is not None else None,
            selected_first_action=selected_first,
            episode_phase=episode_phase,
            delay_selected=delay_selected,
            robot_state=_robot_state,
            grasp_phase=phase_str,
            alpha_effective=round(alpha_eff_best, 4),
            uncertainty_mode=uncertainty_mode,
        )
        return selected, step_record

    # ── Episode lifecycle hooks ────────────────────────────────────────────────

    def _reset_loop_state(self) -> None:
        """Clear per-episode MC buffers.  Called by SmartRobotClient.control_loop()."""
        super()._reset_loop_state()
        self._last_executed_action = None
        self._current_robot_state = None
        self._mc_traj_chunk_count = 0
        self._mc_step_count = 0
        self._mc_ep_start_time = time.time()
        self._recent_spreads.clear()  # O1: spread window is per-episode
        # Anti-repeat: clear at episode boundary so a failed previous episode
        # never pollutes the fresh re-start.
        self._retry_anti_action = None
        self._retry_anti_chunk = None
        self._retry_active_steps = 0
        self._last_selected_chunk = None

    def _on_retry_triggered(self, decision: GripperDecision) -> None:
        """Store the arm state and last selected chunk at the moment a retry fires.

        Called by SmartRobotClient BEFORE the recovery/lift/rewind trajectory
        begins, so _last_feedback_state and _last_selected_chunk still reflect
        the failure-site state (most recent feedback + last dispatched chunk).

        The stored tensors are consumed by Layer 5.5 in _client_select() to
        penalise candidates that replicate the failed approach.
        """
        if self._mc_cfg.retry_anti_repeat_steps <= 0:
            return
        # Store arm state at failure (feedback position; gripper included for context)
        if self._last_executed_action is not None:
            self._retry_anti_action = self._last_executed_action.clone()
        # Store full trajectory of last selected chunk (arm dims only stored separately)
        if self._last_selected_chunk is not None and self._last_selected_chunk.timed_actions:
            try:
                self._retry_anti_chunk = torch.stack(
                    [ta.get_action().float().cpu()
                     for ta in self._last_selected_chunk.timed_actions]
                )  # [T, D]
            except Exception as exc:
                self.logger.debug(f"[mc_so101] _on_retry_triggered: chunk store failed: {exc}")
                self._retry_anti_chunk = None
        self._retry_active_steps = self._mc_cfg.retry_anti_repeat_steps
        self.logger.info(
            f"[mc_so101] ANTI_REPEAT armed | decision={decision.name} "
            f"active_steps={self._retry_active_steps} "
            f"anti_action={'set' if self._retry_anti_action is not None else 'None'} "
            f"anti_chunk={'set T=' + str(self._retry_anti_chunk.shape[0]) if self._retry_anti_chunk is not None else 'None'}"
        )

    def _on_task_done(self) -> None:
        """Called by SmartRobotClient.control_loop() on confirmed TASK_DONE.

        Saves MC trajectory, records episode outcome, and resets per-episode
        state for the next pick-place cycle.
        """
        super()._on_task_done()   # flushes robot_client _traj_recorder if set
        duration_s = time.time() - self._mc_ep_start_time
        steps = self._mc_step_count
        retries = getattr(self, "_ep_retry_count", 0)
        success_after_retry = retries > 0  # any retry + task_done → rescued

        self.logger.info(
            f"[mc_so101] TASK_DONE ep={self._mc_episode_id} | "
            f"steps={steps} duration={duration_s:.1f}s retries={retries} "
            f"overrides={self._client_override_count}"
        )

        if self._mc_traj_recorder is not None:
            path = self._mc_traj_recorder.save(success=True, total_steps=steps)
            self.logger.info(f"[mc_so101] MC trajectory saved → {path}")

        if self._outcome_collector is not None:
            self._outcome_collector.record(
                episode_id=self._mc_episode_id,
                success=True,
                steps=steps,
                duration_s=duration_s,
                sm_retries=retries,
                success_after_retry=success_after_retry,
            )

        # Advance episode for the next cycle
        self._mc_episode_id += 1
        self._mc_traj_chunk_count = 0
        self._mc_step_count = 0
        self._mc_ep_start_time = time.time()
        self._recent_spreads.clear()

        if self._mc_traj_recorder is not None:
            self._mc_traj_recorder.reset(episode_id=self._mc_episode_id, task=self._current_task)

    def stop(self) -> None:
        """Override to save the current (possibly failed) episode before shutting down."""
        duration_s = time.time() - self._mc_ep_start_time
        steps = self._mc_step_count

        # Only save if something was recorded in this episode
        if steps > 0 or self._mc_traj_chunk_count > 0:
            self.logger.info(
                f"[mc_so101] STOP — saving ep={self._mc_episode_id} "
                f"steps={steps} duration={duration_s:.1f}s as FAILED"
            )
            if self._mc_traj_recorder is not None:
                path = self._mc_traj_recorder.save(success=False, total_steps=steps)
                self.logger.info(f"[mc_so101] MC trajectory (failed) saved → {path}")
            if self._outcome_collector is not None:
                retries = getattr(self, "_ep_retry_count", 0)
                self._outcome_collector.record(
                    episode_id=self._mc_episode_id,
                    success=False,
                    steps=steps,
                    duration_s=duration_s,
                    sm_retries=retries,
                    success_after_retry=False,
                )

        super().stop()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _passes_limit_check(chunk: ActionChunk, lo: float, hi: float) -> bool:
        """True if every action in chunk is within [lo, hi] for all joints."""
        for ta in chunk.timed_actions:
            a = ta.get_action()
            if float(a.min()) < lo or float(a.max()) > hi:
                return False
        return True

    @staticmethod
    def _compute_spread_l2(candidates: list[ActionChunk]) -> float:
        """Mean pairwise L2 distance between candidate action trajectories."""
        N = len(candidates)
        if N <= 1:
            return 0.0
        try:
            mats = [
                torch.stack([ta.get_action().float() for ta in c.timed_actions])
                for c in candidates
            ]
            total, count = 0.0, 0
            for i in range(N):
                for j in range(i + 1, N):
                    total += float((mats[i] - mats[j]).norm(dim=-1).mean().item())
                    count += 1
            return total / count if count > 0 else 0.0
        except Exception:
            return 0.0


# ── Summary helpers ────────────────────────────────────────────────────────────


def _save_multicand_summary(
    client: MultiCandSO101Client,
    cfg: MultiCandSO101Config,
    total_t: float,
) -> None:
    """Write aggregate MC + SM stats to <results_dir>/summary.txt."""
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_stats = client._compute_episode_stats()
    nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731

    lines: list[str] = [
        "=" * 72,
        "  SO-101 Multi-Candidate Async-Inference — Summary",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "── Config ──────────────────────────────────────────────────────────",
        f"  task                   : {cfg.task}",
        f"  model                  : {cfg.pretrained_name_or_path}",
        f"  fps                    : {cfg.fps}",
        f"  actions_per_chunk      : {cfg.actions_per_chunk}",
        f"  client_smooth_alpha    : {cfg.client_smooth_alpha}",
        f"  server_score_normalize : {cfg.server_score_normalize}",
        f"  spread_uncertainty_thr : {cfg.spread_uncertainty_threshold}",
        f"  spread_slow_alpha_scale: {cfg.spread_slow_alpha_scale}",
        f"  so101_gripper_open_deg : {cfg.so101_gripper_open_deg}",
        f"  so101_gripper_empty_deg: {cfg.so101_gripper_empty_deg}",
        f"  enable_gripper_sm      : {cfg.enable_gripper_sm}",
    ]
    if cfg.enable_gripper_sm:
        lines += [
            f"  gripper_load_grasp_th  : {cfg.gripper_load_grasp_threshold}",
            f"  gripper_pos_gap_th     : {cfg.gripper_pos_gap_threshold}",
            f"  max_reinfer_retries    : {cfg.max_reinfer_retries}",
            f"  max_empty_grasp_retries: {cfg.max_empty_grasp_retries}",
            f"  lift_retry             : {cfg.empty_grasp_lift_retry_enabled}",
            f"  rewind_retry           : {cfg.empty_grasp_rewind_enabled}",
        ]

    lines += [
        "",
        "── MC Telemetry ─────────────────────────────────────────────────────",
        f"  mc_bundles_recv        : {client._bundle_recv_count}",
        f"  mc_client_overrides    : {client._client_override_count}",
        f"  mc_spread_slow_acts    : {client._spread_slow_activations}",
        f"  total_episodes (MC)    : {client._mc_episode_id}",
        f"  total_time             : {total_t:.1f}s",
    ]

    if ep_stats:
        lines += [
            "",
            "── SM Episode Stats ─────────────────────────────────────────────────",
            f"  total_episodes (SM)    : {ep_stats.get('total_episodes', 0)}",
            f"  overall_sr             : {nan_fmt(ep_stats.get('overall_sr', float('nan')))}",
            f"  total_retries          : {ep_stats.get('total_retries', 0)}",
            f"  eps_with_retry         : {ep_stats.get('eps_with_retry', 0)} / {ep_stats.get('total_episodes', 0)}",
            f"  sr_with_retry          : {nan_fmt(ep_stats.get('sr_with_retry', float('nan')))}"
            f"  ← final SR of retried episodes",
            f"  sr_no_retry            : {nan_fmt(ep_stats.get('sr_no_retry', float('nan')))}"
            f"  ← final SR of clean episodes",
            f"  success_after_retry    : {ep_stats.get('success_after_retry', 0)}"
            f"  ← episodes saved by SM retry",
            f"  rescue_rate            : {nan_fmt(ep_stats.get('rescue_rate', float('nan')))}"
            f"  ← success_after_retry / eps_with_retry",
            f"  sr_lift (SM→no-SM)     : +{ep_stats.get('sr_lift', 0):.1%}"
            f"  ← overall SR improvement from SM",
        ]

    lines += ["", "=" * 72, ""]
    txt = "\n".join(lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(txt, encoding="utf-8")
    logging.getLogger("mc_so101").info(f"[mc_so101] Summary saved → {summary_path}")

    # Also save aggregate JSON for programmatic access
    aggregate = {
        "generated": datetime.now().isoformat(),
        # Top-level aliases for analyze_sweep compatibility (reads overall_success_rate + total_episodes)
        "overall_success_rate": ep_stats.get("overall_sr", float("nan")) if ep_stats else float("nan"),
        "total_episodes": ep_stats.get("total_episodes", 0) if ep_stats else 0,
        "config": {
            "task": cfg.task,
            "pretrained_name_or_path": cfg.pretrained_name_or_path,
            "fps": cfg.fps,
            "actions_per_chunk": cfg.actions_per_chunk,
            "client_smooth_alpha": cfg.client_smooth_alpha,
            "server_score_normalize": cfg.server_score_normalize,
            "spread_uncertainty_threshold": cfg.spread_uncertainty_threshold,
            "so101_gripper_open_deg": cfg.so101_gripper_open_deg,
            "so101_gripper_empty_deg": cfg.so101_gripper_empty_deg,
        },
        "mc_telemetry": {
            "bundles_recv": client._bundle_recv_count,
            "client_overrides": client._client_override_count,
            "spread_slow_activations": client._spread_slow_activations,
            "total_mc_episodes": client._mc_episode_id,
        },
        "sm_stats": ep_stats if ep_stats else {},
        "total_time_s": round(total_t, 2),
    }
    agg_path = out_dir / "aggregate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    logging.getLogger("mc_so101").info(f"[mc_so101] Aggregate JSON saved → {agg_path}")


# ── Entry point ────────────────────────────────────────────────────────────────


@draccus.wrap()
def run_so101_multicand_client(cfg: MultiCandSO101Config) -> None:
    init_logging(console_level=cfg.log_level.upper())
    logging.info("[mc_so101] Config:\n" + pformat(asdict(cfg)))

    client = MultiCandSO101Client(cfg)

    if not client.start():
        client.logger.error("[mc_so101] Failed to connect to policy server. Aborting.")
        return

    if cfg.timing_output_dir:
        client.enable_timing(cfg.timing_output_dir)

    from .helpers import QueueSizeMonitor
    queue_monitor = None
    if cfg.queue_size_monitor_interval > 0:
        queue_monitor = QueueSizeMonitor(
            data=client.action_queue_size,
            interval=cfg.queue_size_monitor_interval,
            path=cfg.queue_size_monitor_path,
        )
        queue_monitor.start()
        client.logger.info(
            f"[mc_so101] Queue monitor started — "
            f"PNG every {cfg.queue_size_monitor_interval}s → {cfg.queue_size_monitor_path}"
        )

    receiver = threading.Thread(
        target=client.receive_actions, daemon=True, name="mc-action-receiver"
    )
    receiver.start()
    client.logger.info("[mc_so101] Action receiver thread started")

    t_start = time.perf_counter()
    try:
        client.control_loop(task=cfg.task)
    finally:
        total_t = time.perf_counter() - t_start

        if queue_monitor is not None:
            try:
                queue_monitor.stop()
            except Exception as exc:
                client.logger.warning(f"queue_monitor.stop() raised: {exc}")

        try:
            client.stop()
        except Exception as exc:
            client.logger.warning(f"client.stop() raised: {exc}")

        try:
            receiver.join(timeout=5.0)
        except Exception:
            pass

        try:
            client.save_timing()
        except Exception as exc:
            client.logger.warning(f"save_timing() raised: {exc}")

        # ── Final summary ─────────────────────────────────────────────────────
        ep_stats = client._compute_episode_stats()
        nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731
        sm_block = ""
        if ep_stats:
            sm_block = (
                f"\n  ── SM Episode Stats ──"
                f"\n  total_episodes (SM)    : {ep_stats.get('total_episodes', 0)}"
                f"\n  overall_sr             : {nan_fmt(ep_stats.get('overall_sr', float('nan')))}"
                f"\n  total_retries          : {ep_stats.get('total_retries', 0)}"
                f"\n  eps_with_retry         : {ep_stats.get('eps_with_retry', 0)}/{ep_stats.get('total_episodes', 0)}"
                f"\n  sr_with_retry          : {nan_fmt(ep_stats.get('sr_with_retry', float('nan')))}"
                f"  (harder eps)"
                f"\n  sr_no_retry            : {nan_fmt(ep_stats.get('sr_no_retry', float('nan')))}"
                f"  (clean eps)"
                f"\n  success_after_retry    : {ep_stats.get('success_after_retry', 0)}"
                f"\n  rescue_rate            : {nan_fmt(ep_stats.get('rescue_rate', float('nan')))}"
                f"\n  sr_lift (SM→no-SM)     : +{ep_stats.get('sr_lift', 0):.1%}"
            )

        client.logger.info(
            f"[mc_so101] ═══ Final summary ═══\n"
            f"  task                   : {cfg.task}\n"
            f"  total_time             : {total_t:.2f}s\n"
            f"  mc_bundles_recv        : {client._bundle_recv_count}\n"
            f"  mc_client_overrides    : {client._client_override_count}\n"
            f"  mc_spread_slow_acts    : {client._spread_slow_activations}\n"
            f"  total_mc_episodes      : {client._mc_episode_id}"
            f"{sm_block}"
        )

        try:
            _save_multicand_summary(client, cfg, total_t)
        except Exception as exc:
            client.logger.warning(f"_save_multicand_summary() raised: {exc}")


if __name__ == "__main__":
    run_so101_multicand_client()
