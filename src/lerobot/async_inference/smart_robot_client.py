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
SmartRobotClient — RobotClient + gripper-feedback state machine.

Subclasses RobotClient without modifying any base-class code.
Set --enable_gripper_sm=false to fall back to vanilla RobotClient behavior via super().

Design: two completely independent read paths
──────────────────────────────────────────────
  Inference path  │ robot.get_observation()          → server → policy → action
                  │   reads: Present_Position (all motors) + camera frames
                  │   unchanged from plain RobotClient
  ────────────────┼──────────────────────────────────────────────────────────
  SM feedback     │ robot.bus.sync_read(register, motor_names)
  path            │   reads: Present_Load / Present_Current (gripper only)
                  │   called every control step; no camera, no position remapping
                  │   does NOT require record_motor_state in robot config
                  │   does NOT affect observation_features / action_features / lerobot_features

Why not use record_motor_state + get_observation()?
  • action_features (= _motors_ft) would include gripper.load/.current.
    _action_tensor_to_action_dict() maps by index → dimension mismatch crash.
  • get_observation() also captures camera frames (10–33ms per frame).
    Calling it every control step just for load/current is ~10× too expensive.
  • lerobot_features sent to server would include load/current dims → policy
    normalizer dimension mismatch.

State machine pipeline:
    policy outputs action_chunk
        ↓
    scan future action.gripper.pos  (action queue lookahead)
        ↓
    infer intended phase: CLOSING / HOLDING / OPENING / APPROACHING
        ↓
    bus.sync_read(Present_Load/Current, ["gripper"])   ← cheap, ~1–2 ms
        ↓
    classify: empty_grasp / grasp_success / slip / normal
        ↓
    decide: CONTINUE / REINFER / RECOVERY / STOP

Fallback:
    --enable_gripper_sm=false  → super().control_loop() directly,
                                  zero overhead, identical to plain robot_client.py

Example:
    # No --robot.record_motor_state needed.
    python -m lerobot.async_inference.smart_robot_client \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyUSB0 \\
        --task="pick up the cube" \\
        --server_address=127.0.0.1:8080 \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=user/model \\
        --actions_per_chunk=50 \\
        --enable_gripper_sm=true \\
        --gripper_load_grasp_threshold=80.0 \\
        --max_empty_grasp_retries=3
"""

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, field
from dataclasses import dataclass as _dataclass
from enum import Enum, auto
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import numpy as np

import draccus
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

from .configs import RobotClientConfig
from .helpers import QueueSizeMonitor, TimedAction, _is_image_array
from .robot_client import RobotClient
from .timing import GripperSMEventRecord, TimingRecorder


# ── Enums ─────────────────────────────────────────────────────────────────────

class GraspPhase(Enum):
    """Intended gripper phase inferred from the upcoming action chunk.

    Full lifecycle: APPROACHING → CLOSING → HOLDING → DROPPING → OPENING

    SLIP  (failure): queue stays HOLDING while load drops unexpectedly.
    DROP  (success): queue transitions from HOLDING to OPENING while load drops.
    The two are distinguished by whether the policy is commanding an opening
    (DROPPING) or still commanding hold (slip → _opening_in_queue guard fires).
    """
    APPROACHING = auto()   # gripper open, arm moving toward object
    CLOSING     = auto()   # gripper transitioning from open → closed (grasp attempt)
    HOLDING     = auto()   # gripper closed, object confirmed in hand
    DROPPING    = auto()   # confirmed hold → policy now commanding open (intentional release)
                           # obs.pos ≈ 10–15 (object gap) while action.pos rises from 0→30;
                           # both open further together — distinguishes DROP from SLIP
    OPENING     = auto()   # gripper freely opening — no object, post-drop, or approaching


class GripperDecision(Enum):
    """Action decided by the state machine for the current step."""
    CONTINUE   = auto()   # nominal path — no intervention
    REINFER    = auto()   # drain queue + force re-inference (empty grasp / stale)
    RECOVERY   = auto()   # slip detected — open gripper then reinfer
    STOP       = auto()   # too many retries — halt the run
    TASK_DONE  = auto()   # gripper closed after confirmed place → reset SM for next cycle
    LIFT_RETRY   = auto()   # first empty_grasp: lift arm via FK/IK+dZ, open, then reinfer
    REWIND_RETRY = auto()   # second empty_grasp: replay action history in reverse, then reinfer


# ── ANSI colour helpers for SM event logs ─────────────────────────────────────
# Applied only to the human-readable bracket prefix so the message body stays
# grep-friendly.  Falls back gracefully in non-TTY environments (colours are
# just invisible escape chars; the text remains intact).
_CR = "\033[91m"    # bright red    — failure events (slip / empty-grasp / recovery)
_CG = "\033[92m"    # bright green  — success events (grasp success / recovery done)
_CY = "\033[93m"    # bright yellow — soft interventions (reinfer / force-reinference)
_CB = "\033[1;91m"  # bold red      — STOP (most severe)
_CX = "\033[0m"     # reset


# ── State machine ─────────────────────────────────────────────────────────────

class GripperStateMonitor:
    """Per-step gripper state classifier.

    Reads a slim feedback dict (load / current keyed by motor_name.suffix)
    built by SmartRobotClient._read_gripper_feedback(), NOT from get_observation().
    Also scans the action queue to infer what phase the policy intends.

    All thresholds are in raw Feetech motor units (same scale as the plots).
    Tune via SmartRobotClientConfig fields.
    """

    def __init__(
        self,
        load_grasp_threshold: float,
        pos_empty_threshold: float,
        pos_open_threshold: float,
        pos_gap_threshold: float,
        pos_stable_threshold: float,
        slip_drop_ratio: float,
        max_reinfer_retries: int,
        max_total_retries: int,
        lookahead_steps: int,
        confirm_steps: int,
        grasp_confirm_steps: int,
        lift_retry_enabled: bool,
        rewind_retry_enabled: bool,
        load_key: str,
        pos_key: str,
        logger: logging.Logger,
    ):
        self._load_th           = load_grasp_threshold
        self._pos_empty         = pos_empty_threshold
        self._pos_open          = pos_open_threshold
        # Minimum gap (obs.pos − cmd.pos) to classify as "object in hand".
        # When the policy commands fully-closed (cmd≈0) but the gripper is
        # physically blocked at pos≈10–15 by an object, the gap is large.
        # When the gripper closes on empty it reaches cmd≈0 → gap≈0.
        self._pos_gap_th        = pos_gap_threshold
        # Maximum per-step decrease in obs.pos before we consider the gripper
        # still in transit rather than blocked by an object.  During a real hold
        # obs.pos is stable (±0–1 per step); during empty-close transit it drops
        # quickly (>2–5 per step).  Prevents false GRASP SUCCESS while closing.
        self._pos_stable_th     = pos_stable_threshold
        self._slip_ratio        = slip_drop_ratio
        self._max_reinfer_retries = max_reinfer_retries
        self._max_total_retries   = max_total_retries
        self._lookahead         = lookahead_steps
        self._confirm_n         = confirm_steps
        # Separate (longer) debounce for GRASP SUCCESS.  Approach-phase side
        # contacts (gripper partially open, touching container edge) can produce
        # a sustained gap + load signal for ~150 ms.  Requiring a longer window
        # (default 2× confirm_n) filters those transients while still catching
        # real grasps whose HOLDING phase lasts seconds.
        self._grasp_confirm_n   = grasp_confirm_steps
        self._lift_retry_enabled    = lift_retry_enabled
        self._rewind_retry_enabled  = rewind_retry_enabled

        self._load_key = load_key   # e.g. "gripper.load"
        self._pos_key  = pos_key    # e.g. "gripper.pos"

        self.logger = logger

        # Runtime state
        self._phase: GraspPhase = GraspPhase.APPROACHING
        # Last phase computed from a non-empty queue.  Returned when the queue
        # is transiently empty so a brief inter-chunk gap does not reset slip/
        # empty-grasp detection mid-hold.  Cleared by reset() (reinfer/episode).
        self._last_known_phase: GraspPhase = GraspPhase.APPROACHING
        self._peak_load: float  = 0.0
        self._hold_confirmed: bool = False
        # Consecutive failures (empty-grasp + slip combined).  Drives the
        # REINFER→RECOVERY escalation ladder.  NOT reset by reset() so that
        # repeated failures within a run accumulate toward RECOVERY/STOP.
        # Reset explicitly on: grasp success, recovery completion, episode start.
        self._failure_count: int = 0
        self._last_failure_type: str = ""   # "empty_grasp" | "slip" — set before _failure_decision()
        self._anomaly_steps: int = 0
        # Set True on the first CLOSING step of a genuine grasp attempt.
        # Set False when a confirmed hold transitions to OPENING/APPROACHING
        # (object was placed) and when reset() is called.
        # Guards empty-grasp detection so go-home with a naturally-closed
        # gripper (home pose pos ≈ 2–5) is not falsely flagged as a missed grasp.
        self._in_grasp_phase: bool = False
        # Set True when a confirmed hold transitions to non-CLOSING/HOLDING
        # (the object was successfully placed).  On the NEXT CLOSING phase entry
        # this signals "go-home with closed gripper" → TASK_DONE instead of
        # starting a new grasp cycle.  Cleared by reset() (new episode/reinfer).
        self._place_occurred: bool = False
        # Set True the first time scan_intended_phase() sees any commanded
        # gripper position ≥ pos_open_threshold in the action queue.
        # Prevents _in_grasp_phase from being armed when the episode starts with
        # the gripper already closed at home (the queue never shows an open cmd).
        # Cleared by reset() so each REINFER retry re-checks from scratch.
        self._cmd_was_open: bool = False
        # Reflects whether the CURRENT queue scan saw any pos ≥ pos_open_threshold.
        # Re-evaluated on every scan_intended_phase() call (not latched like
        # _cmd_was_open).  Used to suppress slip detection when the policy is
        # already commanding an opening — load drop during placing is expected,
        # not a slip.  False when queue is empty (transient gap); see below.
        self._opening_in_queue: bool = False
        # Consecutive steps where the pos-gap and load conditions both hold.
        # GRASP SUCCESS is declared only after confirm_n consecutive steps so
        # that a 1–2 step transient load spike from the empty-close mechanical
        # stop does not falsely set _hold_confirmed=True.
        self._grasp_success_steps: int = 0
        # Most recent commanded gripper position from scan_intended_phase().
        # Updated whenever the queue is non-empty; kept at previous value during
        # transient empty gaps.  Used for the pos-gap check in GRASP SUCCESS.
        self._current_cmd_pos: float = 0.0
        # Raw-load sign tracking for the sign-flip empty-grasp detector.
        # The Feetech load register is SIGNED:
        #   negative  →  motor driving in opening direction  (gripper opens against spring)
        #   positive  →  motor driving in closing direction  (gripper closes)
        # When the load transitions negative→positive within the CLOSING/HOLDING
        # phase the gripper has reversed from open to close.  If pos ≤ pos_empty
        # at or after that moment the gripper closed on empty.
        self._prev_raw_load_positive: bool = True   # sign of load at previous bus read
        # Latched True when a neg→pos transition is observed.  Stays armed until
        # the closing cycle completes (pos ≤ pos_empty fires) or is reset by
        # GRASP SUCCESS, leaving CLOSING/HOLDING, or reset().
        self._load_flip_seen: bool = False
        # True once the PHYSICALLY OBSERVED gripper pos has been ≥ pos_open
        # (fully open threshold) in the current CLOSING/HOLDING cycle.  Required
        # before EMPTY_GRASP can fire.  Two false-positive scenarios suppressed:
        # (A) post-TASK_DONE: queue shows OPEN→CLOSE before robot physically opens
        #     from home; obs.pos stays ≈7 (<< pos_open) → stays False → suppressed.
        # (B) during APPROACHING: gripper briefly opens to 14–19° as part of the
        #     approach trajectory, then closes; pos stays < pos_open → suppressed.
        # Only a genuine pre-grasp open (gripper physically reaches ≥ pos_open=20°)
        # sets this True and enables detection.  Reset on leaving CLOSING/HOLDING
        # and by reset().
        self._grasp_ever_open: bool = False
        # Previous obs.pos used for the GRASP SUCCESS stability gate.
        # None when no reading has been taken in the current CLOSING/HOLDING phase.
        # Cleared on leaving CLOSING/HOLDING and by reset().
        self._prev_pos: float | None = None
        # Consecutive steps where _load_flip_seen is True AND pos ≤ pos_empty.
        # The flip-based empty-grasp detector is debounced by confirm_n steps so
        # that a pre-close during approach (gripper closes before arm reaches the
        # object) does not fire EMPTY_GRASP immediately.  If the arm arrives at
        # the object the gripper is physically pushed above pos_empty and the
        # counter resets via the else-branch below.  Reset on leaving CLOSING/
        # HOLDING and by reset().
        self._flip_confirm_steps: int = 0
        # Latched True on the first LIFT_RETRY so the same empty_grasp event
        # cannot loop back to LIFT_RETRY indefinitely.
        # NOT reset by reset() — persists like _failure_count across reinfer cycles.
        # Reset explicitly: recovery completion and episode start.
        self._lift_retry_attempted: bool = False
        # Latched True on the first REWIND_RETRY (same semantics as _lift_retry_attempted).
        self._rewind_retry_attempted: bool = False

    def reset(self) -> None:
        """Reset transient state (phase, peak, hold flag, debounce).

        _failure_count is intentionally NOT reset here — it persists across
        REINFER cycles so repeated failures accumulate toward RECOVERY/STOP.
        Call reset_failure_count() explicitly after recovery completion or at
        episode start (control_loop does this via _gripper_monitor._failure_count = 0).
        """
        self._phase = GraspPhase.APPROACHING
        self._last_known_phase = GraspPhase.APPROACHING
        self._peak_load = 0.0
        self._hold_confirmed = False
        self._last_failure_type = ""
        self._anomaly_steps = 0
        self._in_grasp_phase = False
        self._place_occurred = False
        self._cmd_was_open = False
        self._opening_in_queue = False
        self._grasp_success_steps = 0
        self._current_cmd_pos = 0.0
        self._prev_raw_load_positive = True
        self._load_flip_seen = False
        self._grasp_ever_open = False
        self._prev_pos = None
        self._flip_confirm_steps = 0

    def scan_intended_phase(
        self,
        action_queue: Queue,
        action_queue_lock: threading.Lock,
        gripper_axis_idx: int | None,
    ) -> GraspPhase:
        """Peek at the next N actions to infer what the policy intends to do.

        When the queue is transiently empty (inter-chunk gap), returns
        ``_last_known_phase`` instead of APPROACHING so that slip/empty-grasp
        detection is not interrupted mid-hold.  ``_last_known_phase`` is reset
        to APPROACHING by reset() (called on reinfer/recovery/episode start).
        """
        if gripper_axis_idx is None:
            return GraspPhase.APPROACHING

        # Fast-path: skip lock acquisition when queue is visibly empty.
        # Queue.empty() is non-atomic but safe here — worst case we acquire the lock
        # on a race and still get an empty list, falling through to the early return.
        if action_queue.empty():
            return self._last_known_phase

        with action_queue_lock:
            raw: list[TimedAction] = list(action_queue.queue)[: self._lookahead]

        if not raw:
            # Queue is transiently empty — preserve last known phase so a brief
            # inter-chunk gap does not silently reset _hold_confirmed / _peak_load.
            return self._last_known_phase

        positions = [a.get_action()[gripper_axis_idx].item() for a in raw]
        # Track the commanded gripper position for the gap-based grasp check.
        # positions[0] is the next queued action — a good proxy for what was just
        # sent.  Not updated when the queue is empty so a brief gap keeps the
        # last known cmd value (consistent with _last_known_phase retention).
        self._current_cmd_pos = float(positions[0])
        has_open   = any(p >= self._pos_open  for p in positions)
        has_closed = any(p <  self._pos_empty for p in positions)
        all_closed = all(p <  self._pos_open  for p in positions)
        all_open   = all(p >= self._pos_open  for p in positions)

        # Latch: once an open command has appeared in the queue we know the
        # policy intends to open/approach before grasping.  Never cleared except
        # by reset() so that a transient empty-queue gap doesn't un-arm detection.
        if has_open:
            self._cmd_was_open = True
        # Live flag: True while the queue contains a command ABOVE pos_empty.
        # Uses pos_empty (not pos_open) as the threshold so that DROPPING commands
        # in the 8–20° range (gradual ramp from closed→open) also set this flag.
        # Without this broader threshold, commands like [10°, 12°, 15°, ...] are
        # classified as "closed" by has_open (needs ≥ pos_open=20°), leaving
        # _opening_in_queue=False while the phase is wrongly HOLDING → false
        # SLIP and EMPTY_GRASP during the early stage of a DROPPING sequence.
        # NOT set when the queue is transiently empty (early return above keeps
        # the previous value so a brief inter-chunk gap doesn't re-enable checks).
        self._opening_in_queue = any(p > self._pos_empty for p in positions)

        # Phase classification.
        # DROPPING is a special sub-case of the OPENING direction: the queue has
        # a mix of closed (cmd≈0, from in-progress hold) and open (cmd≈30, the
        # policy just started commanding release) positions, AND we were previously
        # in HOLDING or DROPPING.  This explicitly marks the intentional-release
        # window so the SM can log it and so the _opening_in_queue guard on SLIP
        # detection has a named counterpart.
        if has_open and has_closed:
            if positions[0] >= self._pos_open:
                phase = GraspPhase.CLOSING     # queue starts open, trending closed → grasping
            elif self._last_known_phase in (GraspPhase.HOLDING, GraspPhase.DROPPING):
                phase = GraspPhase.DROPPING    # queue starts closed (hold), trending open → dropping
            else:
                phase = GraspPhase.OPENING     # queue starts closed, trending open, not from HOLDING
        elif all_closed:
            phase = GraspPhase.HOLDING
        elif all_open:
            phase = GraspPhase.OPENING
        else:
            phase = GraspPhase.APPROACHING

        self._last_known_phase = phase
        return phase

    def update(
        self,
        feedback: dict[str, float] | None,
        action_queue: Queue,
        action_queue_lock: threading.Lock,
        gripper_axis_idx: int | None,
        *,
        _precomputed_phase: "GraspPhase | None" = None,
    ) -> GripperDecision:
        """Classify current gripper state and return a decision.

        Args:
            feedback: slim dict from SmartRobotClient._read_gripper_feedback().
                      Pass None for APPROACHING/OPENING — load/pos checks are
                      skipped for those phases and the bus read is not needed.
            _precomputed_phase: result of scan_intended_phase() if the caller
                      already ran it (to decide whether to read the bus);
                      avoids a redundant second queue scan.
        """
        intended = (
            _precomputed_phase
            if _precomputed_phase is not None
            else self.scan_intended_phase(action_queue, action_queue_lock, gripper_axis_idx)
        )
        self._phase = intended

        if intended not in (GraspPhase.CLOSING, GraspPhase.HOLDING):
            if self._hold_confirmed:
                # Confirmed hold transitioning away = place event.
                if intended == GraspPhase.DROPPING:
                    # Intentional release: policy is commanding open from a held position.
                    # obs.pos (≈10–15, object gap) and action.pos (rising from 0→30) are both
                    # "open relative to fully-closed" and continue to open further together.
                    # This is NOT a slip — slip would have _opening_in_queue=False and
                    # the queue would stay at cmd≈0.
                    self.logger.info(
                        f"{_CG}[gripper_sm] DROP{_CX} — confirmed hold → intentional release "
                        f"(policy commanding open from held position)"
                    )
                self._in_grasp_phase = False
                self._place_occurred = True
            self._hold_confirmed = False
            self._peak_load = 0.0
            self._anomaly_steps = 0
            self._grasp_success_steps = 0   # reset on leaving CLOSING/HOLDING
            self._load_flip_seen = False     # stale flip from previous closing cycle
            self._flip_confirm_steps = 0    # stale debounce counter from previous cycle
            self._grasp_ever_open = False    # reset open-confirmation for next cycle
            self._prev_pos = None            # stale transit baseline from previous cycle
            return GripperDecision.CONTINUE

        # ── Entering CLOSING / HOLDING ────────────────────────────────────────
        if self._place_occurred:
            # Any CLOSING/HOLDING after a confirmed place is the go-home gripper
            # close, not a new grasp attempt.  Return TASK_DONE regardless of
            # _in_grasp_phase: the go-home trajectory may have opened the gripper
            # (pos ≥ pos_open → _cmd_was_open=True → _in_grasp_phase=True on
            # the next CLOSING entry), which would otherwise bypass this check and
            # fall through to the EMPTY_GRASP detector below — a false positive.
            self.logger.debug(
                "[gripper_sm] TASK_DONE candidate — gripper closing after place; "
                "awaiting home-position confirmation"
            )
            return GripperDecision.TASK_DONE

        if not self._in_grasp_phase:
            # No prior place in this episode — arm detection only if the queue
            # has previously shown an open command (gripper_pos ≥ pos_open_threshold).
            # At episode start the queue contains home-position actions with the
            # gripper already closed; _cmd_was_open is still False → skip arming
            # so a closed-at-home start is not mis-classified as an empty grasp.
            if self._cmd_was_open:
                self._in_grasp_phase = True

        # feedback must be provided for CLOSING/HOLDING; fall back to zeros if missing.
        _fb = feedback or {}
        _raw_load = float(_fb.get(self._load_key, 0.0))
        load = abs(_raw_load)   # magnitude — used for all threshold checks
        pos  = float(_fb.get(self._pos_key, 0.0))
        self._peak_load = max(self._peak_load, load)
        _cmd = self._current_cmd_pos
        self.logger.debug(
            f"[gripper_sm] step | phase={intended.name} "
            f"pos={pos:.1f} cmd={_cmd:.1f} gap={pos - _cmd:.1f} "
            f"load={load:.1f}({'+'if _raw_load>=0 else '-'}) peak={self._peak_load:.1f}"
        )

        # ── Load sign tracking for the flip-based empty-grasp detector ──────────
        # Feetech load is signed: negative = motor driving open, positive = driving closed.
        # A neg→pos transition marks the moment the motor reversed from opening to closing.
        # We latch it so the detector can fire when pos finally reaches pos_empty, even
        # if the gripper takes several steps to close after the flip.
        _load_now_positive = (_raw_load >= 0)
        if not self._prev_raw_load_positive and _load_now_positive:
            self._load_flip_seen = True   # latch: will fire as soon as pos ≤ pos_empty
        self._prev_raw_load_positive = _load_now_positive

        # Track whether the gripper has physically reached the full-open threshold
        # (pos ≥ pos_open) in this CLOSING cycle.  Required before EMPTY_GRASP can
        # fire, ruling out two approach-phase false positives:
        #   (A) queue shows CLOSE before robot physically opens from home → pos≈7
        #   (B) gripper briefly opens to 14–19° during approach → pos < pos_open
        # A genuine pre-grasp requires the gripper to physically reach ≥ pos_open.
        if not self._grasp_ever_open and pos >= self._pos_open:
            self._grasp_ever_open = True

        # ── Slip detection ─────────────────────────────────────────────────
        # _opening_in_queue guard: when the policy is already commanding an opening,
        # a load drop is an intentional placement, not a slip.  Without this guard the
        # SM fires SLIP during the HOLDING→OPENING transition (load drops before the
        # queue phase shifts to APPROACHING / OPENING).
        if (
            self._hold_confirmed
            and self._peak_load > self._load_th
            and load < self._peak_load * self._slip_ratio
            and not self._opening_in_queue
        ):
            self._anomaly_steps += 1
            if self._anomaly_steps >= self._confirm_n:
                self._failure_count += 1
                self._hold_confirmed = False
                _peak_at_slip = self._peak_load   # capture before reset
                self._peak_load = 0.0
                self._anomaly_steps = 0
                self._last_failure_type = "slip"
                self.logger.warning(
                    f"{_CR}[gripper_sm] SLIP #{self._failure_count}{_CX} | "
                    f"load={load:.1f} peak={_peak_at_slip:.1f} "
                    f"(dropped to {100*load/max(_peak_at_slip,1e-6):.0f}%)"
                )
                return self._failure_decision()
            return GripperDecision.CONTINUE

        # ── Empty-grasp detection ───────────────────────────────────────────
        # _in_grasp_phase guard: skip when gripper closes for go-home
        # (natural rest position) rather than an actual grasp attempt.
        # _opening_in_queue guard: skip when policy is already commanding above
        # pos_empty — this covers the DROPPING ramp (8–20°) where the phase may
        # be misclassified as HOLDING while the gripper is actually opening.
        # _grasp_ever_open guard: skip when the gripper has never been physically
        # observed fully open (obs.pos ≥ pos_open=20°) in this CLOSING cycle.
        # Gates out APPROACHING-phase false positives:
        #   (A) post-TASK_DONE: queue shows CLOSE before robot physically opens from home
        #   (B) brief partial open (14–19°) during approach trajectory before grasping
        # A real empty-grasp attempt always starts with the gripper fully open.
        if (
            self._in_grasp_phase
            and self._grasp_ever_open
            and not self._opening_in_queue
            and pos <= self._pos_empty
        ):
            # Primary: sign-flip detector (debounced).
            # Motor drove open (negative load) then reversed to close (positive load).
            # We require confirm_n consecutive steps at pos ≤ pos_empty after the flip
            # before declaring empty grasp.  This prevents false positives when the
            # gripper pre-closes during approach (arm not yet at object): if the arm
            # arrives and the object physically pushes pos above pos_empty, the counter
            # resets via the else-branch and the grasp-success detector takes over.
            if self._load_flip_seen:
                self._flip_confirm_steps += 1
                if self._flip_confirm_steps >= self._confirm_n:
                    self._load_flip_seen = False   # consumed; reset for next cycle
                    self._flip_confirm_steps = 0
                    self._failure_count += 1
                    self._anomaly_steps = 0
                    self._last_failure_type = "empty_grasp"
                    self.logger.warning(
                        f"{_CR}[gripper_sm] EMPTY GRASP #{self._failure_count}{_CX} | "
                        f"pos={pos:.1f} load_sign=neg→pos (gripper closed on empty)"
                    )
                    return self._failure_decision()
                return GripperDecision.CONTINUE
            else:
                self._flip_confirm_steps = 0   # reset when flip is no longer active
            # Fallback: absolute threshold (covers cases where the gripper starts
            # from rest so no sign flip is observed, e.g. first episode step).
            if load < self._load_th:
                self._anomaly_steps += 1
                if self._anomaly_steps >= self._confirm_n:
                    self._failure_count += 1
                    self._anomaly_steps = 0
                    self._last_failure_type = "empty_grasp"
                    self.logger.warning(
                        f"{_CR}[gripper_sm] EMPTY GRASP #{self._failure_count}{_CX} | "
                        f"pos={pos:.1f} load={load:.1f} (abs-threshold fallback)"
                    )
                    return self._failure_decision()
            return GripperDecision.CONTINUE

        # ── Successful grasp (debounced, gap-based) ──────────────────────────
        # Primary condition: obs.pos − cmd.pos > pos_gap_threshold.
        #   Policy commands fully-closed (cmd≈0); object physically blocks the
        #   gripper at obs≈10–15 → gap≈10–15 (large, sustained).
        #   Empty close: obs reaches cmd≈0 → gap≈0–2 (small) → no GRASP SUCCESS.
        #   This is more reliable than absolute obs.pos alone because it is
        #   invariant to object size / softness; both a small hard object (obs≈9)
        #   and a large soft one (obs≈18) show a clear, sustained gap.
        # Secondary condition: load ≥ load_th.
        #   Motor is exerting real force — not just a transient contact spike.
        #   Tune gripper_load_grasp_threshold above the mechanical-stop peak
        #   (typically 80–120) to avoid false positives; real grasps show 300–500.
        # Open-position guard: obs.pos must be below pos_open_threshold.
        #   If the gripper is still in the "open" range (obs ≥ pos_open), no real
        #   grasp can have occurred regardless of gap or load.  Without this guard,
        #   a brief side-contact during approach (gripper partially open, touching
        #   the object or container edge) can hold obs stable at pos≈23° with
        #   load=500 for 6+ steps, falsely triggering GRASP SUCCESS.  The contact
        #   then breaks immediately → SLIP fires.  Real grasps always close to
        #   obs ≤ 15° (object in hand); obs ≥ pos_open=20° means still open.
        # Stability gate: obs.pos must NOT be falling fast.
        #   During empty-close transit the gripper is still physically moving;
        #   obs.pos drops quickly (e.g. 28→19→10) while cmd is already at 0 so
        #   the gap is large but the gripper hasn't touched anything — it just
        #   hasn't caught up yet.  Reject that transient by requiring pos to be
        #   stable (|Δpos| < pos_stable_threshold) before counting toward confirm_n.
        # Debounce: confirm_n consecutive qualifying (and stable) steps required
        #   before _hold_confirmed is set.
        _pos_change = (pos - self._prev_pos) if self._prev_pos is not None else 0.0
        self._prev_pos = pos

        _pos_gap = pos - self._current_cmd_pos
        if _pos_gap > self._pos_gap_th and load >= self._load_th and pos < self._pos_open:
            if _pos_change < -self._pos_stable_th:
                # Pos still dropping → gripper in transit, not yet blocked by object
                self.logger.debug(
                    f"[gripper_sm] grasp_check TRANSIT | "
                    f"pos={pos:.1f} cmd={self._current_cmd_pos:.1f} gap={_pos_gap:.1f} "
                    f"load={load:.1f} Δpos={_pos_change:+.2f} (>{-self._pos_stable_th:.1f} threshold) "
                    f"steps_reset={self._grasp_success_steps}→0"
                )
                self._grasp_success_steps = 0
                self._anomaly_steps = 0
                return GripperDecision.CONTINUE
            self._grasp_success_steps += 1
            self._anomaly_steps = 0   # forming grasp → not an anomaly
            self.logger.debug(
                f"[gripper_sm] grasp_check OK | "
                f"pos={pos:.1f} cmd={self._current_cmd_pos:.1f} gap={_pos_gap:.1f} "
                f"load={load:.1f} Δpos={_pos_change:+.2f} "
                f"steps={self._grasp_success_steps}/{self._grasp_confirm_n}"
            )
            if self._grasp_success_steps >= self._grasp_confirm_n:
                if not self._hold_confirmed:
                    self.logger.info(
                        f"{_CG}[gripper_sm] GRASP SUCCESS{_CX} | "
                        f"obs={pos:.1f} cmd={self._current_cmd_pos:.1f} "
                        f"gap={_pos_gap:.1f} load={load:.1f} "
                        f"(confirmed over {self._grasp_confirm_n} steps)"
                    )
                self._hold_confirmed = True
                self._failure_count = 0
                self._last_failure_type = ""
                self._load_flip_seen = False   # grasp confirmed; disarm the empty-grasp latch
            return GripperDecision.CONTINUE

        # Neither anomaly nor successful grasp — nominal CLOSING/HOLDING step.
        self.logger.debug(
            f"[gripper_sm] grasp_check MISS | "
            f"pos={pos:.1f} cmd={self._current_cmd_pos:.1f} gap={_pos_gap:.1f} "
            f"load={load:.1f} Δpos={_pos_change:+.2f} "
            f"{'open_fail ' if pos >= self._pos_open else ''}"
            f"{'gap_fail ' if _pos_gap <= self._pos_gap_th else ''}"
            f"{'load_fail' if load < self._load_th else ''}"
            f" steps_reset={self._grasp_success_steps}→0"
        )
        self._grasp_success_steps = 0
        self._anomaly_steps = 0
        return GripperDecision.CONTINUE

    def _failure_decision(self) -> GripperDecision:
        """Map current failure count to LIFT_RETRY / REINFER / RECOVERY / STOP.

        Pre-ladder:
          first empty_grasp + lift_retry_enabled → LIFT_RETRY (FK/IK lift + open + reinfer)
        Ladder (all other cases):
          count <= max_reinfer_retries  → REINFER  (hold + reinfer from current pos)
          count <= max_total_retries    → RECOVERY (return to home, then reinfer)
          count >  max_total_retries    → STOP
        """
        if (
            self._lift_retry_enabled
            and not self._lift_retry_attempted
            and self._last_failure_type == "empty_grasp"
        ):
            self._lift_retry_attempted = True
            self.logger.warning(
                f"{_CY}[gripper_sm] LIFT_RETRY{_CX} — "
                f"failure #{self._failure_count} empty_grasp: "
                "lifting arm then retrying"
            )
            return GripperDecision.LIFT_RETRY

        if (
            self._rewind_retry_enabled
            and not self._rewind_retry_attempted
            and self._last_failure_type == "empty_grasp"
        ):
            self._rewind_retry_attempted = True
            self.logger.warning(
                f"{_CY}[gripper_sm] REWIND_RETRY{_CX} — "
                f"failure #{self._failure_count} empty_grasp: "
                "rewinding action history then retrying"
            )
            return GripperDecision.REWIND_RETRY

        if self._failure_count > self._max_total_retries:
            self.logger.error(
                f"{_CB}[gripper_sm] STOP{_CX} — "
                f"max failures ({self._max_total_retries}) exceeded"
            )
            return GripperDecision.STOP
        elif (
            self._failure_count > self._max_reinfer_retries
            or self._last_failure_type in ("empty_grasp", "slip")
        ):
            # empty_grasp / slip: REINFER from the same spot is not useful —
            # the gripper is empty or the object has already fallen, so re-grasping
            # from the same mid-air position would fail immediately again.
            # Always return home for a clean starting pose.
            self.logger.warning(
                f"{_CR}[gripper_sm] RECOVERY{_CX} — "
                f"failure #{self._failure_count} type='{self._last_failure_type}' "
                f"→ returning to home"
            )
            return GripperDecision.RECOVERY
        else:
            self.logger.warning(
                f"{_CY}[gripper_sm] REINFER{_CX} — "
                f"failure #{self._failure_count} / {self._max_reinfer_retries} "
                f"→ hold + reinfer from current position"
            )
            return GripperDecision.REINFER


# ── Config ─────────────────────────────────────────────────────────────────────

@_dataclass
class SmartRobotClientConfig(RobotClientConfig):
    """RobotClientConfig extended with gripper state-machine parameters.

    All new fields have defaults so vanilla draccus CLI usage is unchanged.
    Set --enable_gripper_sm=false to fall back to plain RobotClient behavior.

    NOTE: --robot.record_motor_state is NOT needed for the state machine.
    Feedback is read via robot.bus.sync_read() independently of the inference
    pipeline, which only uses Present_Position + camera frames.
    """

    # ── Master switch ──────────────────────────────────────────────────────────
    enable_gripper_sm: bool = field(
        default=True,
        metadata={"help": "Enable gripper state machine. false = vanilla RobotClient behavior."},
    )

    # ── Direct bus read config (no record_motor_state required) ───────────────
    # Motor names to read feedback from (passed to bus.sync_read as motor_names).
    gripper_sm_motor_names: list[str] = field(
        default_factory=lambda: ["gripper"],
        metadata={
            "help": (
                "Motor names to read feedback from via bus.sync_read(). "
                "Extend this list to monitor additional motors (e.g. ['gripper', 'wrist_roll']). "
                "Does NOT require --robot.record_motor_state."
            )
        },
    )
    # Feetech register name → feedback dict key suffix.
    # bus.sync_read(register, motor_names) returns {motor: value}.
    # Combined key in feedback dict: "{motor_name}.{suffix}".
    # Example: {"Present_Load": "load"} → feedback["gripper.load"] = float(val).
    gripper_sm_feedback_registers: dict[str, str] = field(
        default_factory=lambda: {"Present_Load": "load"},
        metadata={
            "help": (
                "Feetech register → feedback key suffix mapping. "
                "Keys must be valid FeetechMotorsBus register names. "
                "Default reads only Present_Load (addr 60) + Present_Position (addr 56, always). "
                "Present_Current (addr 69) is non-contiguous (+9 bytes gap) so adding it costs "
                "an extra serial round-trip (~1-2ms). Only add if current-based logic is needed."
            )
        },
    )

    # Key names in the feedback dict consumed by GripperStateMonitor.
    # Derived from gripper_sm_motor_names[0] + suffix, but configurable.
    gripper_sm_load_key: str = field(
        default="gripper.load",
        metadata={"help": "Feedback dict key for gripper load (must match motor_name.suffix)."},
    )
    gripper_sm_pos_key: str = field(
        default="gripper.pos",
        metadata={
            "help": (
                "Feedback dict key for gripper position used in empty-grasp detection. "
                "NOTE: gripper.pos IS available in the inference obs (get_observation() always "
                "reads Present_Position). _read_gripper_feedback() adds it from the bus read "
                "so the SM can use it without an extra sync_read call."
            )
        },
    )

    # ── Gripper axis in action tensor ──────────────────────────────────────────
    # Key to search in robot.action_features to find the gripper dimension.
    gripper_action_key: str = field(
        default="gripper",
        metadata={"help": "Substring to match in robot.action_features for gripper axis."},
    )

    # ── Thresholds (tune from real-robot plots) ────────────────────────────────
    gripper_load_grasp_threshold: float = field(
        default=150.0,
        metadata={
            "help": (
                "Min |gripper.load| to classify as 'object in hand'. "
                "From plots: successful grasp ~300-500, empty close mechanical stop ~80-120. "
                "Set ABOVE the mechanical-stop peak to avoid phantom GRASP SUCCESS on empty "
                "close (default 150 filters the stop spike; real grasps are 300-500)."
            )
        },
    )
    gripper_pos_gap_threshold: float = field(
        default=7.0,
        metadata={
            "help": (
                "Min gap (obs.gripper.pos − action.gripper.pos) to classify as 'object in hand'. "
                "When policy commands fully-closed (action≈0) but the gripper is physically "
                "blocked by an object (obs≈8–15), the gap is large and sustained. "
                "When closing on empty, obs converges to cmd → gap→0. "
                "This is the primary grasp indicator — more reliable than absolute obs.pos "
                "alone because it is invariant to object size and softness. "
                "Works alongside load_threshold: both must hold for confirm_steps. "
                "Lowered from 10 to 7: with obs≈8–10 for small/hard objects and cmd≈0, "
                "a threshold of 10 is never exceeded (gap=10 fails gap>10)."
            )
        },
    )
    gripper_pos_stable_thresh: float = field(
        default=2.5,
        metadata={
            "help": (
                "Max per-step decrease in obs.gripper.pos that is still considered 'stable'. "
                "Used by the GRASP SUCCESS stability gate: if obs.pos drops by more than "
                "this amount in a single step the gripper is still closing through transit "
                "(obs lag, no object contact) rather than blocked by an object. "
                "In that case _grasp_success_steps is reset so a phantom GRASP SUCCESS "
                "during empty-close transit cannot accumulate to confirm_n. "
                "At 40Hz, real holds vary ±0–1/step; empty transit drops 2–9/step. "
                "Raised from 1.5 to 2.5: 1.5°/step at 40Hz = 60°/s rejection which is "
                "too aggressive for a settling hold that may drop 1–2° as it stabilises."
            )
        },
    )
    gripper_pos_empty_threshold: float = field(
        default=8.0,
        metadata={
            "help": (
                "Max gripper.pos to classify as 'fully closed (no object)'. "
                "From third plot: empty-closed ~2-5, holding ~10-15, open ~28-30."
            )
        },
    )
    gripper_pos_open_threshold: float = field(
        default=20.0,
        metadata={"help": "Min gripper.pos to classify as 'open'. Used for phase detection."},
    )
    gripper_slip_drop_ratio: float = field(
        default=0.4,
        metadata={
            "help": (
                "Slip threshold: load drops below (peak_load × ratio) while in HOLDING phase. "
                "0.4 = load fell to <40%% of peak → slip."
            )
        },
    )

    # ── Retry / recovery ladder ────────────────────────────────────────────────
    max_reinfer_retries: int = field(
        default=2,
        metadata={
            "help": (
                "Consecutive failures (empty-grasp or slip) before escalating to RECOVERY. "
                "Below this threshold: hold + reinfer from current position (REINFER). "
                "At/above threshold: return to home then reinfer (RECOVERY). "
                "Must be < max_empty_grasp_retries."
            )
        },
    )
    max_empty_grasp_retries: int = field(
        default=6,
        metadata={
            "help": (
                "Total consecutive failures (empty-grasp + slip combined) before STOP. "
                "Failures above max_reinfer_retries trigger RECOVERY (home return); "
                "above this threshold the run is halted entirely."
            )
        },
    )
    recovery_return_to_home: bool = field(
        default=True,
        metadata={
            "help": (
                "On RECOVERY, return the robot to its home position (captured at connect "
                "time) before re-inferring. "
                "False = reinfer from current position immediately (legacy behavior)."
            )
        },
    )
    recovery_home_steps: int = field(
        default=90, # ~3 s at 30 fps, ~1 s at 90 fps
        metadata={
            "help": (
                "Number of control steps for the non-blocking home-return trajectory. "
                "At 30 fps: 30 steps ≈ 1 s return time. "
                "Larger values give a smoother, slower return."
            )
        },
    )
    recovery_home_settle_time: float = field(
        default=0.5,
        metadata={
            "help": (
                "Seconds to wait after completing the home-return trajectory before "
                "arming must_go=True and triggering re-inference. "
                "Allows the arm to mechanically settle at home (oscillations dampen) "
                "so the next observation is captured from a stable pose."
            )
        },
    )
    recovery_warmup_steps: int = field(
        default=25,
        metadata={
            "help": (
                "Number of control steps to hold at home and send observations (without "
                "executing policy actions) after recovery completes. "
                "Purpose: fill the policy server's context window with clean home-state "
                "frames before the actual must_go inference fires.  Without this, the "
                "server context contains [failed_grasp_obs…, home_obs] — an abrupt "
                "out-of-distribution jump — causing fast erratic motions on the first "
                "post-recovery chunk.  Set 0 to disable (must_go fires immediately). "
                "At 30 fps, 25 steps ≈ 0.83s; at 90 fps (interp×3), ≈ 0.28s. "
                "Each step sends obs unconditionally so all 25 unique IDs are flushed "
                "into the server context window (covers infer_delay up to ~25 frames)."
            )
        },
    )
    recovery_smooth_steps: int = field(
        default=60,
        metadata={
            "help": (
                "Number of action sub-steps to apply per-axis velocity clamping after "
                "recovery warmup completes. Prevents violent motion on the first policy "
                "chunk when the policy context and actual robot position are mismatched. "
                "Each step the commanded position is clamped to ±recovery_smooth_max_delta "
                "relative to the previous sent position (anchored to home at step 0). "
                "At 90 Hz (30 fps × interp×3), 60 steps ≈ 0.67s. Set 0 to disable."
            )
        },
    )
    recovery_smooth_max_delta: float = field(
        default=3.0,
        metadata={
            "help": (
                "Maximum per-axis position change (degrees) allowed per control sub-step "
                "during post-recovery smoothing. Lower = slower / smoother ramp-in. "
                "At 90 Hz control: 3.0 deg/step ≈ 270 deg/s max angular velocity. "
                "Typical fast policy motion: ~500 deg/s → this halves peak velocity."
            )
        },
    )
    gripper_lookahead_steps: int = field(
        default=15,
        metadata={"help": "How many upcoming queue actions to scan for phase detection."},
    )
    gripper_confirm_steps: int = field(
        default=3,
        metadata={
            "help": (
                "Consecutive steps an anomaly (empty-grasp / slip) must persist before "
                "triggering REINFER/RECOVERY.  Debounces transient load spikes. "
                "At 30fps: 3 steps ≈ 100ms.  Scaled by interpolation_multiplier at runtime."
            )
        },
    )
    gripper_grasp_confirm_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Consecutive steps the gap+load conditions must hold before GRASP SUCCESS "
                "is declared.  0 = auto: 2 × gripper_confirm_steps (after Hz scaling). "
                "Set higher than gripper_confirm_steps so that brief side-contacts during "
                "approach (gripper touching container edge, lasting ~150 ms) cannot "
                "accumulate to confirmation while a real sustained hold (lasting seconds) "
                "easily exceeds the threshold.  Also scaled by interpolation_multiplier."
            )
        },
    )
    # ── Empty-grasp lift retry (Option C: Cartesian Z FK/IK) ─────────────────
    empty_grasp_lift_retry_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable LIFT_RETRY on first EMPTY_GRASP: lift the arm by Cartesian Z delta "
                "(FK→+dZ→IK), open the gripper, then reinfer. "
                "Falls through to RECOVERY on second empty_grasp (or any other failure). "
                "Requires --empty_grasp_lift_urdf_path or the default so101_kinematics.urdf. "
                "Only supported for SO-101."
            )
        },
    )
    empty_grasp_lift_delta_xyz_m: list[float] = field(
        default_factory=lambda: [-0.03, 0.0, 0.08],
        metadata={
            "help": (
                "Cartesian [X, Y, Z] delta in metres applied to the end-effector during LIFT_RETRY. "
                "Default [0, 0, 0.08] = lift 8 cm straight up. "
                "X: negative = retreat toward base, positive = extend forward. "
                "Y: lateral offset. "
                "Z: positive = up. "
                "Example: [-0.03, 0, 0.08] = lift 8 cm + retreat 3 cm."
            )
        },
    )
    empty_grasp_lift_steps: int = field(
        default=60,
        metadata={
            "help": (
                "Control steps for the LIFT_RETRY FK/IK lift trajectory. "
                "At 30 fps: 60 steps ≈ 2 s.  At 90 fps (interp×3): ≈ 0.67 s."
            )
        },
    )
    empty_grasp_lift_warmup_steps: int = field(
        default=25,
        metadata={
            "help": (
                "Context warmup steps at lift position before arming must_go. "
                "Fills server context window with lift-position frames. "
                "0 = skip (must_go fires immediately after the lift trajectory)."
            )
        },
    )
    empty_grasp_lift_settle_time: float = field(
        default=0.5,
        metadata={
            "help": (
                "Seconds to wait after the lift trajectory completes before starting "
                "context warmup (or arming must_go when warmup is disabled). "
                "Lets the arm mechanically settle at the lift position so the first "
                "warmup observation is from a stable pose. "
                "Analogous to recovery_home_settle_time. Set 0 to disable."
            )
        },
    )
    empty_grasp_lift_gripper_open_deg: float = field(
        default=30.0,
        metadata={"help": "Gripper target position (degrees) during LIFT_RETRY (default 30 = fully open)."},
    )
    empty_grasp_lift_urdf_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to a mesh-stripped URDF for FK/IK during LIFT_RETRY. "
                "Empty = auto-resolve to so101_kinematics.urdf alongside the SO-follower driver. "
                "The URDF must have <visual>/<collision> blocks removed so placo can load it "
                "without requiring absent STL files."
            )
        },
    )

    # ── REWIND_RETRY config ───────────────────────────────────────────────────
    empty_grasp_rewind_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, on a second consecutive empty_grasp (after LIFT_RETRY), replay the "
                "last empty_grasp_rewind_steps policy actions in reverse to retrace the arm's "
                "path, then warmup + reinfer. Ring-buffer size auto-scales with "
                "interpolation_multiplier."
            )
        },
    )
    empty_grasp_rewind_steps: int = field(
        default=20,
        metadata={
            "help": (
                "Maximum number of policy-Hz steps to replay when REWIND_RETRY fires. "
                "Acts as an upper cap on the rewind trajectory length. "
                "When empty_grasp_rewind_min_displacement_deg > 0, fewer steps may be "
                "used if the distance target is reached earlier. "
                "Ring-buffer capacity is governed by empty_grasp_rewind_buffer_steps."
            )
        },
    )
    empty_grasp_rewind_buffer_steps: int = field(
        default=60,
        metadata={
            "help": (
                "History ring-buffer capacity in policy-Hz steps. "
                "Decoupled from empty_grasp_rewind_steps so the buffer can store a longer "
                "look-back window than the maximum replay length. "
                "Increase this when the robot hovers near the target for many steps before "
                "a failed grasp, so that the buffer contains earlier meaningful motion. "
                "Actual deque maxlen = buffer_steps × max(1, interpolation_multiplier)."
            )
        },
    )
    empty_grasp_rewind_min_displacement_deg: float = field(
        default=0.0,
        metadata={
            "help": (
                "Minimum net joint-space L2 displacement (in action-space units) the "
                "rewind trajectory endpoint must be from the start (current position). "
                "Uses NET displacement — ||endpoint - start||₂ — NOT cumulative path "
                "length. Cumulative path length is unreliable because hover phases "
                "accumulate many tiny oscillating steps that sum to a large path length "
                "while barely moving the arm (net ≈ 0). "
                "When > 0, _build_rewind_trajectory() scans backward and stops as soon "
                "as the candidate endpoint is >= this distance from the start, or when "
                "empty_grasp_rewind_steps / the buffer runs out. "
                "Set to 0 (default) to use all available history up to rewind_steps. "
                "Units match robot.action_features (degrees for SO-101/SO-100). "
                "Typical starting value: 15–20 deg for a 6-DOF arm in degree-space "
                "(net L2 ≈ 3–5 deg per policy step; 0.8 deg is satisfied in 1 step "
                "and produces no visible motion)."
            )
        },
    )
    empty_grasp_rewind_warmup_steps: int = field(
        default=10,
        metadata={
            "help": (
                "Observation-only steps sent after the rewind trajectory completes to fill "
                "the policy context window before re-inference. "
                "Analogous to empty_grasp_lift_warmup_steps."
            )
        },
    )
    empty_grasp_rewind_settle_time: float = field(
        default=0.0,
        metadata={
            "help": (
                "Seconds to pause at the rewind endpoint before starting context warmup. "
                "Analogous to empty_grasp_lift_settle_time. Set 0 to skip."
            )
        },
    )
    empty_grasp_rewind_gripper_open_deg: float = field(
        default=30.0,
        metadata={
            "help": (
                "Gripper target position (degrees) injected into every step of the "
                "REWIND_RETRY trajectory (default 30 = fully open). "
                "Keeps the gripper open while the arm retraces its path backward."
            )
        },
    )
    bg_obs_sender_send_image: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether _BackgroundObsSender includes camera images in trajectory-phase obs. "
                "True (default): full obs with images sent — required if the server ever needs "
                "image content from trajectory frames (e.g. for logging or future context use). "
                "False (lightweight): images are stripped before gRPC transmission, reducing "
                "payload from ~100-200 KB to < 1 KB per step. Safe because all bg_obs_sender "
                "obs carry skip_inference=True so the server never runs inference on them."
            )
        },
    )

    enable_recapture_home_positions: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to re-capture the home position at the start of each episode "
                "(after start_barrier.wait()).  "
                "When False (default), the snapshot taken at __init__ / connect time is kept "
                "for the entire session — _is_at_home() and recovery trajectories always target "
                "that fixed reference.  "
                "When True, _recapture_home_positions() is called once per episode so the "
                "reference tracks wherever the operator has physically placed the arm before "
                "each trial.  Enable when the robot's resting pose varies between episodes "
                "(e.g. manual resets) and you want TASK_DONE / recovery to reflect the "
                "actual per-episode start pose."
            )
        },
    )
    task_done_home_tolerance: float = field(
        default=5.0,
        metadata={
            "help": (
                "Max per-joint deviation from the home snapshot for the arm to be classified as "
                "'at home' during TASK_DONE detection.  Tighter values (≤5°) prevent false "
                "triggers mid go-home trajectory where some joints reach threshold before others. "
                "Units: degrees when use_degrees=True, [-100, 100] otherwise."
            )
        },
    )
    task_done_home_confirm_steps: int = field(
        default=2,
        metadata={
            "help": (
                "Number of consecutive control steps that must pass _is_at_home() before "
                "TASK_DONE fires.  Guards against single-frame threshold crossings that occur "
                "while the arm is still in transit on the go-home trajectory.  Set to 1 to "
                "restore the original single-sample behaviour."
            )
        },
    )
    task_done_home_check_mode: str = field(
        default="joint",
        metadata={
            "help": (
                "How to determine whether the arm has returned home for TASK_DONE detection. "
                "'joint': per-joint deviation ≤ task_done_home_tolerance (degrees). "
                "'ee': FK → end-effector Cartesian distance ≤ task_done_home_ee_tolerance_m. "
                "Requires placo + so101_kinematics.urdf (same as LIFT_RETRY); falls back to "
                "'joint' with a warning if FK is unavailable."
            )
        },
    )
    task_done_home_ee_tolerance_m: float = field(
        default=0.02,
        metadata={
            "help": (
                "EE-mode home tolerance in metres.  The end-effector must be within this "
                "Cartesian L2 distance of the home EE position for "
                "task_done_home_confirm_steps consecutive steps before TASK_DONE fires. "
                "Default 0.02 m (2 cm).  Only used when task_done_home_check_mode='ee'. "
                "When task_done_home_ee_tolerance_xyz_m is also set, BOTH checks must pass."
            )
        },
    )
    task_done_home_ee_tolerance_xyz_m: list[float] = field(
        default_factory=list,
        metadata={
            "help": (
                "Per-axis EE home tolerances [tol_x, tol_y, tol_z] in metres. "
                "When non-empty (length must be 3), each Cartesian axis is checked "
                "independently: |ee_x - home_x| <= tol_x AND |ee_y - home_y| <= tol_y "
                "AND |ee_z - home_z| <= tol_z.  Useful when the arm has large reach "
                "(X) variance but small lateral (Y) variance — per-axis thresholds "
                "prevent a small Y/Z deviation from masking a large X offset in the "
                "L2 norm.  When empty (default), only task_done_home_ee_tolerance_m "
                "is used.  When both are set, BOTH the L2 check and the per-axis "
                "check must pass.  Example: [0.04, 0.06, 0.04] = 4 cm X/Z, 6 cm Y."
            )
        },
    )
    task_done_home_check_gripper: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to include gripper position in home detection. "
                "joint mode: when True, gripper.pos must be within task_done_home_tolerance "
                "of the home snapshot; when False (default), gripper is excluded from the "
                "joint check so that a closing gripper (TASK_DONE via SM) does not block "
                "home detection even if home was captured with the gripper open. "
                "ee mode: when True, gripper.pos deviation from home is checked separately "
                "using task_done_home_gripper_tolerance_deg; when False (default), gripper "
                "is ignored (consistent with FK not including the gripper joint). "
                "Set True only if your home pose requires a specific gripper state."
            )
        },
    )
    task_done_home_gripper_tolerance_deg: float = field(
        default=10.0,
        metadata={
            "help": (
                "Gripper position tolerance (degrees) for home detection when "
                "task_done_home_check_gripper=True. Applied to both joint and ee modes. "
                "Checked against gripper.pos in _home_positions. "
                "Default 10° is intentionally loose — gripper repeatability is ±5-10°."
            )
        },
    )
    log_level: str = field(
        default="INFO",
        metadata={"help": "Python logging level: DEBUG / INFO / WARNING / ERROR. DEBUG enables per-step loop timing."},
    )

    def __post_init__(self):
        super().__post_init__()
        if len(self.empty_grasp_lift_delta_xyz_m) != 3:
            raise ValueError(
                f"empty_grasp_lift_delta_xyz_m must have exactly 3 elements [X, Y, Z], "
                f"got {self.empty_grasp_lift_delta_xyz_m}"
            )
        if self.gripper_slip_drop_ratio <= 0 or self.gripper_slip_drop_ratio >= 1:
            raise ValueError(
                f"gripper_slip_drop_ratio must be in (0, 1), got {self.gripper_slip_drop_ratio}"
            )
        if self.gripper_pos_empty_threshold >= self.gripper_pos_open_threshold:
            raise ValueError(
                f"gripper_pos_empty_threshold ({self.gripper_pos_empty_threshold}) must be "
                f"< gripper_pos_open_threshold ({self.gripper_pos_open_threshold})"
            )
        if self.max_reinfer_retries < 0:
            raise ValueError(
                f"max_reinfer_retries must be >= 0, got {self.max_reinfer_retries}"
            )
        if self.max_reinfer_retries >= self.max_empty_grasp_retries:
            raise ValueError(
                f"max_reinfer_retries ({self.max_reinfer_retries}) must be "
                f"< max_empty_grasp_retries ({self.max_empty_grasp_retries})"
            )
        if self.recovery_home_steps < 1:
            raise ValueError(
                f"recovery_home_steps must be >= 1, got {self.recovery_home_steps}"
            )


# ── Background obs sender ──────────────────────────────────────────────────────

class _BackgroundObsSender:
    """Fire-and-forget gRPC observation sender for trajectory execution phases.

    During RECOVERY / LIFT_RETRY / REWIND_RETRY the control loop must run at
    strict 10 Hz but send_observation() blocks for 60–400 ms (synchronous gRPC).
    This helper decouples the two: the main thread captures obs cheaply (~30 ms)
    and enqueues a pre-built TimedObservation; a daemon thread drains the queue
    and sends each obs over gRPC without affecting control timing.

    All obs sent here carry skip_inference=True so the server immediately returns
    False without running inference, similarity checks, or updating last_processed_obs.
      - skip_inference = True  → server skips inference (no GPU, no similarity check)
      - must_go = False        → backward-compat signal for old servers
      - leftover = None        → correct: trajectory start always follows queue drain
      - infer_delay = 0        → ignored for skip_inference obs

    When send_image=False (lightweight mode) camera images are stripped from the obs
    before transmission, reducing gRPC payload from ~100-200 KB to < 1 KB per step.
    This is safe because skip_inference obs are never used for inference on the server.

    Lifecycle per trajectory phase:
        start() → enqueue() × N steps → drain() → [warmup] → must_go inference
    """

    def __init__(self, client: "SmartRobotClient", maxsize: int = 128, send_image: bool = True) -> None:
        self._client = client
        self._maxsize = maxsize
        self._send_image = send_image
        self._q: Queue = Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """(Re)start the background sender thread, draining any prior run first."""
        if self._thread is not None and self._thread.is_alive():
            try:
                self._q.put(None, timeout=0.5)
            except Exception:
                pass
            self._thread.join(timeout=3.0)
        self._q = Queue(maxsize=self._maxsize)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="bg-obs-sender"
        )
        self._thread.start()

    def enqueue(self, raw_obs: dict, timestep: int) -> None:
        """Build and enqueue a TimedObservation from already-captured raw obs.

        Called from the main control thread immediately after _capture_raw_obs().
        Preprocessing (~2–5 ms) runs here so the background thread only does gRPC.
        Drops silently when the queue is full (bounded back-pressure).
        """
        try:
            if self._send_image:
                obs_for_send = raw_obs
            else:
                # Lightweight mode: strip image arrays to save ~100-200 KB per step.
                # Joint state and metadata are kept so the server can still log timesteps.
                obs_for_send = {k: v for k, v in raw_obs.items() if not _is_image_array(v)}
            processed = self._client._preprocess_obs(obs_for_send)
            obs = self._client._build_timed_observation(processed, timestep, 0, None)
            obs.must_go = False
            obs.skip_inference = True
            self._q.put_nowait(obs)
        except Exception:
            pass  # queue full or build error — never block the control loop

    def drain(self, timeout: float = 5.0) -> None:
        """Send a sentinel, wait for the sender thread to finish, then join.

        Call this at the end of the trajectory phase (before warmup) to ensure
        all captured obs are delivered to the server before inference is armed.
        Safe to call even when start() was never called.
        """
        if self._thread is None or not self._thread.is_alive():
            self._thread = None
            return
        try:
            self._q.put(None, timeout=1.0)
        except Exception:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        sent = 0
        while True:
            try:
                obs = self._q.get(timeout=1.0)
            except Exception:
                continue
            if obs is None:
                self._client.logger.info(
                    f"[bg_obs_sender] sender done — sent {sent} obs"
                )
                return
            try:
                self._client.send_observation(obs)
                sent += 1
            except Exception as exc:
                self._client.logger.debug(f"[bg_obs_sender] send failed: {exc}")


# ── Client ─────────────────────────────────────────────────────────────────────

class SmartRobotClient(RobotClient):
    """RobotClient with a gripper-feedback state machine.

    Two independent data paths:

      Inference:  robot.get_observation()  (positions + cameras) → server
      SM:         robot.bus.sync_read()    (load/current only)   → GripperStateMonitor

    record_motor_state is NOT required.  The SM reads motor feedback directly
    via the Feetech bus without touching the observation/action feature maps.
    """

    prefix = "smart_robot_client"

    def __init__(self, config: SmartRobotClientConfig):
        super().__init__(config)

        self._gripper_axis_idx: int | None = None
        self._gripper_monitor: GripperStateMonitor | None = None
        self._gripper_sm_recorder: TimingRecorder | None = None

        # Non-blocking recovery trajectory state
        self._home_positions: dict[str, float] | None = None
        self._home_ee_pos: "np.ndarray | None" = None  # EE XYZ (3,) in metres, for 'ee' check mode
        self._recovery_remaining: int = 0
        self._recovery_traj: list[dict[str, float]] | None = None
        self._recovery_warmup_remaining: int = 0
        self._recovery_count: int = 0  # total recoveries this run (never reset mid-run)
        # Post-recovery soft velocity clamping (prevents violent first chunk motion)
        self._recovery_smooth_remaining: int = 0
        self._recovery_smooth_prev: dict[str, float] | None = None
        # Non-blocking LIFT_RETRY trajectory state (FK/IK Cartesian Z lift)
        self._lift_traj: list[dict[str, float]] | None = None
        self._lift_remaining: int = 0
        self._lift_warmup_remaining: int = 0
        self._lift_kinematics = None  # RobotKinematics; None if disabled/unavailable
        self._lift_retry_count: int = 0  # total LIFT_RETRY triggers this run
        # Non-blocking REWIND_RETRY trajectory state (reverse action-history replay)
        self._rewind_traj: list[dict[str, float]] | None = None
        self._rewind_remaining: int = 0
        self._rewind_warmup_remaining: int = 0
        self._rewind_retry_count: int = 0  # total REWIND_RETRY triggers this run
        # Action history ring buffer: auto-sized to buffer_steps × max(1, interpolation_multiplier).
        # buffer_steps is decoupled from rewind_steps so the look-back window can extend
        # further than the maximum replay length (important when the robot hovers for many
        # steps before a failed grasp).
        _buf_len = config.empty_grasp_rewind_buffer_steps * max(1, config.interpolation_multiplier)
        self._action_history: deque[dict[str, float]] | None = (
            deque(maxlen=_buf_len)
            if config.enable_gripper_sm and config.empty_grasp_rewind_enabled
            else None
        )
        # Background obs sender shared by all three trajectory phases (RECOVERY / LIFT_RETRY /
        # REWIND_RETRY).  Each phase calls start() before arming the trajectory and drain()
        # when the last step completes.  The background thread sends context-filling obs
        # (must_go=False, leftover=None) over gRPC without blocking the 10 Hz control loop.
        self._bg_obs_sender = _BackgroundObsSender(self, send_image=config.bg_obs_sender_send_image)

        # Grasp-success attribution counters (never reset mid-run)
        # _last_intervention: most recent major intervention before the current grasp.
        # "" = clean attempt, "lift_retry" = preceded by LIFT_RETRY, "recovery" = preceded by RECOVERY.
        # Overwritten on each new intervention; reset to "" after each GRASP SUCCESS or TASK_DONE.
        self._last_intervention: str = ""
        self._grasp_success_total: int = 0
        self._grasp_success_after_lift_retry: int = 0
        self._grasp_success_after_recovery: int = 0

        # Per-episode retry + success tracking (episode = one pick-place cycle).
        # success=True  → episode ended with TASK_DONE
        # success=False → episode ended with STOP (max retries exceeded)
        self._ep_retry_count: int = 0
        self._ep_records: list[dict] = []   # list of {"retries": int, "success": bool}

        # Consecutive at-home confirmation counter — incremented each step _is_at_home()
        # returns True while a TASK_DONE candidate is active; reset on any False reading
        # or when TASK_DONE fires.  Prevents single-frame threshold crossings mid-trajectory.
        self._at_home_confirm_count: int = 0

        if config.enable_gripper_sm:
            # confirm_steps is defined at policy Hz; scale to actual control Hz so
            # the debounce window stays ~100 ms regardless of interpolation_multiplier.
            _effective_confirm = max(1, config.gripper_confirm_steps * config.interpolation_multiplier)
            # grasp_confirm_steps: 0 → auto = 2 × confirm (approach contacts typically
            # last < 2× window; real grasps last seconds).  Always ≥ confirm.
            _effective_grasp_confirm = (
                max(1, config.gripper_grasp_confirm_steps * config.interpolation_multiplier)
                if config.gripper_grasp_confirm_steps > 0
                else _effective_confirm * 2
            )
            self._gripper_monitor = GripperStateMonitor(
                load_grasp_threshold = config.gripper_load_grasp_threshold,
                pos_empty_threshold  = config.gripper_pos_empty_threshold,
                pos_open_threshold   = config.gripper_pos_open_threshold,
                pos_gap_threshold    = config.gripper_pos_gap_threshold,
                pos_stable_threshold = config.gripper_pos_stable_thresh,
                slip_drop_ratio      = config.gripper_slip_drop_ratio,
                max_reinfer_retries  = config.max_reinfer_retries,
                max_total_retries    = config.max_empty_grasp_retries,
                lookahead_steps      = config.gripper_lookahead_steps,
                confirm_steps        = _effective_confirm,
                grasp_confirm_steps  = _effective_grasp_confirm,
                lift_retry_enabled   = config.empty_grasp_lift_retry_enabled,
                rewind_retry_enabled = config.empty_grasp_rewind_enabled,
                load_key             = config.gripper_sm_load_key,
                pos_key              = config.gripper_sm_pos_key,
                logger               = self.logger,
            )
            if config.interpolation_multiplier > 1:
                self.logger.info(
                    f"[gripper_sm] confirm_steps scaled: {config.gripper_confirm_steps} × "
                    f"{config.interpolation_multiplier} = {_effective_confirm} "
                    f"(control Hz = {config.fps * config.interpolation_multiplier})"
                )
            self._gripper_axis_idx = self._find_gripper_axis(config.gripper_action_key)

            # Capture home position (robot already connected by super().__init__)
            try:
                raw = self.robot.bus.sync_read("Present_Position")
                self._home_positions = {f"{m}.pos": float(v) for m, v in raw.items()}
                self.logger.info(f"[gripper_sm] Home position captured: {self._home_positions}")
            except Exception as exc:
                self.logger.warning(
                    f"[gripper_sm] Could not capture home position: {exc} "
                    "— recovery_return_to_home will be skipped"
                )

            if config.empty_grasp_lift_retry_enabled:
                self._init_lift_kinematics()

            # EE home-check mode: ensure FK solver is available (reuse or lazy-init),
            # then capture the home EE position from the already-read _home_positions.
            if config.task_done_home_check_mode == "ee":
                if self._lift_kinematics is None:
                    # LIFT_RETRY not enabled — init FK solver purely for EE home check.
                    self._init_lift_kinematics()
                self._capture_home_ee()

            self.logger.info(
                f"[gripper_sm] ENABLED | "
                f"axis_idx={self._gripper_axis_idx} | "
                f"sm_motors={config.gripper_sm_motor_names} | "
                f"registers={list(config.gripper_sm_feedback_registers.keys())} | "
                f"load_th={config.gripper_load_grasp_threshold} | "
                f"pos_gap_th={config.gripper_pos_gap_threshold} | "
                f"pos_empty={config.gripper_pos_empty_threshold} | "
                f"pos_open={config.gripper_pos_open_threshold} | "
                f"max_reinfer={config.max_reinfer_retries} | "
                f"max_total={config.max_empty_grasp_retries} | "
                f"confirm_steps={config.gripper_confirm_steps} | "
                f"grasp_confirm_steps={_effective_grasp_confirm} | "
                f"lift_retry={config.empty_grasp_lift_retry_enabled} | "
                f"rewind_retry={config.empty_grasp_rewind_enabled} | "
                f"rewind_steps={config.empty_grasp_rewind_steps} | "
                f"rewind_buf={config.empty_grasp_rewind_buffer_steps} | "
                f"rewind_min_disp={config.empty_grasp_rewind_min_displacement_deg}° | "
                f"return_to_home={config.recovery_return_to_home} | "
                f"home_steps={config.recovery_home_steps}"
            )
            self.logger.info(
                "[gripper_sm] Feedback path: robot.bus.sync_read() — "
                "record_motor_state NOT required"
            )
        else:
            self.logger.info("[gripper_sm] DISABLED — vanilla RobotClient behavior")

    # ── Gripper axis discovery ─────────────────────────────────────────────────

    def _find_gripper_axis(self, gripper_action_key: str) -> int | None:
        """Find the gripper dimension index in the action tensor via robot.action_features.

        record_motor_state=[] (default) means action_features only contains
        .pos keys → no contamination from load/current.
        """
        features = list(self.robot.action_features)
        for idx, key in enumerate(features):
            if gripper_action_key.lower() in key.lower():
                self.logger.info(
                    f"[gripper_sm] Gripper action axis: index={idx} key='{key}' | "
                    f"action_features={features}"
                )
                return idx
        fallback = len(features) - 1
        self.logger.warning(
            f"[gripper_sm] Key '{gripper_action_key}' not found in action_features={features}. "
            f"Falling back to last axis (index={fallback})."
        )
        return fallback

    # ── LIFT_RETRY kinematics helpers ─────────────────────────────────────────

    def _init_lift_kinematics(self) -> None:
        """Load SO-101 FK/IK solver for LIFT_RETRY Cartesian Z lifting."""
        from pathlib import Path as _Path
        cfg: SmartRobotClientConfig = self.config

        if cfg.empty_grasp_lift_urdf_path:
            urdf_path = cfg.empty_grasp_lift_urdf_path
        else:
            # Default: mesh-stripped so101_kinematics.urdf alongside SO-follower driver
            urdf_path = str(
                _Path(__file__).parent.parent / "robots" / "so_follower" / "so101_kinematics.urdf"
            )

        if not _Path(urdf_path).exists():
            self.logger.warning(
                f"[gripper_sm] LIFT_RETRY: URDF not found at '{urdf_path}'. "
                "Set --empty_grasp_lift_urdf_path or create so101_kinematics.urdf. "
                "LIFT_RETRY will fall back to RECOVERY."
            )
            return

        try:
            from lerobot.model.kinematics import RobotKinematics
            self._lift_kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            )
            self.logger.info(
                f"[gripper_sm] LIFT_RETRY kinematics loaded: urdf='{urdf_path}' | "
                f"joints={self._lift_kinematics.joint_names} | "
                f"delta_xyz={cfg.empty_grasp_lift_delta_xyz_m}m"
            )
        except Exception as exc:
            self.logger.warning(
                f"[gripper_sm] LIFT_RETRY: kinematics init failed: {exc} "
                "— LIFT_RETRY will fall back to RECOVERY"
            )

    def _capture_home_ee(self) -> None:
        """Compute and store the home end-effector position via FK.

        Called once at init when task_done_home_check_mode='ee'.  Uses the
        already-captured _home_positions joint angles as input to FK.  On any
        failure (missing joints, FK error) _home_ee_pos is left as None and
        _is_at_home_ee() will fall back to joint-mode.
        """
        if self._lift_kinematics is None or self._home_positions is None:
            self.logger.warning(
                "[gripper_sm] EE home check: FK solver or home joint positions unavailable "
                "— 'ee' mode will fall back to 'joint' check"
            )
            return
        try:
            home_joints = np.array(
                [self._home_positions[f"{jn}.pos"] for jn in self._lift_kinematics.joint_names]
            )
        except KeyError as exc:
            self.logger.warning(
                f"[gripper_sm] EE home check: joint key {exc} missing from home snapshot "
                "— 'ee' mode will fall back to 'joint' check"
            )
            return
        try:
            T = self._lift_kinematics.forward_kinematics(home_joints)
            self._home_ee_pos = T[0:3, 3].copy()
            self.logger.info(
                f"[gripper_sm] Home EE captured (FK): "
                f"xyz=[{self._home_ee_pos[0]*100:.1f}, {self._home_ee_pos[1]*100:.1f}, "
                f"{self._home_ee_pos[2]*100:.1f}] cm"
            )
        except Exception as exc:
            self.logger.warning(
                f"[gripper_sm] EE home check: FK failed: {exc} "
                "— 'ee' mode will fall back to 'joint' check"
            )

    def _recapture_home_positions(self) -> None:
        """Re-read current joint positions as the home reference.

        Called at control_loop() start (after start_barrier.wait()), ensuring
        the home reference reflects where the operator has actually positioned
        the robot — not the arbitrary pose at __init__ (connect) time.

        Updates both _home_positions (joint-mode check + recovery trajectory
        target) and _home_ee_pos (EE-mode check) atomically.
        """
        try:
            raw = self.robot.bus.sync_read("Present_Position")
            self._home_positions = {f"{m}.pos": float(v) for m, v in raw.items()}
            self.logger.info(
                f"[gripper_sm] Home position re-captured at loop start: {self._home_positions}"
            )
        except Exception as exc:
            self.logger.warning(
                f"[gripper_sm] Home re-capture failed: {exc} — keeping __init__ snapshot"
            )
            return

        if self.config.task_done_home_check_mode == "ee":
            self._capture_home_ee()

    def _build_lift_trajectory(self) -> list[dict[str, float]] | None:
        """FK(current joints) → +dZ → IK → linear trajectory + gripper open.

        Returns a list of position dicts (one per step) or None on failure.
        """
        cfg: SmartRobotClientConfig = self.config

        if self._lift_kinematics is None:
            return None

        n = cfg.empty_grasp_lift_steps
        dx, dy, dz = cfg.empty_grasp_lift_delta_xyz_m

        try:
            current_raw = self.robot.bus.sync_read("Present_Position")
            current = {f"{m}.pos": float(v) for m, v in current_raw.items()}
        except Exception as exc:
            self.logger.warning(f"[gripper_sm] LIFT_RETRY: position read failed: {exc}")
            return None

        ik_joints = self._lift_kinematics.joint_names
        try:
            current_joints = np.array([current[f"{jn}.pos"] for jn in ik_joints])
        except KeyError as exc:
            self.logger.warning(f"[gripper_sm] LIFT_RETRY: joint key missing: {exc}")
            return None

        T_current = self._lift_kinematics.forward_kinematics(current_joints)
        T_target = T_current.copy()
        T_target[0, 3] += dx
        T_target[1, 3] += dy
        T_target[2, 3] += dz
        target_joints = self._lift_kinematics.inverse_kinematics(
            current_joints, T_target,
            position_weight=1.0,
            orientation_weight=0.01,
        )

        target = dict(current)
        for i, jn in enumerate(ik_joints):
            target[f"{jn}.pos"] = float(target_joints[i])
        target["gripper.pos"] = cfg.empty_grasp_lift_gripper_open_deg

        traj = [
            {k: current[k] + (i / n) * (target.get(k, current[k]) - current[k])
             for k in current}
            for i in range(1, n + 1)
        ]

        _axes = " | ".join(
            f"{ax}: {T_current[i,3]:.3f}→{T_target[i,3]:.3f}m"
            for i, (ax, d) in enumerate(zip("xyz", (dx, dy, dz)))
            if d != 0.0
        ) or "no translation delta"
        self.logger.info(
            f"[gripper_sm] LIFT_RETRY trajectory: {n} steps | {_axes} | "
            f"gripper: {current.get('gripper.pos', '?'):.1f}→{cfg.empty_grasp_lift_gripper_open_deg}°"
        )
        return traj

    def _reset_interpolator_to_lift_end(self) -> None:
        """Anchor the interpolator's prev state to the lift-trajectory endpoint.

        After the lift trajectory completes (and optional warmup), the interpolator
        still holds the last policy action from *before* LIFT_RETRY fired (a closing
        action).  When the first post-lift policy chunk arrives, the interpolator
        blends from that stale closing position to the new target — producing a brief
        spurious closing motion that looks like a second grasp attempt.

        Setting _prev to the actual lift endpoint (last step of _lift_traj) makes the
        first post-lift chunk interpolate from where the arm physically is.
        Same pattern as recovery's home-position reset.
        """
        if self._lift_traj is None:
            return
        import torch as _torch
        _lift_end = self._lift_traj[-1]
        self.interpolator._prev = _torch.tensor(
            [_lift_end.get(k, 0.0) for k in self.robot.action_features],
            dtype=_torch.float32,
        )
        self.logger.debug(
            "[gripper_sm] interpolator._prev reset to lift endpoint "
            f"(gripper={_lift_end.get('gripper.pos', '?'):.1f}°)"
        )

    # ── REWIND_RETRY: reverse action-history replay ────────────────────────────

    def _build_rewind_trajectory(self) -> list[dict[str, float]] | None:
        """Reverse the action history ring buffer into a rewind trajectory.

        Steps are taken from _action_history (oldest→newest), reversed so the arm
        retraces its path backward to the position it was in rewind_steps ago.
        The gripper axis is overridden to empty_grasp_rewind_gripper_open_deg on
        every step so the gripper opens fully during the backward motion.

        When empty_grasp_rewind_min_displacement_deg > 0, the trajectory is
        extended backward until the cumulative joint-space L2 displacement of arm
        joints (gripper excluded) reaches the threshold, or empty_grasp_rewind_steps
        is exhausted, or the buffer runs out — whichever comes first.  The
        displacement and step count are logged so users can tune the threshold.

        Returns None when the history buffer is empty or not allocated.
        """
        if self._action_history is None or len(self._action_history) == 0:
            return None

        cfg: SmartRobotClientConfig = self.config
        # Snapshot the ring buffer then reverse: newest entry first → arm moves
        # from its current position backward toward the oldest recorded position.
        #
        # Skip the newest entry: in the REWIND_RETRY iteration control_loop_action()
        # has already appended A_N to history before the SM decision fires.
        # Including A_N as traj[0] would re-send the current commanded position,
        # holding the arm in place for one extra control interval before it moves.
        history = list(self._action_history)          # [oldest, …, newest]
        if len(history) <= 1:
            return None
        history = history[:-1]                         # drop just-appended A_N
        reversed_steps = list(reversed(history))       # [second-newest, …, oldest]

        max_steps = cfg.empty_grasp_rewind_steps
        min_disp = cfg.empty_grasp_rewind_min_displacement_deg
        gripper_key = f"{cfg.gripper_action_key}.pos"

        # Determine arm joint keys (exclude all gripper fields) from the first step.
        _gripper_prefix = f"{cfg.gripper_action_key}."
        arm_keys = [k for k in reversed_steps[0] if not k.startswith(_gripper_prefix)]

        if min_disp > 0.0 and len(reversed_steps) >= 2:
            # Scan backward until the net arm L2 displacement from the trajectory
            # start (current position) to the current candidate endpoint is >= min_disp.
            # Net displacement = ||endpoint - start||₂, NOT cumulative path length.
            # This ensures the arm actually ends up far from where it started rather
            # than oscillating in place (hover phases accumulate path length without
            # meaningful net movement, fooling a cumulative check).
            start = reversed_steps[0]
            net_disp = 0.0
            cutoff = min(len(reversed_steps), max_steps)  # default: full budget
            for i in range(1, min(len(reversed_steps), max_steps)):
                curr = reversed_steps[i]
                net_disp = sum(
                    (curr.get(k, 0.0) - start.get(k, 0.0)) ** 2 for k in arm_keys
                ) ** 0.5
                if net_disp >= min_disp:
                    cutoff = i + 1
                    break
            # Compute final net disp at chosen cutoff for logging (in case the loop
            # exhausted without reaching the target).
            end = reversed_steps[cutoff - 1]
            net_disp_final = sum(
                (end.get(k, 0.0) - start.get(k, 0.0)) ** 2 for k in arm_keys
            ) ** 0.5
            reached = net_disp_final >= min_disp
            selected = reversed_steps[:cutoff]
            _reach_tag = "reached" if reached else f"buf exhausted, target={min_disp:.1f}"
            self.logger.info(
                f"[gripper_sm] REWIND_RETRY trajectory: {len(selected)} steps "
                f"(max={max_steps}, buf={len(reversed_steps)}) | "
                f"net_disp={net_disp_final:.2f} ({_reach_tag}) | "
                f"gripper→{cfg.empty_grasp_rewind_gripper_open_deg}°"
            )
        else:
            # Original behaviour: use all history up to max_steps.
            selected = reversed_steps[:max_steps]
            self.logger.info(
                f"[gripper_sm] REWIND_RETRY trajectory: {len(selected)} steps | "
                f"gripper→{cfg.empty_grasp_rewind_gripper_open_deg}° | "
                f"history_len={len(reversed_steps)}"
            )

        traj: list[dict[str, float]] = []
        for step in selected:
            s = dict(step)
            if gripper_key in s:
                s[gripper_key] = cfg.empty_grasp_rewind_gripper_open_deg
            traj.append(s)
        return traj

    def _execute_rewind_retry(self) -> None:
        """Handle empty_grasp after LIFT_RETRY by rewinding action history.

        Non-blocking: builds the reverse-replay trajectory from _action_history
        and stores it in _rewind_traj. The control_loop() executes one step per
        iteration until _rewind_remaining reaches zero, then arms must_go=True
        to trigger a fresh inference from the rewound position.

        Falls back to immediate re-inference when the history buffer is empty.
        """
        self._rewind_retry_count += 1
        _hist_len = len(self._action_history) if self._action_history else 0
        self.logger.warning(
            f"{_CY}[gripper_sm] REWIND_RETRY triggered{_CX} — "
            f"replaying {_hist_len} steps backward then re-inferring"
        )

        # Drain queue, increment generation, reset SM phase state.
        # must_go will be cleared below once the trajectory is armed so that the
        # background sender's context-filling obs (must_go=False) do not trigger
        # premature inference during rewind.  must_go=True is re-armed after drain().
        self._force_reinference(reason="rewind_retry")

        traj = self._build_rewind_trajectory()
        # Clear history after building the trajectory so a subsequent REWIND_RETRY
        # in the same episode does not replay stale pre-rewind actions.
        if self._action_history is not None:
            self._action_history.clear()

        if traj is None:
            self.logger.warning(
                "[gripper_sm] REWIND_RETRY: history buffer empty — "
                "falling back to immediate re-inference"
            )
            self.must_go.set()
            return

        self._rewind_traj = traj
        self._rewind_remaining = len(traj)
        # Clear must_go now that the trajectory is armed: rewind obs are sent via
        # the background sender (must_go=False).  The final must_go=True is armed
        # after drain() + warmup completes, triggering a clean post-rewind inference.
        self.must_go.clear()
        # Start background obs sender for context-filling during the rewind trajectory.
        self._bg_obs_sender.start()

    def _reset_interpolator_to_rewind_end(self) -> None:
        """Anchor the interpolator's prev state to the rewind-trajectory endpoint.

        After the rewind trajectory completes the interpolator still holds the last
        policy action from before REWIND_RETRY fired.  Setting _prev to the actual
        rewind endpoint makes the first post-rewind policy chunk interpolate from
        where the arm physically is.  Same pattern as _reset_interpolator_to_lift_end.
        """
        if self._rewind_traj is None:
            return
        import torch as _torch
        _rewind_end = self._rewind_traj[-1]
        self.interpolator._prev = _torch.tensor(
            [_rewind_end.get(k, 0.0) for k in self.robot.action_features],
            dtype=_torch.float32,
        )
        self.logger.debug(
            "[gripper_sm] interpolator._prev reset to rewind endpoint "
            f"(gripper={_rewind_end.get('gripper.pos', '?'):.1f}°)"
        )

    def _execute_lift_retry(self) -> None:
        """Lift arm Cartesian Z + open gripper, then reinfer.

        Non-blocking: FK/IK trajectory stored in _lift_traj / _lift_remaining;
        control_loop() executes one step per iteration.
        Falls back to _execute_recovery() if trajectory build fails.
        NOTE: does NOT reset _failure_count — the count persists so a second
        empty_grasp after LIFT_RETRY correctly escalates to RECOVERY.
        """
        self.logger.warning(f"{_CY}[gripper_sm] LIFT_RETRY executing{_CX}")

        traj = self._build_lift_trajectory()
        if traj is None:
            self.logger.warning(
                "[gripper_sm] LIFT_RETRY: trajectory build failed — falling back to RECOVERY"
            )
            self._execute_recovery()
            return

        # Drain queue, bump generation, reset SM phase state.
        # Clear must_go so obs are not sent during the lift trajectory.
        self._force_reinference(reason="lift_retry")
        self.must_go.clear()

        self._lift_traj = traj
        self._lift_remaining = len(traj)
        self.logger.info(f"[gripper_sm] LIFT_RETRY armed: {self._lift_remaining} steps")
        # Start background obs sender for context-filling during the lift trajectory.
        self._bg_obs_sender.start()

    # ── Direct bus feedback read ───────────────────────────────────────────────

    def _read_gripper_feedback(self) -> dict[str, float]:
        """Read motor state registers directly via bus.sync_read().

        Completely independent of the inference pipeline:
          • Does NOT call get_observation() (no camera capture, no pos remapping)
          • Does NOT require record_motor_state in robot config
          • Does NOT affect observation_features / action_features / lerobot_features

        Returns a slim dict keyed as "{motor_name}.{register_suffix}", e.g.:
            {"gripper.load": 123.4, "gripper.current": 56.7, "gripper.pos": 8.9}

        Cost: one sync_read per register (~1–2ms each); negligible at 30fps.
        """
        cfg: SmartRobotClientConfig = self.config
        feedback: dict[str, float] = {}

        for register, suffix in cfg.gripper_sm_feedback_registers.items():
            try:
                vals = self.robot.bus.sync_read(register, cfg.gripper_sm_motor_names)
                for motor, val in vals.items():
                    feedback[f"{motor}.{suffix}"] = float(val)
            except Exception as exc:
                self.logger.warning(
                    f"[gripper_sm] bus.sync_read('{register}', {cfg.gripper_sm_motor_names}) "
                    f"failed: {exc} — substituting 0.0"
                )
                for motor in cfg.gripper_sm_motor_names:
                    feedback.setdefault(f"{motor}.{suffix}", 0.0)

        # Present_Position for gripper.pos — needed for empty-grasp detection
        # (pos ≤ pos_empty_threshold → gripper closed on air).
        try:
            pos_vals = self.robot.bus.sync_read("Present_Position", cfg.gripper_sm_motor_names)
            for motor, val in pos_vals.items():
                feedback[f"{motor}.pos"] = float(val)
        except Exception as exc:
            self.logger.debug(
                f"[gripper_sm] bus.sync_read('Present_Position') failed: {exc}"
            )
            for motor in cfg.gripper_sm_motor_names:
                feedback.setdefault(f"{motor}.pos", 0.0)

        return feedback

    # ── Home-position check ───────────────────────────────────────────────────

    def _is_at_home(self) -> bool:
        """Dispatcher: routes to joint-space or EE Cartesian check based on config."""
        if self.config.task_done_home_check_mode == "ee":
            return self._is_at_home_ee()
        return self._is_at_home_joint()

    def _check_gripper_at_home(self, current: dict[str, float]) -> tuple[bool, str]:
        """Check whether gripper position matches home snapshot within its own tolerance.

        Returns (ok, log_fragment).  Called only when task_done_home_check_gripper=True.
        Uses task_done_home_gripper_tolerance_deg (separate, looser than arm tolerance).
        """
        gripper_key = "gripper.pos"
        home_val = (self._home_positions or {}).get(gripper_key)
        if home_val is None:
            return True, ""   # no home snapshot for gripper → skip silently
        cur_val = current.get(gripper_key)
        if cur_val is None:
            return True, ""   # no current reading → skip silently
        dev = abs(cur_val - home_val)
        g_tol = self.config.task_done_home_gripper_tolerance_deg
        ok = dev <= g_tol
        frag = f"  gripper:{dev:+.1f}° ({'✓' if ok else f'✗ tol=±{g_tol:.0f}°'})"
        return ok, frag

    def _is_at_home_joint(self) -> bool:
        """True when every arm joint is within task_done_home_tolerance of _home_positions.

        Gripper is excluded from the arm-joint check by default
        (task_done_home_check_gripper=False) so that a closing gripper during the
        SM TASK_DONE path does not block home detection when home was captured with
        the gripper open.  Set task_done_home_check_gripper=True to re-include it.

        Returns False when:
          - _home_positions is None (capture failed at connect time)
          - bus read fails
          - any arm joint exceeds the tolerance
          - gripper check enabled and gripper exceeds its own tolerance
        """
        if self._home_positions is None:
            return False
        try:
            raw = self.robot.bus.sync_read("Present_Position")
            current = {f"{m}.pos": float(v) for m, v in raw.items()}
        except Exception as exc:
            self.logger.debug(f"[gripper_sm] _is_at_home_joint() bus read failed: {exc}")
            return False
        tol = self.config.task_done_home_tolerance
        deviations: dict[str, float] = {}
        for key, home_val in self._home_positions.items():
            if key == "gripper.pos":
                continue   # gripper handled separately below
            cur_val = current.get(key)
            deviations[key] = abs(cur_val - home_val) if cur_val is not None else float("inf")
        at_home = all(dev <= tol for dev in deviations.values())
        _dev_str = "  ".join(
            f"{k.replace('.pos', '')}:{deviations[k]:+.1f}"
            for k in deviations
        )
        _gripper_str = ""
        if self.config.task_done_home_check_gripper:
            g_ok, _gripper_str = self._check_gripper_at_home(current)
            at_home = at_home and g_ok
        if at_home:
            self.logger.info(
                f"[gripper_sm] _is_at_home ✓ (joint) | tol=±{tol}°  {_dev_str}{_gripper_str}"
            )
        else:
            if deviations:
                _worst = max(deviations, key=lambda k: deviations[k])
                self.logger.debug(
                    f"[gripper_sm] _is_at_home ✗ (joint) | tol=±{tol}°  {_dev_str}{_gripper_str}"
                    f"  (worst: {_worst.replace('.pos','')} Δ={deviations[_worst]:.1f})"
                )
            else:
                self.logger.debug(
                    f"[gripper_sm] _is_at_home ✗ (joint) | tol=±{tol}°  {_dev_str}{_gripper_str}"
                )
        return at_home

    def _is_at_home_ee(self) -> bool:
        """True when end-effector Cartesian distance to home is within tolerance.

        Falls back to joint-mode if _home_ee_pos or _lift_kinematics is unavailable.
        Bus read is shared with FK input so cost ≈ sync_read (~2 ms) + FK (< 0.1 ms).
        """
        if self._home_ee_pos is None or self._lift_kinematics is None:
            self.logger.warning(
                "[gripper_sm] EE home check unavailable — falling back to joint check"
            )
            return self._is_at_home_joint()
        try:
            raw = self.robot.bus.sync_read("Present_Position")
        except Exception as exc:
            self.logger.debug(f"[gripper_sm] _is_at_home_ee() bus read failed: {exc}")
            return False
        try:
            joints = np.array(
                [float(raw[jn]) for jn in self._lift_kinematics.joint_names]
            )
        except KeyError as exc:
            self.logger.debug(f"[gripper_sm] _is_at_home_ee() missing joint {exc}")
            return False
        T = self._lift_kinematics.forward_kinematics(joints)
        ee_pos = T[0:3, 3]
        diff = ee_pos - self._home_ee_pos
        dist = float(np.linalg.norm(diff))
        tol = self.config.task_done_home_ee_tolerance_m
        l2_ok = dist <= tol

        # Optional per-axis check: each axis must be within its own tolerance.
        xyz_tols = self.config.task_done_home_ee_tolerance_xyz_m
        if len(xyz_tols) == 3:
            abs_diff = np.abs(diff)
            xyz_ok = bool(np.all(abs_diff <= np.array(xyz_tols)))
            at_home = l2_ok and xyz_ok
            _xyz_str = (
                f" xyz_diff=[{abs_diff[0]*100:.1f},{abs_diff[1]*100:.1f},{abs_diff[2]*100:.1f}]cm"
                f" xyz_tol=[{xyz_tols[0]*100:.0f},{xyz_tols[1]*100:.0f},{xyz_tols[2]*100:.0f}]cm"
                f" xyz={'✓' if xyz_ok else '✗'}"
            )
        else:
            at_home = l2_ok
            _xyz_str = ""

        # Optional gripper check: EE position does not include gripper joint, so
        # check it separately when task_done_home_check_gripper=True.
        _gripper_str = ""
        if self.config.task_done_home_check_gripper:
            current = {f"{m}.pos": float(v) for m, v in raw.items()}
            g_ok, _gripper_str = self._check_gripper_at_home(current)
            at_home = at_home and g_ok

        _ee_str = (
            f"ee=[{ee_pos[0]*100:.1f},{ee_pos[1]*100:.1f},{ee_pos[2]*100:.1f}]cm "
            f"home=[{self._home_ee_pos[0]*100:.1f},{self._home_ee_pos[1]*100:.1f},"
            f"{self._home_ee_pos[2]*100:.1f}]cm dist={dist*100:.1f}cm tol=±{tol*100:.0f}cm"
            f"{_xyz_str}{_gripper_str}"
        )
        if at_home:
            self.logger.info(f"[gripper_sm] _is_at_home ✓ (ee) | {_ee_str}")
        else:
            self.logger.debug(f"[gripper_sm] _is_at_home ✗ (ee) | {_ee_str}")
        return at_home

    # ── Force re-inference (RTC-safe) ─────────────────────────────────────────

    def _force_reinference(self, reason: str = "", task_suffix: str = "") -> None:
        """Drain the action queue and trigger must_go=True on the next observation.

        Correctly increments _action_generation so receive_actions() discards any
        in-flight chunk that completed before the reinfer request (prevents stale
        actions from re-populating the freshly drained queue).

        _orig_buf is cleared: leftover=None tells the server to generate a fresh
        chunk rather than continuing the aborted trajectory prefix.
        """
        self.logger.warning(f"{_CY}[gripper_sm] FORCE-REINFER{_CX} | reason='{reason}'")

        # Increment generation BEFORE draining so receive_actions() can detect
        # and discard any GetActions() result that was in-flight.
        self._action_generation += 1

        with self.action_queue_lock:
            self.action_queue = Queue()

        # Clear RTC leftover: reinfer starts fresh, not from an aborted trajectory.
        self._orig_buf.clear()

        # Clear round-trip timing buffers (keyed by timestep, now stale).
        with self._send_wall_buf_lock:
            self._send_wall_buf.clear()
            self._obs_infer_delay_buf.clear()

        # Reset empty-queue counter so _force_must_go doesn't fire prematurely
        # before the new inference result arrives (~inference_latency ms).
        self._queue_empty_steps = 0

        # Optionally hint to VLA policies what went wrong.
        if task_suffix:
            self._current_task = self.config.task + " " + task_suffix
            self.logger.info(f"[gripper_sm] Task hint updated: '{self._current_task}'")

        # Arm must_go: queue empty + event set → next control_loop_observation()
        # sends must_go=True → server bypasses similarity/timestep filters.
        self.must_go.set()

        if self._gripper_monitor is not None:
            self._gripper_monitor.reset()
        self._at_home_confirm_count = 0

    def _on_retry_triggered(self, decision: "GripperDecision") -> None:
        """Hook called when a retry decision fires (RECOVERY / LIFT_RETRY / REWIND_RETRY).

        No-op by default.  Subclasses (e.g. MultiCandSO101Client) override this to
        store the arm state at failure for anti-repeat candidate scoring.
        Called BEFORE the recovery/lift/rewind trajectory begins, so
        _last_feedback_state and action queue still reflect the failure-site state.
        """

    def _execute_recovery(self) -> None:
        """Handle a repeated failure by returning to home position then re-inferring.

        Non-blocking: builds a linear interpolation trajectory from current position
        to ``_home_positions`` (captured at connect time) and stores it in
        ``_recovery_traj``.  The control_loop() executes one step per iteration
        until ``_recovery_remaining`` reaches zero, then arms must_go=True to
        trigger a fresh inference from the home position.

        Falls back to immediate re-inference when:
          - recovery_return_to_home=False (config)
          - _home_positions is None (capture failed at connect time)
          - bus.sync_read() fails at recovery start
        """
        self._recovery_count += 1
        self.logger.warning(f"{_CR}[gripper_sm] RECOVERY triggered{_CX} — returning to home position")

        # Drain queue, increment generation, reset SM phase state.
        # must_go is set by _force_reinference but we'll clear it below so obs are
        # not sent during the home-return trajectory.
        self._force_reinference(reason="recovery")
        self.must_go.clear()

        if not self.config.recovery_return_to_home or self._home_positions is None:
            self.logger.info(
                "[gripper_sm] recovery_return_to_home disabled or home unavailable "
                "— reinferring from current position"
            )
            self.must_go.set()
            return

        try:
            current_raw = self.robot.bus.sync_read("Present_Position")
            current = {f"{m}.pos": float(v) for m, v in current_raw.items()}
        except Exception as exc:
            self.logger.warning(
                f"[gripper_sm] Could not read current position for recovery: {exc} "
                "— reinferring from current position"
            )
            self.must_go.set()
            return

        n = self.config.recovery_home_steps
        # Use .get(k, current[k]) so that any motor absent from _home_positions
        # (e.g. a motor that was not responding at connect time) stays at its
        # current position rather than raising a KeyError.
        _missing = set(current) - set(self._home_positions)
        if _missing:
            self.logger.warning(
                f"[gripper_sm] Recovery: motors {_missing} absent from home snapshot — "
                "holding them at current position"
            )
        self._recovery_traj = [
            {k: current[k] + (i / n) * (self._home_positions.get(k, current[k]) - current[k])
             for k in current}
            for i in range(1, n + 1)
        ]
        self._recovery_remaining = n
        self.logger.info(
            f"[gripper_sm] Recovery trajectory built: {n} steps to home | "
            f"current≈{list(current.values())} → home≈{list(self._home_positions.values())}"
        )
        # Start background obs sender: captures camera+joints each trajectory step
        # and sends them asynchronously so the policy context contains real motion
        # frames instead of static endpoint obs from warmup alone.
        self._bg_obs_sender.start()

    # ── Post-recovery action smoothing ────────────────────────────────────────

    def control_loop_action(self, verbose: bool = False) -> Any:
        """Execute one interpolated action step, with post-recovery velocity clamping.

        When _recovery_smooth_remaining > 0 (first N *policy actions* after recovery
        warmup), each commanded position is clamped to ±recovery_smooth_max_delta
        relative to the previously sent position.  This prevents violent motion when
        the first policy chunk arrives with large displacements from home.

        The window is counted in policy actions (one decrement per dequeue), not in
        control sub-steps, so the coverage is independent of interpolation_multiplier.
        All sub-steps of the last smooth policy action are still clamped before
        transitioning to super() — the top-level guard requires needs_new_action() to
        be True so no partial-buffer jump occurs when remaining hits 0 mid-action.

        On the first call after the smooth window is fully exhausted, interpolator._prev
        is synced to the last actually-sent (clamped) position so the next add() does
        not jump from the raw policy action to the new target.

        Outside the smoothing window this delegates to super().control_loop_action().

        Every non-None result is appended to _action_history for REWIND_RETRY.
        """
        # Transition to normal control only once smooth window is consumed AND the
        # last smooth action's interpolation buffer is fully drained.
        if self._recovery_smooth_remaining <= 0 and self.interpolator.needs_new_action():
            # One-time sync: align interpolator anchor to the last physically-sent
            # (clamped) position so the first unclamped add() interpolates from where
            # the arm actually is, not from the raw policy action value.
            # _recovery_smooth_prev is set to None after sync to guard against
            # repeated syncs on subsequent calls.
            if self._recovery_smooth_prev is not None:
                import torch as _torch
                _clamped_tensor = _torch.tensor(
                    [self._recovery_smooth_prev.get(k, 0.0)
                     for k in self.robot.action_features],
                    dtype=_torch.float32,
                )
                self.interpolator._prev = _clamped_tensor
                self._recovery_smooth_prev = None
                self.logger.info(
                    f"{_CG}[gripper_sm] Post-recovery smoothing complete{_CX} — "
                    "interpolator synced to sent position"
                )
            result = super().control_loop_action(verbose)
        else:
            # Smooth path: dequeue new policy action when buffer exhausted.
            # Decrement per dequeue (per policy action) so recovery_smooth_steps
            # covers exactly that many actions regardless of interpolation_multiplier.
            if self.interpolator.needs_new_action():
                try:
                    with self.action_queue_lock:
                        self.action_queue_size.append(self.action_queue.qsize())
                        timed_action = self.action_queue.get_nowait()
                        self._orig_buf.pop(timed_action.get_timestep(), None)
                except Exception:
                    return None
                self.interpolator.add(timed_action.get_action().cpu())
                with self.latest_action_lock:
                    self.latest_action = timed_action.get_timestep()
                self._recovery_smooth_remaining -= 1

            action_tensor = self.interpolator.get()
            if action_tensor is None:
                return None

            raw_action = self._action_tensor_to_action_dict(action_tensor)

            # Clamp each axis: at most ±max_delta from previous sent position.
            # _recovery_smooth_prev is initialised to the endpoint position at each
            # arming site (home / lift endpoint / rewind endpoint) so the ramp starts
            # from the correct physical pose.
            max_delta = self.config.recovery_smooth_max_delta
            if self._recovery_smooth_prev is not None:
                clamped: dict[str, float] = {}
                for key, target in raw_action.items():
                    prev = self._recovery_smooth_prev.get(key, target)
                    diff = target - prev
                    diff = max(-max_delta, min(max_delta, diff))
                    clamped[key] = prev + diff
            else:
                clamped = dict(raw_action)

            result = self.robot.send_action(clamped)
            self._recovery_smooth_prev = clamped

        # Append every executed action to the ring buffer for REWIND_RETRY.
        if result is not None and self._action_history is not None:
            self._action_history.append(dict(result))
        return result

    # ── Timing overrides ──────────────────────────────────────────────────────

    def enable_timing(self, output_dir: str) -> None:
        """Extend base timing with a dedicated SM event recorder."""
        super().enable_timing(output_dir)
        self._gripper_sm_recorder = TimingRecorder(output_dir, "gripper_sm_events")
        self.logger.info(f"[gripper_sm] SM event recorder enabled → {output_dir}")

    def _compute_episode_stats(self) -> dict:
        """Compute per-episode retry/success stats from self._ep_records.

        Returns a dict with keys matching the sim_smart summary stats:
          total_episodes, overall_sr, eps_with_retry, eps_no_retry,
          sr_with_retry, sr_no_retry, success_after_retry, rescue_rate, sr_lift.
        """
        recs = self._ep_records
        if not recs:
            return {}
        total = len(recs)
        overall_sr      = sum(r["success"] for r in recs) / total
        with_retry      = [r for r in recs if r["retries"] > 0]
        no_retry        = [r for r in recs if r["retries"] == 0]
        sar             = sum(1 for r in with_retry if r["success"])  # success_after_retry
        sr_with         = (sum(r["success"] for r in with_retry) / len(with_retry)
                           if with_retry else float("nan"))
        sr_no           = (sum(r["success"] for r in no_retry)   / len(no_retry)
                           if no_retry   else float("nan"))
        rescue          = sar / len(with_retry) if with_retry else float("nan")
        sr_lift         = sar / total
        total_retries   = sum(r["retries"] for r in recs)
        return dict(
            total_episodes   = total,
            overall_sr       = overall_sr,
            total_retries    = total_retries,
            eps_with_retry   = len(with_retry),
            eps_no_retry     = len(no_retry),
            sr_with_retry    = sr_with,
            sr_no_retry      = sr_no,
            success_after_retry = sar,
            rescue_rate      = rescue,
            sr_lift          = sr_lift,
        )

    def _write_timing_summary_txt(
        self,
        output_dir: "Path",
        event_counts: dict[str, int],
        n_fail: int,
    ) -> None:
        """Write a human-readable consolidated timing summary to timing_summary.txt.

        Reads the per-recorder *_summary.json files already saved by save_timing()
        and formats them as the same percentile tables printed to the log, then
        appends the gripper SM event breakdown including recovery count.
        """
        import json as _json
        from pathlib import Path as _Path

        output_dir = _Path(output_dir)
        txt_lines: list[str] = []

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

        for prefix in (
            "client_obs_sent",
            "client_chunk_recv",
            "client_chunk_action",
            "client_aggregate",
            "gripper_sm_events",
        ):
            tbl = _fmt_table(output_dir / f"{prefix}_summary.json")
            if tbl:
                txt_lines.extend(tbl)
                txt_lines.append("")

        txt_lines.append(
            f"[gripper_sm] SM event breakdown  "
            f"(recovery={self._recovery_count}  lift_retry={self._lift_retry_count}  failures={n_fail}):"
        )
        for et in ("grasp_success", "task_done", "empty_grasp", "slip", "stop"):
            txt_lines.append(f"  {et:<28}: {event_counts.get(et, 0)}")
        txt_lines.append(f"  {'recovery':<28}: {self._recovery_count}")
        txt_lines.append(f"  {'lift_retry':<28}: {self._lift_retry_count}")
        txt_lines.append(f"  {'grasp_success_total':<28}: {self._grasp_success_total}")
        txt_lines.append(f"  {'grasp_success_after_lift_retry':<28}: {self._grasp_success_after_lift_retry}")
        txt_lines.append(f"  {'grasp_success_after_recovery':<28}: {self._grasp_success_after_recovery}")

        # ── Per-episode retry / success stats ──
        ep_stats = self._compute_episode_stats()
        if ep_stats:
            nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731
            txt_lines += [
                "",
                "── Episode Stats (pick-place cycles) ──────────────────────────────",
                f"  {'total_episodes':<30}: {ep_stats['total_episodes']}",
                f"  {'overall_sr':<30}: {nan_fmt(ep_stats['overall_sr'])}",
                f"  {'total_retries':<30}: {ep_stats['total_retries']}",
                f"  {'eps_with_retry':<30}: {ep_stats['eps_with_retry']} / {ep_stats['total_episodes']}",
                f"  {'eps_no_retry':<30}: {ep_stats['eps_no_retry']} / {ep_stats['total_episodes']}",
                f"  {'sr_with_retry':<30}: {nan_fmt(ep_stats['sr_with_retry'])}"
                f"  ← final SR of episodes that needed retry",
                f"  {'sr_no_retry':<30}: {nan_fmt(ep_stats['sr_no_retry'])}"
                f"  ← final SR of clean episodes (no retry)",
                f"  {'success_after_retry':<30}: {ep_stats['success_after_retry']}"
                f"  ← episodes saved by SM",
                f"  {'rescue_rate':<30}: {nan_fmt(ep_stats['rescue_rate'])}"
                f"  ← success_after_retry / eps_with_retry (SM effectiveness)",
                f"  {'sr_lift (SM→no-SM)':<30}: +{ep_stats['sr_lift']:.1%}"
                f"  ← overall SR improvement vs baseline without SM",
            ]

        txt_path = output_dir / "timing_summary.txt"
        txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        self.logger.info(f"[save_timing] Consolidated summary → {txt_path}")

    def save_timing(self) -> None:
        """Save base timing records, SM event JSONL, and event-type breakdown."""
        super().save_timing()
        if self._gripper_sm_recorder is None:
            return
        try:
            self._gripper_sm_recorder.save()
        except BaseException as exc:
            self.logger.warning(f"[save_timing] gripper_sm_recorder.save() raised: {exc}")
        try:
            self._gripper_sm_recorder.log_summary()
        except BaseException as exc:
            self.logger.warning(f"[save_timing] gripper_sm_recorder.log_summary() raised: {exc}")
        _counts: dict[str, int] = {}
        _n_fail: int = 0
        try:
            _counts = self._gripper_sm_recorder.count_by("event_type")
            _total  = sum(_counts.values())
            _n_fail = sum(_counts.get(et, 0) for et in ("empty_grasp", "slip", "stop"))
            lines = [f"[gripper_sm] {_total} SM events total "
                     f"(success={_counts.get('grasp_success', 0)}  "
                     f"task_done={_counts.get('task_done', 0)}  "
                     f"recovery={self._recovery_count}  "
                     f"lift_retry={self._lift_retry_count}  "
                     f"failures={_n_fail}):"]
            for et in ("grasp_success", "task_done", "empty_grasp", "slip", "stop"):
                lines.append(f"  {et:<28}: {_counts.get(et, 0)}")
            lines.append(f"  {'recovery':<28}: {self._recovery_count}")
            lines.append(f"  {'lift_retry':<28}: {self._lift_retry_count}")
            lines.append(f"  {'grasp_success_total':<28}: {self._grasp_success_total}")
            lines.append(f"  {'grasp_success_after_lift_retry':<28}: {self._grasp_success_after_lift_retry}")
            lines.append(f"  {'grasp_success_after_recovery':<28}: {self._grasp_success_after_recovery}")
            self.logger.info("\n".join(lines))
        except BaseException as exc:
            self.logger.warning(f"[save_timing] SM event breakdown failed: {exc}")
        try:
            self._write_timing_summary_txt(
                self._gripper_sm_recorder.output_dir, _counts, _n_fail
            )
        except BaseException as exc:
            self.logger.warning(f"[save_timing] txt summary write failed: {exc}")

    # ── Main control loop ──────────────────────────────────────────────────────

    def control_loop(self, task: str, verbose: bool = False) -> Any:
        """Run the control loop.

        enable_gripper_sm=False → super().control_loop() directly (zero overhead).
        enable_gripper_sm=True  → inserts _read_gripper_feedback() + SM update
                                   after each action step.
        """
        if not self.config.enable_gripper_sm:
            return super().control_loop(task, verbose)

        self._current_task = task
        self._recovery_remaining = 0        # discard any in-progress recovery from prior episode
        self._recovery_warmup_remaining = 0  # discard warmup from prior episode
        self._recovery_traj = None
        self._lift_remaining = 0            # discard any in-progress lift from prior episode
        self._lift_warmup_remaining = 0
        self._lift_traj = None
        self._rewind_remaining = 0          # discard any in-progress rewind from prior episode
        self._rewind_warmup_remaining = 0
        self._rewind_traj = None
        if self._action_history is not None:
            self._action_history.clear()
        # Stop any in-flight background obs sender from the previous episode so it
        # does not deliver stale obs after the new episode's context has been reset.
        self._bg_obs_sender.drain(timeout=1.0)
        self._last_intervention = ""
        self._reset_loop_state()
        # Full SM reset for new episode: phase/peak/hold/anomaly + failure counter.
        self._gripper_monitor.reset()
        self._gripper_monitor._failure_count = 0
        self._gripper_monitor._lift_retry_attempted = False
        self._gripper_monitor._rewind_retry_attempted = False
        self._at_home_confirm_count = 0
        self.start_barrier.wait()
        if self.config.enable_recapture_home_positions:
            # Re-capture home reference now that the operator has positioned the arm.
            # Overwrites the __init__-time snapshot so _is_at_home() and recovery
            # trajectories target the actual task start pose, not connect-time pose.
            self._recapture_home_positions()
        else:
            self.logger.info(
                f"[gripper_sm] Home position kept from init snapshot "
                f"(enable_recapture_home_positions=False): {self._home_positions}"
            )
        self.logger.info("[smart_robot_client] State-machine control loop starting")

        control_interval = self.interpolator.get_control_interval(self.config.fps)

        # Transition tracking: log _loop_line at INFO only when phase or decision changes.
        _prev_loop_phase:    GraspPhase | None    = None
        _prev_loop_decision: GripperDecision | None = None

        while self.running:
            t_loop = time.perf_counter()

            # ── Non-blocking recovery: execute one trajectory step per iteration ──
            # Background obs sender (_bg_obs_sender) captures camera+joint obs on
            # each step and sends them asynchronously so the policy context contains
            # real motion frames of the home-return trajectory rather than static
            # endpoint frames.  The main thread only calls send_action() + camera
            # capture (~35 ms total) so 10 Hz timing is maintained.
            if self._recovery_remaining > 0:
                idx = len(self._recovery_traj) - self._recovery_remaining
                self.robot.send_action(self._recovery_traj[idx])
                # Capture obs at policy Hz (every interpolation_multiplier sub-steps).
                # At interp×1 this fires every step; at interp×3 every 3rd sub-step,
                # keeping camera capture off the sub-step critical path.
                # latest_action is always incremented (unique timestep per sub-step).
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                    _bg_ts = self.latest_action
                _interp = max(1, self.config.interpolation_multiplier)
                if idx % _interp == 0:
                    _raw = self._capture_raw_obs()
                    self._on_obs_captured(_raw)
                    self._bg_obs_sender.enqueue(_raw, _bg_ts)
                self._recovery_remaining -= 1
                if self._recovery_remaining == 0:
                    self._gripper_monitor._failure_count = 0
                    self._gripper_monitor._lift_retry_attempted = False   # re-arm for next cycle
                    self._gripper_monitor._rewind_retry_attempted = False  # re-arm for next cycle
                    _settle = self.config.recovery_home_settle_time
                    if _settle > 0:
                        self.logger.info(
                            f"[gripper_sm] Recovery: settling at home for {_settle:.2f}s ..."
                        )
                        time.sleep(_settle)
                    # Drain background sender: wait for all trajectory obs to be delivered
                    # to the server before we arm must_go and trigger inference.
                    self.logger.info(
                        f"{_CY}[gripper_sm] Recovery: draining background obs sender{_CX} ..."
                    )
                    _t_drain = time.perf_counter()
                    self._bg_obs_sender.drain()
                    self.logger.info(
                        f"{_CG}[gripper_sm] Recovery: bg obs drain complete{_CX} — "
                        f"{(time.perf_counter() - _t_drain) * 1000:.0f} ms"
                    )
                    if self._gripper_sm_recorder is not None:
                        with self.latest_action_lock:
                            _ts = self.latest_action
                        self._gripper_sm_recorder.add(GripperSMEventRecord(
                            wall_time=time.time(),
                            episode=self._current_episode,
                            timestep=_ts,
                            event_type="recovery_home_ready",
                            phase="HOME",
                            gripper_load=0.0,
                            gripper_pos=0.0,
                            peak_load=0.0,
                            failure_count=0,  # was just reset above
                            queue_size=0,     # queue will be cleared below
                            settle_ms=_settle * 1000,
                        ))
                    # Drain queue and bump generation to discard any actions that
                    # arrived from receive_actions() during the recovery trajectory.
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    # Context warmup: send N obs from home so context window is topped
                    # up with endpoint frames (trajectory obs already in context via
                    # bg sender; warmup mainly fills remaining context window slots).
                    if self.config.recovery_warmup_steps > 0:
                        self._recovery_warmup_remaining = self.config.recovery_warmup_steps
                        self.logger.info(
                            f"{_CG}[gripper_sm] Recovery complete{_CX} — at home, "
                            f"warming up context for {self.config.recovery_warmup_steps} steps"
                        )
                    else:
                        self.must_go.set()
                        self.logger.info(
                            f"{_CG}[gripper_sm] Recovery complete{_CX} — at home position, inference armed"
                        )
                        self.logger.info(
                            f"\033[1;32m[MUST_GO→ARMED]\033[0m RECOVERY (no warmup) "
                            f"gen={self._action_generation}"
                        )
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue

            # ── Context warmup after recovery ─────────────────────────────────────
            # Sends obs from home WITHOUT executing policy actions.  This fills the
            # policy server's context window with consistent home-state frames before
            # the actual must_go inference fires.  Without this, the server context
            # contains [failed_grasp_obs…, home_obs] — an abrupt OOD jump that causes
            # fast erratic motions on the first post-recovery action chunk.
            if self._recovery_warmup_remaining > 0:
                self._recovery_warmup_remaining -= 1
                # Two issues prevented real context flushing in the old warmup:
                #
                # (A) latest_action is frozen during warmup (control_loop_action() is
                #     never reached here).  obs timestep = latest_action+1 is therefore
                #     the same constant for ALL N warmup steps.  The server deduplicates
                #     by timestep → only 1 unique home-state frame enters its context.
                #     Fix: manually advance latest_action +1 per step.
                #
                # (B) _queue_empty_steps keeps incrementing (queue is empty throughout
                #     warmup).  After _MUST_GO_EMPTY_THRESHOLD=10 steps it triggers
                #     force_must_go=True mid-warmup → server infers from stale context
                #     → the returned chunk is the source of the violent post-recovery motion.
                #     Fix: reset _queue_empty_steps = 0 each warmup step.
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                self._queue_empty_steps = 0
                self.control_loop_observation()
                if self._recovery_warmup_remaining == 0:
                    # Context window now dominated by home-state frames.
                    # Drain any stale server-inferred chunks from the warmup period,
                    # then set must_go so the NEXT obs triggers clean inference.
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    self.must_go.set()
                    self.logger.info(
                        f"\033[1;32m[MUST_GO→ARMED]\033[0m RECOVERY warmup complete "
                        f"({self.config.recovery_warmup_steps} steps) "
                        f"gen={self._action_generation}"
                    )
                    # Arm post-recovery velocity smoothing so the first policy chunk
                    # ramps in gently from the home position rather than jumping to
                    # the first target in a single step.
                    if self.config.recovery_smooth_steps > 0 and self._home_positions:
                        self._recovery_smooth_remaining = self.config.recovery_smooth_steps
                        self._recovery_smooth_prev = dict(self._home_positions)
                        # Reset interpolator anchor to home so the first add() blends
                        # from actual home position, not stale pre-recovery position.
                        import torch as _torch
                        _home_tensor = _torch.tensor(
                            [self._home_positions.get(k, 0.0)
                             for k in self.robot.action_features],
                            dtype=_torch.float32,
                        )
                        self.interpolator._prev = _home_tensor
                        self.logger.info(
                            f"{_CG}[gripper_sm] Warmup complete{_CX} — server context refreshed, "
                            f"inference armed | smoothing {self.config.recovery_smooth_steps} steps "
                            f"max_delta={self.config.recovery_smooth_max_delta}°"
                        )
                    else:
                        self.logger.info(
                            f"{_CG}[gripper_sm] Warmup complete{_CX} — server context refreshed, "
                            "inference armed"
                        )
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue

            # ── Non-blocking LIFT_RETRY: execute one trajectory step per iteration ──
            # Background obs sender captures camera+joints each step asynchronously.
            if self._lift_remaining > 0:
                idx = len(self._lift_traj) - self._lift_remaining
                self.robot.send_action(self._lift_traj[idx])
                # Capture at policy Hz only (every interpolation_multiplier sub-steps)
                # so camera capture does not blow the sub-step control interval.
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                    _bg_ts = self.latest_action
                _interp = max(1, self.config.interpolation_multiplier)
                if idx % _interp == 0:
                    _raw = self._capture_raw_obs()
                    self._on_obs_captured(_raw)
                    self._bg_obs_sender.enqueue(_raw, _bg_ts)
                self._lift_remaining -= 1
                if self._lift_remaining == 0:
                    # Drain any actions that arrived during the lift trajectory.
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    # Let the arm settle mechanically before draining the bg sender.
                    _settle = self.config.empty_grasp_lift_settle_time
                    if _settle > 0:
                        self.logger.info(
                            f"[gripper_sm] LIFT_RETRY: settling at lift position for {_settle:.2f}s ..."
                        )
                        time.sleep(_settle)
                    # Drain background sender: ensure all lift obs reach the server
                    # before warmup begins so context has real lift-trajectory frames.
                    self.logger.info(
                        f"{_CY}[gripper_sm] LIFT_RETRY: draining background obs sender{_CX} ..."
                    )
                    _t_drain = time.perf_counter()
                    self._bg_obs_sender.drain()
                    self.logger.info(
                        f"{_CG}[gripper_sm] LIFT_RETRY: bg obs drain complete{_CX} — "
                        f"{(time.perf_counter() - _t_drain) * 1000:.0f} ms"
                    )
                    if self._gripper_sm_recorder is not None:
                        with self.latest_action_lock:
                            _ts = self.latest_action
                        self._gripper_sm_recorder.add(GripperSMEventRecord(
                            wall_time=time.time(),
                            episode=self._current_episode,
                            timestep=_ts,
                            event_type="lift_position_ready",
                            phase="LIFT",
                            gripper_load=0.0,
                            gripper_pos=0.0,
                            peak_load=0.0,
                            failure_count=self._gripper_monitor._failure_count,
                            queue_size=0,  # queue will be cleared below
                            settle_ms=_settle * 1000,
                        ))
                    if self.config.empty_grasp_lift_warmup_steps > 0:
                        self._lift_warmup_remaining = self.config.empty_grasp_lift_warmup_steps
                        self.logger.info(
                            f"{_CY}[gripper_sm] LIFT_RETRY: at lift position, "
                            f"warming up context for {self.config.empty_grasp_lift_warmup_steps} "
                            f"steps{_CX}"
                        )
                    else:
                        self.must_go.set()
                        self.logger.info(
                            f"\033[1;32m[MUST_GO→ARMED]\033[0m LIFT_RETRY (no warmup) "
                            f"gen={self._action_generation}"
                        )
                        self._reset_interpolator_to_lift_end()
                        if self.config.recovery_smooth_steps > 0 and self._lift_traj:
                            self._recovery_smooth_remaining = self.config.recovery_smooth_steps
                            self._recovery_smooth_prev = dict(self._lift_traj[-1])
                            self.logger.info(
                                f"{_CY}[gripper_sm] LIFT_RETRY: at lift position, "
                                f"inference armed | smoothing {self.config.recovery_smooth_steps} steps "
                                f"max_delta={self.config.recovery_smooth_max_delta}°{_CX}"
                            )
                        else:
                            self.logger.info(
                                f"{_CY}[gripper_sm] LIFT_RETRY: at lift position, "
                                f"inference armed{_CX}"
                            )
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue

            # ── Context warmup after LIFT_RETRY ─────────────────────────────────────
            # Sends obs from lift position without executing policy actions.
            # Fills the server context window so the first post-lift chunk starts clean.
            if self._lift_warmup_remaining > 0:
                self._lift_warmup_remaining -= 1
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                self._queue_empty_steps = 0
                self.control_loop_observation()
                if self._lift_warmup_remaining == 0:
                    # Drain server-inferred chunks from the warmup period, then arm must_go.
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    self.must_go.set()
                    self.logger.info(
                        f"\033[1;32m[MUST_GO→ARMED]\033[0m LIFT_RETRY warmup complete "
                        f"({self.config.empty_grasp_lift_warmup_steps} steps) "
                        f"gen={self._action_generation}"
                    )
                    # Reset interpolator anchor to the lift endpoint so the first
                    # post-lift policy action interpolates FROM the actual lift
                    # position, not the stale pre-LIFT_RETRY closing position.
                    # Without this, interpolator._prev holds the last closing action
                    # from before LIFT_RETRY fired, causing the arm to briefly
                    # interpolate through a "closing" state on the first new chunk.
                    self._reset_interpolator_to_lift_end()
                    # Arm velocity smoothing anchored to the lift endpoint so the
                    # first post-lift policy chunk ramps in gently (same mechanism
                    # as post-recovery smoothing, reusing its config parameters).
                    if self.config.recovery_smooth_steps > 0 and self._lift_traj:
                        self._recovery_smooth_remaining = self.config.recovery_smooth_steps
                        self._recovery_smooth_prev = dict(self._lift_traj[-1])
                        self.logger.info(
                            f"{_CY}[gripper_sm] LIFT_RETRY warmup complete{_CX} — "
                            f"inference armed | smoothing {self.config.recovery_smooth_steps} steps "
                            f"max_delta={self.config.recovery_smooth_max_delta}°"
                        )
                    else:
                        self.logger.info(
                            f"{_CY}[gripper_sm] LIFT_RETRY warmup complete{_CX} — "
                            "inference armed"
                        )
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue

            # ── Non-blocking REWIND_RETRY: execute one trajectory step per iteration ──
            # Replays action history in reverse. Background obs sender captures
            # camera+joints each step so the policy context contains the real
            # backward-motion frames.  Main thread only does send_action() + camera
            # capture (~35 ms) so 10 Hz timing is unaffected.
            if self._rewind_remaining > 0:
                idx = len(self._rewind_traj) - self._rewind_remaining
                self.robot.send_action(self._rewind_traj[idx])
                # Capture at policy Hz only (every interpolation_multiplier sub-steps).
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                    _bg_ts = self.latest_action
                _interp = max(1, self.config.interpolation_multiplier)
                if idx % _interp == 0:
                    _raw = self._capture_raw_obs()
                    self._on_obs_captured(_raw)
                    self._bg_obs_sender.enqueue(_raw, _bg_ts)
                self._rewind_remaining -= 1
                if self._rewind_remaining == 0:
                    # Drain any chunks that arrived during rewind (stale — endpoint
                    # obs after drain will trigger the clean inference we want).
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    _settle = self.config.empty_grasp_rewind_settle_time
                    if _settle > 0:
                        self.logger.info(
                            f"[gripper_sm] REWIND_RETRY: settling at rewind endpoint for {_settle:.2f}s ..."
                        )
                        time.sleep(_settle)
                    # Drain background sender: wait for all rewind obs to be delivered
                    # before warmup begins so context has real backward-motion frames.
                    self.logger.info(
                        f"{_CY}[gripper_sm] REWIND_RETRY: draining background obs sender{_CX} ..."
                    )
                    _t_drain = time.perf_counter()
                    self._bg_obs_sender.drain()
                    self.logger.info(
                        f"{_CG}[gripper_sm] REWIND_RETRY: bg obs drain complete{_CX} — "
                        f"{(time.perf_counter() - _t_drain) * 1000:.0f} ms"
                    )
                    if self._gripper_sm_recorder is not None:
                        with self.latest_action_lock:
                            _ts = self.latest_action
                        self._gripper_sm_recorder.add(GripperSMEventRecord(
                            wall_time=time.time(),
                            episode=self._current_episode,
                            timestep=_ts,
                            event_type="rewind_position_ready",
                            phase="REWIND",
                            gripper_load=0.0,
                            gripper_pos=0.0,
                            peak_load=0.0,
                            failure_count=self._gripper_monitor._failure_count,
                            queue_size=0,  # queue will be cleared below
                            settle_ms=_settle * 1000,
                        ))
                    if self.config.empty_grasp_rewind_warmup_steps > 0:
                        self._rewind_warmup_remaining = self.config.empty_grasp_rewind_warmup_steps
                        self.logger.info(
                            f"{_CY}[gripper_sm] REWIND_RETRY: at rewind endpoint, "
                            f"warming up context for {self.config.empty_grasp_rewind_warmup_steps} "
                            f"steps{_CX}"
                        )
                    else:
                        self.must_go.set()
                        self.logger.info(
                            f"\033[1;32m[MUST_GO→ARMED]\033[0m REWIND_RETRY (no warmup) "
                            f"gen={self._action_generation}"
                        )
                        self._reset_interpolator_to_rewind_end()
                        if self.config.recovery_smooth_steps > 0 and self._rewind_traj:
                            self._recovery_smooth_remaining = self.config.recovery_smooth_steps
                            self._recovery_smooth_prev = dict(self._rewind_traj[-1])
                            self.logger.info(
                                f"{_CY}[gripper_sm] REWIND_RETRY: at rewind endpoint, "
                                f"inference armed | smoothing {self.config.recovery_smooth_steps} steps "
                                f"max_delta={self.config.recovery_smooth_max_delta}°{_CX}"
                            )
                        else:
                            self.logger.info(
                                f"{_CY}[gripper_sm] REWIND_RETRY: at rewind endpoint, "
                                f"inference armed{_CX}"
                            )
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue

            # ── Context warmup after REWIND_RETRY ────────────────────────────────────
            # Sends obs from the rewind endpoint without executing policy actions.
            # Fills the server context window so the first post-rewind chunk starts clean.
            if self._rewind_warmup_remaining > 0:
                self._rewind_warmup_remaining -= 1
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                self._queue_empty_steps = 0
                self.control_loop_observation()
                if self._rewind_warmup_remaining == 0:
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    self.must_go.set()
                    self.logger.info(
                        f"\033[1;32m[MUST_GO→ARMED]\033[0m REWIND_RETRY warmup complete "
                        f"({self.config.empty_grasp_rewind_warmup_steps} steps) "
                        f"gen={self._action_generation}"
                    )
                    self._reset_interpolator_to_rewind_end()
                    if self.config.recovery_smooth_steps > 0 and self._rewind_traj:
                        self._recovery_smooth_remaining = self.config.recovery_smooth_steps
                        self._recovery_smooth_prev = dict(self._rewind_traj[-1])
                        self.logger.info(
                            f"{_CY}[gripper_sm] REWIND_RETRY warmup complete{_CX} — "
                            f"inference armed | smoothing {self.config.recovery_smooth_steps} steps "
                            f"max_delta={self.config.recovery_smooth_max_delta}°"
                        )
                    else:
                        self.logger.info(
                            f"{_CY}[gripper_sm] REWIND_RETRY warmup complete{_CX} — "
                            "inference armed"
                        )
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue

            # 1. Execute one (interpolated) action — identical to parent
            if not self.interpolator.needs_new_action() or self.actions_available():
                self.control_loop_action(verbose)

            # 1b. Full-arm joint feedback for trajectory recording (zero overhead when disabled).
            if self._traj_recorder is not None:
                self._read_feedback_state()

            # 2. State machine: phase scan (memory-only) then conditional bus read.
            #    Bus reads (~3–6ms: Load+Current+Position) only for CLOSING/HOLDING;
            #    skipped for APPROACHING/DROPPING/OPENING (~60–70% of steps).
            _phase = self._gripper_monitor.scan_intended_phase(
                self.action_queue, self.action_queue_lock, self._gripper_axis_idx
            )
            _fb: dict[str, float] | None = (
                self._read_gripper_feedback()
                if _phase in (GraspPhase.CLOSING, GraspPhase.HOLDING)
                else None
            )
            # Snapshot SM state before update() — slip path resets _peak_load to 0
            # before returning RECOVERY, so we must capture it beforehand.
            _hold_before = self._gripper_monitor._hold_confirmed
            _peak_before = self._gripper_monitor._peak_load
            decision = self._gripper_monitor.update(
                _fb,
                self.action_queue,
                self.action_queue_lock,
                self._gripper_axis_idx,
                _precomputed_phase=_phase,
            )

            # ── SM event logging (no-op when timing is disabled) ─────────
            if self._gripper_sm_recorder is not None:
                # Failure events use _last_failure_type ("empty_grasp" | "slip") so
                # the record captures WHAT failed regardless of whether the decision
                # was REINFER or RECOVERY.  failure_count distinguishes them implicitly.
                _event_type: str | None = (
                    self._gripper_monitor._last_failure_type
                    if decision in (GripperDecision.REINFER, GripperDecision.RECOVERY,
                                    GripperDecision.LIFT_RETRY, GripperDecision.REWIND_RETRY)
                    else "stop"          if decision == GripperDecision.STOP
                    else "task_done"     if decision == GripperDecision.TASK_DONE
                    else "grasp_success" if (not _hold_before and self._gripper_monitor._hold_confirmed)
                    else None
                )
                if _event_type is not None:
                    with self.action_queue_lock:
                        _qsize = self.action_queue.qsize()
                    with self.latest_action_lock:
                        _ts = self.latest_action
                    self._gripper_sm_recorder.add(GripperSMEventRecord(
                        wall_time=time.time(),
                        episode=self._current_episode,
                        timestep=_ts,
                        event_type=_event_type,
                        phase=_phase.name,
                        gripper_load=abs(float((_fb or {}).get(self.config.gripper_sm_load_key, 0.0))),
                        gripper_pos=float((_fb or {}).get(self.config.gripper_sm_pos_key, 0.0)),
                        peak_load=_peak_before,
                        failure_count=self._gripper_monitor._failure_count,
                        queue_size=_qsize,
                    ))

            # ── Grasp-success attribution (runs even when timing recorder is off) ──
            if not _hold_before and self._gripper_monitor._hold_confirmed:
                self._grasp_success_total += 1
                if self._last_intervention == "lift_retry":
                    self._grasp_success_after_lift_retry += 1
                    self.logger.info(
                        f"[gripper_sm] grasp_success_after_lift_retry={self._grasp_success_after_lift_retry}"
                    )
                elif self._last_intervention == "recovery":
                    self._grasp_success_after_recovery += 1
                    self.logger.info(
                        f"[gripper_sm] grasp_success_after_recovery={self._grasp_success_after_recovery}"
                    )
                self._last_intervention = ""  # reset — next grasp starts clean

            if decision == GripperDecision.REINFER:
                self._ep_retry_count += 1
                _suffix = (
                    "(gripper missed, retry grasp)"
                    if self._gripper_monitor._last_failure_type == "empty_grasp"
                    else "(object slipped, retry grasp)"
                )
                self._force_reinference(reason=self._gripper_monitor._last_failure_type,
                                        task_suffix=_suffix)
            elif decision == GripperDecision.RECOVERY:
                self._ep_retry_count += 1
                self._last_intervention = "recovery"
                self._on_retry_triggered(decision)
                self._execute_recovery()
                # Skip obs-send on this iteration: _execute_recovery() drains the queue
                # so _ready_to_send_observation() would fire on queue-size threshold and
                # send an obs from the failed-grasp position (not home).  The server would
                # infer on that stale state; the resulting actions arrive during the
                # recovery trajectory and cause abnormal motion when executed.
                # Subsequent iterations hit _recovery_remaining > 0 → continue at the top.
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue
            elif decision == GripperDecision.LIFT_RETRY:
                self._ep_retry_count += 1
                self._lift_retry_count += 1
                self._last_intervention = "lift_retry"
                self._on_retry_triggered(decision)
                self._execute_lift_retry()
                # Skip obs-send: queue drained, _lift_remaining > 0 on the next iteration.
                # If _execute_lift_retry() fell back to RECOVERY, _recovery_remaining > 0
                # and the recovery block at the top of the loop will handle it.
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue
            elif decision == GripperDecision.REWIND_RETRY:
                self._ep_retry_count += 1
                self._last_intervention = "rewind_retry"
                self._on_retry_triggered(decision)
                self._execute_rewind_retry()
                # Skip obs-send: queue drained, _rewind_remaining > 0 on the next iteration.
                time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))
                continue
            elif decision == GripperDecision.TASK_DONE:
                if self._is_at_home():
                    self._at_home_confirm_count += 1
                    _confirm_need = self.config.task_done_home_confirm_steps
                    if self._at_home_confirm_count >= _confirm_need:
                        self.logger.info(
                            f"{_CG}[gripper_sm] TASK DONE{_CX} — pick-place cycle complete "
                            f"| source=SM | phase={_phase.name} "
                            f"| confirm={self._at_home_confirm_count}/{_confirm_need} "
                            f"| resetting SM for next task"
                        )
                        self._at_home_confirm_count = 0
                        self._ep_records.append({"retries": self._ep_retry_count, "success": True})
                        self._ep_retry_count = 0
                        self._force_reinference(reason="task_done")
                        self._on_task_done()   # flush trajectory + open new episode file
                        self._gripper_monitor._failure_count = 0
                        self._gripper_monitor._lift_retry_attempted = False   # re-arm for next cycle
                        self._gripper_monitor._rewind_retry_attempted = False  # re-arm for next cycle
                        self._last_intervention = ""  # new cycle starts clean
                        # Continue loop — SM is clean, must_go armed for fresh inference
                    else:
                        self.logger.debug(
                            f"[gripper_sm] TASK_DONE candidate | source=SM | phase={_phase.name}"
                            f" — at_home confirm {self._at_home_confirm_count}/{_confirm_need}, waiting..."
                        )
                else:
                    # Arm still moving home — reset confirm count so a transient threshold
                    # crossing does not carry over to the next step.
                    self._at_home_confirm_count = 0
                    self.logger.debug(
                        f"[gripper_sm] TASK_DONE candidate | source=SM | phase={_phase.name}"
                        " — arm not yet at home, polling..."
                    )
            # ── Open-gripper go-home: SM's TASK_DONE only fires in CLOSING/HOLDING phase.
            # Some policies keep the gripper OPEN during the return trajectory; in that
            # case the phase stays OPENING and the SM never emits TASK_DONE, causing the
            # control loop to keep running and the server to start a new grasp cycle.
            # Poll _is_at_home() here to detect task completion regardless of gripper phase.
            elif (
                decision == GripperDecision.CONTINUE
                and self._gripper_monitor._place_occurred
                and _phase not in (GraspPhase.CLOSING, GraspPhase.HOLDING)
            ):
                if self._is_at_home():
                    self._at_home_confirm_count += 1
                    _confirm_need = self.config.task_done_home_confirm_steps
                    if self._at_home_confirm_count >= _confirm_need:
                        self.logger.info(
                            f"{_CG}[gripper_sm] TASK DONE{_CX} — arm at home after place "
                            f"| source=backup | phase={_phase.name} "
                            f"| confirm={self._at_home_confirm_count}/{_confirm_need} "
                            f"| resetting SM for next task"
                        )
                        self._at_home_confirm_count = 0
                        self._ep_records.append({"retries": self._ep_retry_count, "success": True})
                        self._ep_retry_count = 0
                        self._force_reinference(reason="task_done")
                        self._on_task_done()   # flush trajectory + open new episode file
                        self._gripper_monitor._failure_count = 0
                        self._gripper_monitor._lift_retry_attempted = False   # re-arm for next cycle
                        self._gripper_monitor._rewind_retry_attempted = False  # re-arm for next cycle
                        self._last_intervention = ""  # new cycle starts clean
                        # Continue loop — must_go armed for fresh inference
                    else:
                        self.logger.debug(
                            f"[gripper_sm] TASK_DONE candidate | source=backup | phase={_phase.name}"
                            f" — at_home confirm {self._at_home_confirm_count}/{_confirm_need}, waiting..."
                        )
                else:
                    self._at_home_confirm_count = 0
            elif decision == GripperDecision.STOP:
                self.logger.error(
                    f"{_CB}[gripper_sm] STOP{_CX} — max empty-grasp retries exceeded. "
                    "Check thresholds or object placement."
                )
                self._ep_records.append({"retries": self._ep_retry_count, "success": False})
                self._ep_retry_count = 0
                self.stop()
                break

            # 3. Send observation if queue below threshold — identical to parent
            if self._ready_to_send_observation():
                self.control_loop_observation()

            _fb_str = (
                f"load={_fb.get(self.config.gripper_sm_load_key, 0):.0f} "
                f"pos={_fb.get(self.config.gripper_sm_pos_key, 0):.1f}"
                if _fb is not None else "no_read"
            )
            _loop_line = (
                f"[smart_robot_client] loop={(time.perf_counter() - t_loop) * 1000:.1f}ms | "
                f"phase={_phase.name} | decision={decision.name} | {_fb_str}"
            )
            # INFO on phase/decision transition; DEBUG for repeated steps in same state.
            if _phase != _prev_loop_phase or decision != _prev_loop_decision:
                self.logger.info(_loop_line)
                _prev_loop_phase    = _phase
                _prev_loop_decision = decision
            else:
                self.logger.debug(_loop_line)
            time.sleep(max(0, control_interval - (time.perf_counter() - t_loop)))

        return None, None


# ── Entry point ────────────────────────────────────────────────────────────────

@draccus.wrap()
def smart_async_client(cfg: SmartRobotClientConfig):
    init_logging(console_level=cfg.log_level.upper())
    logging.info(pformat(asdict(cfg)))

    client = SmartRobotClient(cfg)

    if not client.start():
        client.logger.error("Failed to connect to policy server. Aborting.")
        return

    client.logger.info("Starting action receiver thread...")

    if cfg.timing_output_dir:
        client.enable_timing(cfg.timing_output_dir)

    action_receiver_thread = threading.Thread(
        target=client.receive_actions, daemon=True, name="action-receiver"
    )
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

        # ── Final episode-stats summary ────────────────────────────────────
        try:
            _ep_stats = client._compute_episode_stats()
            if _ep_stats:
                _nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731
                client.logger.info(
                    f"[smart_robot_client] ═══ Final episode summary ═══\n"
                    f"  total_episodes       : {_ep_stats['total_episodes']}\n"
                    f"  overall_sr           : {_nan_fmt(_ep_stats['overall_sr'])}\n"
                    f"  total_retries        : {_ep_stats['total_retries']}\n"
                    f"  eps_with_retry       : {_ep_stats['eps_with_retry']} / {_ep_stats['total_episodes']}\n"
                    f"  sr_with_retry        : {_nan_fmt(_ep_stats['sr_with_retry'])}"
                    f"  (harder eps, retry triggered)\n"
                    f"  sr_no_retry          : {_nan_fmt(_ep_stats['sr_no_retry'])}"
                    f"  (clean eps, no retry)\n"
                    f"  success_after_retry  : {_ep_stats['success_after_retry']}\n"
                    f"  rescue_rate          : {_nan_fmt(_ep_stats['rescue_rate'])}"
                    f"  (SM saved/retried)\n"
                    f"  sr_lift (SM→no-SM)   : +{_ep_stats['sr_lift']:.1%}"
                    f"  (overall SR gain from SM)"
                )
                # Save sm_summary.txt alongside timing outputs (or cwd if no timing dir).
                _out_dir = Path(cfg.timing_output_dir) if cfg.timing_output_dir else Path(".")
                _out_dir.mkdir(parents=True, exist_ok=True)
                _sm_path = _out_dir / "sm_summary.txt"
                from datetime import datetime as _dt
                _lines = [
                    "=" * 72,
                    "  SmartRobotClient — Episode Summary",
                    f"  Generated : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 72,
                    "",
                    "── Config ──────────────────────────────────────────────────────────",
                    f"  task                 : {cfg.task}",
                    f"  enable_gripper_sm    : {cfg.enable_gripper_sm}",
                    f"  max_reinfer_retries  : {cfg.max_reinfer_retries}",
                    f"  max_empty_grasp_retries : {cfg.max_empty_grasp_retries}",
                    "",
                    "── Episode Stats (pick-place cycles) ──────────────────────────────",
                    f"  total_episodes       : {_ep_stats['total_episodes']}",
                    f"  overall_sr           : {_nan_fmt(_ep_stats['overall_sr'])}",
                    f"  total_retries        : {_ep_stats['total_retries']}",
                    f"  eps_with_retry       : {_ep_stats['eps_with_retry']} / {_ep_stats['total_episodes']}",
                    f"  eps_no_retry         : {_ep_stats['eps_no_retry']} / {_ep_stats['total_episodes']}",
                    f"  sr_with_retry        : {_nan_fmt(_ep_stats['sr_with_retry'])}"
                    f"  ← final SR of episodes that needed retry (harder episodes)",
                    f"  sr_no_retry          : {_nan_fmt(_ep_stats['sr_no_retry'])}"
                    f"  ← final SR of clean episodes (no retry)",
                    f"  sr_no_retry > sr_with_retry is expected: retried eps are harder.",
                    f"  success_after_retry  : {_ep_stats['success_after_retry']}  ← episodes saved by SM",
                    f"  rescue_rate          : {_nan_fmt(_ep_stats['rescue_rate'])}"
                    f"  ← success_after_retry / eps_with_retry (SM effectiveness)",
                    f"  sr_lift (SM→no-SM)   : +{_ep_stats['sr_lift']:.1%}"
                    f"  ← overall SR improvement vs baseline without SM",
                    "",
                    "── Per-Episode Records ─────────────────────────────────────────────",
                ]
                for i, r in enumerate(client._ep_records):
                    status = "SUCCESS" if r["success"] else "FAILED"
                    _lines.append(f"  ep{i:04d}  [{status}]  retries={r['retries']}")
                _lines += ["", "=" * 72, ""]
                _sm_path.write_text("\n".join(_lines), encoding="utf-8")
                client.logger.info(f"[smart_robot_client] SM summary saved → {_sm_path}")
        except BaseException as exc:
            client.logger.warning(f"Episode stats summary failed: {exc}")

        client.logger.info("SmartRobotClient stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    smart_async_client()
