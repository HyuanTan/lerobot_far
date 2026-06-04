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

"""SimSmartClient — SimRobotClient with gripper state machine for LIBERO evaluation.

Detects empty_grasp in simulation using gripper finger qpos (no load sensor needed),
then uses MuJoCo set_state() to teleport the robot back to a saved physics snapshot
and runs context warmup + re-inference.  Supports LIBERO object / spatial / goal /
libero_10 / libero_90 task suites.

Key differences from SmartRobotClient (real robot):
  • Gripper detection: finger qpos gap (no motor load/current available in sim)
  • Rewind:           MuJoCo set_state() + sim.forward() — exact, instantaneous,
                      restores robot AND object positions simultaneously
  • Background sender: not needed — env.step() is synchronous
  • Warmup:           N env.step() with hold action to advance sim + fill context

Gripper action convention (Robosuite / LIBERO):
  action[-1] > 0  → close gripper (grasping)
  action[-1] < 0  → open  gripper (releasing)
  get_libero_dummy_action() = [0,0,0,0,0,0,-1]  → hold position, open gripper

Usage::

    python -m lerobot.async_inference.sim_test.run_libero_smart_test \\
        --env_task=libero_object \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=<path> \\
        --server_address=localhost:8080 \\
        --enable_gripper_sm=true \\
        --rewind_buffer_steps=60 \\
        --rewind_warmup_steps=10
"""

from __future__ import annotations

import enum
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

import numpy as np
import torch

from .configs import LiberoSimConfig
from .sim_client import EpisodeResult, SimRobotClient, _extract_success


logger = logging.getLogger(__name__)

# ANSI terminal colours (gracefully no-op when piped to a file)
_CY = "\033[33m"   # yellow  — CLOSING / REWIND
_CG = "\033[32m"   # green   — GRASP_SUCCESS / warmup done / SUCCESS
_CR = "\033[31m"   # red     — EMPTY_GRASP / SLIP / max-retries / FAILED
_CX = "\033[0m"    # reset


# ── Extended episode result with retry statistics ─────────────────────────────

@dataclass
class SmartEpisodeResult(EpisodeResult):
    """EpisodeResult with gripper-SM retry bookkeeping."""
    retries: int = 0
    success_after_retry: bool = False   # success=True AND retries > 0


# ── Gripper state machine phases ──────────────────────────────────────────────

class _GraspPhase(enum.Enum):
    NORMAL  = "NORMAL"   # no active grasp attempt
    CLOSING = "CLOSING"  # gripper commanded closed, waiting to confirm grasp or empty
    HOLDING = "HOLDING"  # object confirmed in hand
    REWIND  = "REWIND"   # executing set_state() + warmup (placeholder, not used at runtime)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SimSmartClientConfig(LiberoSimConfig):
    """LiberoSimConfig extended with gripper state machine + set_state() rewind.

    All SM fields are ignored when enable_gripper_sm=False (default), so this
    config is a drop-in replacement for LiberoSimConfig.

    Requires obs_type='pixels_agent_pos' for gripper qpos access.
    The SM is auto-disabled with a warning when obs_type='pixels'.
    """

    # ── Gripper SM toggle ────────────────────────────────────────────────────
    enable_gripper_sm: bool = field(
        default=False,
        metadata={"help": "Enable gripper state machine for empty_grasp detection and rewind."},
    )

    # ── Closing detection ────────────────────────────────────────────────────
    gripper_close_action_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "Gripper action value strictly above this triggers CLOSING phase. "
                "Robosuite/LIBERO convention: action[-1] > 0 = close gripper. "
                "Default 0.0 fires when action[-1] > 0 (any close command)."
            )
        },
    )
    gripper_pos_sum_empty_threshold: float = field(
        default=0.008,
        metadata={
            "help": (
                "abs(qpos[0]) + abs(qpos[1]) below which gripper is considered fully closed "
                "/ empty. Franka mirrored joints: abs_sum=0 (closed) to ~0.08 (open). "
                "Default 0.008 = fingers nearly touching (< 4 mm total gap)."
            )
        },
    )
    gripper_pos_sum_open_threshold: float = field(
        default=0.05,
        metadata={
            "help": (
                "abs_sum above which gripper is considered open. Used to (1) set "
                "_gripper_was_open flag and (2) skip CLOSING evaluation while the "
                "gripper is still transitioning from open. Default 0.05 (5 cm total gap)."
            )
        },
    )
    gripper_pos_sum_grasp_threshold: float = field(
        default=0.02,
        metadata={
            "help": (
                "abs_sum above which a confirmed grasp is detected (object between "
                "fingers while gripper commanded closed). Default 0.02 (~1 cm per "
                "finger) — triggers for objects wider than ~2 cm."
            )
        },
    )
    gripper_confirm_steps: int = field(
        default=3,
        metadata={
            "help": (
                "Consecutive steps finger qpos must stay below empty_threshold (while "
                "gripper commanded closed) to confirm empty_grasp. "
                "Also used to confirm slip in HOLDING phase."
            )
        },
    )
    sm_activation_delay: int = field(
        default=10,
        metadata={
            "help": (
                "Minimum executed policy steps before SM activates. Prevents false "
                "empty_grasp detections during the first few steps of an episode when "
                "the gripper may still be transitioning from the reset state."
            )
        },
    )
    closing_qpos_velocity_epsilon: float = field(
        default=0.001,
        metadata={
            "help": (
                "Minimum per-step decrease in qpos_sum that signals the gripper is still "
                "in active transit (fingers still moving toward each other). "
                "In the CLOSING phase, if (prev_qpos_sum - qpos_sum) > this value the "
                "empty_grasp confirm_count is HELD (not incremented) — the fingers have "
                "not yet settled so the grasp outcome is unknown.  The count is NOT reset "
                "to zero so that progress already accumulated on stable steps is preserved. "
                "Prevents false EMPTY_GRASP when the gripper descends through "
                "empty_threshold fast enough that MuJoCo contact hasn't registered yet. "
                "Default 0.001 m/step (1 mm/step): below this the fingers are considered "
                "settled and the empty count may accumulate. "
                "Set to 0.0 to disable (original behaviour: count immediately on "
                "qpos_sum < empty_threshold regardless of velocity)."
            )
        },
    )
    max_empty_grasp_retries: int = field(
        default=3,
        metadata={
            "help": (
                "Max empty_grasp + slip detections per episode before the SM stops "
                "retrying (episode continues but SM goes dormant for the rest of the "
                "episode)."
            )
        },
    )

    # ── Home-reset recovery (triggered after the first rewind also fails) ──────
    enable_home_reset: bool = field(
        default=True,
        metadata={
            "help": (
                "After the first rewind retry fails (failure_count >= 2), restore the "
                "robot and scene to the saved initial-episode state instead of rewinding "
                "again. This gives the policy a clean slate from the episode start. "
                "Set to False to keep rewinding on every retry (original behaviour)."
            )
        },
    )
    home_reset_warmup_steps: int = field(
        default=15,
        metadata={
            "help": (
                "Context-warmup steps executed after a home-reset before re-inference "
                "is armed.  Typically longer than rewind_warmup_steps because the policy "
                "context window needs to be filled from a fully reset starting state."
            )
        },
    )

    # ── Rewind mode ──────────────────────────────────────────────────────────
    rewind_mode: str = field(
        default="set_state",
        metadata={
            "help": (
                "Rewind strategy after empty_grasp / slip detection. "
                "'set_state'     (default) — MuJoCo sim.set_state() teleports robot AND "
                "                objects back to the saved snapshot instantly. Exact, "
                "                zero-step, restores the full scene state. "
                "'action_replay' — Reverse the executed action history by negating EEF "
                "                delta dims and stepping env.step() in a loop.  Less "
                "                precise but produces smooth visual context for the policy."
            )
        },
    )

    # ── Rewind buffer / step-back (both modes) ────────────────────────────────
    rewind_buffer_steps: int = field(
        default=60,
        metadata={
            "help": (
                "set_state mode : number of (MjSimState, obs_raw) snapshots to keep. "
                "action_replay  : size of the action history ring buffer. "
                "Larger values allow rewinding further back into the approach trajectory."
            )
        },
    )
    rewind_warmup_steps: int = field(
        default=10,
        metadata={
            "help": (
                "Context-warmup steps after the rewind (both modes). The sim is stepped "
                "with a hold action (no EEF delta, gripper open) to fill the policy "
                "context window with post-rewind observations before re-inference fires."
            )
        },
    )

    # ── set_state mode only ───────────────────────────────────────────────────
    rewind_step_back: int = field(
        default=0,
        metadata={
            "help": (
                "(set_state only) How many steps back to rewind. "
                "0 = use the oldest available snapshot (maximum rewind distance). "
                "Positive values pick a specific index from the end of the buffer."
            )
        },
    )

    # ── action_replay mode only ───────────────────────────────────────────────
    rewind_gripper_open_val: float = field(
        default=-1.0,
        metadata={
            "help": (
                "(action_replay only) Gripper action value used on every rewind step. "
                "-1.0 = fully open (Robosuite/LIBERO convention: action[-1] < 0 opens "
                "the gripper). The gripper opens during the backward replay so the robot "
                "is ready to attempt a fresh grasp after reinference."
            )
        },
    )
    rewind_min_net_displacement: float = field(
        default=0.0,
        metadata={
            "help": (
                "(action_replay only) Minimum L2 net EEF displacement (metres) required "
                "before the rewind trajectory is considered long enough.  0 = use all "
                "history up to rewind_buffer_steps.  Setting e.g. 0.05 ensures the arm "
                "moves at least 5 cm away from the failed grasp position before stopping."
            )
        },
    )

    def __post_init__(self):
        # Preserve parent validation (fps / chunk_size_threshold / ... + aggregate_fn).
        super().__post_init__()

        # Gripper-SM parameter validation (only when the SM is actually enabled, so
        # SM-disabled runs with stale values are not blocked).
        if not self.enable_gripper_sm:
            return

        # Finger-qpos-sum thresholds must be ordered:  0 < empty <= grasp < open.
        #   empty == grasp is allowed: it removes the ambiguous "still closing" dead-zone
        #     [empty, grasp) so any sub-grasp qpos (no contact) is treated as empty
        #     (aggressive detection).
        #   empty > grasp is REJECTED: the GRASP_SUCCESS check (qpos >= grasp OR contact)
        #     runs BEFORE the empty check in the CLOSING phase, so empty > grasp silently
        #     degenerates to empty == grasp and the configured empty value is ineffective —
        #     exactly the inverted-ordering bug this guard exists to catch.
        e = self.gripper_pos_sum_empty_threshold
        g = self.gripper_pos_sum_grasp_threshold
        o = self.gripper_pos_sum_open_threshold
        if not (0.0 < e <= g < o):
            raise ValueError(
                "gripper_pos_sum thresholds must satisfy 0 < empty <= grasp < open, got "
                f"empty={e}, grasp={g}, open={o}. "
                "(empty > grasp is the inverted-ordering bug: the grasp check runs first, "
                "so empty > grasp behaves identically to empty == grasp and the empty value "
                "has no effect. Set empty <= grasp.)"
            )
        if self.gripper_confirm_steps < 1:
            raise ValueError(
                f"gripper_confirm_steps must be >= 1, got {self.gripper_confirm_steps}"
            )
        if self.sm_activation_delay < 0:
            raise ValueError(
                f"sm_activation_delay must be >= 0, got {self.sm_activation_delay}"
            )
        if self.max_empty_grasp_retries < 0:
            raise ValueError(
                f"max_empty_grasp_retries must be >= 0, got {self.max_empty_grasp_retries}"
            )
        if self.rewind_buffer_steps < 1:
            raise ValueError(
                f"rewind_buffer_steps must be >= 1, got {self.rewind_buffer_steps}"
            )
        if self.rewind_mode not in ("set_state", "action_replay"):
            raise ValueError(
                f"rewind_mode must be 'set_state' or 'action_replay', got {self.rewind_mode!r}"
            )


# ── Client ────────────────────────────────────────────────────────────────────

class SimSmartClient(SimRobotClient):
    """SimRobotClient with MuJoCo set_state() rewind on empty_grasp detection.

    All finger-qpos comparisons use abs_sum = abs(qpos[0]) + abs(qpos[1]) because
    Franka's parallel gripper has mirrored joints (finger1 = +x, finger2 = -x when
    open) — the signed sum is always ≈ 0 regardless of gripper state.

    abs_sum ranges (Franka):  0.0 = fully closed,  ~0.08 = fully open

    State machine phases:
        NORMAL  → [gripper_was_open AND action > 0] → CLOSING
        CLOSING → [abs_sum >= open_th]               → skip (still transitioning from open)
        CLOSING → [abs_sum >= grasp_th OR contact]          → HOLDING
        CLOSING → [abs_sum < empty_th AND no contact
                   AND qpos_delta <= velocity_eps × N]      → EMPTY_GRASP → REWIND
        HOLDING → [abs_sum < empty_th AND no contact × N]  → SLIP → REWIND

    Velocity guard (closing_qpos_velocity_epsilon):
        During CLOSING, if (prev_qpos_sum - qpos_sum) > epsilon the fingers are still
        in active transit — empty_grasp confirm_count is HELD (not incremented, not reset).
        This prevents false EMPTY_GRASP when the gripper descends quickly through
        empty_threshold before MuJoCo registers contact (1–3 step physics lag).
        Holding (not resetting) preserves progress already made on stable steps and
        avoids permanent suppression when qpos noise keeps delta just above epsilon.
        REWIND (set_state)     → sim.set_state() + warmup N steps  → NORMAL
        REWIND (action_replay) → reversed env.step() × M + warmup  → NORMAL

    rewind_mode='set_state' (default): exact physics teleport — restores robot AND
        object positions instantly.  Requires rewind_buffer_steps snapshots.
    rewind_mode='action_replay': negates EEF delta dims of the action history and
        replays via env.step() — produces smooth visual transition for the policy
        context, but cannot restore object positions.

    Gripper was-open guard:
        The SM tracks whether the gripper was observed open (abs_sum > open_th)
        at least once since the last reset/rewind.  This prevents false empty_grasp
        detections at episode start when the Franka gripper initialises closed.
    """

    def __init__(
        self,
        config: SimSmartClientConfig,
        env: Any,
        env_preprocessor: Any,
        lerobot_features: dict,
        task_description: str = "",
    ) -> None:
        super().__init__(config, env, env_preprocessor, lerobot_features, task_description)

        # Warn when obs_type won't carry gripper state
        if config.enable_gripper_sm and getattr(config, "obs_type", "pixels") != "pixels_agent_pos":
            self.logger.warning(
                "[sim_smart] enable_gripper_sm=True but obs_type != 'pixels_agent_pos'. "
                "Gripper qpos is not available in obs — SM disabled."
            )
            self._sm_enabled = False
        else:
            self._sm_enabled = config.enable_gripper_sm

        # Per-mode rewind buffers (only one is active at a time)
        use_set_state    = self._sm_enabled and config.rewind_mode == "set_state"
        use_action_replay = self._sm_enabled and config.rewind_mode == "action_replay"

        # set_state mode: ring buffer of (MjSimState, obs_raw_copy) tuples
        self._state_buffer: deque | None = (
            deque(maxlen=config.rewind_buffer_steps) if use_set_state else None
        )
        # action_replay mode: ring buffer of executed action np.ndarray (shape (7,))
        self._action_history: deque | None = (
            deque(maxlen=config.rewind_buffer_steps) if use_action_replay else None
        )

        if self._sm_enabled and config.rewind_mode not in ("set_state", "action_replay"):
            self.logger.warning(
                f"[sim_smart] Unknown rewind_mode='{config.rewind_mode}'. "
                "Falling back to 'set_state'."
            )

        # SM state (reset each episode)
        self._grasp_phase = _GraspPhase.NORMAL
        self._confirm_count: int = 0
        self._failure_count: int = 0
        self._sm_step_count: int = 0    # steps since last SM reset
        self._gripper_was_open: bool = False
        self._warmup_remaining: int = 0
        self._rewind_triggered: bool = False
        self._home_reset_triggered: bool = False
        self._prev_qpos_sum: float = 0.0  # qpos_sum from the previous SM step (velocity guard)

        # Saved initial episode state for home-reset recovery
        # Captured at env.reset() so we can teleport back to the exact start
        # when the first rewind retry also yields an empty grasp.
        self._episode_home_state: tuple | None = None  # (MjSimState, obs_raw_copy)

        # Cumulative rescue-rate counters (NOT reset between episodes)
        self._run_eps_retried: int = 0         # episodes where retries > 0
        self._run_success_after_retry: int = 0  # episodes saved by SM

        # Frames captured during action_replay rewind (cleared after each rewind)
        self._rewind_frames: list = []

    # ── Sim access helpers ────────────────────────────────────────────────────

    def _get_robosuite_env(self):
        """Return the OffScreenRenderEnv (Robosuite env) inside the SyncVectorEnv."""
        return self.env.envs[0]._env

    def _get_sim(self):
        """Return the MjSim instance."""
        return self._get_robosuite_env().sim

    # ── Snapshot management (set_state mode) ─────────────────────────────────

    def _save_snapshot(self) -> None:
        """Save current (MjSimState, obs_raw) to the ring buffer."""
        sim = self._get_sim()
        state = sim.get_state()
        obs_copy = _copy_obs_raw(self._last_obs_raw)
        self._state_buffer.append((state, obs_copy))

    # ── Action history (action_replay mode) ──────────────────────────────────

    def _build_rewind_trajectory(self) -> list[np.ndarray] | None:
        """Build a reversed action trajectory from _action_history.

        Each step in the returned list is the recorded action with:
          • dims [0:6] (EEF deltas) negated  — moves arm in the opposite direction
          • dim  [6]   (gripper) overridden to cfg.rewind_gripper_open_val

        When rewind_min_net_displacement > 0, the trajectory is extended backward
        until the L2 norm of the cumulative EEF position offset (dims [0:3]) from
        the start reaches the threshold, or rewind_buffer_steps is exhausted.

        Returns None when the history buffer is empty or has fewer than 2 steps.
        """
        if self._action_history is None or len(self._action_history) < 2:
            return None

        cfg: SimSmartClientConfig = self.config
        history = list(self._action_history)   # [oldest, …, newest]
        history = history[:-1]                  # drop just-appended step (already executed)
        reversed_steps = list(reversed(history))  # [second-newest, …, oldest]

        max_steps = cfg.rewind_buffer_steps
        min_disp  = cfg.rewind_min_net_displacement

        if min_disp > 0.0 and len(reversed_steps) >= 2:
            # Scan backward until net EEF displacement (position dims 0:3) ≥ min_disp.
            # Net (not cumulative) displacement to avoid hover phases fooling the check.
            start = reversed_steps[0]
            cutoff = min(len(reversed_steps), max_steps)
            net_disp = 0.0
            for i in range(1, cutoff):
                curr = reversed_steps[i]
                net_disp = float(np.linalg.norm(curr[:3] - start[:3]))
                if net_disp >= min_disp:
                    cutoff = i + 1
                    break
            end_disp = float(np.linalg.norm(reversed_steps[cutoff - 1][:3] - start[:3]))
            reached = end_disp >= min_disp
            selected = reversed_steps[:cutoff]
            self.logger.info(
                f"{_CY}[sim_smart] REWIND traj{_CX}: {len(selected)} steps "
                f"(action_replay) | net_disp={end_disp:.4f}m "
                f"({'reached' if reached else f'buf exhausted, target={min_disp:.3f}'}) | "
                f"gripper→{cfg.rewind_gripper_open_val}"
            )
        else:
            selected = reversed_steps[:max_steps]
            self.logger.info(
                f"{_CY}[sim_smart] REWIND traj{_CX}: {len(selected)} steps "
                f"(action_replay) | gripper→{cfg.rewind_gripper_open_val}"
            )

        traj: list[np.ndarray] = []
        for step in selected:
            s = step.copy()
            s[:6] = -s[:6]                          # negate EEF delta to move backward
            s[6]  = cfg.rewind_gripper_open_val     # open gripper during rewind
            traj.append(s)
        return traj

    # ── Rewind dispatcher ─────────────────────────────────────────────────────

    def _execute_rewind(self) -> None:
        """Dispatch rewind to set_state or action_replay based on config."""
        cfg: SimSmartClientConfig = self.config
        if cfg.rewind_mode == "action_replay":
            self._execute_rewind_action_replay()
        else:
            self._execute_rewind_set_state()

    def _drain_queue_and_arm_warmup(self, warmup_steps: int | None = None) -> None:
        """Shared post-rewind/reset bookkeeping: drain queue, clear SM, arm warmup.

        Args:
            warmup_steps: override warmup length.  None = use cfg.rewind_warmup_steps.
                          Pass cfg.home_reset_warmup_steps for home-reset recovery.
        """
        cfg: SimSmartClientConfig = self.config
        self._action_generation += 1
        with self.action_queue_lock:
            self.action_queue = Queue()
        self._queue_empty_steps = 0
        self.must_go.clear()          # warmup obs must NOT trigger early inference
        self._rewind_triggered = False
        self._home_reset_triggered = False
        self._warmup_remaining = warmup_steps if warmup_steps is not None else cfg.rewind_warmup_steps
        # Re-arm gripper-was-open guard for the fresh attempt
        self._grasp_phase = _GraspPhase.NORMAL
        self._confirm_count = 0
        self._gripper_was_open = False
        self._sm_step_count = 0
        self._prev_qpos_sum = 0.0

    def _execute_rewind_set_state(self) -> None:
        """Mode B: Restore saved MuJoCo state snapshot + arm context warmup."""
        cfg: SimSmartClientConfig = self.config

        if not self._state_buffer:
            self.logger.warning(
                "[sim_smart] set_state rewind: snapshot buffer empty — "
                "falling back to immediate re-inference"
            )
            self._action_generation += 1
            with self.action_queue_lock:
                self.action_queue = Queue()
            self._queue_empty_steps = 0
            self.must_go.set()
            self._rewind_triggered = False
            return

        # Pick target snapshot (oldest = furthest back, or specific step_back)
        step_back = cfg.rewind_step_back
        if step_back <= 0 or step_back > len(self._state_buffer):
            step_back = len(self._state_buffer)
        idx = len(self._state_buffer) - step_back
        saved_state, saved_obs_raw = self._state_buffer[idx]

        # 1. Restore MuJoCo physics
        sim = self._get_sim()
        sim.set_state(saved_state)
        sim.forward()

        # 2. Reset controller goals to prevent 1-step position jump
        try:
            for robot in self._get_robosuite_env().robots:
                robot.controller.reset_goal()
        except Exception as exc:
            self.logger.debug(f"[sim_smart] controller.reset_goal() skipped: {exc}")

        # 3. Restore client's obs snapshot
        self._last_obs_raw = saved_obs_raw
        self._state_buffer.clear()

        self.logger.info(
            f"{_CY}[sim_smart] REWIND set_state{_CX}: restored {step_back} steps back | "
            f"warmup={cfg.rewind_warmup_steps} steps"
        )
        self._drain_queue_and_arm_warmup()

    def _execute_rewind_action_replay(self) -> None:
        """Mode A: Reverse action history via synchronous env.step() loop."""
        cfg: SimSmartClientConfig = self.config

        traj = self._build_rewind_trajectory()
        if traj is None:
            self.logger.warning(
                "[sim_smart] action_replay rewind: history too short — "
                "falling back to immediate re-inference"
            )
            self._action_generation += 1
            with self.action_queue_lock:
                self.action_queue = Queue()
            self._queue_empty_steps = 0
            self.must_go.set()
            self._rewind_triggered = False
            if self._action_history is not None:
                self._action_history.clear()
            return

        # ── 1. Drain queue + clear must_go BEFORE the loop ───────────────────
        # Prevents stale action chunks from firing during the backward motion.
        # must_go is cleared so backward-motion obs reach the server as context
        # frames (must_go=False) without triggering premature inference.
        self._action_generation += 1
        with self.action_queue_lock:
            self.action_queue = Queue()
        self._queue_empty_steps = 0
        self.must_go.clear()
        if self._action_history is not None:
            self._action_history.clear()

        # ── 2. Execute reversed trajectory: step + video + obs to server ─────
        self._rewind_frames.clear()
        for action_np in traj:
            obs_raw, _, _, _, _ = self.env.step(action_np[np.newaxis])
            self._last_obs_raw = obs_raw
            # Capture frame so backward motion appears in the saved video
            if self._record_video:
                frame = self._extract_frame(obs_raw)
                if frame is not None:
                    self._rewind_frames.append(frame)
            # Send obs to server: fills policy context with backward-motion frames.
            # must_go is cleared so this does NOT trigger inference yet.
            self.control_loop_observation()

        self.logger.info(
            f"{_CY}[sim_smart] REWIND action_replay{_CX}: replayed {len(traj)} steps | "
            f"warmup={cfg.rewind_warmup_steps} steps"
        )

        # ── 3. Reset SM state + arm warmup ────────────────────────────────────
        self._rewind_triggered = False
        self._warmup_remaining = cfg.rewind_warmup_steps
        self._grasp_phase = _GraspPhase.NORMAL
        self._confirm_count = 0
        self._gripper_was_open = False
        self._sm_step_count = 0
        self._prev_qpos_sum = 0.0

    def _execute_home_reset(self) -> None:
        """Restore the saved initial-episode state and arm context warmup.

        Called when the first rewind retry also yields an empty_grasp (failure_count >= 2).
        Teleports the robot AND all objects back to the exact state captured at env.reset()
        via sim.set_state(), then runs home_reset_warmup_steps obs-only steps to refill
        the policy context window before triggering fresh inference.

        Falls back to immediate re-inference if _episode_home_state was not saved
        (enable_home_reset=False path or obs_type mismatch).
        """
        cfg: SimSmartClientConfig = self.config

        if self._episode_home_state is None:
            self.logger.warning(
                "[sim_smart] HOME_RESET: no saved initial state — "
                "falling back to immediate re-inference"
            )
            self._action_generation += 1
            with self.action_queue_lock:
                self.action_queue = Queue()
            self._queue_empty_steps = 0
            self.must_go.set()
            self._home_reset_triggered = False
            return

        saved_state, saved_obs_raw = self._episode_home_state

        # 1. Restore full MuJoCo physics state (robot joints + all object poses)
        sim = self._get_sim()
        sim.set_state(saved_state)
        sim.forward()

        # 2. Reset controller goals to prevent a 1-step position jump
        try:
            for robot in self._get_robosuite_env().robots:
                robot.controller.reset_goal()
        except Exception as exc:
            self.logger.debug(f"[sim_smart] HOME_RESET: controller.reset_goal() skipped: {exc}")

        # 3. Restore client's obs snapshot
        self._last_obs_raw = saved_obs_raw

        self.logger.info(
            f"{_CY}[sim_smart] HOME_RESET{_CX} — restored episode initial state "
            f"(failure #{self._failure_count}) | "
            f"warmup={cfg.home_reset_warmup_steps} steps"
        )
        self._drain_queue_and_arm_warmup(warmup_steps=cfg.home_reset_warmup_steps)

    # ── Gripper state machine ─────────────────────────────────────────────────

    def _trigger_recovery(self, failure_no: int) -> None:
        """Set the appropriate recovery flag based on the failure number.

        Recovery hierarchy:
            failure_no == 1 (first empty_grasp/slip):
                → rewind retry  (go back N steps from where the grasp failed)
            failure_no >= 2  (second or later, if enable_home_reset=True):
                → home reset    (restore the initial episode state from env.reset())
            enable_home_reset=False:
                → always rewind (original behaviour)
        """
        cfg: SimSmartClientConfig = self.config
        if failure_no == 1 or not cfg.enable_home_reset:
            self._rewind_triggered = True
            self.logger.info(
                f"{_CY}[sim_smart] REWIND RETRY triggered{_CX} "
                f"(failure #{failure_no})"
            )
        else:
            self._home_reset_triggered = True
            self.logger.info(
                f"{_CY}[sim_smart] HOME RESET triggered{_CX} "
                f"(failure #{failure_no} — rewind already tried)"
            )

    def _get_gripper_qpos_sum(self, obs_raw: dict) -> float:
        """Return abs-sum of gripper finger qpos (0 = closed, ~0.08 = fully open).

        Franka's parallel gripper uses mirrored joints: finger1 = +x, finger2 = -x
        when open.  The signed sum is always ~0; abs().sum() gives the correct
        total spread (0 when closed, ~0.08 when fully open).

        obs_raw["robot_state"]["gripper"]["qpos"] shape: (2,) per env, batched to
        (n_envs, 2) by SyncVectorEnv.  Falls back to -1.0 (below all thresholds,
        treated as closed) on any access error.
        """
        try:
            gripper = obs_raw["robot_state"]["gripper"]["qpos"]
            arr = np.asarray(gripper)
            # Shape: (n_envs, 2) → take first env
            if arr.ndim == 2:
                arr = arr[0]
            return float(np.abs(arr[:2]).sum())
        except Exception:
            return -1.0  # safe default: treat as closed (no false "open" detection)

    def _is_gripper_contacting_object(self) -> bool:
        """Return True if any MuJoCo contact involves both a gripper geom and a non-robot geom.

        Uses sim.data.contact (ncon active contacts).  Returns False on any error
        so SM falls back to qpos-only logic when this method is unavailable.
        """
        try:
            sim = self._get_sim()
            gripper_geom_ids: set[int] = set()
            # Collect geom IDs whose names contain "finger" or "gripper"
            for i in range(sim.model.ngeom):
                name = sim.model.geom_id2name(i) or ""
                if "finger" in name or "gripper" in name:
                    gripper_geom_ids.add(i)
            if not gripper_geom_ids:
                return False
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                # Contact between a gripper geom and a non-gripper geom
                if (g1 in gripper_geom_ids) != (g2 in gripper_geom_ids):
                    return True
            return False
        except Exception:
            return False

    def _update_gripper_sm(self, action_tensor: torch.Tensor, obs_raw: dict) -> None:
        """Update gripper SM after one executed policy action."""
        if not self._sm_enabled:
            return

        cfg: SimSmartClientConfig = self.config
        self._sm_step_count += 1

        # Activation delay: ignore first N steps to let the episode stabilise
        if self._sm_step_count < cfg.sm_activation_delay:
            return

        gripper_action = float(action_tensor.cpu().flatten()[-1])
        qpos_sum = self._get_gripper_qpos_sum(obs_raw)
        qpos_delta = self._prev_qpos_sum - qpos_sum  # > 0 while fingers are closing

        # Periodically log raw sensor values for threshold tuning / diagnostics
        if self._sm_step_count % 20 == 0:
            self.logger.info(
                f"[sim_smart] SM step={self._sm_step_count} "
                f"phase={self._grasp_phase.value} "
                f"action={gripper_action:.3f} qpos_sum={qpos_sum:.4f} "
                f"qpos_delta={qpos_delta:+.4f} "
                f"was_open={self._gripper_was_open}"
            )

        # Track whether gripper has been open since last reset
        if qpos_sum >= cfg.gripper_pos_sum_open_threshold:
            self._gripper_was_open = True

        if self._grasp_phase == _GraspPhase.NORMAL:
            # Transition to CLOSING only if gripper was open previously
            # Robosuite/LIBERO: action > 0 = close command
            if self._gripper_was_open and gripper_action > cfg.gripper_close_action_threshold:
                self._grasp_phase = _GraspPhase.CLOSING
                self._confirm_count = 0
                self.logger.debug(
                    f"{_CY}[sim_smart] → CLOSING{_CX} | action={gripper_action:.2f} qpos_sum={qpos_sum:.4f}"
                )

        elif self._grasp_phase == _GraspPhase.CLOSING:
            if gripper_action <= cfg.gripper_close_action_threshold:
                # Policy opened gripper (action <= 0) — abort closing attempt.
                # Fix B: before exiting, check if qpos was already below empty_threshold
                # with a partial confirm count and no contact.  This catches the pattern
                # where the gripper closed empty, the velocity guard held the count just
                # below threshold, and the policy immediately re-opens to retry — without
                # this check the empty grasp would be silently swallowed.
                if (
                    self._confirm_count > 0
                    and qpos_sum < cfg.gripper_pos_sum_empty_threshold
                    and not self._is_gripper_contacting_object()
                ):
                    self._failure_count += 1
                    self.logger.warning(
                        f"{_CR}[sim_smart] EMPTY_GRASP #{self._failure_count}{_CX} "
                        f"(exit-CLOSING) | qpos_sum={qpos_sum:.4f} confirm={self._confirm_count}"
                    )
                    if self._failure_count <= cfg.max_empty_grasp_retries:
                        self._trigger_recovery(failure_no=self._failure_count)
                    else:
                        self.logger.warning(
                            f"{_CR}[sim_smart] Max retries ({cfg.max_empty_grasp_retries}) reached{_CX} — "
                            "SM dormant for rest of episode"
                        )
                        self._sm_enabled = False
                self._grasp_phase = _GraspPhase.NORMAL
                self._confirm_count = 0
            elif qpos_sum >= cfg.gripper_pos_sum_open_threshold:
                # Gripper still in open position (transitioning from open → closed).
                # Skip evaluation to avoid false GRASP_SUCCESS at the start of a close.
                self._confirm_count = 0
            else:
                # abs_sum has dropped below open_threshold: gripper is settling.
                # Evaluate grasp outcome with a single contact query.
                contact = self._is_gripper_contacting_object()
                if qpos_sum >= cfg.gripper_pos_sum_grasp_threshold or contact:
                    # Object confirmed:
                    #   • qpos: fingers held apart by object width (≥ grasp_threshold)
                    #   • contact: any gripper↔object contact, covers edge/one-sided
                    #     grasps where only one finger displaces and abs_sum is small
                    self._grasp_phase = _GraspPhase.HOLDING
                    self._confirm_count = 0
                    via = "contact" if qpos_sum < cfg.gripper_pos_sum_grasp_threshold else "qpos"
                    self.logger.info(
                        f"{_CG}[sim_smart] GRASP_SUCCESS{_CX} | qpos_sum={qpos_sum:.4f} via={via}"
                    )
                elif qpos_sum < cfg.gripper_pos_sum_empty_threshold:
                    # Fingers nearly closed AND no contact.
                    # Velocity guard: if qpos_sum is still actively decreasing
                    # (fingers in transit, not yet settled), defer counting — the
                    # grasp outcome is not yet determined.  MuJoCo contact registration
                    # can lag 1–3 steps behind the physical finger position, so a fast-
                    # closing gripper can briefly cross empty_threshold before the
                    # contact normal force appears.
                    # qpos_delta = prev - current; positive = still closing.
                    if (
                        cfg.closing_qpos_velocity_epsilon > 0.0
                        and qpos_delta > cfg.closing_qpos_velocity_epsilon
                    ):
                        # Hold: fingers still in transit — do NOT increment, but also do
                        # NOT reset to zero.  Resetting would discard progress already
                        # accumulated on stable steps and could permanently suppress
                        # detection when noise keeps qpos_delta just above epsilon.
                        self.logger.debug(
                            f"[sim_smart] CLOSING: qpos still falling "
                            f"(delta={qpos_delta:+.4f} > epsilon={cfg.closing_qpos_velocity_epsilon}) "
                            f"— holding empty_grasp count at {self._confirm_count}"
                        )
                    else:
                        self._confirm_count += 1
                        if self._confirm_count >= cfg.gripper_confirm_steps:
                            self._failure_count += 1
                            self.logger.warning(
                                f"{_CR}[sim_smart] EMPTY_GRASP #{self._failure_count}{_CX} | "
                                f"qpos_sum={qpos_sum:.4f} delta={qpos_delta:+.4f}"
                            )
                            if self._failure_count <= cfg.max_empty_grasp_retries:
                                self._trigger_recovery(failure_no=self._failure_count)
                            else:
                                self.logger.warning(
                                    f"{_CR}[sim_smart] Max retries ({cfg.max_empty_grasp_retries}) reached{_CX} — "
                                    "SM dormant for rest of episode"
                                )
                                self._sm_enabled = False
                            self._grasp_phase = _GraspPhase.NORMAL
                            self._confirm_count = 0
                else:
                    # abs_sum in (empty_threshold, grasp_threshold), no contact — still closing
                    self._confirm_count = 0

        elif self._grasp_phase == _GraspPhase.HOLDING:
            if gripper_action <= cfg.gripper_close_action_threshold:
                # Policy opened gripper intentionally (action <= 0)
                self._grasp_phase = _GraspPhase.NORMAL
                self._confirm_count = 0
            elif qpos_sum < cfg.gripper_pos_sum_empty_threshold:
                # abs_sum near zero — check contact before counting as slip.
                # Edge/one-sided grasps keep abs_sum small while still in contact.
                if self._is_gripper_contacting_object():
                    # Object still in contact despite small abs_sum → no slip
                    self._confirm_count = 0
                else:
                    self._confirm_count += 1
                    if self._confirm_count >= cfg.gripper_confirm_steps:
                        self._failure_count += 1
                        self.logger.warning(
                            f"{_CR}[sim_smart] SLIP #{self._failure_count}{_CX} | "
                            f"qpos_sum={qpos_sum:.4f}"
                        )
                        if self._failure_count <= cfg.max_empty_grasp_retries:
                            self._trigger_recovery(failure_no=self._failure_count)
                        else:
                            self.logger.warning(
                                f"{_CR}[sim_smart] Max retries ({cfg.max_empty_grasp_retries}) reached{_CX} — "
                                "SM dormant for rest of episode"
                            )
                            self._sm_enabled = False
                        self._grasp_phase = _GraspPhase.NORMAL
                        self._confirm_count = 0
            else:
                self._confirm_count = 0

        # Always update prev qpos for velocity computation next step.
        # Use max(0, qpos_sum) to keep the value non-negative.
        self._prev_qpos_sum = max(0.0, qpos_sum)

    # ── _execute_action override ──────────────────────────────────────────────

    def _execute_action(self, timed_action) -> Any:
        """Save snapshot / record history → execute → update gripper SM."""
        # set_state mode: save physics snapshot BEFORE the step (so we can teleport back)
        if self._sm_enabled and self._state_buffer is not None:
            self._save_snapshot()

        result = super()._execute_action(timed_action)

        if result is not None and self._sm_enabled:
            obs_raw, _done, _info = result
            self._update_gripper_sm(timed_action.get_action(), obs_raw)

            # action_replay mode: record executed action AFTER the step
            if self._action_history is not None:
                action_np = timed_action.get_action().cpu().numpy().flatten().astype(np.float32)
                self._action_history.append(action_np)

        return result

    # ── Episode SM reset ──────────────────────────────────────────────────────

    def _reset_sm_state(self) -> None:
        self._grasp_phase = _GraspPhase.NORMAL
        self._confirm_count = 0
        self._failure_count = 0
        self._sm_step_count = 0
        self._gripper_was_open = False
        self._warmup_remaining = 0
        self._rewind_triggered = False
        self._home_reset_triggered = False
        self._sm_enabled = (
            self.config.enable_gripper_sm
            and getattr(self.config, "obs_type", "pixels") == "pixels_agent_pos"
        )
        if self._state_buffer is not None:
            self._state_buffer.clear()
        if self._action_history is not None:
            self._action_history.clear()

    # ── Episode loop ──────────────────────────────────────────────────────────

    def run_episode(
        self,
        episode_id: int,
        max_steps: int,
        first_episode: bool = False,
        task_description: str | None = None,
    ) -> EpisodeResult:
        """Run one episode with gripper SM monitoring and set_state() rewind."""
        if task_description is not None:
            self._task_str = task_description

        self._reset_sm_state()
        self._reset_loop_state()

        ep_start = time.perf_counter()
        self.logger.info(
            f"[sim_smart] ── Episode {episode_id} start ── "
            f"task='{self._task_str}'  sm={'on' if self._sm_enabled else 'off'}"
        )

        frames: list[np.ndarray] = []
        obs_raw, _info = self.env.reset()
        self._last_obs_raw = obs_raw

        # Save the initial episode state for home-reset recovery.
        # This snapshot is restored if the first rewind retry also yields an
        # empty_grasp (failure_count >= 2), giving the policy a clean slate.
        if self._sm_enabled and self.config.enable_home_reset:
            try:
                init_state = self._get_sim().get_state()
                self._episode_home_state = (init_state, _copy_obs_raw(obs_raw))
                self.logger.debug(
                    "[sim_smart] Saved initial episode state for home-reset recovery"
                )
            except Exception as exc:
                self.logger.warning(
                    f"[sim_smart] Could not save initial episode state: {exc}. "
                    "Home reset will fall back to immediate re-inference."
                )
                self._episode_home_state = None
        else:
            self._episode_home_state = None

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

            # ── Context warmup after rewind ───────────────────────────────
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
                with self.latest_action_lock:
                    self.latest_action = max(self.latest_action, 0) + 1
                self._queue_empty_steps = 0

                # Step with hold action (zero delta, open gripper) to advance sim
                # and obtain a fresh observation from the restored position.
                hold = np.zeros((1, 7), dtype=np.float32)
                hold[0, -1] = -1.0  # open gripper (action < 0 = open in Robosuite)
                obs_raw, _, _, _, _ = self.env.step(hold)
                self._last_obs_raw = obs_raw
                if self._record_video:
                    frame = self._extract_frame(obs_raw)
                    if frame is not None:
                        frames.append(frame)
                self.control_loop_observation()

                if self._warmup_remaining == 0:
                    # Drain any server-inferred chunks from warmup, arm must_go
                    self._action_generation += 1
                    with self.action_queue_lock:
                        self.action_queue = Queue()
                    self._queue_empty_steps = 0
                    self.must_go.set()
                    self.logger.info(
                        f"{_CG}[sim_smart] Warmup complete{_CX} — inference armed "
                        f"(failures so far: {self._failure_count})"
                    )

                work_t = time.perf_counter() - t_loop
                time.sleep(max(0.0, self.config.environment_dt - work_t))
                continue

            # ── Home reset (failure >= 2): restore initial episode state ──────
            if self._home_reset_triggered:
                self._execute_home_reset()
                if self._record_video and self._rewind_frames:
                    frames.extend(self._rewind_frames)
                    self._rewind_frames.clear()
                work_t = time.perf_counter() - t_loop
                time.sleep(max(0.0, self.config.environment_dt - work_t))
                continue

            # ── Rewind (failure == 1): step back N steps from grasp point ────
            if self._rewind_triggered:
                self._execute_rewind()
                # action_replay mode captures backward-motion frames during rewind;
                # splice them into the episode video here.
                if self._record_video and self._rewind_frames:
                    frames.extend(self._rewind_frames)
                    self._rewind_frames.clear()
                work_t = time.perf_counter() - t_loop
                time.sleep(max(0.0, self.config.environment_dt - work_t))
                continue

            # ── Normal control: action + observation ──────────────────────
            if self.actions_available():
                result = self.control_loop_action()
                if result is not None:
                    obs_raw, done, last_info = result
                    step += 1
                    if self._record_video and isinstance(obs_raw, dict):
                        frame = self._extract_frame(obs_raw)
                        if frame is not None:
                            frames.append(frame)
                    if done:
                        break

            if self._ready_to_send_observation():
                self.control_loop_observation()

            if step > 0 and step % 20 == 0:
                with self.action_queue_lock:
                    qsz = self.action_queue.qsize()
                self.logger.info(
                    f"[sim_smart] step={step}/{max_steps}  queue={qsz}  "
                    f"phase={self._grasp_phase.value}  failures={self._failure_count}"
                )

            work_t = time.perf_counter() - t_loop
            time.sleep(max(0.0, self.config.environment_dt - work_t))

        success = _extract_success(last_info)
        duration = time.perf_counter() - ep_start
        retries = self._failure_count

        if self._record_video:
            self.logger.info(
                f"[sim_smart] Video: {len(frames)} total frames "
                f"(retries={retries}, rewind_frames_in_video={len(frames) - step})"
            )
            extra_tag = "_retry" if retries > 0 else ""
            self._save_episode_video(frames, episode_id, success, extra_tag=extra_tag)
        success_after_retry = success and retries > 0

        # Update cumulative rescue-rate counters
        if retries > 0:
            self._run_eps_retried += 1
        if success_after_retry:
            self._run_success_after_retry += 1
        rescue_rate = (
            self._run_success_after_retry / self._run_eps_retried
            if self._run_eps_retried > 0 else float("nan")
        )

        _col = _CG if success else _CR
        status_tag = "SUCCESS" if success else "FAILED"
        retry_tag  = f"  retries={retries}" if retries > 0 else ""
        rescue_tag = (
            f"  rescue_rate={rescue_rate:.1%}"
            f"({self._run_success_after_retry}/{self._run_eps_retried})"
            if self._run_eps_retried > 0 else ""
        )
        self.logger.info(
            f"{_col}[sim_smart] ══ Episode {episode_id} [{status_tag}]{_CX}  "
            f"steps={step}  duration={duration:.2f}s{retry_tag}{rescue_tag}"
            f"  task='{self._task_str}'"
        )

        return SmartEpisodeResult(
            episode_id=episode_id,
            task_description=self._task_str,
            success=success,
            steps=step,
            duration_s=duration,
            retries=retries,
            success_after_retry=success_after_retry,
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _copy_obs_raw(obs_raw: dict) -> dict:
    """Shallow-copy the top-level obs dict, deep-copying numpy arrays."""
    out: dict = {}
    for k, v in obs_raw.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, dict):
            out[k] = _copy_obs_raw(v)
        else:
            out[k] = v
    return out
