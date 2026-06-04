"""Offline gripper state machine for LIBERO simulation.

Adapted from lerobot.async_inference.sim_test.sim_smart_client for use in
offline (non-gRPC) simulation: detects empty_grasp from gripper qpos and
triggers MuJoCo set_state() rewind + warmup steps.

Key differences from SimSmartClient:
- No gRPC / action queue: actions come from predict_action_chunk directly
- Rewind: calls env.envs[0]._env.sim.set_state() (SyncVectorEnv path)
- No background sender thread
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CY = "\033[33m"
_CG = "\033[32m"
_CR = "\033[31m"
_CX = "\033[0m"


class _GraspPhase(enum.Enum):
    NORMAL  = "NORMAL"
    CLOSING = "CLOSING"
    HOLDING = "HOLDING"


@dataclass
class OfflineSMConfig:
    """Configuration for offline gripper state machine."""

    gripper_close_action_threshold: float = 0.0
    gripper_pos_sum_empty_threshold: float = 0.008
    gripper_pos_sum_open_threshold: float = 0.05
    gripper_pos_sum_grasp_threshold: float = 0.02
    gripper_confirm_steps: int = 3
    sm_activation_delay: int = 10
    closing_qpos_velocity_epsilon: float = 0.001
    max_empty_grasp_retries: int = 3

    enable_home_reset: bool = True
    home_reset_warmup_steps: int = 15

    rewind_buffer_steps: int = 60
    rewind_warmup_steps: int = 10
    rewind_step_back: int = 0


class OfflineGripperSM:
    """Offline gripper SM: tracks phase, detects empty_grasp, schedules rewind.

    Usage::

        sm = OfflineGripperSM(config)
        sm.reset()

        # Before each env.step():
        sm.save_snapshot(env, obs_raw)

        # After env.step():
        gripper_qpos = obs_raw["robot_state"]["gripper"]["qpos"]  # (1, 2)
        action_gripper = action[0, -1]  # scalar gripper action
        rewind = sm.update(action_gripper, gripper_qpos)
        if rewind:
            sm.execute_rewind(env)
            warmup_steps = sm.warmup_remaining  # caller drives hold steps
    """

    def __init__(self, config: OfflineSMConfig):
        self.cfg = config
        self._snapshots: deque = deque(maxlen=config.rewind_buffer_steps)
        self._initial_snapshot: Any = None
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._phase = _GraspPhase.NORMAL
        self._confirm_count: int = 0
        self._failure_count: int = 0
        self._step_count: int = 0
        self._prev_qpos_sum: float | None = None
        self._gripper_was_open: bool = False
        self._in_grasp_phase: bool = False
        self._snapshots.clear()
        self.warmup_remaining: int = 0
        self._home_reset_triggered: bool = False

    def save_initial_snapshot(self, env) -> None:
        """Call once after env.reset() to save the episode-start state."""
        if not self.cfg.enable_home_reset:
            return
        try:
            sim = env.envs[0]._env.sim
            self._initial_snapshot = (sim.get_state(), None)
        except Exception as e:
            logger.warning(f"[SM] Could not save initial snapshot: {e}")
            self._initial_snapshot = None

    def save_snapshot(self, env) -> None:
        """Save a MuJoCo state snapshot to the rewind buffer."""
        try:
            sim = env.envs[0]._env.sim
            self._snapshots.append(sim.get_state())
        except Exception as e:
            logger.debug(f"[SM] Snapshot failed: {e}")

    # ── State machine update ──────────────────────────────────────────────────

    def update(self, action_gripper: float, gripper_qpos: np.ndarray, env=None) -> bool:
        """Update SM state after one env.step().

        Mirrors SimSmartClient._update_gripper_sm: gripper-finger qpos thresholds
        + MuJoCo contact query for robust empty-grasp / slip detection.

        Args:
            action_gripper: The gripper action scalar (last dimension of action).
            gripper_qpos:   Gripper joint positions, shape (2,) or (1, 2).
            env:            SyncVectorEnv — used for MuJoCo contact query. When None
                            the SM falls back to qpos-only logic (contact treated as
                            False), reproducing the previous simplified behaviour.

        Returns:
            True if a recovery (rewind or home_reset) should be executed.
        """
        # Dormant once retries are exhausted (matches SimSmartClient._sm_enabled=False).
        if self._failure_count > self.cfg.max_empty_grasp_retries:
            return False

        self._step_count += 1
        if self._step_count < self.cfg.sm_activation_delay:
            return False

        if gripper_qpos.ndim == 2:
            gripper_qpos = gripper_qpos[0]
        qpos_sum = float(abs(gripper_qpos[0]) + abs(gripper_qpos[1]))
        prev = self._prev_qpos_sum if self._prev_qpos_sum is not None else qpos_sum
        qpos_delta = prev - qpos_sum  # > 0 while fingers are closing

        cmd_close = action_gripper > self.cfg.gripper_close_action_threshold

        # Track whether gripper has been open since last reset
        if qpos_sum >= self.cfg.gripper_pos_sum_open_threshold:
            self._gripper_was_open = True

        should_recover = False

        # ── Phase transitions ────────────────────────────────────────────────
        if self._phase == _GraspPhase.NORMAL:
            if self._gripper_was_open and cmd_close:
                self._phase = _GraspPhase.CLOSING
                self._confirm_count = 0
                self._in_grasp_phase = True
                logger.debug(f"{_CY}[SM] NORMAL→CLOSING  qpos_sum={qpos_sum:.4f}{_CX}")

        elif self._phase == _GraspPhase.CLOSING:
            if not cmd_close:
                # Policy opened gripper (action <= 0) — abort closing attempt.
                # Fix B: catch an empty grasp that the velocity guard held just
                # below threshold before the policy re-opens to retry.
                if (self._confirm_count > 0
                        and qpos_sum < self.cfg.gripper_pos_sum_empty_threshold
                        and not self._is_gripper_contacting_object(env)):
                    should_recover = self._trigger_failure("EMPTY_GRASP(exit-CLOSING)", qpos_sum)
                self._phase = _GraspPhase.NORMAL
                self._confirm_count = 0
            elif qpos_sum >= self.cfg.gripper_pos_sum_open_threshold:
                # Gripper still open / transitioning — skip evaluation to avoid
                # a false GRASP_SUCCESS at the very start of a close.
                self._confirm_count = 0
            else:
                # abs_sum dropped below open_threshold — evaluate grasp outcome.
                contact = self._is_gripper_contacting_object(env)
                if qpos_sum >= self.cfg.gripper_pos_sum_grasp_threshold or contact:
                    # Object confirmed: fingers held apart (qpos) OR any gripper↔object
                    # contact (covers edge / one-sided grasps with small abs_sum).
                    self._phase = _GraspPhase.HOLDING
                    self._confirm_count = 0
                    via = "contact" if qpos_sum < self.cfg.gripper_pos_sum_grasp_threshold else "qpos"
                    logger.info(f"{_CG}[SM] GRASP_SUCCESS  qpos_sum={qpos_sum:.4f} via={via}{_CX}")
                elif qpos_sum < self.cfg.gripper_pos_sum_empty_threshold:
                    # Fingers nearly closed AND no contact → candidate empty grasp.
                    # Velocity guard: if fingers still actively closing, defer (don't
                    # increment, don't reset) — MuJoCo contact can lag 1-3 steps.
                    if (self.cfg.closing_qpos_velocity_epsilon > 0.0
                            and qpos_delta > self.cfg.closing_qpos_velocity_epsilon):
                        logger.debug(
                            f"[SM] CLOSING: qpos still falling (delta={qpos_delta:+.4f}) "
                            f"— holding empty count at {self._confirm_count}"
                        )
                    else:
                        self._confirm_count += 1
                        if self._confirm_count >= self.cfg.gripper_confirm_steps:
                            should_recover = self._trigger_failure("EMPTY_GRASP", qpos_sum)
                            self._phase = _GraspPhase.NORMAL
                            self._confirm_count = 0
                else:
                    # abs_sum in (empty, grasp), no contact — still settling
                    self._confirm_count = 0

        elif self._phase == _GraspPhase.HOLDING:
            if not cmd_close:
                # Policy opened gripper intentionally (action <= 0) → release
                self._phase = _GraspPhase.NORMAL
                self._confirm_count = 0
                self._in_grasp_phase = False
                self._gripper_was_open = False
            elif qpos_sum < self.cfg.gripper_pos_sum_empty_threshold:
                # abs_sum collapsed near zero — slip unless still in contact.
                if self._is_gripper_contacting_object(env):
                    self._confirm_count = 0  # edge/one-sided grasp still holding
                else:
                    self._confirm_count += 1
                    if self._confirm_count >= self.cfg.gripper_confirm_steps:
                        should_recover = self._trigger_failure("SLIP", qpos_sum)
                        self._phase = _GraspPhase.NORMAL
                        self._confirm_count = 0
            else:
                self._confirm_count = 0

        # Always update prev qpos (non-negative) for next-step velocity guard.
        self._prev_qpos_sum = max(0.0, qpos_sum)
        return should_recover

    # ── Failure / recovery dispatch ───────────────────────────────────────────

    def _trigger_failure(self, kind: str, qpos_sum: float) -> bool:
        """Bump failure count and select recovery (rewind vs home_reset).

        Recovery hierarchy (matches SimSmartClient._trigger_recovery):
            failure_no == 1 (or enable_home_reset=False) → rewind retry
            failure_no >= 2  (enable_home_reset=True)     → home reset
        Beyond max_empty_grasp_retries the SM goes dormant (returns False).

        Returns True if a recovery should be executed this step.
        """
        self._failure_count += 1
        if self._failure_count > self.cfg.max_empty_grasp_retries:
            logger.info(
                f"{_CR}[SM] {kind} #{self._failure_count} — max retries "
                f"({self.cfg.max_empty_grasp_retries}) reached, SM dormant{_CX}"
            )
            return False

        if self.cfg.enable_home_reset and self._failure_count >= 2:
            self._home_reset_triggered = True
            logger.info(
                f"{_CR}[SM] {kind} #{self._failure_count}{_CX} qpos_sum={qpos_sum:.4f} "
                f"→ {_CY}HOME_RESET{_CX} (rewind already tried)"
            )
        else:
            self._home_reset_triggered = False
            logger.info(
                f"{_CR}[SM] {kind} #{self._failure_count}{_CX} qpos_sum={qpos_sum:.4f} "
                f"→ {_CY}REWIND retry{_CX}"
            )
        return True

    # ── MuJoCo contact query ──────────────────────────────────────────────────

    def _is_gripper_contacting_object(self, env) -> bool:
        """True if any active MuJoCo contact links a gripper geom to a non-gripper geom.

        Mirrors SimSmartClient._is_gripper_contacting_object. Falls back to False
        (qpos-only logic) when env is None or the sim is unavailable.
        """
        if env is None:
            return False
        try:
            sim = env.envs[0]._env.sim
            gripper_geom_ids: set[int] = set()
            for i in range(sim.model.ngeom):
                name = sim.model.geom_id2name(i) or ""
                if "finger" in name or "gripper" in name:
                    gripper_geom_ids.add(i)
            if not gripper_geom_ids:
                return False
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                if (g1 in gripper_geom_ids) != (g2 in gripper_geom_ids):
                    return True
            return False
        except Exception:
            return False

    # ── Rewind execution ──────────────────────────────────────────────────────

    def execute_rewind(self, env) -> bool:
        """Restore MuJoCo state from the rewind buffer (or initial snapshot)."""
        if self._home_reset_triggered and self._initial_snapshot is not None:
            logger.info(f"{_CY}[SM] HOME RESET (failure_count={self._failure_count}){_CX}")
            self._home_reset_triggered = False
            mj_state = self._initial_snapshot[0]
            self.warmup_remaining = self.cfg.home_reset_warmup_steps
        else:
            if not self._snapshots:
                logger.warning("[SM] No snapshots available for rewind")
                return False

            if self.cfg.rewind_step_back > 0:
                idx = max(0, len(self._snapshots) - self.cfg.rewind_step_back)
                mj_state = list(self._snapshots)[idx]
            else:
                mj_state = self._snapshots[0]  # oldest (maximum rewind distance)
            self.warmup_remaining = self.cfg.rewind_warmup_steps
            logger.info(
                f"{_CY}[SM] REWIND  buffer_len={len(self._snapshots)}  "
                f"warmup={self.warmup_remaining}{_CX}"
            )

        try:
            sim = env.envs[0]._env.sim
            sim.set_state(mj_state)
            sim.forward()
            # Reset controller goals so the next action does not snap the EEF from
            # the post-rewind pose to a stale goal (1-step position jump).
            # Matches SimSmartClient._execute_rewind_set_state step 2.
            try:
                for robot in env.envs[0]._env.robots:
                    robot.controller.reset_goal()
            except Exception as exc:
                logger.debug(f"[SM] controller.reset_goal() skipped: {exc}")
            self._snapshots.clear()
            # Reset SM state for the fresh attempt; keep failure_count.
            self._confirm_count = 0
            self._prev_qpos_sum = None
            self._gripper_was_open = False
            self._in_grasp_phase = False
            self._step_count = 0          # re-arm activation delay after rewind
            self._phase = _GraspPhase.NORMAL
        except Exception as e:
            logger.error(f"[SM] set_state failed: {e}")
            return False

        return True

    @property
    def is_active(self) -> bool:
        return self._failure_count <= self.cfg.max_empty_grasp_retries

    @property
    def failure_count(self) -> int:
        return self._failure_count
