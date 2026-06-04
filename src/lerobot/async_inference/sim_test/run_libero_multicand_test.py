"""Multi-candidate async-inference LIBERO evaluation (Phase 2).

Connects to a running multi_candidate_server.py.  When the server returns an
ActionBundle (top_k > 1), this client applies a second selection pass based on
execution continuity before committing to a candidate.

Phase summary
─────────────
Phase 1 (top_k=1, server selects best):
  Server returns ActionBundle.selected only.
  This client falls back to bundle.selected (identical to run_libero_test.py).

Phase 2 (top_k>1, client co-selects):
  Server returns top-K candidates in ActionBundle.candidates (ranked by server
  composite score).  This client re-ranks using:
    final_score = (1 - alpha) * server_score + alpha * continuity_score
  where continuity_score = -||candidate_first_action - last_executed_action||₂.
  If all candidates fail a joint-limit check they are silently discarded and
  bundle.selected (server's best-1) is used as a safe fallback.

Phase 3 (data collection):
  After each episode this client writes a one-line JSON record to
  <data_collect_dir>/client_outcomes.jsonl for offline joining with the
  server's candidates.jsonl (keyed by episode_id).

Usage::

    # 1. Start multi-candidate server (in another terminal):
    python -m lerobot.async_inference.multi_candidate_server \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=lerobot/smolvla_base \\
        --host=localhost --port=8080 \\
        --n_candidates=4 --top_k=2 --delay_delta=1 \\
        --data_collect_dir=./mc_data

    # 2. Run Phase 2 evaluation:
    python -m lerobot.async_inference.sim_test.run_libero_multicand_test \\
        --env_task=libero_10 \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=lerobot/smolvla_base \\
        --server_address=localhost:8080 \\
        --actions_per_chunk=16 --fps=30 \\
        --client_smooth_alpha=0.3 \\
        --data_collect_dir=./mc_data \\
        --results_dir=./mc_results

    # Offline join (example):
    # python -c "
    #   import pandas as pd
    #   cands  = pd.read_json('./mc_data/candidates.jsonl', lines=True)
    #   outcs  = pd.read_json('./mc_data/client_outcomes.jsonl', lines=True)
    #   merged = cands.merge(outcs, on='episode_id')
    #   print(merged.groupby('episode_id')['success'].mean())
    # "
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import torch
from lerobot.utils.utils import init_logging

from .run_libero_test import _read_timing_tables
from .sim_client import EpisodeResult, SimRobotClient, _get_task_description
from .sim_smart_client import SimSmartClient, SimSmartClientConfig, SmartEpisodeResult, _GraspPhase
from .trajectory_recorder import TrajectoryRecorder
from ..helpers import ActionBundle, ActionChunk, TimedAction, get_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_actions(chunk: ActionChunk) -> list[list[float]]:
    """Convert ActionChunk.timed_actions → list of D-dim float lists (T × D)."""
    return [ta.get_action().float().tolist() for ta in chunk.timed_actions]


def _build_all_cands_data(bundle: "ActionBundle") -> list[dict]:
    """Build the all_candidates list for trajectory recording.

    Returns a list of dicts for all N server-generated candidates (ranked by
    server score, rank 0 = best).  Empty list when record_all_candidates=False
    on the server (bundle.all_candidates will be empty).

    Each entry includes an ``in_top_k`` flag so readers can distinguish the
    top_k candidates that were sent for client re-ranking from the rest.
    """
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiCandLiberoConfig(SimSmartClientConfig):
    """LiberoSimConfig extended with Phase-2/3 multi-candidate client settings."""

    # ── Phase 2: client-side re-ranking ─────────────────────────────────────
    client_smooth_alpha: float = field(
        default=0.3,
        metadata={
            "help": (
                "Weight for execution-continuity score in client-side re-ranking. "
                "0.0 = use server ranking unchanged; "
                "1.0 = pure continuity (ignore server score). "
                "Continuity score = -||candidate[0] - last_executed_action||₂."
            )
        },
    )
    action_limit_min: float = field(
        default=-1.5,
        metadata={"help": "Minimum allowed action value for joint-limit check (post-processed space)."},
    )
    action_limit_max: float = field(
        default=1.5,
        metadata={"help": "Maximum allowed action value for joint-limit check (post-processed space)."},
    )

    # ── Phase 3: data collection ─────────────────────────────────────────────
    data_collect_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "If set, write per-episode outcome records to "
                "<data_collect_dir>/client_outcomes.jsonl for Phase 3 data joining. "
                "Should match the server's --data_collect_dir so the two files can be "
                "joined offline on episode_id."
            )
        },
    )

    # ── Candidate stats logging ──────────────────────────────────────────────
    log_candidate_stats: bool = field(
        default=True,
        metadata={"help": "Log per-bundle candidate stats at INFO level."},
    )

    # ── O1: spread-based uncertainty slow mode ────────────────────────────────
    spread_uncertainty_threshold: float = field(
        default=0.08,
        metadata={
            "help": (
                "Rolling mean spread_l2 threshold above which high-uncertainty slow mode "
                "activates in NORMAL phase: alpha is scaled up by spread_slow_alpha_scale "
                "to emphasise continuity and produce slower/safer pre-grasp movements. "
                "Set to 0.0 to disable (original behaviour)."
            )
        },
    )
    spread_slow_alpha_scale: float = field(
        default=1.5,
        metadata={
            "help": (
                "Alpha multiplier applied when rolling spread exceeds "
                "spread_uncertainty_threshold (NORMAL phase only). Capped at 0.9. "
                "Default 1.5 → e.g. base_alpha=0.3 becomes 0.45 under high uncertainty."
            )
        },
    )
    spread_slow_mode_window: int = field(
        default=5,
        metadata={
            "help": (
                "Number of most-recent chunk spread_l2 values to average for the rolling "
                "uncertainty signal compared against spread_uncertainty_threshold. Default 5."
            )
        },
    )

    # ── O3: per-bundle server score normalisation ─────────────────────────────
    server_score_normalize: str = field(
        default="softmax",
        metadata={
            "help": (
                "Per-bundle server + continuity score normalisation before combining. "
                "'none'    — raw scores (original behaviour). "
                "'softmax' — softmax across candidates; makes alpha scale-invariant (default). "
                "'minmax'  — rescale each score set to [0,1] within the bundle."
            )
        },
    )

    # ── Trajectory recording ─────────────────────────────────────────────────
    record_trajectory: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, write per-episode trajectory JSON files to trajectory_dir. "
                "Each file contains per-chunk candidate arrays (actions + scores) and "
                "per-step executed actions + robot_state for offline visualization with "
                "analyze_multicand_trajectory.py."
            )
        },
    )
    trajectory_dir: str = field(
        default="./mc_trajectories",
        metadata={"help": "Output directory for per-episode trajectory JSON files (--record_trajectory=true)."},
    )

    # ── Output root ───────────────────────────────────────────────────────────
    save_root_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional root directory prepended to all output paths "
                "(data_collect_dir, trajectory_dir, results_dir, timing_output_dir, "
                "video_dir, queue_size_monitor_path). "
                "Empty string = use paths as-is. "
                "Example: --save_root_path=./outputs/eval/libero/run1"
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
            self.video_dir = str(root / self.video_dir)
            self.queue_size_monitor_path = str(root / self.queue_size_monitor_path)


# ---------------------------------------------------------------------------
# Data collectors (client-side)
# ---------------------------------------------------------------------------


class ClientOutcomeCollector:
    """Appends per-episode outcome records to <output_dir>/client_outcomes.jsonl.

    Schema: {"episode_id", "success", "steps", "duration_s", "sm_retries", "success_after_retry"}
    Join with server's candidates.jsonl on ``episode_id`` for Phase 3 training.
    sm_retries / success_after_retry are 0/False when enable_gripper_sm=False.
    """

    def __init__(self, output_dir: str):
        self._path = Path(output_dir) / "client_outcomes.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logging.getLogger("ClientOutcomeCollector").info(
            f"[Phase 3] Client outcomes → {self._path}"
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

    Covers P0 (client selection), P1 (continuity), P2 (robot state).

    Schema::

        {
          "episode_id":           int,
          "timestep":             int,
          # P0 ─ client selection
          "client_selected_idx":  int,     # -1 = Phase-1 server-only
          "client_override":      bool,    # True when client differs from server rank-0
          "n_candidates":         int,
          "server_scores":        [float], # per-candidate server composite scores
          "continuity_scores":    [float], # per-candidate -||cand[0]-prev_action||₂
          "combined_scores":      [float], # (1-α)*server + α*continuity
          # P1 ─ continuity / delay
          "execution_continuity": float | null,  # ||selected[0] - prev_action||₂
          "prev_action":          [float] | null, # last executed action (postprocessed)
          "selected_first_action":[float] | null, # chosen candidate's first action
          "episode_phase":        float,   # timestep / max_steps  ∈ [0,1]
          "delay_selected":       int | null, # inference_delay of chosen candidate
          # P2 ─ robot state
          "robot_state":          [float] | null,  # current joint state from obs
          # P3 ─ SM phase / scoring
          "grasp_phase":          str,    # NORMAL | CLOSING | HOLDING (SM phase at selection time)
          "alpha_effective":      float   # actual alpha used after phase + latency adaptation
        }

    Join with server's candidates.jsonl on (episode_id, timestep).
    """

    def __init__(self, output_dir: str):
        self._path = Path(output_dir) / "client_steps.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logging.getLogger("ClientStepCollector").info(
            f"[P0-P2] Client steps → {self._path}"
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
            "execution_continuity": round(execution_continuity, 6) if execution_continuity is not None else None,
            "prev_action": [round(v, 6) for v in prev_action] if prev_action is not None else None,
            "selected_first_action": [round(v, 6) for v in selected_first_action] if selected_first_action is not None else None,
            "episode_phase": round(episode_phase, 4),
            "delay_selected": delay_selected,
            "robot_state": [round(v, 6) for v in robot_state] if robot_state is not None else None,
            "grasp_phase": grasp_phase,
            "alpha_effective": round(alpha_effective, 4),
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Multi-candidate client  (P0 + P1 + P2)
# ---------------------------------------------------------------------------


class MultiCandSimClient(SimSmartClient):
    """SimSmartClient that performs client-side selection when ActionBundle arrives.

    Overrides _resolve_raw_chunk() to intercept ActionBundle from
    MultiCandidatePolicyServer, then:

    P0 — records client_selected_idx + client_override per step.
    P1 — scores candidates by continuity (||cand[0]-last_action||) and records
         execution_continuity, prev_action, episode_phase, delay_selected.
    P2 — records current robot_state (from VectorEnv obs) per step.

    All records are written asynchronously via ClientStepCollector.
    Episode outcomes are written via ClientOutcomeCollector by the test runner.
    """

    prefix = "mc_sim_client"
    logger = get_logger(prefix)

    def __init__(
        self,
        config: MultiCandLiberoConfig,
        env: Any,
        env_preprocessor: Any,
        lerobot_features: dict,
        task_description: str = "",
    ):
        super().__init__(config, env, env_preprocessor, lerobot_features, task_description)
        self._mc_cfg = config
        self._episode_max_steps: int = (config.max_episode_steps or 500)

        # P1: continuity tracking (written by control-loop thread, read by receiver thread)
        self._last_executed_action: torch.Tensor | None = None
        # P2: robot state snapshot (written by control-loop, read by receiver)
        self._current_robot_state: list[float] | None = None

        # Step collector for P0/P1/P2 data
        self._step_collector: ClientStepCollector | None = (
            ClientStepCollector(config.data_collect_dir) if config.data_collect_dir else None
        )

        # Trajectory recorder (--record_trajectory=true)
        self._traj_recorder: TrajectoryRecorder | None = (
            TrajectoryRecorder(config.trajectory_dir) if config.record_trajectory else None
        )
        self._traj_chunk_count: int = 0  # per-episode chunk sequence number

        # O1: rolling spread for uncertainty detection (reset per episode)
        self._recent_spreads: deque = deque(maxlen=max(1, config.spread_slow_mode_window))
        self._spread_slow_activations: int = 0  # cumulative counter, not reset per episode

        # Telemetry counters
        self._client_override_count: int = 0
        self._bundle_recv_count: int = 0

        # O4: warn when action_replay is used with a short rewind buffer
        if (
            config.enable_gripper_sm
            and getattr(config, "rewind_mode", "set_state") == "action_replay"
            and config.rewind_buffer_steps < 40
        ):
            self.logger.warning(
                f"[mc_client] rewind_mode=action_replay with rewind_buffer_steps="
                f"{config.rewind_buffer_steps} (<40). "
                "Recommend rewind_buffer_steps≥40 so the arm moves far enough from the "
                "failed grasp site before re-approaching. "
                "Alternatively, use rewind_mode=set_state to restore object positions exactly."
            )

    # ------------------------------------------------------------------
    # Core hook: ActionBundle → ActionChunk  (receiver thread)
    # ------------------------------------------------------------------

    def _resolve_raw_chunk(self, raw: Any) -> Any:
        """Intercept ActionBundle, run client selection, record step data."""
        if not isinstance(raw, ActionBundle):
            return raw  # plain ActionChunk or legacy list

        self._bundle_recv_count += 1
        bundle: ActionBundle = raw

        try:
            result = self._resolve_action_bundle(bundle)
        except Exception as exc:
            import traceback as _tb
            self.logger.error(
                f"[mc_client] _resolve_raw_chunk failed (returning bundle.selected as fallback): "
                f"{exc}\n{_tb.format_exc()}"
            )
            result = bundle.selected
        # Propagate server inference time from ActionBundle to ActionChunk so that
        # base_client's latency tracker (raw.inference_time_s > 0) updates correctly.
        if isinstance(result, ActionChunk) and bundle.inference_time_s > 0:
            result.inference_time_s = bundle.inference_time_s
        return result

    def _resolve_action_bundle(self, bundle: ActionBundle) -> ActionChunk:
        """Core bundle → ActionChunk logic, called from _resolve_raw_chunk() inside try/except."""
        # Derive timestep + episode_phase from the bundle's action timestamps
        _ts = (
            bundle.selected.timed_actions[0].get_timestep()
            if bundle.selected and bundle.selected.timed_actions
            else -1
        )
        _phase = max(0.0, min(1.0, _ts / self._episode_max_steps)) if _ts >= 0 else 0.0

        candidates = bundle.candidates
        chunk_idx = self._traj_chunk_count
        self._traj_chunk_count += 1

        if not candidates:
            # Phase 1 (top_k=1): server already picked best-1; no re-ranking
            if self._mc_cfg.log_candidate_stats:
                self.logger.info(
                    f"[mc_client] Bundle #{self._bundle_recv_count} (Phase 1 t={_ts}): "
                    f"server best-1 only"
                )
            if self._step_collector is not None:
                # Record a Phase-1 step with client_selected_idx=-1
                self._write_step_record(
                    timestep=_ts,
                    episode_phase=_phase,
                    client_selected_idx=-1,
                    client_override=False,
                    n_candidates=1,
                    server_scores=[bundle.selected_score],
                    continuity_scores=[0.0],
                    combined_scores=[bundle.selected_score],
                    selected_chunk=bundle.selected,
                    delay_selected=None,
                )
            if self._traj_recorder is not None:
                _ph = getattr(self, "_grasp_phase", None)
                self._traj_recorder.record_chunk({
                    "chunk_idx": chunk_idx,
                    "first_timestep": _ts,
                    "n_candidates": 1,
                    "selected_candidate_idx": -1,
                    "client_override": False,
                    "server_score": round(bundle.selected_score, 6),
                    "spread_l2": 0.0,
                    "grasp_phase": _ph.value if _ph else "NORMAL",
                    "alpha_effective": self._mc_cfg.client_smooth_alpha,
                    "selected_actions": _chunk_actions(bundle.selected),
                    "candidates": [],
                    "all_candidates": _build_all_cands_data(bundle),
                })
            return bundle.selected

        # O1: compute spread before _client_select so uncertainty mode can read it
        _spread = self._compute_spread_l2(candidates)
        self._recent_spreads.append(_spread)

        # Phase 2: re-rank candidates
        selected, step_data = self._client_select(bundle, _ts, _phase)
        if self._step_collector is not None:
            self._step_collector.record(**step_data)
        if self._traj_recorder is not None:
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
            self._traj_recorder.record_chunk({
                "chunk_idx": chunk_idx,
                "first_timestep": _ts,
                "n_candidates": len(candidates),
                "selected_candidate_idx": best_rank,
                "client_override": client_override,
                "server_score": round(bundle.selected_score, 6),
                "spread_l2": round(_spread, 6),  # reuse value computed for O1
                "grasp_phase": step_data.get("grasp_phase", "NORMAL"),
                "alpha_effective": step_data.get("alpha_effective", self._mc_cfg.client_smooth_alpha),
                "selected_actions": _chunk_actions(selected),
                "candidates": cands_data,
                "all_candidates": _build_all_cands_data(bundle),
            })
        return selected

    def _client_select(
        self, bundle: ActionBundle, timestep: int, episode_phase: float
    ) -> tuple[ActionChunk, dict]:
        """Re-rank bundle.candidates with phase-aware scoring; return (chosen_chunk, step_record_dict).

        Scoring layers applied in order:
          1. Dynamic alpha — CLOSING: trust server quality; HOLDING: trust continuity.
             O1: Uncertainty slow mode — alpha scaled up when rolling spread_l2 exceeds
                 spread_uncertainty_threshold (high model uncertainty → slower/safer arm).
          2. Latency-corrected continuity reference — compare against timed_actions[delay_steps].
          3. High-latency alpha attenuation — reduce continuity weight when delay is large.
             O2: CLOSING phase EE-only continuity — exclude gripper dim so the close
                 command itself does not penalise continuity; EE XYZ stability is kept.
             O3: Per-bundle softmax/minmax normalisation of server and continuity scores
                 so alpha's weight is scale-invariant across model versions and tasks.
          4. CLOSING EE-stability penalties — gripper flip (−0.5) + high vel_peak penalty.
          5. HOLDING slip gate — hard-discard candidates that open the gripper mid-chunk.
        """
        import math

        candidates = bundle.candidates
        base_alpha = self._mc_cfg.client_smooth_alpha
        lim_lo = self._mc_cfg.action_limit_min
        lim_hi = self._mc_cfg.action_limit_max
        chunk_len = len(candidates[0].timed_actions) if candidates else 1

        # SM phase (receiver thread reads, control thread writes — GIL safe for heuristics)
        phase = getattr(self, "_grasp_phase", None)
        sm_on = getattr(self, "_sm_enabled", False)

        # Layer 1: phase-adaptive base alpha
        if sm_on and phase == _GraspPhase.CLOSING:
            alpha_phase = base_alpha * 0.3
        elif sm_on and phase == _GraspPhase.HOLDING:
            alpha_phase = min(base_alpha * 1.8, 0.7)
        else:
            alpha_phase = base_alpha

        # O1: spread-based uncertainty slow mode (NORMAL / non-critical phases only)
        uncertainty_mode = False
        if (
            self._mc_cfg.spread_uncertainty_threshold > 0.0
            and self._recent_spreads
            and phase not in (_GraspPhase.CLOSING, _GraspPhase.HOLDING)
        ):
            mean_spread = sum(self._recent_spreads) / len(self._recent_spreads)
            if mean_spread > self._mc_cfg.spread_uncertainty_threshold:
                alpha_phase = min(alpha_phase * self._mc_cfg.spread_slow_alpha_scale, 0.9)
                uncertainty_mode = True
                self._spread_slow_activations += 1
                if self._mc_cfg.log_candidate_stats:
                    self.logger.debug(
                        f"[mc_client] t={timestep} UNCERTAINTY_SLOW: "
                        f"mean_spread={mean_spread:.4f} > thr={self._mc_cfg.spread_uncertainty_threshold:.3f}"
                        f" → alpha_phase={alpha_phase:.3f}"
                    )

        # Snapshot P1/P2 inputs — safe under CPython GIL
        _prev_action: torch.Tensor | None = self._last_executed_action
        _robot_state: list[float] | None = self._current_robot_state

        # ── Pass 1: raw server scores, continuity scores, per-candidate alpha_eff ──
        server_scores_raw: list[float] = []
        cont_scores_raw: list[float] = []
        alpha_effs: list[float] = []

        for rank, cand in enumerate(candidates):
            meta = bundle.candidate_meta[rank] if bundle.candidate_meta else None
            srv = meta.server_score if meta is not None else float(-rank)
            server_scores_raw.append(srv)

            # Layer 2: latency-corrected reference point
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
                if sm_on and phase == _GraspPhase.CLOSING and ref_a.shape[0] > 1:
                    # O2: exclude gripper (last) dim — EE XYZ continuity only.
                    # This prevents the gripper close command from being penalised as
                    # a "discontinuity" vs the previous open-gripper action.
                    cont = -float((ref_a[:-1] - _prev_action.float().cpu()[:-1]).norm())
                else:
                    cont = -float((ref_a - _prev_action.float().cpu()).norm())
            cont_scores_raw.append(cont)

        # ── O3: per-bundle score normalisation ──────────────────────────────────
        norm_mode = self._mc_cfg.server_score_normalize

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

        # ── Pass 2: combined scores, penalties, limit gates ──────────────────────
        combined_scores: list[float] = []
        passed_limit: list[bool] = []

        for rank, cand in enumerate(candidates):
            meta = bundle.candidate_meta[rank] if bundle.candidate_meta else None
            alpha_eff = alpha_effs[rank]
            srv  = server_scores_norm[rank]
            cont = cont_scores_norm[rank]

            combined = (1.0 - alpha_eff) * srv + alpha_eff * cont

            # Layer 4: CLOSING EE-stability penalties
            if sm_on and phase == _GraspPhase.CLOSING and cand.timed_actions:
                gripper_cmds = [float(ta.get_action()[-1]) for ta in cand.timed_actions]
                if any(g < 0 for g in gripper_cmds):
                    combined -= 0.5
                if meta and meta.vel_peak > 0:
                    combined -= 0.05 * meta.vel_peak

            combined_scores.append(combined)

            # Joint-limit gate
            ok = self._passes_limit_check(cand, lim_lo, lim_hi)
            if not ok and self._mc_cfg.log_candidate_stats:
                self.logger.debug(
                    f"[mc_client] t={timestep} Candidate #{rank} out of "
                    f"[{lim_lo:.2f},{lim_hi:.2f}]"
                )

            # Layer 5: HOLDING slip gate
            if sm_on and phase == _GraspPhase.HOLDING and cand.timed_actions:
                if any(float(ta.get_action()[-1]) < 0 for ta in cand.timed_actions):
                    ok = False
                    if self._mc_cfg.log_candidate_stats:
                        self.logger.debug(
                            f"[mc_client] t={timestep} Candidate #{rank} opens gripper "
                            "in HOLDING phase — discarded"
                        )

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
                f"[mc_client] t={timestep}: all candidates failed checks — "
                "falling back to bundle.selected"
            )
            best_rank = 0
            selected = bundle.selected
        else:
            selected = candidates[best_rank]

        client_override = best_rank != 0
        if client_override:
            self._client_override_count += 1

        alpha_eff_best = alpha_effs[best_rank] if alpha_effs and best_rank < len(alpha_effs) else alpha_phase
        phase_str = phase.value if phase else "NORMAL"

        if self._mc_cfg.log_candidate_stats:
            srv_str  = "  ".join(f"c{i}:{s:.3f}" for i, s in enumerate(server_scores_raw))
            cont_str = "  ".join(f"c{i}:{s:.3f}" for i, s in enumerate(cont_scores_raw))
            slow_tag = " [SLOW]" if uncertainty_mode else ""
            self.logger.info(
                f"[mc_client] t={timestep} Bundle #{self._bundle_recv_count} | "
                f"n={len(candidates)} chosen={best_rank} "
                f"phase={phase_str} α={alpha_eff_best:.3f}{slow_tag} | "
                f"srv=[{srv_str}] cont=[{cont_str}] | "
                f"overrides={self._client_override_count}"
            )

        # P1: execution continuity of chosen candidate (timed_actions[0] for reporting)
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
            episode_id=self._current_episode,
            timestep=timestep,
            client_selected_idx=best_rank,
            client_override=client_override,
            n_candidates=len(candidates),
            server_scores=server_scores_raw,   # always raw for offline analysis
            continuity_scores=cont_scores_raw,
            combined_scores=combined_scores,   # reflects normalisation + penalties
            execution_continuity=exec_cont,
            prev_action=_prev_action.tolist() if _prev_action is not None else None,
            selected_first_action=selected_first,
            episode_phase=episode_phase,
            delay_selected=delay_selected,
            robot_state=_robot_state,
            grasp_phase=phase_str,
            alpha_effective=round(alpha_eff_best, 4),
        )
        return selected, step_record

    # ------------------------------------------------------------------
    # P1 + P2: per-step state tracking  (control-loop thread)
    # ------------------------------------------------------------------

    def _execute_action(self, timed_action: TimedAction) -> Any:
        """Track last-executed action (P1), robot state (P2), and trajectory steps."""
        result = super()._execute_action(timed_action)
        # P1: continuity
        self._last_executed_action = timed_action.get_action().clone().cpu()
        # P2: robot state from VectorEnv obs — extract EE pos + quat + gripper
        try:
            import numpy as _np
            raw = self._last_obs_raw
            rs = raw.get("robot_state") if isinstance(raw, dict) else None
            if isinstance(rs, dict):
                # LIBERO nested structure: {"eef": {"pos", "quat"}, "gripper": {"qpos"}, ...}
                eef = rs.get("eef", {})
                pos  = eef.get("pos")   # (3,) ndarray
                quat = eef.get("quat")  # (4,) ndarray
                gpos = rs.get("gripper", {}).get("qpos")  # (2,) ndarray
                parts = [p for p in (pos, quat, gpos) if p is not None]
                if parts:
                    self._current_robot_state = _np.concatenate(
                        [_np.asarray(p).flatten() for p in parts]
                    ).tolist()  # [eef_pos(3), eef_quat(4), gripper_qpos(2)] = 9-dim
            elif rs is not None:
                arr = _np.asarray(rs)
                if arr.ndim >= 1 and arr.dtype.kind in ("f", "i", "u"):
                    self._current_robot_state = arr.flatten().tolist()
        except Exception:
            pass
        # Trajectory: record executed step
        if self._traj_recorder is not None:
            _ts_step = timed_action.get_timestep()
            _ph = getattr(self, "_grasp_phase", None)
            self._traj_recorder.record_step({
                "timestep": _ts_step,
                "action": timed_action.get_action().float().tolist(),
                "robot_state": self._current_robot_state,
                "episode_phase": max(0.0, min(1.0, _ts_step / self._episode_max_steps)),
                "grasp_phase": _ph.value if _ph else "NORMAL",
            })
        return result

    def _reset_loop_state(self) -> None:
        """Clear per-episode P1/P2 buffers and trajectory chunk counter."""
        super()._reset_loop_state()
        self._last_executed_action = None
        self._current_robot_state = None
        self._traj_chunk_count = 0
        self._recent_spreads.clear()  # O1: spread window is per-episode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_step_record(
        self,
        timestep: int,
        episode_phase: float,
        client_selected_idx: int,
        client_override: bool,
        n_candidates: int,
        server_scores: list[float],
        continuity_scores: list[float],
        combined_scores: list[float],
        selected_chunk: ActionChunk,
        delay_selected: int | None,
    ) -> None:
        """Write a step record for Phase-1 bundles (no candidate re-ranking)."""
        assert self._step_collector is not None
        first_action: list[float] | None = None
        if selected_chunk.timed_actions:
            first_action = selected_chunk.timed_actions[0].get_action().float().tolist()
        prev = self._last_executed_action
        exec_cont: float | None = None
        if prev is not None and first_action is not None:
            import torch as _torch
            exec_cont = float((_torch.tensor(first_action) - prev.float()).norm())
        _ph = getattr(self, "_grasp_phase", None)
        self._step_collector.record(
            episode_id=self._current_episode,
            timestep=timestep,
            client_selected_idx=client_selected_idx,
            client_override=client_override,
            n_candidates=n_candidates,
            server_scores=server_scores,
            continuity_scores=continuity_scores,
            combined_scores=combined_scores,
            execution_continuity=exec_cont,
            prev_action=prev.tolist() if prev is not None else None,
            selected_first_action=first_action,
            episode_phase=episode_phase,
            delay_selected=delay_selected,
            robot_state=self._current_robot_state,
            grasp_phase=_ph.value if _ph else "NORMAL",
            alpha_effective=self._mc_cfg.client_smooth_alpha,
        )

    @staticmethod
    def _passes_limit_check(chunk: ActionChunk, lo: float, hi: float) -> bool:
        """Return True if all actions in chunk are within [lo, hi]."""
        for ta in chunk.timed_actions:
            a = ta.get_action()
            if float(a.min()) < lo or float(a.max()) > hi:
                return False
        return True

    @staticmethod
    def _compute_spread_l2(candidates: list[ActionChunk]) -> float:
        """Mean pairwise L2 distance between candidate action chunks (client-side)."""
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


# ---------------------------------------------------------------------------
# Results helpers (mirrors run_libero_test.py)
# ---------------------------------------------------------------------------


def _save_multicand_summary(
    all_results: list,
    cfg: "MultiCandLiberoConfig",
    total_t: float,
    mc_bundles_recv: int,
    mc_client_overrides: int,
    mc_spread_slow_activations: int = 0,
    timing_output_dir: str | None = None,
) -> None:
    """Write aggregate + MC stats + optional retry stats (+ timing tables) to <results_dir>/summary.txt."""
    if not all_results:
        return

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_sr = sum(r.success for r in all_results) / len(all_results)
    avg_steps  = sum(r.steps   for r in all_results) / len(all_results)

    by_task: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_task[r.task_description].append(r)

    nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731

    lines: list[str] = [
        "=" * 72,
        "  LIBERO Multi-Candidate Evaluation — Summary",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "── Config ──────────────────────────────────────────────────────────",
        f"  suite                : {cfg.env_task}",
        f"  model                : {cfg.pretrained_name_or_path}",
        f"  obs_type             : {cfg.obs_type}",
        f"  fps                  : {cfg.fps}",
        f"  actions_per_chunk    : {cfg.actions_per_chunk}",
        f"  episodes_per_task    : {cfg.episodes_per_task}",
        f"  client_smooth_alpha  : {cfg.client_smooth_alpha}",
        f"  server_score_normalize: {cfg.server_score_normalize}",
        f"  spread_uncertainty_thr: {cfg.spread_uncertainty_threshold}",
        f"  spread_slow_alpha_scale: {cfg.spread_slow_alpha_scale}",
        f"  enable_gripper_sm    : {cfg.enable_gripper_sm}",
    ]
    if cfg.enable_gripper_sm:
        lines += [
            f"  rewind_mode          : {cfg.rewind_mode}",
            f"  rewind_buffer_steps  : {cfg.rewind_buffer_steps}",
            f"  rewind_warmup_steps  : {cfg.rewind_warmup_steps}",
            f"  max_empty_grasp_retries : {cfg.max_empty_grasp_retries}",
        ]

    lines += [
        "",
        "── Overall ─────────────────────────────────────────────────────────",
        f"  total_episodes       : {len(all_results)}",
        f"  overall_sr           : {overall_sr:.1%}",
        f"  avg_steps            : {avg_steps:.1f}",
        f"  total_time           : {total_t:.1f}s",
        f"  mc_bundles_recv      : {mc_bundles_recv}",
        f"  mc_client_overrides  : {mc_client_overrides}",
        f"  mc_spread_slow_acts  : {mc_spread_slow_activations}",
    ]

    # Retry stats — only populated when enable_gripper_sm=True
    smart_results = [r for r in all_results if isinstance(r, SmartEpisodeResult)]
    if smart_results and cfg.enable_gripper_sm:
        eps_with_retry = [r for r in smart_results if r.retries > 0]
        eps_no_retry   = [r for r in smart_results if r.retries == 0]
        total_retries  = sum(r.retries for r in smart_results)
        sr_with_retry  = (
            sum(r.success for r in eps_with_retry) / len(eps_with_retry)
            if eps_with_retry else float("nan")
        )
        sr_no_retry    = (
            sum(r.success for r in eps_no_retry) / len(eps_no_retry)
            if eps_no_retry else float("nan")
        )
        success_after_retry = sum(r.success_after_retry for r in smart_results)
        rescue_rate = (
            success_after_retry / len(eps_with_retry)
            if eps_with_retry else float("nan")
        )
        sr_lift = success_after_retry / len(all_results) if all_results else 0.0
        lines += [
            "",
            "── Retry Stats ─────────────────────────────────────────────────────",
            f"  total_retries        : {total_retries}",
            f"  eps_with_retry       : {len(eps_with_retry)} / {len(all_results)}",
            f"  sr_with_retry        : {nan_fmt(sr_with_retry)}"
            f"  ← final SR of episodes that needed retry (harder episodes)",
            f"  sr_no_retry          : {nan_fmt(sr_no_retry)}"
            f"  ← final SR of clean episodes (no retry triggered)",
            f"  sr_no_retry > sr_with_retry is expected: retried eps are harder.",
            f"  success_after_retry  : {success_after_retry}  ← episodes saved by SM",
            f"  rescue_rate          : {nan_fmt(rescue_rate)}"
            f"  ← success_after_retry / eps_with_retry (SM effectiveness)",
            f"  sr_lift (SM→no-SM)   : +{sr_lift:.1%}"
            f"  ← overall SR improvement vs baseline without SM",
        ]

    lines += [
        "",
        "── Per-Task ────────────────────────────────────────────────────────",
    ]
    for desc, eps in sorted(by_task.items()):
        sr = sum(r.success for r in eps) / len(eps) if eps else 0.0
        smart_eps = [r for r in eps if isinstance(r, SmartEpisodeResult)]
        retries_str = ""
        if smart_eps and cfg.enable_gripper_sm:
            task_retries = sum(r.retries for r in smart_eps)
            task_sar = sum(r.success_after_retry for r in smart_eps)
            retries_str = f"  retries={task_retries}  success_after_retry={task_sar}"
        lines.append(
            f"  [{sr:5.1%}]  eps={len(eps)}{retries_str}"
            f"\n           {desc}"
        )

    timing_lines = _read_timing_tables(timing_output_dir)
    if timing_lines:
        lines += ["", "── Timing Tables ────────────────────────────────────────────────────", ""]
        lines.extend(timing_lines)

    lines += ["", "=" * 72, ""]

    txt = "\n".join(lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(txt, encoding="utf-8")
    logging.info(f"[MultiCandTest] Summary saved → {summary_path}")


def _save_results(results: list[EpisodeResult], cfg: MultiCandLiberoConfig) -> dict:
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_description].append(r)

    task_stats = []
    for desc, eps in sorted(by_task.items()):
        sr = sum(r.success for r in eps) / len(eps) if eps else 0.0
        task_stats.append({
            "task_description": desc,
            "episodes": len(eps),
            "success_rate": sr,
            "avg_steps": sum(r.steps for r in eps) / len(eps) if eps else 0,
            "avg_duration_s": sum(r.duration_s for r in eps) / len(eps) if eps else 0,
        })

    overall_sr = sum(r.success for r in results) / len(results) if results else 0.0
    aggregate = {
        "total_episodes": len(results),
        "overall_success_rate": overall_sr,
        "per_task": task_stats,
        "config": {
            "policy_type": cfg.policy_type,
            "pretrained_name_or_path": cfg.pretrained_name_or_path,
            "env_task": cfg.env_task,
            "actions_per_chunk": cfg.actions_per_chunk,
            "fps": cfg.fps,
            "episodes_per_task": cfg.episodes_per_task,
            "client_smooth_alpha": cfg.client_smooth_alpha,
            "data_collect_dir": cfg.data_collect_dir,
        },
    }

    (out_dir / "episodes.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    logging.info(f"[MultiCandTest] Results saved to {out_dir}")
    return aggregate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@draccus.wrap()
def run_libero_multicand_test(cfg: MultiCandLiberoConfig):
    init_logging(console_level=cfg.log_level.upper())
    logging.info("[MultiCandTest] Config:\n" + pformat(asdict(cfg)))

    # ── Build envs ───────────────────────────────────────────────────────────
    from lerobot.envs.factory import make_env, make_env_config
    from lerobot.envs.utils import env_to_policy_features

    env_cfg = make_env_config(
        "libero",
        task=cfg.env_task,
        obs_type=cfg.obs_type,
        camera_name=cfg.camera_name,
    )
    if cfg.task_ids is not None:
        env_cfg.task_ids = cfg.task_ids
    if cfg.max_episode_steps is not None and hasattr(env_cfg, "episode_length"):
        env_cfg.episode_length = cfg.max_episode_steps

    logging.info(f"[MultiCandTest] Building LIBERO envs for suite '{cfg.env_task}' ...")
    envs_dict = make_env(env_cfg, n_envs=1)
    env_preprocessor, _ = env_cfg.get_env_processors()
    try:
        lerobot_features = env_to_policy_features(env_cfg)
    except Exception as exc:
        logging.warning(f"[MultiCandTest] Could not build lerobot features: {exc}. Using {{}}.")
        lerobot_features = {}

    task_list = [
        (suite, tid, env)
        for suite, task_envs in envs_dict.items()
        for tid, env in sorted(task_envs.items())
    ]
    logging.info(f"[MultiCandTest] Built {len(task_list)} task env(s)")

    if not task_list:
        logging.error("[MultiCandTest] No task environments created. Aborting.")
        return

    # ── Phase 3 outcome collector ─────────────────────────────────────────────
    outcome_collector: ClientOutcomeCollector | None = (
        ClientOutcomeCollector(cfg.data_collect_dir) if cfg.data_collect_dir else None
    )

    # ── Create client ────────────────────────────────────────────────────────
    first_suite, first_tid, first_env = task_list[0]
    first_task_desc = _get_task_description(first_env)

    client = MultiCandSimClient(
        config=cfg,
        env=first_env,
        env_preprocessor=env_preprocessor,
        lerobot_features=lerobot_features,
        task_description=first_task_desc,
    )

    all_results: list[EpisodeResult] = []

    if not client.start():
        logging.error("[MultiCandTest] Could not connect to policy server. Aborting.")
        for _, _, env in task_list:
            env.close()
        return

    if cfg.timing_output_dir:
        client.enable_timing(cfg.timing_output_dir)

    queue_monitor = None
    if cfg.queue_size_monitor_interval > 0:
        from ..helpers import QueueSizeMonitor
        queue_monitor = QueueSizeMonitor(
            data=client.action_queue_size,
            interval=cfg.queue_size_monitor_interval,
            path=cfg.queue_size_monitor_path,
        )
        queue_monitor.start()
        logging.info(
            f"[MultiCandTest] Queue size monitor started — "
            f"saving PNG every {cfg.queue_size_monitor_interval}s "
            f"to {cfg.queue_size_monitor_path}"
        )

    receiver = threading.Thread(
        target=client.receive_actions, daemon=True, name="mc-action-receiver"
    )
    receiver.start()

    # ── Task × episode loop ───────────────────────────────────────────────────
    t_all_start = time.perf_counter()
    global_ep = 0

    try:
        for task_idx, (suite_name, task_id, task_env) in enumerate(task_list):
            task_desc = _get_task_description(task_env)
            logging.info(
                f"[MultiCandTest] ══ Task {task_idx + 1}/{len(task_list)} "
                f"| suite={suite_name} | task_id={task_id} | desc='{task_desc}' ══"
            )
            client.env = task_env

            task_results: list[EpisodeResult] = []
            for ep_local in range(cfg.episodes_per_task):
                # ── Trajectory recorder: reset before episode ─────────────────
                if client._traj_recorder is not None:
                    client._traj_recorder.reset(global_ep, task_desc)

                result = client.run_episode(
                    episode_id=global_ep,
                    max_steps=cfg.max_episode_steps or 500,
                    first_episode=(global_ep == 0),
                    task_description=task_desc,
                )
                task_results.append(result)
                all_results.append(result)

                # ── Phase 3: record episode outcome ──────────────────────────
                if outcome_collector is not None:
                    outcome_collector.record(
                        episode_id=global_ep,
                        success=result.success,
                        steps=result.steps,
                        duration_s=result.duration_s,
                        sm_retries=getattr(result, "retries", 0),
                        success_after_retry=getattr(result, "success_after_retry", False),
                    )

                # ── Trajectory recorder: save after episode ───────────────────
                if client._traj_recorder is not None:
                    traj_path = client._traj_recorder.save(result.success, result.steps)
                    logging.info(f"[MultiCandTest] Trajectory saved → {traj_path}")

                global_ep += 1
                _retries = getattr(result, "retries", 0)
                _retry_tag = f"  sm_retries={_retries}" if _retries > 0 else ""
                logging.info(
                    f"[MultiCandTest] task={task_id} ep={ep_local}/{cfg.episodes_per_task - 1} "
                    f"success={result.success}  steps={result.steps}  "
                    f"duration={result.duration_s:.2f}s  "
                    f"mc_overrides={client._client_override_count}{_retry_tag}"
                )

            task_sr = sum(r.success for r in task_results) / len(task_results) if task_results else 0.0
            logging.info(
                f"[MultiCandTest] Task {task_id} summary: "
                f"success_rate={task_sr:.1%}  episodes={len(task_results)}"
            )

    finally:
        if queue_monitor is not None:
            queue_monitor.stop()
        client.stop()
        receiver.join(timeout=5.0)
        client.save_timing()
        for _, _, env in task_list:
            try:
                env.close()
            except Exception:
                pass

    # ── Final summary ─────────────────────────────────────────────────────────
    total_t = time.perf_counter() - t_all_start
    if all_results:
        overall_sr = sum(r.success for r in all_results) / len(all_results)
        avg_steps = sum(r.steps for r in all_results) / len(all_results)

        # Retry stats — populated only when enable_gripper_sm=True
        smart_results = [r for r in all_results if isinstance(r, SmartEpisodeResult)]
        retry_lines = ""
        if smart_results and cfg.enable_gripper_sm:
            nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731
            total_retries      = sum(r.retries for r in smart_results)
            eps_with_retry     = [r for r in smart_results if r.retries > 0]
            eps_no_retry       = [r for r in smart_results if r.retries == 0]
            sr_with_retry      = (
                sum(r.success for r in eps_with_retry) / len(eps_with_retry)
                if eps_with_retry else float("nan")
            )
            sr_no_retry        = (
                sum(r.success for r in eps_no_retry) / len(eps_no_retry)
                if eps_no_retry else float("nan")
            )
            success_after_retry = sum(r.success_after_retry for r in smart_results)
            rescue_rate = (
                success_after_retry / len(eps_with_retry) if eps_with_retry else float("nan")
            )
            sr_lift = success_after_retry / len(all_results) if all_results else 0.0
            retry_lines = (
                f"\n  ── Retry Stats ──"
                f"\n  total_retries        : {total_retries}"
                f"\n  eps_with_retry       : {len(eps_with_retry)}/{len(all_results)}"
                f"\n  sr_with_retry        : {nan_fmt(sr_with_retry)}"
                f"  (harder eps, retry triggered)"
                f"\n  sr_no_retry          : {nan_fmt(sr_no_retry)}"
                f"  (clean eps, no retry)"
                f"\n  success_after_retry  : {success_after_retry}"
                f"\n  rescue_rate          : {nan_fmt(rescue_rate)}"
                f"  (SM saved/retried)"
                f"\n  sr_lift (SM→no-SM)   : +{sr_lift:.1%}"
                f"  (overall SR gain from SM)"
            )

        logging.info(
            f"[MultiCandTest] ═══ Final summary ═══\n"
            f"  suite             : {cfg.env_task}\n"
            f"  total_episodes    : {len(all_results)}\n"
            f"  overall_sr        : {overall_sr:.1%}\n"
            f"  avg_steps         : {avg_steps:.1f}\n"
            f"  total_time        : {total_t:.2f}s\n"
            f"  mc_bundles_recv   : {client._bundle_recv_count}\n"
            f"  mc_client_overrides: {client._client_override_count}"
            f"{retry_lines}"
        )
        _save_multicand_summary(
            all_results, cfg, total_t,
            client._bundle_recv_count, client._client_override_count,
            client._spread_slow_activations,
            timing_output_dir=cfg.timing_output_dir,
        )
        if cfg.save_results:
            _save_results(all_results, cfg)
        if cfg.data_collect_dir:
            logging.info(
                f"[MultiCandTest] Data written to {cfg.data_collect_dir}/\n"
                "  server: candidates.jsonl   — per-step: jerk/vel_peak/score/spread_l2/spread_std\n"
                "  client: client_steps.jsonl — per-step: P0 selection, P1 continuity, P2 state\n"
                "  client: client_outcomes.jsonl — per-episode: success/steps/duration\n"
                "  Join server+client on (episode_id, timestep); outcomes on episode_id.\n"
                "  See analysis example: python -m lerobot.async_inference.sim_test.analyze_multicand"
            )
        if cfg.record_trajectory:
            logging.info(
                f"[MultiCandTest] Trajectories written to {cfg.trajectory_dir}/\n"
                "  ep{N:04d}.json — per-episode: chunk candidate arrays + executed steps\n"
                "  Visualize: python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory"
                f" --traj_dir={cfg.trajectory_dir} --out_dir=./mc_viz"
            )
    else:
        logging.warning("[MultiCandTest] No episodes completed.")


if __name__ == "__main__":
    run_libero_multicand_test()
