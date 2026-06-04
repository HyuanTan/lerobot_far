"""Multi-candidate action chunk policy server.

Drop-in replacement for policy_server.py that generates N action chunk
candidates per inference call, scores them server-side, and returns either
the single best candidate (Phase 1) or a ranked shortlist for client-side
selection (Phase 2).

Candidate diversity comes from two orthogonal axes:
  A1 — Gaussian noise diversity: batch_size > 1 causes sample_noise() to draw
       independent noise vectors → naturally different trajectories.
  B  — RTC inference_delay variants: different delays bias the denoiser toward
       conservative (long delay) or aggressive (short delay) re-planning.

For N=4 candidates the server runs 2 sequential batched forward passes:
  pass 0 : batch_size=n_per_delay, delay = max(0, base_delay - delay_delta)
  pass 1 : batch_size=n_per_delay, delay = base_delay + delay_delta
giving 2 × n_per_delay = N candidates with both noise and delay diversity.

Phase 1 (default):
  Server picks the top-1 by composite score, packs it into ActionBundle with
  empty candidates list, serialises the bundle in the existing Actions.data
  gRPC field.  Clients that know ActionBundle unpack it; legacy clients that
  expect ActionChunk fall back gracefully (the server can be configured to
  return a plain ActionChunk for full backward compat).

Phase 2 (top_k > 1):
  Server returns ActionBundle.candidates with the top-K ranked chunks so the
  client can apply its own safety / state-machine filter and pick one.

Data collection for Phase 3 (value function training):
  A CandidateDataCollector writes per-step JSONL records containing candidate
  stats and server scores.  The client fills in the episode outcome at the
  end of each episode via a lightweight TCP callback or a shared JSONL file.

Usage::

    # Phase 1 — server picks best-1, client unchanged
    python -m lerobot.async_inference.multi_candidate_server \\
        --host=127.0.0.1 --port=8080 --fps=30 \\
        --n_candidates=4 --top_k=1 \\
        --delay_delta=1 \\
        --data_collect_dir=./multicand_data

    # Phase 2 — server returns top-2, client selects
    python -m lerobot.async_inference.multi_candidate_server \\
        --host=127.0.0.1 --port=8080 --fps=30 \\
        --n_candidates=4 --top_k=2 \\
        --delay_delta=1

    # Then launch run_libero_multicand_test.py (Phase 2) or the regular
    # run_libero_test.py (Phase 1 — no changes needed there).
"""

import json
import logging
import math
import threading
import time
from collections import deque
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import grpc
import torch

from lerobot.transport import services_pb2_grpc  # type: ignore
from lerobot.utils.utils import init_logging

from .configs import PolicyServerConfig
from .helpers import ActionBundle, ActionChunk, CandidateMeta, TimedObservation, get_logger
from .policy_server import PolicyServer, _RTC_CAPABLE_POLICIES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiCandidateServerConfig(PolicyServerConfig):
    """PolicyServerConfig extended with multi-candidate generation settings."""

    # ── Candidate generation ─────────────────────────────────────────────────
    n_candidates: int = field(
        default=4,
        metadata={
            "help": (
                "Total number of action chunk candidates to generate per inference call. "
                "Must be even: n_candidates // 2 candidates per delay variant. "
                "Set to 1 to disable multi-candidate (identical to PolicyServer)."
            )
        },
    )
    delay_delta: int = field(
        default=1,
        metadata={
            "help": (
                "Half-range for inference_delay variants. "
                "Two delay values are used: max(0, base-delta) and base+delta. "
                "Set to 0 to use only the base delay (noise diversity only)."
            )
        },
    )

    # ── Server-side scoring ──────────────────────────────────────────────────
    w_jerk: float = field(
        default=1.0,
        metadata={"help": "Weight for jerk (smoothness) penalty in composite score."},
    )
    w_vel_peak: float = field(
        default=0.5,
        metadata={"help": "Weight for peak joint-velocity penalty in composite score."},
    )
    w_consistency: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for consistency penalty (deviation from ensemble mean). "
                "Penalises outlier candidates. "
                "CAUTION: set to 0.0 (default) for pick-place tasks — non-zero values "
                "select 'closest-to-average' trajectories which at policy decision points "
                "(multi-modal output) picks the least decisive candidate and causes repeated "
                "hesitation.  Enable only when the policy is known to be unimodal or "
                "when outlier rejection (e.g. failed denoising) is the primary concern."
            )
        },
    )

    # ── Return policy ────────────────────────────────────────────────────────
    top_k: int = field(
        default=1,
        metadata={
            "help": (
                "Number of top-ranked candidates to include in ActionBundle.candidates "
                "for client-side selection (Phase 2). "
                "1 = server picks best-1 (Phase 1, backward-compatible with plain clients)."
            )
        },
    )
    return_plain_action_chunk: bool = field(
        default=False,
        metadata={
            "help": (
                "When True AND top_k==1, serialise the result as a plain ActionChunk "
                "instead of ActionBundle.  Enables full backward compatibility with "
                "clients that were not updated for multi-candidate support."
            )
        },
    )

    # ── Phase 3 data collection ──────────────────────────────────────────────
    data_collect_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "If set, write per-step candidate metadata JSONL to this directory "
                "for Phase 3 value-function training. None = disabled."
            )
        },
    )
    record_all_candidates: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, include ALL N server-generated candidates (not just top_k) "
                "in ActionBundle.all_candidates / all_candidate_meta. "
                "The client's trajectory recorder saves them in the episode JSON so "
                "non-top_k candidates are visible for offline analysis. "
                "Increases gRPC payload by ~(N-top_k)×T×D×4 bytes per inference step."
            )
        },
    )

    # ── Output root ──────────────────────────────────────────────────────────
    save_root_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional root directory prepended to all output paths "
                "(data_collect_dir, timing_output_dir). "
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
            if self.timing_output_dir is not None:
                self.timing_output_dir = str(root / self.timing_output_dir)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ActionChunkScorer:
    """Scores a batch of candidate action chunks using analytic metrics.

    All metrics are computed directly from the [N, T, D] candidates tensor —
    no model calls, no extra GPU memory.  Total cost ≈ 0.5 ms for N=4, T=16, D=7.

    Scoring space
    ─────────────
    Jerk and peak-velocity are computed in physical action space (degrees for
    SO-101) when ``action_unnorm_scale`` is provided.  Without it they fall
    back to normalised-model space, which is scale-inconsistent across joints
    and systematically disadvantages decisive high-amplitude motions.

    Scale extraction per normalization mode
    ────────────────────────────────────────
    The inverse transform for differences (velocity = Δaction) only involves
    the scale factor; the mean/offset cancels:
      MEAN_STD   : physical_vel = normalised_vel × std          → scale = std
      MIN_MAX     : physical_vel = normalised_vel × (max−min)/2 → scale = (max−min)/2
      QUANTILES   : physical_vel = normalised_vel × (q99−q01)/2 → scale = (q99−q01)/2
      QUANTILE10  : physical_vel = normalised_vel × (q90−q10)/2 → scale = (q90−q10)/2
      IDENTITY    : no conversion needed                         → scale = None

    Models:
      SmolVLA ACTION → MEAN_STD  → scale = std  [D]
      PI05    ACTION → QUANTILES → scale = (q99−q01)/2  [D]
    """

    def __init__(
        self,
        cfg: "MultiCandidateServerConfig",
        action_unnorm_scale: torch.Tensor | None = None,
    ):
        self.w_jerk = cfg.w_jerk
        self.w_vel_peak = cfg.w_vel_peak
        self.w_consistency = cfg.w_consistency
        # [D] float32 on CPU; moved to device on first score() call.
        # None → score in normalised space (legacy / IDENTITY normalisation).
        self._unnorm_scale: torch.Tensor | None = (
            action_unnorm_scale.float().cpu() if action_unnorm_scale is not None else None
        )

    def _get_vel_phys(
        self, vel_norm: torch.Tensor
    ) -> torch.Tensor:
        """Scale [N, T-1, D] normalised velocity to physical space if scale is known."""
        if self._unnorm_scale is None:
            return vel_norm
        scale = self._unnorm_scale.to(device=vel_norm.device, dtype=vel_norm.dtype)
        return vel_norm * scale.unsqueeze(0).unsqueeze(0)  # broadcast [D] → [N, T-1, D]

    @torch.no_grad()
    def score(self, candidates: torch.Tensor) -> torch.Tensor:
        """Compute composite score for each candidate (higher = better).

        Args:
            candidates: [N, T, D] float32 tensor in model (normalised) space.

        Returns:
            scores: [N] float32 tensor.
        """
        N = candidates.shape[0]
        dev = candidates.device
        scores = torch.zeros(N, dtype=torch.float32, device=dev)

        vel_norm = candidates[:, 1:] - candidates[:, :-1]   # [N, T-1, D] normalised
        vel      = self._get_vel_phys(vel_norm)              # physical space (or normalised)

        # ── Smoothness: L2 jerk (2nd-order finite difference, physical space) ──
        if vel.shape[1] > 1:
            jerk = (vel[:, 1:] - vel[:, :-1]).norm(dim=-1).mean(dim=-1)  # [N]
        else:
            jerk = torch.zeros(N, device=dev)
        scores -= self.w_jerk * jerk

        # ── Peak joint velocity (physical space) ──────────────────────────────
        vel_peak = vel.abs().amax(dim=(1, 2))                # [N]
        scores -= self.w_vel_peak * vel_peak

        # ── Consistency: deviation from ensemble mean (normalised space) ──────
        # NOTE: w_consistency defaults to 0.0 — see MultiCandidateServerConfig.
        # This term selects for "closest-to-average" candidates, which at policy
        # decision points (multi-modal output) selects the least decisive trajectory
        # and causes repeated hesitation.  Enable only with careful tuning.
        if self.w_consistency > 0.0:
            mean_chunk = candidates.mean(dim=0, keepdim=True)      # [1, T, D]
            deviation  = (candidates - mean_chunk).norm(dim=-1).mean(dim=-1)  # [N]
            scores -= self.w_consistency * deviation

        return scores  # higher is better

    @torch.no_grad()
    def per_candidate_stats(
        self, candidates: torch.Tensor, scores: torch.Tensor
    ) -> list[dict]:
        """Return per-candidate stat dicts (for logging + Phase 3 collection)."""
        N = candidates.shape[0]
        vel_norm = candidates[:, 1:] - candidates[:, :-1]
        vel      = self._get_vel_phys(vel_norm)
        if vel.shape[1] > 1:
            jerk = (vel[:, 1:] - vel[:, :-1]).norm(dim=-1).mean(dim=-1)
        else:
            jerk = torch.zeros(N, device=candidates.device)
        vel_peak = vel.abs().amax(dim=(1, 2))
        return [
            {
                "jerk": jerk[i].item(),
                "vel_peak": vel_peak[i].item(),
                "server_score": scores[i].item(),
            }
            for i in range(N)
        ]

    @torch.no_grad()
    def compute_spread(self, candidates: torch.Tensor) -> dict[str, float]:
        """Compute inter-candidate diversity metrics (P0).

        High spread indicates high model uncertainty at this observation —
        candidate selection has larger expected impact on outcome.

        Args:
            candidates: [N, T, D] float32 tensor in model (normalised) space.

        Returns:
            dict with:
              spread_l2  — mean pairwise L2 distance between candidates
                           (averaged over time steps); proxy for ensemble entropy.
              spread_std — std of per-candidate chunk L2-norms;
                           detects outlier candidates.
        """
        N = candidates.shape[0]
        if N <= 1:
            return {"spread_l2": 0.0, "spread_std": 0.0}

        # Mean pairwise L2: [N,1,T,D] - [1,N,T,D] → [N,N,T,D] → norm → mean(T) → [N,N]
        diffs = candidates.unsqueeze(1) - candidates.unsqueeze(0)   # [N,N,T,D]
        pairwise = diffs.norm(dim=-1).mean(dim=-1)                  # [N,N]
        triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=candidates.device), diagonal=1)
        spread_l2 = float(pairwise[triu].mean().item())

        # Std of per-candidate chunk norms
        chunk_norms = candidates.norm(dim=-1).mean(dim=-1)          # [N]
        spread_std = float(chunk_norms.std().item()) if N > 1 else 0.0

        return {"spread_l2": spread_l2, "spread_std": spread_std}


# ---------------------------------------------------------------------------
# Phase 3 data collector
# ---------------------------------------------------------------------------


class CandidateDataCollector:
    """Writes per-step candidate records to a JSONL file for Phase 3 training.

    Record schema::

        {
          "episode_id":    int,
          "timestep":      int,
          "base_delay":    int,
          "n_candidates":  int,
          "candidates": [
            {"delay": int, "noise_idx": int, "jerk": float, "vel_peak": float,
             "server_score": float},
            ...
          ],
          "server_selected_idx": int,   # index into candidates
          "episode_outcome": null        # filled retroactively at episode end
        }

    Call fill_episode_outcome(episode_id, success) at episode end; the
    collector retroactively patches all records for that episode.
    """

    def __init__(self, output_dir: str):
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._out_dir / "candidates.jsonl"
        self._lock = threading.Lock()
        # episode_id → list of line-offsets in the JSONL for retroactive patching
        self._ep_records: dict[int, list[int]] = {}
        self._logger = logging.getLogger("CandidateDataCollector")
        self._logger.info(f"[data] Recording to {self._jsonl_path}")

    def record_step(
        self,
        episode_id: int,
        timestep: int,
        base_delay: int,
        candidate_meta: list[CandidateMeta],
        server_selected_idx: int,
        candidate_spread_l2: float = 0.0,
        candidate_spread_std: float = 0.0,
    ) -> None:
        record = {
            "episode_id": episode_id,
            "timestep": timestep,
            "base_delay": base_delay,
            "n_candidates": len(candidate_meta),
            # P0: inter-candidate diversity (model uncertainty proxy)
            "candidate_spread_l2": round(candidate_spread_l2, 6),
            "candidate_spread_std": round(candidate_spread_std, 6),
            "candidates": [
                {
                    "delay": m.inference_delay,
                    "noise_idx": m.noise_idx,
                    "jerk": m.jerk,
                    "vel_peak": m.vel_peak,
                    "server_score": m.server_score,
                }
                for m in candidate_meta
            ],
            "server_selected_idx": server_selected_idx,
            "episode_outcome": None,
        }
        with self._lock:
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                offset = f.tell()
                f.write(json.dumps(record) + "\n")
            self._ep_records.setdefault(episode_id, []).append(offset)

    def fill_episode_outcome(self, episode_id: int, success: bool) -> None:
        """Patch all records for `episode_id` with the known outcome."""
        offsets = self._ep_records.pop(episode_id, [])
        if not offsets:
            return
        try:
            with self._lock:
                with self._jsonl_path.open("r+b") as f:
                    for offset in offsets:
                        f.seek(offset)
                        line = f.readline().rstrip(b"\n")
                        rec = json.loads(line)
                        rec["episode_outcome"] = bool(success)
                        new_line = json.dumps(rec).encode()
                        # Lines must be same length or shorter (pad if needed)
                        if len(new_line) <= len(line):
                            new_line = new_line.ljust(len(line))
                        f.seek(offset)
                        f.write(new_line + b"\n")
            self._logger.info(
                f"[data] Patched {len(offsets)} records for ep{episode_id} "
                f"outcome={'success' if success else 'failed'}"
            )
        except Exception as exc:
            self._logger.warning(f"[data] fill_episode_outcome failed: {exc}")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class MultiCandidatePolicyServer(PolicyServer):
    """PolicyServer that generates N candidate action chunks per inference call.

    Inherits the full gRPC stack from PolicyServer; only _predict_action_chunk
    is overridden.

    Candidate generation strategy:
      - 2 RTC inference_delay variants: [base-delta, base+delta]
      - n_candidates // 2 noise-diverse samples per delay (one batched forward pass)
      - Total: 2 × (n_candidates // 2) = n_candidates candidates

    Scoring: composite analytic score (jerk + vel_peak + consistency), higher = better.

    Return policy:
      Phase 1 (top_k=1): wraps best-1 in ActionBundle (selected only).
      Phase 2 (top_k>1): wraps top-K ranked chunks in ActionBundle.candidates.

    Data collection: writes per-step JSONL when data_collect_dir is set.
    """

    prefix = "multi_cand_server"
    logger = get_logger(prefix)

    def __init__(self, config: MultiCandidateServerConfig):
        super().__init__(config)
        self._mc_cfg = config
        # Extract per-dim physical scale AFTER super().__init__ (loads policy + normalizer)
        _unnorm_scale = self._extract_action_unnorm_scale()
        self._scorer = ActionChunkScorer(config, action_unnorm_scale=_unnorm_scale)
        self._collector: CandidateDataCollector | None = (
            CandidateDataCollector(config.data_collect_dir)
            if config.data_collect_dir
            else None
        )
        self._ep_counter = 0
        self._ep_lock = threading.Lock()
        _scale_info = (
            f"physical (D={_unnorm_scale.shape[0]})" if _unnorm_scale is not None else "normalised"
        )
        self.logger.info(
            f"[mc] MultiCandidatePolicyServer | n_candidates={config.n_candidates} "
            f"top_k={config.top_k} delay_delta={config.delay_delta} "
            f"scoring_space={_scale_info} "
            f"w_jerk={config.w_jerk} w_vel_peak={config.w_vel_peak} "
            f"w_consistency={config.w_consistency} "
            f"data_collect={'on' if config.data_collect_dir else 'off'}"
        )

    def _extract_action_unnorm_scale(self) -> torch.Tensor | None:
        """Extract per-dimension physical scale for jerk/velocity scoring.

        Returns a [D] float32 CPU tensor such that:
            physical_velocity = normalised_velocity * scale
        for the action normalization mode used by the loaded policy.

        Returns None when the action space is already physical (IDENTITY mode)
        or when the normalizer is unavailable — scoring falls back to normalised
        space (original behaviour).

        Supported modes (covers SmolVLA=MEAN_STD and PI05=QUANTILES):
          MEAN_STD   → scale = std
          MIN_MAX    → scale = (max − min) / 2
          QUANTILES  → scale = (q99 − q01) / 2
          QUANTILE10 → scale = (q90 − q10) / 2
          IDENTITY   → returns None (no conversion)
        """
        from lerobot.configs import FeatureType, NormalizationMode
        from lerobot.utils.constants import ACTION

        if self._normalizer_step is None:
            self.logger.info("[mc] unnorm_scale: normalizer unavailable → scoring in normalised space")
            return None

        norm_mode = self._normalizer_step.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)
        stats = self._normalizer_step._tensor_stats.get(ACTION)

        if norm_mode == NormalizationMode.IDENTITY or stats is None:
            self.logger.info(f"[mc] unnorm_scale: ACTION={norm_mode.value} → scoring in normalised space")
            return None

        scale: torch.Tensor | None = None
        if norm_mode == NormalizationMode.MEAN_STD:
            scale = stats.get("std")
        elif norm_mode == NormalizationMode.MIN_MAX:
            lo, hi = stats.get("min"), stats.get("max")
            if lo is not None and hi is not None:
                scale = (hi - lo) / 2.0
        elif norm_mode == NormalizationMode.QUANTILES:
            q01, q99 = stats.get("q01"), stats.get("q99")
            if q01 is not None and q99 is not None:
                scale = (q99 - q01) / 2.0
        elif norm_mode == NormalizationMode.QUANTILE10:
            q10, q90 = stats.get("q10"), stats.get("q90")
            if q10 is not None and q90 is not None:
                scale = (q90 - q10) / 2.0

        if scale is None:
            self.logger.warning(
                f"[mc] unnorm_scale: ACTION={norm_mode.value} but required stats missing "
                "→ falling back to normalised space"
            )
            return None

        scale_cpu = scale.float().cpu()
        self.logger.info(
            f"[mc] unnorm_scale: ACTION={norm_mode.value} "
            f"scale_min={scale_cpu.min():.4f} scale_max={scale_cpu.max():.4f} "
            f"→ scoring in physical space"
        )
        return scale_cpu

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def _predict_action_chunk(
        self, observation_t: TimedObservation
    ) -> tuple[ActionChunk | ActionBundle, dict]:
        """Override: generate N candidates, score, return best / top-K."""
        n = self._mc_cfg.n_candidates
        rtc_on = self._rtc_enabled and self.policy_type in _RTC_CAPABLE_POLICIES

        # n_candidates=1: identical to base class (no multi-candidate overhead)
        if n <= 1:
            chunk, timings = super()._predict_action_chunk(observation_t)
            bundle = ActionBundle(selected=chunk)
            if self._collector is not None:
                with self._ep_lock:
                    if observation_t.is_episode_start:
                        self._ep_counter += 1
                    ep_id = self._ep_counter
                self._collector.record_step(
                    episode_id=ep_id,
                    timestep=observation_t.get_timestep(),
                    base_delay=observation_t.inference_delay,
                    candidate_meta=[],
                    server_selected_idx=0,
                    candidate_spread_l2=0.0,
                    candidate_spread_std=0.0,
                )
            return self._finalise(bundle, observation_t), timings

        # ── Episode counter (for Phase 3 data collection) ──────────────────
        with self._ep_lock:
            if observation_t.is_episode_start:
                self._ep_counter += 1
            ep_id = self._ep_counter

        # ── Preprocess observation once (shared across all candidates) ──────
        t0 = time.perf_counter()
        observation, raw_state, leftover = self._prepare_obs_and_leftover(observation_t)
        prepare_ms = (time.perf_counter() - t0) * 1000

        # ── Determine delay variants ────────────────────────────────────────
        # Priority (RTC enabled):
        #   1. p50/p95 pair from client LatencyTracker  — data-driven, tracks real spread
        #   2. base_delay ± delay_delta                 — fixed fallback (warmup / no tracker)
        # RTC disabled: single delay, all diversity from stochastic denoising noise.
        base_delay: int = observation_t.inference_delay       # max / p95
        delay_low: int = getattr(observation_t, "inference_delay_low", 0)  # p50 (0=unavailable)
        delta = self._mc_cfg.delay_delta

        if rtc_on:
            n_per_delay = max(1, n // 2)
            if delay_low > 0 and delay_low < base_delay:
                # Data-driven pair: [p50 (optimistic/strong-guidance), max/p95 (conservative/weak)]
                delays = [delay_low, base_delay]
                self.logger.debug(
                    f"[mc] delay pair from latency tracker: low={delay_low} high={base_delay}"
                )
            elif delta > 0:
                # Fallback: fixed ±delta (warmup or p50==p95, latency very stable)
                delays = [max(0, base_delay - delta), base_delay + delta]
            else:
                delays = [base_delay]
                n_per_delay = n
        else:
            # Single delay: all N candidates share delay, noise provides diversity
            delays = [base_delay]
            n_per_delay = n

        # Collapse duplicate delays (e.g. when p50 rounds to same steps as p95)
        if len(delays) == 2 and delays[0] == delays[1]:
            delays = [delays[0]]
            n_per_delay = n

        # ── Generate candidates in batched forward passes ───────────────────
        t1 = time.perf_counter()
        all_model_chunks: list[torch.Tensor] = []   # each [n_per_delay, T, D] model-space
        delay_labels: list[int] = []
        noise_labels: list[int] = []

        for delay in delays:
            batch_obs = self._expand_obs(observation, n_per_delay)
            batch_leftover = (
                leftover.unsqueeze(0).expand(n_per_delay, -1, -1).contiguous()
                if leftover is not None else None
            )
            with torch.no_grad():
                chunk_batch = self.policy.predict_action_chunk(
                    batch_obs,
                    inference_delay=delay,
                    prev_chunk_left_over=batch_leftover,
                )  # [n_per_delay, T, D] or [n_per_delay, T', D]
            if chunk_batch.ndim != 3:
                chunk_batch = chunk_batch.unsqueeze(0)
            chunk_batch = chunk_batch[:, : self.actions_per_chunk, :]  # trim
            all_model_chunks.append(chunk_batch)
            delay_labels.extend([delay] * n_per_delay)
            noise_labels.extend(list(range(n_per_delay)))

        infer_ms = (time.perf_counter() - t1) * 1000

        # [N, T, D] in model-space (normalised, on GPU)
        candidates_model = torch.cat(all_model_chunks, dim=0)  # [N, T, D]
        N_actual = candidates_model.shape[0]

        # ── Score candidates + diversity (P0) ─────────────────────────────
        scores = self._scorer.score(candidates_model.float())           # [N]
        stats  = self._scorer.per_candidate_stats(candidates_model.float(), scores)
        spread = self._scorer.compute_spread(candidates_model.float())  # P0

        # ── Postprocess each candidate ─────────────────────────────────────
        t2 = time.perf_counter()
        action_chunks: list[ActionChunk] = []
        candidate_meta: list[CandidateMeta] = []

        for i in range(N_actual):
            cand_model = candidates_model[i:i+1]  # [1, T, D]
            post_actions = self._postprocess_chunk(cand_model, observation_t)
            original_actions_i = cand_model.squeeze(0).detach().cpu()
            timed = self._time_action_chunk(
                observation_t.get_timestamp(),
                list(post_actions),
                observation_t.get_timestep(),
            )
            action_chunks.append(
                ActionChunk(timed_actions=timed, original_actions=original_actions_i)
            )
            candidate_meta.append(
                CandidateMeta(
                    inference_delay=delay_labels[i],
                    noise_idx=noise_labels[i],
                    jerk=stats[i]["jerk"],
                    vel_peak=stats[i]["vel_peak"],
                    server_score=stats[i]["server_score"],
                )
            )

        postprocess_ms = (time.perf_counter() - t2) * 1000

        # ── Rank and select ────────────────────────────────────────────────
        ranked_indices = scores.argsort(descending=True).tolist()
        top_k = min(self._mc_cfg.top_k, N_actual)
        top_k_indices = ranked_indices[:top_k]
        best_idx = ranked_indices[0]

        bundle = ActionBundle(
            selected=action_chunks[best_idx],
            candidates=[action_chunks[i] for i in top_k_indices],
            candidate_meta=[candidate_meta[i] for i in top_k_indices],
            selected_score=scores[best_idx].item(),
            server_selected_idx=0,  # position 0 in candidates is the best
        )

        # ── Full-N candidates for offline analysis ─────────────────────────
        # Populated only when record_all_candidates=True to avoid unnecessary
        # gRPC payload growth in production runs.
        if self._mc_cfg.record_all_candidates:
            bundle.all_candidates = [action_chunks[i] for i in ranked_indices]
            bundle.all_candidate_meta = [candidate_meta[i] for i in ranked_indices]

        # ── Phase 3 data collection ────────────────────────────────────────
        if self._collector is not None:
            self._collector.record_step(
                episode_id=ep_id,
                timestep=observation_t.get_timestep(),
                base_delay=base_delay,
                candidate_meta=candidate_meta,
                server_selected_idx=best_idx,
                candidate_spread_l2=spread["spread_l2"],
                candidate_spread_std=spread["spread_std"],
            )

        total_ms = prepare_ms + infer_ms + postprocess_ms
        _delay_src = "p50/p95" if (delay_low > 0 and delay_low < base_delay) else f"base±{delta}"
        self.logger.info(
            f"[mc] Obs #{observation_t.get_timestep()} | "
            f"N={N_actual} delays={delays}({_delay_src}) | "
            f"best_idx={best_idx} score={scores[best_idx]:.3f} "
            f"jerk={stats[best_idx]['jerk']:.4f} "
            f"spread_l2={spread['spread_l2']:.4f} | "
            f"prepare={prepare_ms:.1f}ms infer={infer_ms:.1f}ms "
            f"post={postprocess_ms:.1f}ms total={total_ms:.1f}ms"
        )

        timings = {
            "prepare_ms": prepare_ms,
            "preprocess_ms": 0.0,
            "infer_ms": infer_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
        }
        return self._finalise(bundle, observation_t), timings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_obs_and_leftover(
        self, observation_t: TimedObservation
    ) -> tuple[dict, None, torch.Tensor | None]:
        """Preprocess obs and resolve leftover (mirrors PolicyServer logic)."""
        from lerobot.utils.constants import OBS_STATE
        from .helpers import jpeg_decode_images_in_raw_obs, raw_observation_to_observation
        from .policy_server import reanchor_relative_rtc_prefix

        if getattr(observation_t, "obs_pre_mapped", False):
            observation = observation_t.get_observation()
            if getattr(observation_t, "jpeg_images", False):
                observation = jpeg_decode_images_in_raw_obs(observation)
            for key, val in observation.items():
                if "image" in key and isinstance(val, torch.Tensor) and val.dtype == torch.uint8:
                    observation[key] = val.float() / 255.0
            raw_state = None
        else:
            raw_obs = observation_t.get_observation()
            if getattr(observation_t, "jpeg_images", False):
                raw_obs = jpeg_decode_images_in_raw_obs(raw_obs)
            observation = raw_observation_to_observation(
                raw_obs,
                self.lerobot_features,
                self.policy_image_features,
                skip_resize=getattr(observation_t, "skip_server_resize", False),
            )
            raw_state = observation.get(OBS_STATE) if (
                self._rtc_enabled and self._use_relative_actions
                and self._relative_step is not None
            ) else None

        observation = self.preprocessor(observation)

        leftover: torch.Tensor | None = None
        if observation_t.leftover_actions is not None and self._rtc_enabled:
            if self._use_relative_actions and self._relative_step is not None and raw_state is not None:
                leftover = reanchor_relative_rtc_prefix(
                    observation_t.leftover_actions,
                    raw_state,
                    self._relative_step,
                    self._normalizer_step,
                    self.device,
                )
            else:
                leftover = observation_t.leftover_actions.to(self.device)

        return observation, raw_state, leftover

    @staticmethod
    def _expand_obs(observation: dict, n: int) -> dict:
        """Expand a batch_size=1 preprocessed observation to batch_size=n.

        Tensor values are expanded (no copy of data).  Strings and scalars
        are kept as-is (the model handles them per-batch).
        """
        expanded = {}
        for k, v in observation.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1:
                expanded[k] = v.expand(n, *v.shape[1:]).contiguous()
            else:
                expanded[k] = v
        return expanded

    def _postprocess_chunk(
        self,
        model_chunk: torch.Tensor,  # [1, T, D]
        observation_t: TimedObservation,
    ) -> list[torch.Tensor]:
        """Apply postprocessor to a single candidate; return list of action tensors."""
        _, chunk_size, _ = model_chunk.shape
        processed = []
        for i in range(chunk_size):
            processed.append(self.postprocessor(model_chunk[:, i, :]))
        stacked = torch.stack(processed, dim=1).squeeze(0)  # [T, D]
        return list(stacked.detach().cpu())

    def _finalise(
        self, bundle: ActionBundle, observation_t: TimedObservation
    ) -> ActionChunk | ActionBundle:
        """Return plain ActionChunk or ActionBundle depending on config."""
        if self._mc_cfg.return_plain_action_chunk and self._mc_cfg.top_k == 1:
            return bundle.selected
        return bundle

    # ------------------------------------------------------------------
    # Data collection: episode outcome reporting
    # ------------------------------------------------------------------

    def report_episode_outcome(self, episode_id: int, success: bool) -> None:
        """Called by the test runner after an episode ends (Phase 3 patching)."""
        if self._collector is not None:
            self._collector.fill_episode_outcome(episode_id, success)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def serve(cfg: MultiCandidateServerConfig):
    init_logging(console_level=cfg.log_level.upper())
    logging.info(pformat(asdict(cfg)))

    server_instance = MultiCandidatePolicyServer(cfg)
    if cfg.timing_output_dir:
        server_instance.enable_timing(cfg.timing_output_dir)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    server_instance.logger.info(
        f"MultiCandidatePolicyServer started on {cfg.host}:{cfg.port} | "
        f"n_candidates={cfg.n_candidates} top_k={cfg.top_k}"
    )
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server_instance.logger.info("Keyboard interrupt — stopping")
        server.stop(grace=2)
    finally:
        server_instance.save_timing()
        server_instance.stop()
        server_instance.logger.info("MultiCandidatePolicyServer terminated")


if __name__ == "__main__":
    serve()
