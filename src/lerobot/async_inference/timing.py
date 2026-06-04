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

"""Fine-grained per-step timing records for async-inference client and server.

Four record types — two per side:
  Client: ClientObsSentRecord        — per observation sent (prep + serialize + gRPC)
          ClientChunkReceivedRecord   — per action chunk received from server
  Server: ServerRecvRecord           — per observation received over gRPC
          ServerInferRecord          — per inference cycle (queue wait + full pipeline)

TimingRecorder accumulates records across steps and saves on demand to:
  <output_dir>/<prefix>_records.jsonl   — one JSON object per line (all steps)
  <output_dir>/<prefix>_summary.json    — per-field statistics (mean/std/p50/p95/p99/max)

Typical usage (client side)::

    recorder = TimingRecorder("./timing", "client_obs_sent")
    recorder.add(ClientObsSentRecord(wall_time=..., episode=0, ...))
    # ... at run end ...
    recorder.save()
    recorder.log_summary()
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Client-side records ───────────────────────────────────────────────────────

@dataclass
class ClientObsSentRecord:
    """Client-side timing for one observation send cycle.

    Covers four sequential prep stages in control_loop_observation() plus the
    optional JPEG-encode, serialize, and gRPC-send stages in send_observation().

    Timeline (all ms):
      obs_capture_ms       – robot.get_observation() + resize_images_in_raw_obs()
      infer_delay_calc_ms  – LatencyTracker lookup
      leftover_collect_ms  – _orig_buf scan
      obs_build_ms         – _build_timed_observation()
      ────────────────────── total_prep_ms = sum of the four above
      jpeg_encode_ms       – jpeg_encode_images_in_raw_obs() (0.0 when JPEG disabled)
      serialize_ms         – pickle.dumps(TimedObservation) on the (optionally encoded) obs
      grpc_send_ms         – stub.SendObservations() blocking gRPC call

    Note: jpeg_encode_ms is NOT included in total_prep_ms so that total_prep_ms
    remains comparable between runs with and without JPEG encoding.
    """

    wall_time: float            # time.time() when obs send completed
    episode: int
    timestep: int
    obs_capture_ms: float       # _capture_raw_obs() + _preprocess_obs()
    infer_delay_calc_ms: float  # LatencyTracker.max() lookup + math.ceil()
    leftover_collect_ms: float  # scanning _orig_buf for remaining timesteps
    obs_build_ms: float         # _build_timed_observation()
    total_prep_ms: float        # sum of the four prep stages above (excl. JPEG)
    jpeg_encode_ms: float       # jpeg_encode_images_in_raw_obs(); 0.0 if disabled
    serialize_ms: float         # pickle.dumps(TimedObservation) — after JPEG encoding
    grpc_send_ms: float         # stub.SendObservations() call duration
    payload_kb: float           # serialised observation size in kilobytes
    must_go: bool               # whether this obs triggered immediate inference
    infer_delay: int            # RTC inference-delay hint sent to server (post-hysteresis, sent)
    leftover_steps: int         # number of leftover actions bundled in this obs
    # Tier 2 split-estimator control-flow state (defaults keep old recordings parseable):
    split_active: bool = False  # True = split-component formula used; False = bootstrap fallback
    infer_delay_raw: int = -1   # infer_delay BEFORE hysteresis (post-cap); -1 = not recorded.
                                # Differs from infer_delay only when hysteresis held the value.


@dataclass
class ClientChunkReceivedRecord:
    """Client-side timing for one action chunk received from the server.

    Timeline relative to the paired ClientObsSentRecord (same timestep):

      obs.timestamp  ──── jpeg_encode ──── serialize ──── grpc_send ────▶ send_wall
                                                                               │
                     ◀─────────────── round_trip_ms ────────────────────────  │
                     │    (server queue_wait + pipeline + srv_serialize         │
                     │     + net_s2c; does NOT include client send overhead)   │
                     ▼                                                          │
                receive_time (wall_time here; before pickle.loads)             │
                     │                                                          │
                     ├── deser_ms (pickle.loads) ──▶ receive_after_deser       │
                                                                               │
      complete_s  =  jpeg_encode_ms + serialize_ms + grpc_send_ms             │
                   + round_trip_ms + deser_ms                        ◀────────┘
                   (= receive_after_deser − obs.timestamp;
                    the quantity fed to LatencyTracker after Option B)
    """

    wall_time: float                    # time.time() before pickle.loads() (wire receipt)
    episode: int
    first_timestep: int
    last_timestep: int
    chunk_size: int                     # number of TimedActions in the chunk
    round_trip_ms: float                # send_wall → receive_time (before deser);
                                        # covers server queue_wait + pipeline + srv_serialize + net_s2c;
                                        # does NOT include client send overhead (jpeg/serialize/grpc)
    server_infer_ms: float              # full pipeline time reported by ActionChunk.inference_time_s
    deser_ms: float                     # pickle.loads(ActionChunk) duration
    queue_size_at_recv: int             # action queue depth at the moment this chunk arrived
    estimated_first_exec_lag_ms: float  # queue_size_at_recv × environment_dt: approximate time
                                        # before the first action from this chunk is executed.
                                        # For latest_only aggregate_fn this is an overestimate
                                        # (old queue is replaced, so actual lag ≈ 0–1 step).


# ── Server-side records ───────────────────────────────────────────────────────

@dataclass
class ServerRecvRecord:
    """Server-side timing for one observation received over gRPC."""

    wall_time: float            # time.time() when obs arrived at server
    timestep: int
    recv_deser_ms: float        # receive_bytes_in_chunks() + pickle.loads()
    one_way_ms: float           # obs.timestamp (obs-build start) → server arrival;
                                # includes client jpeg_encode + serialize + gRPC overhead
    adj_one_way_ms: float       # one_way_ms - client_send_overhead_ms (jpeg_encode only);
                                # closer to true network one-way latency
    enqueued: bool              # whether obs passed sanity checks and entered the queue


@dataclass
class ServerInferRecord:
    """Server-side timing for one complete inference cycle in GetActions."""

    wall_time: float            # time.time() when action chunk was dispatched (after throttle sleep)
    timestep: int
    queue_wait_ms: float        # time obs spent in observation_queue before GetActions dequeued it
    prepare_ms: float           # raw→lerobot observation format conversion
    preprocess_ms: float        # preprocessor pipeline (tokenize/normalize/device)
    infer_ms: float             # model forward pass only
    postprocess_ms: float       # postprocessor pipeline (unnormalize/device)
    total_pipeline_ms: float    # prepare + preprocess + infer + postprocess
    serialize_ms: float         # pickle.dumps(ActionChunk)
    throttle_sleep_ms: float    # time.sleep() from inference_latency throttle (0.0 when policy is slow)
    infer_delay: int            # inference_delay hint forwarded to the policy
    leftover_used: bool         # whether leftover_actions were sent to the policy


# ── RTC-specific client records ───────────────────────────────────────────────

@dataclass
class ChunkActionRecord:
    """Per-chunk RTC metadata logged by the client when a chunk is received.

    Captures whether RTC was active (original_actions present), how many leftover
    steps were in the request, and L2 statistics of the original (pre-postprocess)
    actions returned by the server.  Together with ClientChunkReceivedRecord this
    supports analyses 4 (leftover pathway health) and 5 (chunk boundary continuity).
    """

    wall_time: float            # time.time() when chunk was received
    episode: int
    first_timestep: int
    chunk_size: int             # number of TimedActions in this chunk
    has_original_actions: bool  # True when server sent original_actions (RTC active)
    leftover_steps: int         # number of leftover_actions bundled in the request obs
    infer_delay_used: int       # infer_delay sent with the obs that triggered this chunk
    orig_action_l2_mean: float  # mean per-step L2 norm of original_actions (0.0 if absent)
    orig_action_l2_max: float   # max per-step L2 norm of original_actions (0.0 if absent)


@dataclass
class AggregateRecord:
    """Per-chunk overlap metrics logged when _aggregate_action_queues() merges a chunk.

    Captures how many timesteps overlapped between the incoming chunk and the
    existing queue, and a corruption proxy for weighted-average blending.  Used for
    analysis 6 (aggregate_fn corruption) and analysis 4 (leftover pathway health).
    """

    wall_time: float            # time.time() at merge time
    episode: int
    first_timestep: int         # first timestep of the incoming chunk
    chunk_size: int             # total steps in the incoming chunk
    n_new: int                  # steps with no overlap (appended as-is)
    n_overlap: int              # steps blended with existing queue content
    old_l2_mean: float          # mean L2 norm of old (existing) actions in overlap region
    new_l2_mean: float          # mean L2 norm of new (incoming) actions in overlap region
    diff_l2_mean: float         # mean ||old - new|| in overlap region (0.0 if n_overlap=0)
    aggregate_fn_name: str      # name of the aggregate function used


# ── Per-step control loop records ────────────────────────────────────────────

@dataclass
class ControlStepRecord:
    """Per control-loop step timestamp.

    Logged once per call to control_loop_action() to capture the actual
    action execution rate, independent of obs-send / inference frequency.

    For RobotClient (with ActionInterpolator) this fires on every sub-step
    call, so it reflects the true motor command rate including interpolated
    sub-steps.  ``timestep`` is the most recently dequeued policy timestep
    (``self.latest_action``); it does not advance on interpolated sub-steps.
    """

    wall_time: float   # time.time() at the start of control_loop_action()
    episode: int
    timestep: int      # latest_action at the time of execution (-1 before first chunk)


# ── Gripper SM event records ──────────────────────────────────────────────────

@dataclass
class GripperSMEventRecord:
    """Per-event record for the gripper state machine in SmartRobotClient.

    Logged on seven event types:
      empty_grasp          — gripper closed on air  (REINFER or RECOVERY triggered)
      slip                 — held object dropped    (REINFER or RECOVERY triggered)
      grasp_success        — first step where load+pos confirm object in hand
      stop                 — max total failures exhausted
      recovery_home_ready  — recovery home trajectory + settle sleep + bg-obs drain
                             completed; obs-send is about to resume from home position.
                             settle_ms = actual recovery_home_settle_time sleep duration.
      lift_position_ready  — LIFT_RETRY trajectory + settle sleep + bg-obs drain
                             completed; obs-send is about to resume from lift position.
                             settle_ms = actual empty_grasp_lift_settle_time sleep duration.
      rewind_position_ready — REWIND_RETRY trajectory + settle sleep + bg-obs drain
                             completed; obs-send is about to resume from rewind endpoint.
                             settle_ms = actual empty_grasp_rewind_settle_time sleep duration.

    The decision (REINFER vs RECOVERY) for empty_grasp/slip is implicit: compare
    failure_count against the config's max_reinfer_retries threshold logged at run start.

    ``recovery_home_ready``, ``lift_position_ready``, and ``rewind_position_ready`` events
    mark the END of silent gaps in the ClientObsSentRecord.wall_time series caused by
    trajectory execution, settle sleeps, and background obs drain.
    Post-hoc: gap duration ≈ (wall_time of ready event) minus
    (wall_time of preceding empty_grasp/slip event).

    Saved to ``<output_dir>/gripper_sm_events_records.jsonl`` (one JSON per
    line) and ``gripper_sm_events_summary.json`` (percentile stats for float
    fields).  Use the JSONL for per-event filtering; the summary for aggregate
    statistics (mean load at failure, typical queue depth, etc.).
    """

    wall_time: float        # time.time() when event was detected
    episode: int            # BaseAsyncClient._current_episode
    timestep: int           # latest executed action timestep at event time
    event_type: str         # "empty_grasp" | "slip" | "grasp_success" | "stop"
                            # | "recovery_home_ready" | "lift_position_ready"
                            # | "rewind_position_ready"
    phase: str              # GraspPhase name at event time
                            # ("CLOSING" | "HOLDING" | "HOME" | "LIFT" | "REWIND")
    gripper_load: float     # |gripper.load| at event (0.0 for structural/ready events)
    gripper_pos: float      # gripper.pos at event (0.0 for structural/ready events)
    peak_load: float        # peak load seen in the current CLOSING/HOLDING phase (0.0 for ready events)
    failure_count: int      # cumulative consecutive failures (empty-grasp + slip) at event time
    queue_size: int         # action queue depth at event time
    settle_ms: float = 0.0  # settle sleep duration that just ended (non-zero for ready events only)


# ── Recorder ─────────────────────────────────────────────────────────────────

class TimingRecorder:
    """Thread-safe accumulator for timing records.

    Records are kept in memory and flushed to disk on save().  Each record
    type should use its own TimingRecorder instance so the JSONL schema is
    uniform within a file.
    """

    def __init__(self, output_dir: str | Path, prefix: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._lock = threading.Lock()
        self._records: list[dict] = []

    def add(self, record: Any) -> None:
        """Append one timing record (must be a dataclass instance)."""
        d = asdict(record)  # convert outside the lock — reduces critical section to O(1) append
        with self._lock:
            self._records.append(d)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def save(self) -> tuple[Path, Path] | None:
        """Write all records to JSONL and a summary JSON with percentile stats.

        Returns (jsonl_path, summary_path), or None when no records exist.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return None

        # ── Per-step JSONL (one JSON object per line) ──────────────────────
        jsonl_path = self.output_dir / f"{self.prefix}_records.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        # ── Summary statistics (percentiles per numeric field) ─────────────
        summary: dict[str, Any] = {"n_records": len(records)}
        for key in records[0]:
            vals = [
                r[key] for r in records
                if isinstance(r[key], (int, float)) and not isinstance(r[key], bool)
            ]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            valid = arr[~np.isnan(arr)]
            n_nan = int(np.isnan(arr).sum())
            if valid.size == 0:
                summary[key] = {"mean": None, "n": len(vals), "n_nan": n_nan}
                continue
            summary[key] = {
                "mean": round(float(np.mean(valid)),             3),
                "std":  round(float(np.std(valid)),              3),
                "min":  round(float(np.min(valid)),              3),
                "p50":  round(float(np.percentile(valid, 50)),  3),
                "p95":  round(float(np.percentile(valid, 95)),  3),
                "p99":  round(float(np.percentile(valid, 99)),  3),
                "max":  round(float(np.max(valid)),              3),
                "n":    int(valid.size),
                "n_nan": n_nan,
            }

        summary_path = self.output_dir / f"{self.prefix}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info(
            f"[TimingRecorder] {self.prefix}: saved {len(records)} records "
            f"→ {jsonl_path} | summary → {summary_path}"
        )
        return jsonl_path, summary_path

    def count_by(self, field: str) -> dict[str, int]:
        """Return a ``{value: count}`` dict grouped by *field*.

        Records where *field* is absent or None are counted under ``None``.
        Useful for string fields (e.g. ``event_type``) that ``log_summary``
        skips because they are not numeric.
        """
        from collections import Counter
        with self._lock:
            return dict(Counter(r.get(field) for r in self._records))

    def log_summary(self, ms_fields: list[str] | None = None) -> None:
        """Log a human-readable percentile table for all float fields.

        Args:
            ms_fields: Explicit list of field names to include.  When None,
                       all float-valued fields are included automatically.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            logger.info(f"[TimingRecorder] {self.prefix}: no records to summarise")
            return

        keys = ms_fields or [k for k in records[0] if isinstance(records[0][k], float)]

        lines = [f"[TimingRecorder] {self.prefix}  ({len(records)} records)"]
        lines.append(f"  {'field':<30} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
        lines.append("  " + "-" * 72)
        for key in keys:
            vals = [
                r[key] for r in records
                if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
            ]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                continue
            nan_tag = f"  [{int(np.isnan(arr).sum())} NaN]" if np.isnan(arr).any() else ""
            lines.append(
                f"  {key:<30} {np.mean(valid):>7.2f}  "
                f"{np.percentile(valid, 50):>7.2f}  "
                f"{np.percentile(valid, 95):>7.2f}  "
                f"{np.percentile(valid, 99):>7.2f}  "
                f"{np.max(valid):>7.2f}{nan_tag}"
            )
        logger.info("\n".join(lines))
