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

"""BaseAsyncClient — shared gRPC client framework for robot_client and sim_client.

Both RobotClient (real robot) and SimRobotClient (gym env) extend this class.
All timing logs, queue management, gRPC communication, and LatencyTracker logic
live here so the two concrete clients stay in sync automatically.

Shared (base class):
  send_observation()          → [CLIENT→SERVER] log
  receive_actions()           → [CLIENT←SERVER] log, _orig_buf population
  control_loop_observation()  → [CLIENT PREP] log (calls _capture_raw_obs +
                                _preprocess_obs + _build_timed_observation)
  control_loop_action()       → pop queue + _execute_action (override for interpolation)
  _aggregate_action_queues(), actions_available(), _ready_to_send_observation()

Concrete hooks (must implement in subclass):
  _build_policy_config()     → RemotePolicyConfig sent to server on start()
  _capture_raw_obs()         → latest raw observation from env/robot
  _preprocess_obs(raw)       → convert raw to TimedObservation.observation format
  _build_timed_observation() → wrap processed obs in TimedObservation

Virtual hooks (may override):
  control_loop_action()      → RobotClient overrides for ActionInterpolator
  _execute_action()          → SimRobotClient implements; robot overrides control_loop_action
"""

from __future__ import annotations

import math
import pickle  # nosec
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace as _dataclass_replace
from queue import Queue
from typing import Any

import grpc
import torch

from lerobot.policies.rtc import LatencyTracker
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .helpers import (
    ActionChunk,
    FPSTracker,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    jpeg_encode_images_in_raw_obs,
)
from .timing import (
    AggregateRecord,
    ChunkActionRecord,
    ClientChunkReceivedRecord,
    ClientObsSentRecord,
    ControlStepRecord,
    TimingRecorder,
)


class BaseAsyncClient(ABC):
    """Abstract base for robot and simulation async-inference clients.

    Subclasses must define a class-level ``prefix`` string that is used for
    the logger name and log line prefixes.
    """

    prefix: str = "base_client"

    def __init__(self, config: Any):
        self.config = config
        self.logger = get_logger(self.prefix)

        # gRPC channel & stub
        self.channel = grpc.insecure_channel(
            config.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"[{self.prefix}] Connecting to {config.server_address}")

        # Shutdown coordination
        self.shutdown_event = threading.Event()

        # Barrier: sync receiver thread + control loop at start of first episode/run
        self.start_barrier = threading.Barrier(2)

        # Action queue
        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.action_chunk_size: int = -1

        # Latest executed action timestep
        self.latest_action_lock = threading.Lock()
        self.latest_action: int = -1

        # must_go: forces immediate inference when queue is empty
        self.must_go = threading.Event()
        self.must_go.set()

        # Telemetry
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        # Legacy single-signal tracker of complete_s — kept as the bootstrap fallback
        # (and for the [CLIENT←SERVER] latency log) until the split trackers warm up.
        self.latency_tracker = LatencyTracker()
        # Tier 2 split-component trackers (see configs.py infer/overhead_latency_quantile):
        #   infer_tracker:    server pipeline time (stable, single-peaked)  → large window
        #   overhead_tracker: complete_s − server_infer (heavy-tailed)       → small window
        self.infer_tracker = LatencyTracker(maxlen=50)
        self.overhead_tracker = LatencyTracker(maxlen=20)
        # Hysteresis state: last infer_delay actually sent, to suppress ±1-step
        # oscillation at ceil() quantization boundaries.  Reset per episode.
        self._last_infer_delay: int = 0

        # Round-trip timing: obs_timestep → wall time obs bytes were fully sent
        self._send_wall_buf: dict[int, float] = {}
        self._send_wall_buf_lock = threading.Lock()
        # infer_delay hint carried by each sent obs, keyed by obs timestep.
        # Looked up in receive_actions() to populate ChunkActionRecord.infer_delay_used.
        # Shares _send_wall_buf_lock (both dicts are always updated together).
        self._obs_infer_delay_buf: dict[int, int] = {}

        # Log-spam guard: track how many times the same timestep has been sent so
        # repeated sends during long warmup inference don't flood logs at INFO level.
        self._last_send_timestep: int | None = None
        self._same_timestep_send_count: int = 0

        # RTC leftover buffer: timestep → pre-postprocessed action tensor
        self._orig_buf: dict[int, torch.Tensor] = {}

        # Episode generation counter: incremented on every _reset_loop_state().
        # receive_actions() snapshots this before each blocking GetActions call and
        # discards any chunk that arrives after a reset (stale cross-episode chunk).
        self._action_generation: int = 0

        # RTC active flag: set the first time a chunk with original_actions arrives.
        # When True, _aggregate_action_queues switches to latest_only automatically.
        self._rtc_active: bool = False

        # must_go re-trigger: counts consecutive control_loop_observation() calls where
        # the action queue was empty.  When the count reaches _MUST_GO_EMPTY_THRESHOLD,
        # a must_go=True obs is force-sent so the server re-runs inference even if the
        # regular must_go event was already cleared (e.g. a prior must_go obs was sent but
        # the resulting chunk hasn't arrived yet and the robot has been stationary long
        # enough that observations_similar keeps filtering the non-must_go obs).
        self._queue_empty_steps: int = 0
        _MUST_GO_EMPTY_THRESHOLD: int = 10  # steps; 10 × (1/30s) ≈ 333 ms at 30 fps
        self._MUST_GO_EMPTY_THRESHOLD = _MUST_GO_EMPTY_THRESHOLD
        # Set by _reset_loop_state(); consumed (cleared) by the first successful
        # control_loop_observation() call to mark the obs as is_episode_start=True.
        # This is distinct from must_go (which fires on post-chunk and _force_must_go too).
        self._next_obs_is_episode_start: bool = False

        # One-time warning flags (avoid log spam across steps).
        # Warn once when server chunk size < configured actions_per_chunk.
        self._chunk_size_mismatch_warned: bool = False

        # Timing statistics (disabled until enable_timing() is called)
        self._current_episode: int = -1
        self._obs_sent_recorder: TimingRecorder | None = None
        self._chunk_recv_recorder: TimingRecorder | None = None
        self._chunk_action_recorder: TimingRecorder | None = None
        self._aggregate_recorder: TimingRecorder | None = None
        self._control_step_recorder: TimingRecorder | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Handshake with server and send policy instructions."""
        try:
            t0 = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            self.logger.debug(
                f"[{self.prefix}] Server ready in {(time.perf_counter() - t0) * 1000:.1f}ms"
            )
            policy_bytes = pickle.dumps(self._build_policy_config())  # nosec
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_bytes))
            self.logger.info(
                f"[{self.prefix}] PolicyInstructions sent | "
                f"actions_per_chunk={self.config.actions_per_chunk}"
            )
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as exc:
            self.logger.error(f"[{self.prefix}] start() failed: {exc}")
            return False

    def stop(self) -> None:
        """Signal shutdown and close the gRPC channel."""
        self.shutdown_event.set()
        self.channel.close()
        self.logger.debug(f"[{self.prefix}] Channel closed")

    def enable_timing(self, output_dir: str) -> None:
        """Enable per-step timing statistics, writing records to *output_dir*.

        Creates four TimingRecorder instances:
          - ``client_obs_sent``    prep + serialize + gRPC timings per sent obs
          - ``client_chunk_recv``  round-trip + deser timings per received chunk
          - ``client_chunk_action`` RTC metadata per received chunk (original_actions stats)
          - ``client_aggregate``   overlap / corruption metrics per queue merge
        """
        self._obs_sent_recorder    = TimingRecorder(output_dir, "client_obs_sent")
        self._chunk_recv_recorder  = TimingRecorder(output_dir, "client_chunk_recv")
        self._chunk_action_recorder = TimingRecorder(output_dir, "client_chunk_action")
        self._aggregate_recorder   = TimingRecorder(output_dir, "client_aggregate")
        self._control_step_recorder = TimingRecorder(output_dir, "client_control_step")
        self.logger.info(f"[{self.prefix}] Timing enabled → {output_dir}")

    def save_timing(self) -> None:
        """Flush accumulated timing records to disk and log percentile summaries.

        All recorder.save() calls happen first so that disk files are written even
        when a subsequent log_summary() is interrupted by KeyboardInterrupt.  Each
        call is individually guarded so one failure cannot skip the others.
        """
        recorders = [r for r in (
            self._obs_sent_recorder,
            self._chunk_recv_recorder,
            self._chunk_action_recorder,
            self._aggregate_recorder,
            self._control_step_recorder,
        ) if r is not None]

        for recorder in recorders:
            try:
                recorder.save()
            except BaseException as exc:
                self.logger.warning(f"[save_timing] {recorder.prefix}.save() raised: {exc}")

        for recorder in recorders:
            try:
                recorder.log_summary()
            except BaseException as exc:
                self.logger.warning(f"[save_timing] {recorder.prefix}.log_summary() raised: {exc}")

    def _reset_loop_state(self) -> None:
        """Reset per-episode/run state: queue, counters, timing.

        Increments _action_generation so receive_actions() can detect and discard
        any in-flight chunks that arrive after this reset (stale cross-episode chunks).
        """
        self._action_generation += 1
        self._current_episode += 1
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.action_chunk_size = -1
        with self.latest_action_lock:
            self.latest_action = -1
        self._orig_buf.clear()
        with self._send_wall_buf_lock:
            self._send_wall_buf.clear()
            self._obs_infer_delay_buf.clear()
        self.must_go.set()
        self.logger.info(
            f"\033[1;32m[MUST_GO→ARMED]\033[0m episode_start "
            f"ep={self._current_episode} gen={self._action_generation}"
        )
        self.fps_tracker.reset()
        # Reset ALL latency trackers + hysteresis.  _reset_loop_state() runs at episode
        # boundaries AND after SM recovery (smart_robot_client drains bg obs then calls
        # this), so resetting here clears the anomalous post-recovery samples that would
        # otherwise pollute the windows for the next ~50 s (orthogonal fix #1).
        self.latency_tracker.reset()
        self.infer_tracker.reset()
        self.overhead_tracker.reset()
        self._last_infer_delay = 0
        self._queue_empty_steps = 0
        self._next_obs_is_episode_start = True

    # ── Abstract hooks ────────────────────────────────────────────────────────

    @abstractmethod
    def _build_policy_config(self) -> RemotePolicyConfig:
        """Build the RemotePolicyConfig sent to the server in start()."""

    @abstractmethod
    def _capture_raw_obs(self) -> dict:
        """Return the latest raw observation dict from the env/robot."""

    @abstractmethod
    def _preprocess_obs(self, raw_obs: dict) -> dict:
        """Convert raw_obs to the format stored in TimedObservation.observation.

        robot_client: identity (server converts via raw_observation_to_observation).
        sim_client:   preprocess_observation + env_preprocessor pipeline.
        """

    @abstractmethod
    def _build_timed_observation(
        self,
        processed_obs: dict,
        timestep: int,
        infer_delay: int,
        leftover: torch.Tensor | None,
    ) -> TimedObservation:
        """Wrap processed obs in a TimedObservation.

        robot_client sets obs_pre_mapped=False; sim_client sets obs_pre_mapped=True
        (when using a real preprocessor pipeline so the server skips raw→lerobot).
        """

    # ── Virtual hooks (may override) ──────────────────────────────────────────

    def _on_chunk_received(self, timed_actions: list, obs_id: int) -> None:
        """Called from receive_actions() after a chunk arrives and is queued.

        Override in subclasses to record received action chunks for analysis.
        Default is a no-op.
        """

    def _on_obs_captured(self, raw_obs: dict) -> None:
        """Called from control_loop_observation() right after _capture_raw_obs().

        Default is a no-op. Subclasses may override for observation-level hooks
        (e.g. image caching); feedback_state recording is handled per control step
        via RobotClient._read_feedback_state() instead.
        """

    def _resolve_raw_chunk(self, raw: Any) -> Any:
        """Hook called immediately after pickle.loads() in receive_actions().

        Override to intercept ActionBundle from MultiCandidatePolicyServer and
        apply client-side candidate selection before the standard ActionChunk
        processing path runs.  Default is identity (no transformation).

        Args:
            raw: Deserialized object — ActionChunk, ActionBundle, or legacy
                 list[TimedAction] depending on the server version.

        Returns:
            The transformed object (typically ActionChunk or list[TimedAction]).
        """
        return raw

    # ── gRPC communication ────────────────────────────────────────────────────

    def send_observation(self, obs: TimedObservation, _timing_out: dict | None = None) -> bool:
        """Serialize and send one observation.  Logs [CLIENT→SERVER].

        Args:
            obs:         The observation to send.
            _timing_out: Optional dict to populate with ``serialize_ms``,
                         ``grpc_send_ms``, and ``payload_kb``.  Used by
                         control_loop_observation() when timing is enabled.
        """
        if not self.running:
            raise RuntimeError(f"[{self.prefix}] Not running. Call start() first.")

        # 方案2: JPEG-compress camera images before pickling when configured.
        # Creates a new TimedObservation with encoded image bytes instead of raw
        # arrays — the original obs is never mutated (safe for callers that keep a ref).
        _jpeg_quality = getattr(self.config, "obs_image_jpeg_quality", None)
        _jpeg_encode_ms = 0.0
        if _jpeg_quality is not None:
            t_jpeg = time.perf_counter()
            obs = _dataclass_replace(
                obs,
                observation=jpeg_encode_images_in_raw_obs(obs.get_observation(), _jpeg_quality),
                jpeg_images=True,
            )
            _jpeg_encode_ms = (time.perf_counter() - t_jpeg) * 1000

        # Stamp the client-side overhead so the server can subtract it from one_way_ms.
        obs = _dataclass_replace(obs, client_send_overhead_ms=_jpeg_encode_ms)
        t_ser = time.perf_counter()
        obs_bytes = pickle.dumps(obs)  # nosec
        serialize_t = time.perf_counter() - t_ser

        try:
            obs_iter = send_bytes_in_chunks(
                obs_bytes, services_pb2.Observation,
                log_prefix=f"[{self.prefix}] Obs", silent=True,
            )
            t_grpc = time.perf_counter()
            _ = self.stub.SendObservations(obs_iter)
            grpc_send_t = time.perf_counter() - t_grpc

            send_wall = time.time()
            with self._send_wall_buf_lock:
                self._send_wall_buf[obs.get_timestep()] = send_wall
                self._obs_infer_delay_buf[obs.get_timestep()] = obs.inference_delay

            if _timing_out is not None:
                _timing_out["jpeg_encode_ms"] = _jpeg_encode_ms
                _timing_out["serialize_ms"] = serialize_t * 1000
                _timing_out["grpc_send_ms"] = grpc_send_t * 1000
                _timing_out["payload_kb"] = len(obs_bytes) / 1024

            ts = obs.get_timestep()
            if ts == self._last_send_timestep:
                self._same_timestep_send_count += 1
            else:
                self._last_send_timestep = ts
                self._same_timestep_send_count = 1

            n = self._same_timestep_send_count
            suffix = f" (repeat #{n})" if n > 1 else ""
            self.logger.debug(
                f"[CLIENT→SERVER] Obs #{ts} sent{suffix} | "
                f"t={send_wall:.3f} | "
                f"serialize={serialize_t * 1000:.1f}ms | "
                f"grpc_send={grpc_send_t * 1000:.1f}ms | "
                f"payload={len(obs_bytes) / 1024:.1f}KB | "
                f"must_go={obs.must_go} | "
                f"infer_delay={obs.inference_delay} | "
                f"leftover={'yes' if obs.leftover_actions is not None else 'no'}"
            )
            return True
        except grpc.RpcError as exc:
            self.logger.error(f"[{self.prefix}] SendObservations error: {exc}")
            return False

    def receive_actions(self) -> None:
        """Background thread: poll GetActions, fill _orig_buf, merge into queue.

        Logs [CLIENT←SERVER].  Fills self._orig_buf from ActionChunk.original_actions
        so leftover_actions can be sent back on the next observation (RTC support).
        """
        self.start_barrier.wait()
        self.logger.info(f"[{self.prefix}] Action receiver thread started")

        _MAX_ERRORS = 5
        _err = 0
        # Track consecutive empty responses to detect a broken server.
        _empty_streak: int = 0
        _EMPTY_WARN_INTERVAL: int = 100  # warn every N consecutive empties

        while self.running:
            try:
                # Snapshot generation before blocking so we can detect a reset that
                # happened while GetActions was waiting (stale cross-episode chunk).
                _gen = self._action_generation
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    _empty_streak += 1
                    if _empty_streak % _EMPTY_WARN_INTERVAL == 0:
                        self.logger.warning(
                            f"[{self.prefix}] {_empty_streak} consecutive empty server responses. "
                            "Server policy may be in a broken state (e.g. CUDA error). "
                            "Consider restarting the policy server."
                        )
                    continue
                _empty_streak = 0

                if self._action_generation != _gen:
                    self.logger.debug(
                        f"[{self.prefix}] Discarding stale chunk — episode reset during GetActions"
                    )
                    # Re-arm must_go: the queue is empty but we never delivered a chunk,
                    # so the next control_loop_observation must send must_go=True to
                    # guarantee the server enqueues it (bypassing similarity filtering).
                    self.must_go.set()
                    continue

                receive_time = time.time()

                t_deser = time.perf_counter()
                raw = pickle.loads(actions_chunk.data)  # nosec
                raw = self._resolve_raw_chunk(raw)  # hook: ActionBundle → ActionChunk
                deser_t = time.perf_counter() - t_deser
                # Stamped after deser so that complete_s below covers the full
                # obs-build → client-deser path (queue_wait + pipeline + net_s2c + deser).
                receive_after_deser = time.time()

                if isinstance(raw, ActionChunk):
                    timed_actions = raw.timed_actions
                    original_actions = raw.original_actions
                    if raw.inference_time_s > 0 and timed_actions:
                        # Option B: measure obs-build-start → after-deser on the client.
                        # timed_actions[0].get_timestamp() == obs.timestamp (set in
                        # _build_timed_observation before send), so complete_s captures
                        # queue_wait + full pipeline + net_s2c + deser — everything
                        # that raw.inference_time_s (server pipeline only) misses.
                        _chunk_lifetime_s = (
                            self.config.actions_per_chunk * self.config.environment_dt
                        )
                        complete_s = receive_after_deser - timed_actions[0].get_timestamp()
                        if 0 < complete_s <= _chunk_lifetime_s:
                            self.latency_tracker.add(complete_s)
                            # Tier 2 split B: separate the stable server pipeline time from
                            # the heavy-tailed overhead.  inference_time_s is a pure server-side
                            # DURATION (perf_counter), so complete_s − inference_time_s mixes two
                            # durations from different clocks but needs no clock sync (both are
                            # elapsed times, not absolute timestamps).  Clamp ≥ 0 for float skew.
                            _server_infer_s = float(raw.inference_time_s)
                            _overhead_s = max(0.0, complete_s - _server_infer_s)
                            self.infer_tracker.add(_server_infer_s)
                            self.overhead_tracker.add(_overhead_s)
                        else:
                            self.logger.warning(
                                f"[{self.prefix}] Skipping warmup/outlier complete latency "
                                f"{complete_s * 1000:.0f}ms from latency_tracker "
                                f"(> chunk_lifetime {_chunk_lifetime_s * 1000:.0f}ms). "
                                "infer_delay will stay 0 until steady-state inference resumes."
                            )
                    server_infer_ms = raw.inference_time_s * 1000
                else:
                    # Backward compat: old server returned bare list[TimedAction]
                    timed_actions = raw
                    original_actions = None
                    server_infer_ms = float("nan")

                if not timed_actions:
                    continue

                first_ts = timed_actions[0].get_timestep()
                last_ts = timed_actions[-1].get_timestep()

                # Round-trip: obs fully sent → actions fully received
                with self._send_wall_buf_lock:
                    send_wall = self._send_wall_buf.pop(first_ts, None)
                    infer_delay_used = self._obs_infer_delay_buf.pop(first_ts, 0)
                    stale = [k for k in self._send_wall_buf if k < first_ts]
                    for k in stale:
                        del self._send_wall_buf[k]
                        self._obs_infer_delay_buf.pop(k, None)
                round_trip_ms = (
                    (receive_time - send_wall) * 1000 if send_wall is not None else float("nan")
                )
                # Split-component percentiles (Tier 2): infer is covered at infer_q, overhead
                # at overhead_q — surfaced here so the calibration can be read off the logs.
                _iq = self.config.infer_latency_quantile
                _oq = self.config.overhead_latency_quantile
                infer_pq    = (self.infer_tracker.percentile(_iq) or 0.0) * 1000
                overhead_pq = (self.overhead_tracker.percentile(_oq) or 0.0) * 1000

                self.logger.info(
                    f"[CLIENT←SERVER] Chunk #{first_ts}–{last_ts} received | "
                    f"t={receive_time:.3f} | "
                    f"round_trip={round_trip_ms:.1f}ms | "
                    f"server_infer={server_infer_ms:.1f}ms | "
                    f"deser={deser_t * 1000:.2f}ms | "
                    f"infer_p{int(_iq * 100)}={infer_pq:.1f}ms "
                    f"overhead_p{int(_oq * 100)}={overhead_pq:.1f}ms"
                )

                # Move to client_device if needed
                if self.config.client_device != "cpu":
                    for ta in timed_actions:
                        if ta.get_action().device.type != self.config.client_device:
                            ta.action = ta.get_action().to(self.config.client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Warn once if the server delivers fewer actions than configured.
                # infer_delay is capped using actions_per_chunk (config), but if the actual
                # chunk is shorter the cap is too loose — infer_delay could exceed real chunk
                # size and trigger RTC assertion failures in the policy.
                if (
                    not self._chunk_size_mismatch_warned
                    and len(timed_actions) < self.config.actions_per_chunk
                ):
                    self._chunk_size_mismatch_warned = True
                    self.logger.warning(
                        f"[{self.prefix}] Server chunk size ({len(timed_actions)}) is smaller than "
                        f"configured actions_per_chunk ({self.config.actions_per_chunk}). "
                        f"infer_delay cap is computed from actions_per_chunk and may exceed the "
                        f"real chunk length, risking RTC index-out-of-bounds. "
                        f"Set --actions_per_chunk<={len(timed_actions)} to match the server."
                    )

                # Populate _orig_buf for RTC leftover tracking (Step 3 of robot_client.py)
                if original_actions is not None and len(timed_actions) > 0:
                    with self.action_queue_lock:
                        for i, ta in enumerate(timed_actions):
                            if i < len(original_actions):
                                self._orig_buf[ta.get_timestep()] = original_actions[i]

                    # First time we see original_actions: server has RTC enabled.
                    # Blending RTC-guided chunks via any strategy other than latest_only
                    # corrupts the prefix continuity that RTC depends on. This should
                    # have been caught by RobotClientConfig.__post_init__ — if we reach
                    # here with a wrong aggregate_fn it means the config bypass was used.
                    if not self._rtc_active:
                        self._rtc_active = True
                        fn_name = getattr(self.config, "aggregate_fn_name", "")
                        if fn_name != "latest_only":
                            raise RuntimeError(
                                f"[{self.prefix}] RTC is active on the server but "
                                f"aggregate_fn_name='{fn_name}'. "
                                "Set --aggregate_fn_name=latest_only to prevent RTC prefix corruption."
                            )

                # Compute original_actions stats BEFORE _aggregate_action_queues clears _orig_buf
                _leftover_steps_pre = len(self._orig_buf)
                # Sample queue depth BEFORE merge to reflect how many actions were
                # already available when this chunk arrived (diagnostic for starvation).
                with self.action_queue_lock:
                    _queue_size_at_recv = self.action_queue.qsize()
                if self._chunk_action_recorder is not None and original_actions is not None:
                    import torch as _torch
                    _orig_l2s = [
                        float(_torch.tensor(a, dtype=_torch.float32).norm().item())
                        if not isinstance(a, _torch.Tensor)
                        else float(a.float().norm().item())
                        for a in original_actions
                    ]
                    _orig_l2_mean = float(sum(_orig_l2s) / len(_orig_l2s)) if _orig_l2s else 0.0
                    _orig_l2_max  = float(max(_orig_l2s)) if _orig_l2s else 0.0
                else:
                    _orig_l2_mean = _orig_l2_max = 0.0

                _agg_stats = self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)

                _agg_fn_name = getattr(self.config, "aggregate_fn_name", "unknown")
                if self._aggregate_recorder is not None and len(timed_actions) > 0:
                    self._aggregate_recorder.add(AggregateRecord(
                        wall_time=receive_time,
                        episode=self._current_episode,
                        first_timestep=first_ts,
                        chunk_size=len(timed_actions),
                        n_new=_agg_stats["n_new"],
                        n_overlap=_agg_stats["n_overlap"],
                        old_l2_mean=_agg_stats["old_l2_mean"],
                        new_l2_mean=_agg_stats["new_l2_mean"],
                        diff_l2_mean=_agg_stats["diff_l2_mean"],
                        aggregate_fn_name=_agg_fn_name,
                    ))

                if self._chunk_action_recorder is not None:
                    self._chunk_action_recorder.add(ChunkActionRecord(
                        wall_time=receive_time,
                        episode=self._current_episode,
                        first_timestep=first_ts,
                        chunk_size=len(timed_actions),
                        has_original_actions=original_actions is not None,
                        leftover_steps=_leftover_steps_pre,
                        infer_delay_used=infer_delay_used,
                        orig_action_l2_mean=_orig_l2_mean,
                        orig_action_l2_max=_orig_l2_max,
                    ))

                if self._chunk_recv_recorder is not None:
                    self._chunk_recv_recorder.add(ClientChunkReceivedRecord(
                        wall_time=receive_time,
                        episode=self._current_episode,
                        first_timestep=first_ts,
                        last_timestep=last_ts,
                        chunk_size=len(timed_actions),
                        round_trip_ms=round_trip_ms,
                        server_infer_ms=server_infer_ms,
                        deser_ms=deser_t * 1000,
                        queue_size_at_recv=_queue_size_at_recv,
                        estimated_first_exec_lag_ms=(
                            _queue_size_at_recv * self.config.environment_dt * 1000
                        ),
                    ))

                self._on_chunk_received(timed_actions, first_ts)
                self.must_go.set()
                _err = 0

            except grpc.RpcError as exc:
                if self.running:
                    self.logger.error(f"[{self.prefix}] GetActions error: {exc}")
                    _err += 1
                    if _err >= _MAX_ERRORS:
                        self.logger.error(
                            f"[{self.prefix}] {_MAX_ERRORS} consecutive failures. Stopping."
                        )
                        self.stop()
                        break
                continue
            except Exception as exc:
                # Catch-all: prevent unexpected exceptions from silently killing the
                # receiver thread (e.g. errors in _resolve_raw_chunk or _aggregate_action_queues).
                # Log with full traceback so the root cause can be diagnosed.
                import traceback as _tb
                self.logger.error(
                    f"[{self.prefix}] Unexpected error in receive loop "
                    f"(receiver thread kept alive): {exc}\n{_tb.format_exc()}"
                )
                continue

    # ── Queue management ──────────────────────────────────────────────────────

    def _aggregate_action_queues(
        self,
        incoming: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> dict:
        """Merge incoming actions into the queue via aggregate_fn.

        Returns overlap stats dict with keys:
          n_new, n_overlap, old_l2_mean, new_l2_mean, diff_l2_mean
        Used by receive_actions() to populate AggregateRecord when timing is on.
        """
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):
                return x2

        future_queue: Queue[TimedAction] = Queue()
        with self.action_queue_lock:
            internal = list(self.action_queue.queue)

        # Read latest_action once before the loop.  Per-action re-reads inside the
        # loop would create a TOCTOU window where control_loop_action dequeues ts=N
        # and starts executing it (no lock held) before updating latest_action.
        # A re-read in that window would see the old latest, letting ts=N pass the
        # filter, re-enter future_queue, and be executed a second time.
        with self.latest_action_lock:
            current_latest = self.latest_action

        current_map = {a.get_timestep(): a.get_action() for a in internal}

        n_new = n_overlap = 0
        old_norms: list[float] = []
        new_norms: list[float] = []
        diff_norms: list[float] = []

        for new_action in incoming:
            if new_action.get_timestep() <= current_latest:
                continue
            if new_action.get_timestep() not in current_map:
                future_queue.put(new_action)
                n_new += 1
            else:
                old_a = current_map[new_action.get_timestep()]
                new_a = new_action.get_action()
                blended = aggregate_fn(old_a, new_a)
                future_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=new_action.get_timestep(),
                        action=blended,
                    )
                )
                n_overlap += 1
                if self._aggregate_recorder is not None:
                    old_norms.append(float(old_a.float().norm().item()))
                    new_norms.append(float(new_a.float().norm().item()))
                    diff_norms.append(float((old_a.float() - new_a.float()).norm().item()))

        with self.action_queue_lock:
            self.action_queue = future_queue

        return {
            "n_new":       n_new,
            "n_overlap":   n_overlap,
            "old_l2_mean": float(sum(old_norms)  / len(old_norms))  if old_norms  else 0.0,
            "new_l2_mean": float(sum(new_norms)  / len(new_norms))  if new_norms  else 0.0,
            "diff_l2_mean":float(sum(diff_norms) / len(diff_norms)) if diff_norms else 0.0,
        }

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self) -> bool:
        if self.action_chunk_size <= 0:
            return True
        with self.action_queue_lock:
            return (
                self.action_queue.qsize() / self.action_chunk_size
                <= self.config.chunk_size_threshold
            )

    # ── Control loop helpers ──────────────────────────────────────────────────

    def control_loop_observation(self) -> None:
        """Capture, preprocess, and send one observation.  Logs [CLIENT PREP].

        The four timing stages mirror robot_client.py exactly:
          Stage 1 capture:      _capture_raw_obs() + _preprocess_obs()
          Stage 2 infer_delay:  LatencyTracker → math.ceil(max_lat / dt)
          Stage 3 leftover:     collect _orig_buf entries for remaining queue
          Stage 4 obs_build:    _build_timed_observation()
        """
        # Stage 1: capture + preprocess
        t0 = time.perf_counter()
        raw_obs = self._capture_raw_obs()
        self._on_obs_captured(raw_obs)
        processed = self._preprocess_obs(raw_obs)
        obs_capture_t = time.perf_counter() - t0

        with self.latest_action_lock:
            latest_action = self.latest_action

        # Stage 2: infer_delay — Tier 2 split-component estimation with asymmetric quantiles.
        #
        # complete_s is split (see receive_actions) into two trackers of different nature:
        #   infer_tracker    = server pipeline time  (stable, single-peaked)
        #   overhead_tracker = complete_s − server_infer  (heavy-tailed: gRPC + queue_wait)
        #
        # infer_delay = ceil( ( Q_infer-quantile(infer) + Q_overhead-quantile(overhead) ) / dt )
        #   - infer component: HIGH quantile (default p90) — covering its tail is cheap.
        #   - overhead component: MODERATE quantile (default p75) — covering the heavy/bimodal
        #     tail would inflate infer_delay (overcorrection); rare misses are caught by
        #     force_must_go.  Keeping it low is what reduces the over-estimation seen in fig8.
        #
        # inference_delay_low (multi-candidate optimistic hint) = ceil((infer.p50 + overhead.p50)/dt).
        #
        # Bootstrap fallback: until BOTH split trackers have _DELAY_MIN_SAMPLES samples, fall
        # back to the legacy single-complete_s formula so behaviour is unchanged at startup /
        # right after a reset (orthogonal fix #2).
        t1 = time.perf_counter()
        _DELAY_MIN_SAMPLES = 3
        _infer_delay_cap = max(0, self.config.actions_per_chunk - 1)
        dt = self.config.environment_dt
        _infer_q = self.config.infer_latency_quantile
        _overhead_q = self.config.overhead_latency_quantile
        # spike_buffer_s → steps, used only by the bootstrap fallback below.
        _SPIKE_BUFFER_STEPS: int = max(1, math.ceil(self.config.spike_buffer_s / dt))

        def _to_delay_steps(lat_s: float) -> int:
            return min(math.ceil(lat_s / dt) if lat_s > 0 else 0, _infer_delay_cap)

        _split_ready = (
            len(self.infer_tracker) >= _DELAY_MIN_SAMPLES
            and len(self.overhead_tracker) >= _DELAY_MIN_SAMPLES
        )
        # Diagnostic fields (referenced by the cap-clamp warning below).
        _p50_lat = self.latency_tracker.percentile(0.5) or 0.0
        _p90_lat = self.latency_tracker.percentile(0.9) or 0.0

        if _split_ready:
            # Tier 2: split B + asymmetric quantiles.
            _infer_lat    = self.infer_tracker.percentile(_infer_q) or 0.0
            _overhead_lat = self.overhead_tracker.percentile(_overhead_q) or 0.0
            _infer_delay  = _to_delay_steps(_infer_lat + _overhead_lat)
            _infer_p50    = self.infer_tracker.percentile(0.5) or 0.0
            _overhead_p50 = self.overhead_tracker.percentile(0.5) or 0.0
            _infer_delay_low = _to_delay_steps(_infer_p50 + _overhead_p50)
        elif len(self.latency_tracker) >= _DELAY_MIN_SAMPLES:
            # Bootstrap fallback: legacy single-complete_s p50+buffer / cap-p90 formula.
            _infer_delay = min(
                _to_delay_steps(_p50_lat) + _SPIKE_BUFFER_STEPS,
                _to_delay_steps(_p90_lat),
            )
            _infer_delay_low = _to_delay_steps(_p50_lat)
        else:
            # Too few samples for any estimate.
            _infer_delay = min(
                _to_delay_steps(_p50_lat) + _SPIKE_BUFFER_STEPS,
                _to_delay_steps(_p90_lat),
            )
            _infer_delay_low = 0

        if _infer_delay > _infer_delay_cap:
            self.logger.warning(
                f"[{self.prefix}] infer_delay {_infer_delay} steps exceeds cap "
                f"({_infer_delay_cap} = actions_per_chunk-1={self.config.actions_per_chunk}-1) "
                f"and has been clamped. Latency tracker p50={_p50_lat * 1000:.1f}ms "
                f"p90={_p90_lat * 1000:.1f}ms, dt={dt * 1000:.1f}ms. "
                "Consider increasing actions_per_chunk or reducing inference latency."
            )
            _infer_delay = _infer_delay_cap

        # Hysteresis (orthogonal fix #3): suppress ±1-step oscillation when the latency
        # estimate sits near a ceil() boundary (e.g. p50 jitters across k·dt).
        # ASYMMETRIC by design:
        #   - upward changes adopt immediately (rising latency → need a larger delay NOW;
        #     never hold a too-small delay, which would risk starvation),
        #   - downward changes require a >= 2-step drop before shrinking (a single-step
        #     dip is treated as jitter and the previous, safer value is held).
        # This also avoids the slow-drift lockup of a symmetric deadband: a sustained
        # +1-per-cycle ramp keeps adopting (each step is "upward").
        _infer_delay_raw = _infer_delay  # pre-hysteresis (post-cap), logged for diagnostics
        if _infer_delay < self._last_infer_delay and (self._last_infer_delay - _infer_delay) < 2:
            _infer_delay = self._last_infer_delay
        self._last_infer_delay = _infer_delay
        infer_delay_t = time.perf_counter() - t1

        # Stage 3: collect leftover pre-postprocessed actions (_orig_buf)
        t2 = time.perf_counter()
        _leftover: torch.Tensor | None = None
        with self.action_queue_lock:
            _remaining_ts = sorted(a.get_timestep() for a in self.action_queue.queue)
            _leftover_parts = [self._orig_buf[ts] for ts in _remaining_ts if ts in self._orig_buf]
        if _leftover_parts:
            _leftover = torch.stack(_leftover_parts)
        leftover_t = time.perf_counter() - t2

        # Stage 4: build TimedObservation
        # timestep = latest_action + 1 so the server always sees a new timestep.
        # max(latest_action, 0) was wrong: after executing action 0 (latest_action=0)
        # it still produced timestep=0, which the server drops as already predicted.
        t3 = time.perf_counter()
        observation = self._build_timed_observation(
            processed, latest_action + 1, _infer_delay, _leftover
        )
        observation.inference_delay_low = _infer_delay_low
        obs_build_t = time.perf_counter() - t3

        # must_go: trigger immediate inference when queue is empty.
        # Also force must_go=True when the queue has been empty for too long —
        # this rescues the case where must_go was already cleared (a prior must_go obs
        # was sent but no chunk arrived yet) and observations_similar keeps filtering
        # the stationary robot's non-must_go obs, creating a silent inference stall.
        with self.action_queue_lock:
            _queue_empty = self.action_queue.empty()

        if _queue_empty:
            self._queue_empty_steps += 1
        else:
            self._queue_empty_steps = 0

        _force_must_go = _queue_empty and (self._queue_empty_steps >= self._MUST_GO_EMPTY_THRESHOLD)
        if _force_must_go:
            self.logger.warning(
                f"\033[1;33m[FORCE_MUST_GO]\033[0m queue empty "
                f"{self._queue_empty_steps} steps (>{self._MUST_GO_EMPTY_THRESHOLD}) — "
                "forcing must_go=True to break inference stall "
                "(stationary robot + observations_similar filter)"
            )
            self._queue_empty_steps = 0  # reset so the next stall window starts fresh

        observation.must_go = (self.must_go.is_set() and _queue_empty) or _force_must_go

        # is_episode_start: only True for the very first obs after _reset_loop_state().
        # _force_must_go and post-chunk must_go are NOT episode boundaries — they must
        # not increment _episode_generation on the server (see policy_server bug fix).
        if self._next_obs_is_episode_start and not _force_must_go:
            observation.is_episode_start = True
            self._next_obs_is_episode_start = False

        total_prep_t = obs_capture_t + infer_delay_t + leftover_t + obs_build_t
        self.logger.debug(
            f"[CLIENT PREP] Obs #{observation.get_timestep()} | "
            f"capture={obs_capture_t * 1000:.1f}ms | "
            f"infer_delay_calc={infer_delay_t * 1000:.2f}ms | "
            f"leftover_collect={leftover_t * 1000:.2f}ms (steps={len(_leftover_parts)}) | "
            f"obs_build={obs_build_t * 1000:.2f}ms | "
            f"total_prep={total_prep_t * 1000:.1f}ms"
        )

        _send_timing: dict = {}
        _ = self.send_observation(
            observation,
            _timing_out=_send_timing if self._obs_sent_recorder is not None else None,
        )

        if self._obs_sent_recorder is not None and _send_timing:
            self._obs_sent_recorder.add(ClientObsSentRecord(
                wall_time=time.time(),
                episode=self._current_episode,
                timestep=observation.get_timestep(),
                obs_capture_ms=obs_capture_t * 1000,
                infer_delay_calc_ms=infer_delay_t * 1000,
                leftover_collect_ms=leftover_t * 1000,
                obs_build_ms=obs_build_t * 1000,
                total_prep_ms=total_prep_t * 1000,
                jpeg_encode_ms=_send_timing.get("jpeg_encode_ms", 0.0),
                serialize_ms=_send_timing.get("serialize_ms", 0.0),
                grpc_send_ms=_send_timing.get("grpc_send_ms", 0.0),
                payload_kb=_send_timing.get("payload_kb", 0.0),
                must_go=observation.must_go,
                infer_delay=observation.inference_delay,
                leftover_steps=len(_leftover_parts),
                split_active=_split_ready,
                infer_delay_raw=_infer_delay_raw,
            ))

        if observation.must_go:
            self.must_go.clear()

        fps = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
        self.logger.debug(
            f"[{self.prefix}] Obs #{observation.get_timestep()} | "
            f"avg_fps={fps['avg_fps']:.2f}/{fps['target_fps']:.2f}"
        )

    def control_loop_action(self) -> Any:
        """Pop one action from the queue and execute it.

        RobotClient overrides this to use ActionInterpolator (sub-step control).
        SimRobotClient uses this default: pop once per step and call _execute_action.
        """
        if self._control_step_recorder is not None:
            with self.latest_action_lock:
                _ts_now = self.latest_action
            self._control_step_recorder.add(ControlStepRecord(
                wall_time=time.time(),
                episode=self._current_episode,
                timestep=_ts_now,
            ))

        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
            # Remove from leftover buffer (Step 3: not included in next leftover tensor)
            self._orig_buf.pop(timed_action.get_timestep(), None)

        result = self._execute_action(timed_action)

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        return result

    def _execute_action(self, timed_action: TimedAction) -> Any:
        """Execute one action on the robot/env.

        SimRobotClient implements this (env.step).
        RobotClient overrides control_loop_action instead (uses ActionInterpolator).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must either implement _execute_action() "
            "or override control_loop_action()."
        )
