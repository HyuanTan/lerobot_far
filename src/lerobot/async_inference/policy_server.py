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
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1 \
     --obs_similarity_atol=1.0 \
     --log_level=INFO
```
"""

import logging
import pickle  # nosec
import signal
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.utils import init_logging
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    ActionChunk,
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    jpeg_decode_images_in_raw_obs,
    observations_similar,
    observations_similarity_norm,
    raw_observation_to_observation,
)
from .timing import ServerInferRecord, ServerRecvRecord, TimingRecorder

# Policies that support RTC's predict_action_chunk(inference_delay, prev_chunk_left_over) API.
_RTC_CAPABLE_POLICIES = {"smolvla", "pi0", "pi05"}


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Timing statistics (disabled until enable_timing() is called)
        self._recv_recorder: TimingRecorder | None = None
        self._infer_recorder: TimingRecorder | None = None
        # Maps obs timestep → perf_counter() at enqueue time, for queue-wait measurement
        self._enqueue_perf: dict[int, float] = {}
        self._enqueue_perf_lock = threading.Lock()

        # Consecutive inference failure tracking.  After _MAX_CONSECUTIVE_INFERENCE_FAILURES
        # the policy is marked broken and all GetActions calls return Empty immediately.
        self._inference_failure_count: int = 0
        self._policy_broken: bool = False
        # Incremented each time must_go=True clears _predicted_timesteps.  GetActions
        # snapshots this before inference and only adds the timestep back if the episode
        # hasn't changed, preventing a cross-episode race where a long inference completes
        # after the next episode's must_go=True clear and re-pollutes _predicted_timesteps.
        self._episode_generation: int = 0

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self._rtc_enabled: bool = False
        # Set by SendPolicyInstructions when policy uses relative actions + RTC.
        self._use_relative_actions: bool = False
        self._relative_step: RelativeActionsProcessorStep | None = None
        self._normalizer_step: NormalizerProcessorStep | None = None
        # Bug 4: warn once when inference_latency throttle has no effect.
        self._inference_latency_warned: bool = False

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        with self._enqueue_perf_lock:
            self._enqueue_perf.clear()

        self._inference_failure_count = 0
        self._policy_broken = False
        self._episode_generation = 0

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Initialise RTC processor when the client has requested RTC re-planning.
        rtc_cfg: RTCConfig | None = policy_specs.rtc_config
        self._rtc_enabled = False
        if rtc_cfg is not None and self.policy_type in _RTC_CAPABLE_POLICIES:
            self.policy.config.rtc_config = rtc_cfg
            if hasattr(self.policy, "init_rtc_processor"):
                self.policy.init_rtc_processor()
                self._rtc_enabled = True
                self.logger.info(
                    f"RTC processor initialised | execution_horizon={rtc_cfg.execution_horizon}"
                )
            else:
                self.logger.warning(
                    f"Policy {self.policy_type} listed as RTC-capable but has no init_rtc_processor()"
                )

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        # Extract RelativeActionsProcessorStep / NormalizerProcessorStep for leftover
        # reanchoring when the policy uses relative actions together with RTC.
        self._use_relative_actions = bool(getattr(self.policy.config, "use_relative_actions", False))
        self._relative_step = None
        self._normalizer_step = None
        if self._rtc_enabled and self._use_relative_actions:
            for step in self.preprocessor.steps:
                if isinstance(step, RelativeActionsProcessorStep):
                    self._relative_step = step
                elif isinstance(step, NormalizerProcessorStep):
                    self._normalizer_step = step
            if self._relative_step is None:
                self.logger.warning(
                    "Policy has use_relative_actions=True but no RelativeActionsProcessorStep "
                    "found in preprocessor — leftover reanchoring will be skipped."
                )
            else:
                self.logger.info(
                    "Relative-action RTC: leftover reanchoring enabled via reanchor_relative_rtc_prefix()"
                )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        if received_bytes is None:
            # Server shutting down mid-stream (shutdown_event was set during reception).
            return services_pb2.Empty()
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        one_way_ms = (receive_time - obs_timestamp) * 1000
        # Subtract client-side JPEG encoding overhead to get a closer estimate of
        # true network one-way latency.  client_send_overhead_ms is 0.0 when JPEG
        # is disabled (field default on older clients).
        _client_overhead_ms = getattr(timed_observation, "client_send_overhead_ms", 0.0)
        adj_one_way_ms = one_way_ms - _client_overhead_ms
        self.logger.info(
            f"[SERVER←CLIENT] Obs #{obs_timestep} received | "
            f"t={receive_time:.3f} | "
            f"one_way={one_way_ms:.1f}ms | "
            f"adj_one_way={adj_one_way_ms:.1f}ms | "
            f"deserialize={deserialize_time*1000:.1f}ms | "
            f"must_go={timed_observation.must_go} | "
            f"infer_delay={timed_observation.inference_delay} | "
            f"obs_fps={fps_metrics['avg_fps']:.1f}Hz"
        )

        enqueued = self._enqueue_observation(timed_observation)
        if not enqueued:
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        if self._recv_recorder is not None:
            self._recv_recorder.add(ServerRecvRecord(
                wall_time=receive_time,
                timestep=obs_timestep,
                recv_deser_ms=deserialize_time * 1000,
                one_way_ms=one_way_ms,
                adj_one_way_ms=adj_one_way_ms,
                enqueued=enqueued,
            ))

        return services_pb2.Empty()

    # Maximum consecutive inference failures before the policy is marked broken.
    _MAX_CONSECUTIVE_INFERENCE_FAILURES: int = 5

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # If too many consecutive inference failures occurred, stop retrying to avoid
        # spamming CUDA errors. The client will keep receiving Empty and should detect
        # the stall. Restart the policy server to recover.
        if self._policy_broken:
            self.logger.warning(
                "[policy_server] Policy is in broken state due to consecutive inference failures. "
                "Restart the policy server to recover."
            )
            return services_pb2.Empty()

        # obs is initialised to None so the exception handler can safely reference it
        # even if the failure happened before queue.get() returned.
        obs = None
        try:
            getactions_starts = time.perf_counter()
            try:
                obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            except Empty:
                return services_pb2.Empty()

            # Compute how long this obs waited in the queue before we dequeued it.
            with self._enqueue_perf_lock:
                enqueue_perf = self._enqueue_perf.pop(obs.get_timestep(), getactions_starts)
            queue_wait_ms = (time.perf_counter() - enqueue_perf) * 1000

            self.logger.debug(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            # Snapshot episode generation before the (potentially long) inference.
            # Used below to guard against cross-episode pollution in _predicted_timesteps.
            with self._predicted_timesteps_lock:
                ep_gen_snapshot = self._episode_generation

            inference_start = time.perf_counter()
            action_chunk, pipeline_timings = self._predict_action_chunk(obs)
            # Record total pipeline time (preprocess + infer + postprocess) and attach to response
            # so the client can track latency with LatencyTracker.
            action_chunk.inference_time_s = time.perf_counter() - inference_start

            # Add to predicted set AFTER successful inference. Doing this before inference would
            # block all fresh obs with the same timestep from entering the queue during long
            # first-inference warmup (~46s JIT/GPU cold start), causing the server to act on a
            # stale observation. Moving it here lets the queue be refreshed with the most recent
            # state during warmup, so the next GetActions call sees fresh data.
            #
            # Guard: skip the add if a must_go=True obs arrived during inference (episode reset).
            # Without this, a cross-episode race would re-pollute episode N+1's empty set with
            # episode N's last timestep (e.g. ts=0 blocking episode N+1's Obs #0).
            with self._predicted_timesteps_lock:
                if self._episode_generation == ep_gen_snapshot:
                    self._predicted_timesteps.add(obs.get_timestep())
                else:
                    self.logger.debug(
                        f"[policy_server] Episode reset during inference for obs #{obs.get_timestep()} "
                        "(ep_gen changed) — skipping _predicted_timesteps.add() to avoid "
                        "cross-episode pollution"
                    )

            # Inference succeeded — reset consecutive failure counter.
            self._inference_failure_count = 0

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            # Throttle sleep: ensure complete_s (obs.timestamp → client receive_after_deser)
            # ≈ inference_latency.
            #
            # Method A — obs.timestamp wall-clock anchor:
            #   sleep until (obs.timestamp + inference_latency), where obs.timestamp is the
            #   client wall-clock set before serialization.  This absorbs client-serialize,
            #   grpc_send, recv_deser_srv, queue_wait, infer, serialize automatically so that
            #   complete_s ≈ inference_latency + net_s2c + client_deser ≈ inference_latency + 15 ms.
            #
            #   Old server-side timer gave complete_s ≈ inference_latency + ~129 ms because it
            #   missed client_serialize(~20ms) + grpc_send(~67ms) + recv_deser_srv(~27ms) + deser.
            #
            #   Prerequisite: client and server clocks must be NTP-synced (|offset| < 5 ms).
            #   For same-machine sim (libero) this is exact.  Verify with:
            #     chronyc tracking | grep "System time"   (on both machines)
            _elapsed = time.perf_counter() - getactions_starts
            if self.config.inference_latency > 0:
                _dispatch_target = obs.get_timestamp() + self.config.inference_latency
                _sleep_s = max(0.0, _dispatch_target - time.time())
                if not self._inference_latency_warned and _sleep_s == 0.0:
                    self._inference_latency_warned = True
                    self.logger.warning(
                        f"[policy_server] inference_latency={self.config.inference_latency * 1000:.1f}ms "
                        f"but wall-clock elapsed since obs.timestamp already exceeds target "
                        f"(server pipeline {_elapsed * 1000:.1f}ms + client overhead) — sleep is 0. "
                        "The --inference_latency parameter has no effect for this policy. "
                        "Set --inference_latency=0 to silence this warning."
                    )
            else:
                _sleep_s = 0.0
            time.sleep(_sleep_s)

            # dispatch_wall is stamped AFTER the throttle sleep so it aligns with when
            # the client actually receives the chunk (matching ClientChunkReceivedRecord.wall_time).
            dispatch_wall = time.time()
            first_ts = action_chunk.timed_actions[0].get_timestep()
            last_ts  = action_chunk.timed_actions[-1].get_timestep()
            self.logger.debug(
                f"[SERVER→CLIENT] Chunk #{first_ts}–{last_ts} dispatched | "
                f"t={dispatch_wall:.3f} | "
                f"queue_wait={queue_wait_ms:.1f}ms | "
                f"infer={action_chunk.inference_time_s * 1000:.1f}ms | "
                f"serialize={serialize_time * 1000:.1f}ms | "
                f"throttle={_sleep_s * 1000:.1f}ms | "
                f"infer_delay_used={obs.inference_delay} | "
                f"leftover_used={'yes' if obs.leftover_actions is not None else 'no'}"
            )

            if self._infer_recorder is not None:
                self._infer_recorder.add(ServerInferRecord(
                    wall_time=dispatch_wall,
                    timestep=obs.get_timestep(),
                    queue_wait_ms=queue_wait_ms,
                    prepare_ms=pipeline_timings["prepare_ms"],
                    preprocess_ms=pipeline_timings["preprocess_ms"],
                    infer_ms=pipeline_timings["infer_ms"],
                    postprocess_ms=pipeline_timings["postprocess_ms"],
                    total_pipeline_ms=pipeline_timings["total_ms"],
                    serialize_ms=serialize_time * 1000,
                    throttle_sleep_ms=_sleep_s * 1000,
                    infer_delay=obs.inference_delay,
                    leftover_used=(obs.leftover_actions is not None),
                ))

            return actions

        except Exception as e:
            # _predicted_timesteps.add() is only called on success, so nothing to discard here.
            self._inference_failure_count += 1
            if self._inference_failure_count >= self._MAX_CONSECUTIVE_INFERENCE_FAILURES:
                self._policy_broken = True
                self.logger.error(
                    f"[policy_server] FATAL: {self._MAX_CONSECUTIVE_INFERENCE_FAILURES} consecutive "
                    f"inference failures — policy may be in unrecoverable CUDA/GPU error state. "
                    f"Restart the policy server to recover. Last error: {e}"
                )
            else:
                self.logger.error(
                    f"[policy_server] Inference failed "
                    f"({self._inference_failure_count}/{self._MAX_CONSECUTIVE_INFERENCE_FAILURES}): {e}"
                )
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        # obs_pre_mapped observations are already in lerobot format; lerobot_features may be empty
        # so skip the joint-space similarity check (which requires feature key mapping).
        if getattr(obs, "obs_pre_mapped", False):
            return True

        atol = self.config.obs_similarity_atol
        if atol == 0.0:
            # Similarity check disabled: always pass
            self.logger.debug(
                f"[obs_similar] Obs #{obs.get_timestep()} vs #{previous_obs.get_timestep()} — "
                f"similarity check disabled (atol=0.0) → PASS"
            )
            return True

        norm = observations_similarity_norm(obs, previous_obs, lerobot_features=self.lerobot_features)
        is_similar = norm < atol
        self.logger.debug(
            f"[obs_similar] Obs #{obs.get_timestep()} vs #{previous_obs.get_timestep()} — "
            f"norm={norm:.4f} atol={atol:.4f} → {'SKIP (too similar)' if is_similar else 'PASS'}"
        )
        if is_similar:
            return False

        return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        # skip_inference=True: trajectory-phase obs from _BackgroundObsSender.
        # No inference, no similarity check, no last_processed_obs update.
        # Use getattr for pickle compat with old clients that lack this field.
        if getattr(obs, "skip_inference", False):
            self.logger.debug(
                f"\033[0;33m[OBS→SKIP]\033[0m Obs #{obs.get_timestep()} — "
                f"skip_inference=True (trajectory bg obs, no inference)"
            )
            return False

        # must_go=True: force-run inference for this obs (bypass similarity check + queue).
        # Two sources of must_go=True have different semantics:
        #   is_episode_start=True  — true episode boundary (_reset_loop_state on client).
        #                            Clear _predicted_timesteps AND increment _episode_generation
        #                            so any in-flight inference from the OLD episode skips its
        #                            post-inference _predicted_timesteps.add() (see GetActions).
        #   is_episode_start=False — queue-empty re-trigger (_force_must_go) or post-chunk must_go.
        #                            Clear _predicted_timesteps only (don't change _episode_generation).
        #                            Bug fixed: _force_must_go fires every ~333ms and was incorrectly
        #                            incrementing _episode_generation, causing add() to be skipped
        #                            every time → Obs #0 never entered _predicted_timesteps → infinite
        #                            re-inference loop when RTC was enabled.
        if obs.must_go:
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.clear()
                if obs.is_episode_start:
                    self._episode_generation += 1

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Record enqueue time so GetActions can compute queue-wait duration.
            with self._enqueue_perf_lock:
                self._enqueue_perf[obs.get_timestep()] = time.perf_counter()

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            reason = "must_go" if obs.must_go else ("first_obs" if self.last_processed_obs is None else "passed_similarity")
            delta_str = (
                f" \033[1;33mΔts={obs.get_timestep() - last_obs:+d}\033[0m"
                if isinstance(last_obs, int) else ""
            )
            self.logger.info(
                f"\033[1;36m[OBS→QUEUE]\033[0m Obs #{obs.get_timestep()} enqueued "
                f"| reason={reason} must_go={obs.must_go} ep_start={obs.is_episode_start} "
                f"| prev_obs=#{last_obs}{delta_str}"
            )
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        inference_delay: int = 0,
        leftover: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the policy.

        For RTC-capable policies (smolvla/pi0/pi05) the inference_delay and leftover
        prefix are forwarded so the denoising process can account for execution lag.
        """
        if self._rtc_enabled and self.policy_type in _RTC_CAPABLE_POLICIES:
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=inference_delay,
                prev_chunk_left_over=leftover,
            )
        else:
            chunk = self.policy.predict_action_chunk(observation)

        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # shape → (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> tuple[ActionChunk, dict]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk (with optional RTC guidance)
        4. Apply postprocessor (unnormalization, device movement)
        5. Pack into ActionChunk (timed_actions + original_actions for RTC leftover)
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        if getattr(observation_t, "obs_pre_mapped", False):
            # sim_client path: observation already in lerobot format after client-side preprocessing.
            # Skip raw_observation_to_observation() and use it directly.
            observation: Observation = observation_t.get_observation()
            # 方案2 decode: client may have JPEG-encoded images to reduce payload size.
            if getattr(observation_t, "jpeg_images", False):
                observation = jpeg_decode_images_in_raw_obs(observation)
            # uint8 payload optimisation: client may transmit images as uint8 [0,255] to reduce
            # gRPC payload ~4x.  Convert back to float32 [0,1] before the policy preprocessor.
            for key, val in observation.items():
                if "image" in key and isinstance(val, torch.Tensor) and val.dtype == torch.uint8:
                    observation[key] = val.float() / 255.0
            # obs_pre_mapped means the observation was already preprocessed (normalized) on the
            # client side, so we cannot recover the raw state for leftover reanchoring.
            raw_state: torch.Tensor | None = None
        else:
            # robot_client path: convert raw robot observation to lerobot format.
            # 方案2 decode: client may have JPEG-encoded images to reduce payload size.
            # Decode bytes → numpy HWC uint8 before raw_observation_to_observation() which
            # expects arrays that can be passed to torch.tensor() / PILImage.
            raw_obs = observation_t.get_observation()
            if getattr(observation_t, "jpeg_images", False):
                raw_obs = jpeg_decode_images_in_raw_obs(raw_obs)
            # raw_observation_to_observation() already handles uint8→float32 internally.
            observation: Observation = raw_observation_to_observation(
                raw_obs,
                self.lerobot_features,
                self.policy_image_features,
                skip_resize=getattr(observation_t, "skip_server_resize", False),
            )
            # Save the raw (unnormalized) state before the preprocessor runs so we can
            # reanchor absolute leftover actions to the current joint frame (relative-action RTC).
            raw_state = observation.get(OBS_STATE) if (
                self._rtc_enabled and self._use_relative_actions and self._relative_step is not None
            ) else None
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk — forward RTC hints from the observation when available"""
        # Deserialise leftover from the observation (sent by the client's _orig_buf).
        # For relative-action policies the leftover is stored in absolute coordinates and must
        # be re-expressed relative to the current joint state before entering the denoiser.
        leftover: torch.Tensor | None = None
        if observation_t.leftover_actions is not None and self._rtc_enabled:
            if self._use_relative_actions and self._relative_step is not None and raw_state is not None:
                leftover = reanchor_relative_rtc_prefix(
                    observation_t.leftover_actions,  # absolute coords stored by previous round
                    raw_state,                       # current joint state (unnormalized)
                    self._relative_step,
                    self._normalizer_step,
                    self.device,
                )
                self.logger.debug("Leftover reanchored to current state for relative-action RTC")
            else:
                leftover = observation_t.leftover_actions.to(self.device)

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(
            observation,
            inference_delay=observation_t.inference_delay,
            leftover=leftover,
        )
        inference_time = time.perf_counter() - start_inference
        self.logger.debug(
            f"Model forward done | action shape: {action_tensor.shape} | "
            f"inference={inference_time*1000:.1f}ms"
        )

        # Capture model-space (pre-postprocessed) actions now — needed for non-relative RTC.
        # For relative-action policies we will overwrite this after postprocessing with
        # absolute coords so the client can send them back for correct reanchoring.
        model_space_actions: torch.Tensor | None = None
        if self._rtc_enabled:
            model_space_actions = action_tensor.squeeze(0).detach().cpu()

        """4. Apply postprocessor"""
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        processed_actions = []
        for i in range(chunk_size):
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        # Determine what to store as original_actions for the client's leftover buffer:
        #   • non-relative: model-space (normalized absolute) — same as before
        #   • relative: post-postprocessed absolute coords so the next round can call
        #     reanchor_relative_rtc_prefix() to convert them to the new joint frame
        original_actions: torch.Tensor | None = None
        if self._rtc_enabled:
            if self._use_relative_actions and self._relative_step is not None:
                original_actions = action_tensor  # already on CPU, absolute coords
            else:
                original_actions = model_space_actions  # normalized model-space

        """5. Pack into ActionChunk"""
        timed_actions = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        total_pipeline_ms = 1000 * (postprocess_stops - start_prepare)
        self.logger.debug(
            f"[SERVER PIPELINE] Obs #{observation_t.get_timestep()} | "
            f"prepare={prepare_time*1000:.1f}ms | "
            f"preprocess={preprocessing_time*1000:.1f}ms | "
            f"infer={inference_time*1000:.1f}ms | "
            f"postprocess={postprocessing_time*1000:.1f}ms | "
            f"total={total_pipeline_ms:.1f}ms | "
            f"infer_delay={observation_t.inference_delay} | "
            f"leftover={'yes' if leftover is not None else 'no'}"
        )

        pipeline_timings = {
            "prepare_ms":     prepare_time * 1000,
            "preprocess_ms":  preprocessing_time * 1000,
            "infer_ms":       inference_time * 1000,
            "postprocess_ms": postprocessing_time * 1000,
            "total_ms":       total_pipeline_ms,
        }

        # inference_time_s is set by GetActions after this call returns.
        return ActionChunk(timed_actions=timed_actions, original_actions=original_actions), pipeline_timings

    def enable_timing(self, output_dir: str) -> None:
        """Enable per-step timing statistics, writing records to *output_dir*.

        Creates two TimingRecorder instances:
          - ``server_recv``  records gRPC-receive + deserialize timings per obs
          - ``server_infer`` records queue-wait + full pipeline timings per inference
        """
        self._recv_recorder = TimingRecorder(output_dir, "server_recv")
        self._infer_recorder = TimingRecorder(output_dir, "server_infer")
        self.logger.info(f"[{self.prefix}] Timing enabled → {output_dir}")

    def save_timing(self) -> None:
        """Flush accumulated timing records to disk and log percentile summaries."""
        for recorder in filter(None, (self._recv_recorder, self._infer_recorder)):
            recorder.save()
            recorder.log_summary()

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    # init_logging clears existing handlers and re-adds with the requested console level.
    # Must be called AFTER class definition (which sets logger=get_logger(prefix) with
    # console_level="INFO") so the level override actually takes effect for all loggers.
    init_logging(console_level=cfg.log_level.upper())
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)
    if cfg.timing_output_dir:
        policy_server.enable_timing(cfg.timing_output_dir)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    # server = grpc.server(
    # futures.ThreadPoolExecutor(max_workers=10),
    # options=[
    #     ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    #     ("grpc.max_send_message_length",    8 * 1024 * 1024),
    #    ],
    # )

    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    # Re-raise SIGTERM as KeyboardInterrupt so the finally block runs save_timing().
    # Without this, `kill <pid>` (SIGTERM) terminates Python at the C layer,
    # bypassing try/except/finally entirely → timing records are never flushed to disk.
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info("Shutting down (SIGTERM or keyboard interrupt) — stopping server")
        server.stop(grace=2)
    finally:
        policy_server.save_timing()
        policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
