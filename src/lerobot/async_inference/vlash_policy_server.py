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
VLASHPolicyServer — PolicyServer that loads and runs VLASH policies
without LeRobot's preprocessor/postprocessor pipeline.

VLASH policies (vlash_pi05, vlash_pi0) handle normalization internally:
``predict_action_chunk`` accepts a raw lerobot-format batch and returns
unnormalized actions in robot units.  This server therefore:

  * Skips ``make_pre_post_processors()`` (no LeRobot normalizer/tokenizer).
  * Calls ``policy.predict_action_chunk(batch)`` directly after building a
    minimal batch dict from the raw robot observation.
  * Disables RTC (VLASH uses its own async strategy via future-state injection
    on the client side).

Environment variable
--------------------
VLASH_PACKAGE_PATH
    Filesystem path to the ``vlash_main`` directory that contains the
    ``vlash`` Python package.  Defaults to
    ``~/VLA/vla_asyn_arena_ori_repo/vlash/vlash_main``.

Supported policy_type values (sent by VLASHRobotClient)
--------------------------------------------------------
  vlash_pi05   →  vlash.policies.pi05.modeling_pi05.PI05Policy
  vlash_pi0    →  vlash.policies.pi0.modeling_pi0.PI0Policy

Usage
-----
    VLASH_PACKAGE_PATH=/path/to/vlash_main \\
    python -m lerobot.async_inference.vlash_policy_server \\
        --host=127.0.0.1 \\
        --port=8080 \\
        --fps=30
"""

import inspect
import json
import logging
import os
import pickle  # nosec
import sys
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)

from .configs import PolicyServerConfig
from .helpers import (
    ActionChunk,
    Observation,
    RemotePolicyConfig,
    TimedObservation,
    jpeg_decode_images_in_raw_obs,
    raw_observation_to_observation,
)
from .policy_server import PolicyServer


# Policy type strings handled by this server
VLASH_POLICY_TYPES: frozenset[str] = frozenset({"vlash_pi05", "vlash_pi0"})

# Fields present in LeRobot-trained checkpoints but absent from VLASH checkpoints.
# Used to auto-detect which loader/pipeline to apply.
_LEROBOT_CHECKPOINT_FIELDS: frozenset[str] = frozenset({
    "use_relative_actions",
    "rtc_config",
    "freeze_vision_encoder",
    "train_expert_only",
    "relative_exclude_joints",
    "action_feature_names",
})


# ── Checkpoint origin detection ───────────────────────────────────────────────


def _is_lerobot_checkpoint(pretrained_path: str) -> bool:
    """Return True if the checkpoint was trained by LeRobot (not VLASH).

    Peeks at ``config.json`` without fully instantiating the config class.
    LeRobot checkpoints contain fields like ``use_relative_actions`` and
    ``rtc_config`` that are absent from VLASH checkpoints.
    """
    if os.path.isdir(pretrained_path):
        cfg_file: Path | None = Path(pretrained_path) / "config.json"
        if not cfg_file.exists():
            return False
    else:
        try:
            from transformers.utils import cached_file

            cfg_file = cached_file(
                pretrained_path, "config.json", _raise_exceptions_for_missing_entries=False
            )
        except Exception:
            return False

    if cfg_file is None or not Path(str(cfg_file)).exists():
        return False

    try:
        with open(cfg_file) as f:
            cfg_data = json.load(f)
        return bool(_LEROBOT_CHECKPOINT_FIELDS & set(cfg_data.keys()))
    except Exception:
        return False


# ── VLASH package loading ──────────────────────────────────────────────────────


def _ensure_vlash_on_path() -> None:
    """Add the VLASH package root to sys.path if not already importable."""
    vlash_path = os.environ.get(
        "VLASH_PACKAGE_PATH",
        os.path.expanduser("~/VLA/vla_asyn_arena_ori_repo/vlash/vlash_main"),
    )
    if vlash_path not in sys.path:
        sys.path.insert(0, vlash_path)


def _load_vlash_policy(policy_type: str, pretrained_path: str):
    """Instantiate and return a VLASH policy loaded from *pretrained_path*.

    Triggers VLASH package import (which patches LeRobot's config registry with
    ``vlash_pi05 → PI05Config`` and ``vlash_pi0 → PI0Config``), then resolves the
    policy class via LeRobot's ``get_policy_class`` fallback so the dispatch stays
    consistent with the rest of the framework.

    The returned instance is already in eval mode.  Callers should still call
    ``.to(device)`` if they need to move it.
    """
    _ensure_vlash_on_path()

    # Importing any VLASH symbol triggers vlash/configs/__init__.py which:
    #   1. Patches _choice_registry: "vlash_pi05" → PI05Config, "vlash_pi0" → PI0Config
    #   2. Makes get_policy_class("vlash_pi05") resolve to VLASH's PI05Policy
    import vlash  # type: ignore[import]  # noqa: F401

    from lerobot.policies import get_policy_class

    _logger = logging.getLogger(__name__)

    try:
        policy_cls = get_policy_class(policy_type)
    except (ValueError, KeyError) as e:
        raise ValueError(
            f"Unknown VLASH policy type '{policy_type}'. "
            f"Supported: {sorted(VLASH_POLICY_TYPES)}"
        ) from e

    src_file = inspect.getfile(policy_cls)
    _logger.info(
        f"[VLASH LOAD] policy_type={policy_type!r} → "
        f"{policy_cls.__module__}.{policy_cls.__name__} | "
        f"source={src_file} | "
        f"checkpoint={pretrained_path!r}"
    )

    instance = policy_cls.from_pretrained(pretrained_path)

    model_cls = type(instance.model) if hasattr(instance, "model") else type(instance)
    model_src = inspect.getfile(model_cls)
    _logger.info(
        f"[VLASH LOAD] {policy_cls.__name__} loaded successfully | "
        f"config_type={type(instance.config).__name__} from {type(instance.config).__module__} | "
        f"model_source={model_src}"
    )

    return instance


# ── VLASHPolicyServer ─────────────────────────────────────────────────────────


class VLASHPolicyServer(PolicyServer):
    """PolicyServer variant that runs VLASH policies without LeRobot's pipeline.

    Overrides two methods from PolicyServer:

    ``SendPolicyInstructions``
        Loads a VLASH policy (PI05Policy / PI0Policy) from the checkpoint path
        provided by the client.  Bypasses LeRobot's ``make_pre_post_processors``
        and disables RTC — VLASH handles both normalization and async timing
        internally.

    ``_predict_action_chunk``
        Converts the raw robot observation to a minimal lerobot-format batch
        dict (``observation.state``, ``observation.images.*``, ``task``), then
        calls ``policy.predict_action_chunk(batch)`` directly.  The returned
        actions are already in robot units (VLASH unnormalizes internally), so
        no postprocessor is applied.
    """

    prefix = "vlash_policy_server"

    # ── gRPC: policy setup ────────────────────────────────────────────────────

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Load the VLASH policy and configure server state for inference."""
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(
                f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}"
            )

        if policy_specs.policy_type not in VLASH_POLICY_TYPES:
            raise ValueError(
                f"VLASHPolicyServer only accepts policy types in {sorted(VLASH_POLICY_TYPES)}. "
                f"Got: '{policy_specs.policy_type}'. "
                "Use the standard PolicyServer for LeRobot policies."
            )

        self.logger.info(
            f"Receiving VLASH policy instructions from {client_id} | "
            f"policy_type={policy_specs.policy_type} | "
            f"path={policy_specs.pretrained_name_or_path} | "
            f"actions_per_chunk={policy_specs.actions_per_chunk} | "
            f"device={policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        # Disable RTC — VLASH uses future-state injection on the client instead
        self._rtc_enabled = False
        self._use_relative_actions = False
        self._relative_step = None
        self._normalizer_step = None

        pretrained_path = policy_specs.pretrained_name_or_path
        start = time.perf_counter()

        if _is_lerobot_checkpoint(pretrained_path):
            # ── LeRobot checkpoint: use standard pre/post-processor pipeline ──
            # Map vlash_pi05 → pi05, vlash_pi0 → pi0 for the base policy loader
            lerobot_type = policy_specs.policy_type.removeprefix("vlash_")
            self.logger.info(
                f"Detected LeRobot checkpoint — loading with standard pipeline "
                f"(effective policy_type={lerobot_type})"
            )
            policy_class = get_policy_class(lerobot_type)
            self.policy = policy_class.from_pretrained(pretrained_path)
            self.policy.to(self.device)
            self.policy.eval()

            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=pretrained_path,
                preprocessor_overrides={
                    "device_processor": {"device": self.device},
                    "rename_observations_processor": {
                        "rename_map": getattr(policy_specs, "rename_map", {})
                    },
                },
                postprocessor_overrides={"device_processor": {"device": self.device}},
            )
            self._is_vlash_policy = False
        else:
            # ── VLASH checkpoint: skip pre/post-processor (VLASH normalizes internally) ──
            self.policy = _load_vlash_policy(policy_specs.policy_type, pretrained_path)
            self.policy.to(self.device)
            self.policy.eval()
            self.preprocessor = None
            self.postprocessor = None
            self._is_vlash_policy = True

        elapsed = time.perf_counter() - start
        self.logger.info(
            f"Policy loaded in {elapsed:.2f}s | type={policy_specs.policy_type} | "
            f"vlash_native={self._is_vlash_policy} | device={self.device}"
        )
        return services_pb2.Empty()

    # ── Inference pipeline ────────────────────────────────────────────────────

    def _predict_action_chunk(
        self, observation_t: TimedObservation
    ) -> tuple[ActionChunk, dict]:
        """Run inference and return an ActionChunk.

        Routes to one of two pipelines depending on the checkpoint origin:

        * **VLASH checkpoint** (``_is_vlash_policy=True``): builds a minimal batch
          dict from the raw observation and calls ``policy.predict_action_chunk``
          directly. VLASH handles normalization/unnormalization internally.

        * **LeRobot checkpoint** (``_is_vlash_policy=False``): delegates to
          ``PolicyServer._predict_action_chunk`` which applies the full
          LeRobot pre/post-processor pipeline.  RTC is always disabled here
          (``_rtc_enabled=False``) so the parent method runs the simple path.
        """
        if not getattr(self, "_is_vlash_policy", True):
            # LeRobot checkpoint — use parent's full pre/post-processor pipeline
            return super()._predict_action_chunk(observation_t)

        # ── VLASH-native inference path ────────────────────────────────────────
        start_prepare = time.perf_counter()

        raw_obs = observation_t.get_observation()
        if getattr(observation_t, "jpeg_images", False):
            raw_obs = jpeg_decode_images_in_raw_obs(raw_obs)

        observation: Observation = raw_observation_to_observation(
            raw_obs,
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        batch: dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in observation.items()
        }

        # VLASH handles: normalize → VLM → ODE sample → unnormalize
        start_inference = time.perf_counter()
        action_tensor = self.policy.predict_action_chunk(batch)
        inference_time = time.perf_counter() - start_inference

        self.logger.debug(
            f"VLASH forward done | shape={action_tensor.shape} | "
            f"infer={inference_time * 1000:.1f}ms"
        )

        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)

        action_tensor = action_tensor[:, : self.actions_per_chunk, :].detach().cpu()
        _, chunk_size, _ = action_tensor.shape

        timed_actions = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor.squeeze(0)),
            observation_t.get_timestep(),
        )

        total_ms = (time.perf_counter() - start_prepare) * 1000
        self.logger.info(
            f"[VLASH PIPELINE] Obs #{observation_t.get_timestep()} | "
            f"prepare={prepare_time * 1000:.1f}ms | "
            f"infer={inference_time * 1000:.1f}ms | "
            f"total={total_ms:.1f}ms"
        )

        pipeline_timings = {
            "prepare_ms": prepare_time * 1000,
            "preprocess_ms": 0.0,
            "infer_ms": inference_time * 1000,
            "postprocess_ms": 0.0,
            "total_ms": total_ms,
        }

        # original_actions=None → no RTC leftover tracking
        return ActionChunk(timed_actions=timed_actions, original_actions=None), pipeline_timings


# ── Entry point ───────────────────────────────────────────────────────────────


@draccus.wrap()
def serve_vlash(cfg: PolicyServerConfig):
    """Start the VLASHPolicyServer with the given configuration."""
    logging.info(pformat(asdict(cfg)))

    policy_server = VLASHPolicyServer(cfg)
    if cfg.timing_output_dir:
        policy_server.enable_timing(cfg.timing_output_dir)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(
        f"VLASHPolicyServer started on {cfg.host}:{cfg.port} | "
        f"supported_types={sorted(VLASH_POLICY_TYPES)}"
    )
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info("Keyboard interrupt — stopping server")
        server.stop(grace=2)
    finally:
        policy_server.save_timing()
        policy_server.logger.info("VLASHPolicyServer terminated")


if __name__ == "__main__":
    serve_vlash()
