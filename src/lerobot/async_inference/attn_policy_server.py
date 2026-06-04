"""
Attention-capturing policy server for SmolVLA and PI05.

Drop-in replacement for policy_server.py that additionally captures attention
maps and saves visualizations asynchronously.  Dispatches to the correct probe
based on the policy type announced by the client:

  smolvla — SmolVLAAttentionProbe (cross-attention heatmaps)
  pi05    — PI05FeatureProbe (lang→image + action→image heatmaps)

Usage::

    # SmolVLA attention server:
    python -m lerobot.async_inference.attn_policy_server \\
        --host=127.0.0.1 --port=8080 --fps=30 \\
        --attn_output_dir=/tmp/smolvla_attn \\
        --attn_save_every_n=5

    # PI05 attention server:
    python -m lerobot.async_inference.attn_policy_server \\
        --host=127.0.0.1 --port=8080 --fps=10 \\
        --attn_output_dir=/tmp/pi05_attn \\
        --attn_save_every_n=3

    # Then launch the regular run_libero_test unchanged (no --attn_* flags needed there).

Other policy types fall back to standard PolicyServer behavior without overhead.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from dataclasses import asdict
from concurrent import futures

import draccus
import grpc
import numpy as np
import torch

from lerobot.utils.constants import OBS_IMAGES

from .configs import PolicyServerConfig
from .helpers import TimedObservation, get_logger
from .policy_server import PolicyServer
from .attn_probe import SmolVLAAttentionProbe
from .attn_visualizer import save_inference_attention
from lerobot.transport import (
    services_pb2_grpc,  # type: ignore
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AttnPolicyServerConfig(PolicyServerConfig):
    """PolicyServerConfig extended with attention-visualization settings."""

    attn_output_dir: str | None = field(
        default=None,
        metadata={"help": "Directory to write attention visualization PNGs. None = disabled."},
    )
    attn_save_every_n: int = field(
        default=1,
        metadata={"help": "Save attention visualizations every N inference calls (1 = every call)."},
    )
    attn_keep_cpu_copy: bool = field(
        default=True,
        metadata={"help": "Move captured attention tensors to CPU immediately to avoid GPU memory pressure."},
    )
    attn_num_vis_cameras: int = field(
        default=1,
        metadata={"help": "Number of cameras to visualize (0 = all detected cameras)."},
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class AttnPolicyServer(PolicyServer):
    """PolicyServer subclass that captures attention maps per inference call.

    Supports two probe types dispatched on policy_type:
      * smolvla  → SmolVLAAttentionProbe (cross-attention heatmaps)
      * pi05     → PI05FeatureProbe (lang→image + action→image heatmaps)

    All other policy types behave identically to the base PolicyServer.

    Visualization is performed in a background thread pool so it does not
    block the gRPC response path.
    """

    prefix = "attn_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: AttnPolicyServerConfig):
        super().__init__(config)
        self._attn_cfg = config
        self._attn_probe = None  # SmolVLAAttentionProbe | PI05FeatureProbe | None
        self._attn_output_dir: Path | None = (
            Path(config.attn_output_dir) if config.attn_output_dir else None
        )
        self._attn_save_every_n: int = max(1, config.attn_save_every_n)
        self._attn_call_count: int = 0
        self._episode_counter: int = 0
        self._attn_offsets_built: bool = False

        # Background thread pool for async visualization saves (2 workers enough)
        self._vis_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="attn_vis")
            if config.attn_output_dir
            else None
        )

        # Lock protecting _attn_call_count and _episode_counter
        self._attn_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Probe lifecycle — re-create after each policy load
    # ------------------------------------------------------------------

    def _init_attn_probe(self) -> None:
        """Create (or recreate) the appropriate probe after policy is loaded."""
        if self.policy is None:
            self._attn_probe = None
            return

        try:
            if self.policy_type == "smolvla":
                self._attn_probe = SmolVLAAttentionProbe(
                    self.policy,
                    keep_cpu_copy=self._attn_cfg.attn_keep_cpu_copy,
                )
                self._attn_offsets_built = False
                self.logger.info("[attn] SmolVLAAttentionProbe initialized.")
            elif self.policy_type == "pi05":
                from .pi05_feature_probe import PI05FeatureProbe

                self._attn_probe = PI05FeatureProbe(
                    self.policy,
                    keep_cpu_copy=self._attn_cfg.attn_keep_cpu_copy,
                )
                self.logger.info("[attn] PI05FeatureProbe initialized.")
            else:
                self._attn_probe = None
        except Exception as exc:
            self.logger.warning(f"[attn] Could not create probe for '{self.policy_type}': {exc}")
            self._attn_probe = None

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        result = super().SendPolicyInstructions(request, context)
        self._init_attn_probe()
        return result

    # ------------------------------------------------------------------
    # Override _predict_action_chunk to inject probe
    # ------------------------------------------------------------------

    def _predict_action_chunk(self, observation_t: TimedObservation):
        if self._attn_probe is None or self._attn_output_dir is None:
            # No probe or no output dir — plain inference
            return super()._predict_action_chunk(observation_t)

        with self._attn_lock:
            call_idx = self._attn_call_count
            self._attn_call_count += 1
            if observation_t.is_episode_start:
                self._episode_counter += 1
            episode_idx = self._episode_counter
        obs_timestep = observation_t.get_timestep()

        # Determine whether to save visualizations for this call
        should_vis = (call_idx % self._attn_save_every_n == 0)

        if not should_vis:
            return super()._predict_action_chunk(observation_t)

        # -- Run inference with probe active --
        # We need access to the preprocessed observation for images.
        # Intercept by patching preprocessor temporarily via closure.
        _preprocessed_obs: dict = {}

        original_get_action_chunk = self._get_action_chunk

        def _wrapped_get_action_chunk(observation, inference_delay=0, leftover=None):
            _preprocessed_obs.update(observation)
            with self._attn_probe as probe:
                result = original_get_action_chunk(
                    observation, inference_delay=inference_delay, leftover=leftover
                )
            # Post-inference: build token offsets and submit visualization
            try:
                self._submit_visualization(probe, _preprocessed_obs, episode_idx, obs_timestep)
            except Exception as exc:
                self.logger.warning(f"[attn] Visualization submission failed: {exc}")
            return result

        # Temporarily replace _get_action_chunk
        self._get_action_chunk = _wrapped_get_action_chunk
        try:
            result = super()._predict_action_chunk(observation_t)
        finally:
            self._get_action_chunk = original_get_action_chunk

        return result

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _extract_images_for_vis(
        self, observation: dict[str, torch.Tensor]
    ) -> list[np.ndarray]:
        """Extract camera images from the preprocessed observation dict as numpy HWC."""
        img_keys = sorted(k for k in observation if OBS_IMAGES in k and "empty" not in k)
        images: list[np.ndarray] = []
        for k in img_keys:
            t = observation[k]
            if t.ndim == 4:
                t = t[0]  # take first batch element
            # t: [C, H, W] float in [0, 1]
            arr = t.detach().cpu().float().numpy()
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, C]
            images.append(arr)
        return images

    def _submit_visualization(
        self,
        probe,
        observation: dict[str, torch.Tensor],
        episode: int,
        timestep: int,
    ) -> None:
        """Finalize capture layout/offsets, then submit async visualization save."""
        if self.policy_type == "smolvla":
            self._submit_smolvla(probe, observation, episode, timestep)
        elif self.policy_type == "pi05":
            self._submit_pi05(probe, observation, episode, timestep)

    def _submit_smolvla(
        self,
        probe,
        observation: dict[str, torch.Tensor],
        episode: int,
        timestep: int,
    ) -> None:
        """SmolVLA path: build token offsets, async-save cross-attention heatmaps."""
        capture = probe.last_capture
        if capture is None or not capture.cross_attn:
            return

        try:
            probe.build_token_offsets(observation)
            if capture.token_offsets is not None:
                self._attn_offsets_built = True
        except Exception as exc:
            self.logger.warning(f"[attn/smolvla] build_token_offsets failed: {exc}")

        images = self._extract_images_for_vis(observation)
        img_hw = (images[0].shape[0], images[0].shape[1]) if images else None

        import copy
        capture_copy = copy.deepcopy(capture)
        output_dir = self._attn_output_dir

        def _save():
            try:
                saved = save_inference_attention(
                    capture=capture_copy,
                    output_dir=output_dir,
                    images=images,
                    token_labels=None,
                    img_hw=img_hw,
                    episode=episode,
                    timestep=timestep,
                )
                self.logger.info(
                    f"[attn/smolvla] Saved {len(saved)} plots for ep{episode} t{timestep}"
                )
            except Exception as exc:
                self.logger.warning(f"[attn/smolvla] Failed to save plots: {exc}")

        if self._vis_executor is not None:
            self._vis_executor.submit(_save)

    def _submit_pi05(
        self,
        probe,
        observation: dict[str, torch.Tensor],
        episode: int,
        timestep: int,
    ) -> None:
        """PI05 path: build token layout, async-save lang→image + action→image heatmaps."""
        from .pi05_feature_visualizer import save_step_features

        try:
            probe.set_token_layout(observation)
        except Exception as exc:
            self.logger.warning(f"[attn/pi05] set_token_layout failed: {exc}")

        capture = probe.last_capture
        if capture is None or capture.token_layout is None:
            return

        images = self._extract_images_for_vis(observation)
        img_hw = (images[0].shape[0], images[0].shape[1]) if images else None

        lang_mask = None
        attn_key = next(
            (k for k in observation if "attention_mask" in k and "language" in k), None
        )
        if attn_key is not None:
            lang_mask = observation[attn_key].bool().cpu()

        import copy
        capture_copy = copy.deepcopy(capture)
        output_dir = self._attn_output_dir

        def _save():
            try:
                saved = save_step_features(
                    capture=capture_copy,
                    output_dir=output_dir,
                    images=images,
                    token_labels=None,
                    img_hw=img_hw,
                    lang_mask=lang_mask,
                    episode=episode,
                    timestep=timestep,
                )
                self.logger.info(
                    f"[attn/pi05] Saved {len(saved)} plots for ep{episode} t{timestep}"
                )
            except Exception as exc:
                self.logger.warning(f"[attn/pi05] Failed to save plots: {exc}")

        if self._vis_executor is not None:
            self._vis_executor.submit(_save)

    def stop(self):
        super().stop()
        if self._vis_executor is not None:
            self._vis_executor.shutdown(wait=True)
            self.logger.info("[attn] Visualization executor shut down.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def serve(cfg: AttnPolicyServerConfig):
    """Start the AttnPolicyServer with the given configuration."""
    logging.info(pformat(asdict(cfg)))

    server_instance = AttnPolicyServer(cfg)
    if cfg.timing_output_dir:
        server_instance.enable_timing(cfg.timing_output_dir)
    if cfg.attn_output_dir:
        server_instance.logger.info(
            f"[attn] Attention visualizations → {cfg.attn_output_dir} "
            f"(every {cfg.attn_save_every_n} inference call(s)) "
            f"[supported: smolvla, pi05]"
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    server_instance.logger.info(f"AttnPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server_instance.logger.info("Keyboard interrupt — stopping server")
        server.stop(grace=2)
    finally:
        server_instance.save_timing()
        server_instance.stop()
        server_instance.logger.info("AttnPolicyServer terminated")


if __name__ == "__main__":
    serve()
