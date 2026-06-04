"""Evaluate Inference-Time RTC on LIBERO benchmarks.

Adapted from bt-libero/it_rtc/eval_libero_rtc.py to use LeRobot interfaces:
  - lerobot.policies.factory.{make_policy, make_pre_post_processors}
  - lerobot.envs.libero.LiberoEnv  +  gymnasium.vector.SyncVectorEnv
  - LeRobot native RTC (SmolVLAPolicy.config.rtc_config)

Two extra modes controlled by EvalConfig.enable_sm:
  - enable_sm=False  (default): vectorized rollout, no gripper SM
  - enable_sm=True:            sequential single-env rollout + OfflineGripperSM
                                set_state() rewind on empty_grasp

Usage::

    python -m lerobot.async_libero_inference.it_rtc.eval_libero_rtc \\
        --policy.path=<checkpoint> \\
        --env.type=libero --env.task=libero_spatial \\
        --eval.n_episodes=10 --eval.batch_size=10 \\
        --eval.async_delay=4 --eval.execution_horizon=10 \\
        --eval.method_type=rtc \\
        --eval.enable_sm=false \\
        --output_dir=outputs/eval/it_rtc/...
"""

# NOTE: do NOT add `from __future__ import annotations` here.
# LeRobot's parser.wrap() uses inspect.getfullargspec().annotations to extract
# the config class type. With PEP 563 (from __future__ import annotations),
# all annotations become strings, so argtype = "EvalPipelineConfig" (a string)
# is passed to draccus.parse(config_class=...), causing:
#   TypeError: must be called with a dataclass type or instance
# Python 3.12 natively supports X | Y unions and generic types without PEP 563.

import datetime as dt
import json
import logging
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor
from tqdm import trange

# ── Imports safe without a GPU display ───────────────────────────────────────
from lerobot.configs import parser
import lerobot.envs as lerobot_envs  # noqa: F401  registers EnvConfig subclasses; no libGL
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
from lerobot.utils.random_utils import set_seed
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from lerobot.utils.constants import ACTION

# gripper_sm has no LIBERO/libGL dependency — safe to import at module level
from .gripper_sm import OfflineGripperSM, OfflineSMConfig

# ── LIBERO/robosuite/libGL-dependent imports (deferred) ──────────────────────
# lerobot.envs.libero → libero.libero.envs → robosuite → OpenCV → libGL.so.1
# Deferring allows parse_eval_results_rtc to import this module without a display.
_LIBERO_IMPORTED = False

def _lazy_libero_imports():
    """Load LIBERO env symbols on first call to main(). Requires GPU/libGL."""
    global _LIBERO_IMPORTED
    if _LIBERO_IMPORTED:
        return
    global LiberoEnv, _get_suite, write_video

    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.utils.io_utils import write_video
    _LIBERO_IMPORTED = True


# ---------------------------------------------------------------------------
# Method configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineMethodConfig:
    """Chunked execution with delayed observation; no RTC guidance."""
    pass


@dataclass(frozen=True)
class RTCMethodConfig:
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: str = "EXP"  # ZEROS / ONES / LINEAR / EXP


# ---------------------------------------------------------------------------
# Dataset & eval configs
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfigForEval:
    repo_id: str | None = None
    root: str | None = None
    norm_stats_path: str | None = None


@dataclass
class EvalConfig:
    """Evaluation knobs."""

    n_episodes: int = 50
    batch_size: int = 50

    # Simulated inference delay: policy receives obs from `async_delay` steps ago.
    async_delay: int = 0

    # Actions per chunk executed before re-planning.
    execution_horizon: int = 10

    # RTC parameters (only used when method_type == "rtc")
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: str = "EXP"

    # Method type: "baseline" or "rtc"
    method_type: str = "baseline"

    # ── State machine toggle ────────────────────────────────────────────────
    enable_sm: bool = False

    # SM parameters (from OfflineSMConfig)
    sm_gripper_close_action_threshold: float = 0.0
    sm_gripper_pos_sum_empty_threshold: float = 0.008
    sm_gripper_pos_sum_open_threshold: float = 0.05
    sm_gripper_pos_sum_grasp_threshold: float = 0.02
    sm_gripper_confirm_steps: int = 3
    sm_activation_delay: int = 10
    sm_max_empty_grasp_retries: int = 3
    sm_enable_home_reset: bool = True
    sm_home_reset_warmup_steps: int = 15
    sm_rewind_buffer_steps: int = 60
    sm_rewind_warmup_steps: int = 10
    sm_rewind_step_back: int = 0          # 0 = oldest snapshot (max rewind distance)
    sm_closing_qpos_velocity_epsilon: float = 0.001

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                f"batch_size ({self.batch_size}) > n_episodes ({self.n_episodes})."
            )
        if self.method_type not in ("baseline", "rtc"):
            raise ValueError(
                f"eval.method_type must be 'baseline' or 'rtc' (got {self.method_type!r})."
            )
        if self.method_type == "rtc" and self.async_delay > self.execution_horizon:
            logging.warning(
                f"async_delay ({self.async_delay}) > execution_horizon ({self.execution_horizon}); "
                f"effective horizon will be raised to max(d,s)={self.async_delay}."
            )
        if self.method_type == "rtc" and self.async_delay == self.execution_horizon and self.async_delay > 0:
            logging.warning(
                f"RTC DEGENERATE: async_delay == execution_horizon == {self.async_delay}. "
                f"effective_horizon=max(d,s)={self.async_delay}. "
                f"All executed actions come from the PREVIOUS chunk (new_chunk[d:s] is empty). "
                f"RTC provides guidance but no new-chunk actions are ever executed. "
                f"Consider using execution_horizon > async_delay (s > d)."
            )

    def make_sm_config(self) -> OfflineSMConfig:
        return OfflineSMConfig(
            gripper_close_action_threshold=self.sm_gripper_close_action_threshold,
            gripper_pos_sum_empty_threshold=self.sm_gripper_pos_sum_empty_threshold,
            gripper_pos_sum_open_threshold=self.sm_gripper_pos_sum_open_threshold,
            gripper_pos_sum_grasp_threshold=self.sm_gripper_pos_sum_grasp_threshold,
            gripper_confirm_steps=self.sm_gripper_confirm_steps,
            sm_activation_delay=self.sm_activation_delay,
            max_empty_grasp_retries=self.sm_max_empty_grasp_retries,
            enable_home_reset=self.sm_enable_home_reset,
            home_reset_warmup_steps=self.sm_home_reset_warmup_steps,
            rewind_buffer_steps=self.sm_rewind_buffer_steps,
            rewind_warmup_steps=self.sm_rewind_warmup_steps,
            rewind_step_back=self.sm_rewind_step_back,
            closing_qpos_velocity_epsilon=self.sm_closing_qpos_velocity_epsilon,
        )


@dataclass
class EvalPipelineConfig:
    env: lerobot_envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfigForEval = field(default_factory=DatasetConfigForEval)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    task_description: str | None = ""
    gpu_id: int = 0

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning("No pretrained path provided — policy will have random weights.")

        if not self.job_name:
            self.job_name = f"{self.env.type}_{self.policy.type}" if self.policy else "eval"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_libero_processor() -> PolicyProcessorPipeline:
    """Create the LIBERO env preprocessor (image flip + state reshape)."""
    return PolicyProcessorPipeline(steps=[LiberoProcessorStep()])


def _preprocess_obs_for_policy(
    raw_obs: dict,
    libero_proc,
    policy_preprocessor,
    task_descs: list[str],
    device: torch.device,
) -> dict:
    """Convert raw gym VectorEnv observation to a policy-ready batch.

    Pipeline:
      1. preprocess_observation(): numpy arrays → LeRobot tensors (B, ...)
      2. LiberoProcessorStep: flip images 180°; reshape robot_state →
         observation.state (B, 8) = [eef_pos(3) | eef_axisangle(3) | gripper_qpos(2)]
      3. task injection
      4. policy_preprocessor (policy-specific):
           SmolVLA: NormalizerStep → TokenizerStep → DeviceStep
                    state is passed as a separate tensor to the model
           Pi05:    NormalizerStep → Pi05PrepareStateTokenizerStep → TokenizerStep → DeviceStep
                    state is discretized (256 bins) and prepended to the text:
                    "Task: {desc}, State: {8 bins};\nAction: "
                    then tokenized — state enters the model via language tokens

    Both policies require obs_type='pixels_agent_pos' for robot state access.
    """
    obs = preprocess_observation(raw_obs)
    obs = libero_proc(obs)

    # Task injection: the preprocessor's tokenizer expects list[str]
    if len(task_descs) == 1:
        obs["task"] = task_descs[0]          # single string → AddBatchDim wraps in list
    else:
        obs["task"] = task_descs             # list[str] already batched

    if policy_preprocessor is not None:
        obs = policy_preprocessor(obs)
    else:
        # Fallback: move all tensors to device manually
        obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

    return obs


def _postprocess_actions(actions_norm: Tensor, postprocessor) -> Tensor:
    """Unnormalize a chunk of actions: (B, horizon, action_dim) → env space."""
    if postprocessor is None:
        return actions_norm

    B, H, D = actions_norm.shape
    out = []
    for i in range(H):
        step_action = actions_norm[:, i, :]   # (B, D)
        step_env = postprocessor(step_action)  # unnormalize
        if isinstance(step_env, dict):
            step_env = step_env[ACTION]
        out.append(step_env)
    return torch.stack(out, dim=1)             # (B, H, D)


# ---------------------------------------------------------------------------
# RTC enabling
# ---------------------------------------------------------------------------

def enable_rtc_on_policy(policy: PreTrainedPolicy, rtc_method: RTCMethodConfig,
                          execution_horizon: int) -> None:
    """Configure native LeRobot RTC on SmolVLAPolicy or PI05Policy.

    Both policies share the same RTC interface:
        policy.config.rtc_config = RTCConfig(...)
        policy.init_rtc_processor()
        policy.predict_action_chunk(inference_delay=..., prev_chunk_left_over=..., ...)

    No monkey-patching required (unlike the VLASH version which patched SmolVLA).
    """
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.configs.types import RTCAttentionSchedule

    # Accept any policy that exposes init_rtc_processor (SmolVLA, Pi05, Pi0, …)
    if not hasattr(policy, "init_rtc_processor"):
        logging.warning(
            f"Policy {type(policy).__name__} does not support init_rtc_processor(); "
            "RTC will be skipped — treating as baseline."
        )
        return

    policy.config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=rtc_method.max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[rtc_method.prefix_attention_schedule],
    )
    policy.init_rtc_processor()
    logging.info(
        f"RTC enabled on {type(policy).__name__}: "
        f"execution_horizon={execution_horizon}, "
        f"guidance={rtc_method.max_guidance_weight}, "
        f"schedule={rtc_method.prefix_attention_schedule}"
    )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

MAX_STEP_BY_SUITE = {
    "libero_spatial": 230,
    "libero_object":  290,
    "libero_goal":    310,
    "libero_10":      530,
    "libero_90":      410,
}


def _make_env_fn(suite, suite_name: str, task_id: int, episode_index: int,
                 max_episode_steps: int | None, obs_type: str = "pixels_agent_pos") -> callable:
    """Return a no-arg callable that creates a single LiberoEnv."""
    def _fn():
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=max_episode_steps,
            obs_type=obs_type,
            episode_index=episode_index,
        )
    return _fn


def schedule_envs(suite, suite_name: str, task_ids: list[int],
                  batch_size: int, n_episodes_per_task: int | None,
                  max_episode_steps: int | None, obs_type: str = "pixels_agent_pos"):
    """Build a schedule: list of batches, each batch is a list of (task_id, env_fns, n_eps).

    Returns:
        schedule: list of batches
            Each batch: list of (task_id, task_language, env_fn_list)
    """
    from lerobot.envs.libero import get_task_init_states as _get_init_states

    def _n_available(tid: int) -> int:
        return len(_get_init_states(suite, tid))

    n_tasks = len(task_ids)

    if batch_size < n_tasks:
        # Multiple batches
        n_batches = (n_tasks + batch_size - 1) // batch_size
        schedule = []
        for b in range(n_batches):
            batch_task_ids = task_ids[b * batch_size: (b + 1) * batch_size]
            batch = []
            for tid in batch_task_ids:
                task = suite.get_task(tid)
                available = _n_available(tid)
                n_ep = min(n_episodes_per_task, available) if n_episodes_per_task else available
                env_fns = [_make_env_fn(suite, suite_name, tid, ep, max_episode_steps, obs_type)
                           for ep in range(n_ep)]
                batch.append((tid, task.language, env_fns))
            schedule.append(batch)
        return schedule

    # All tasks fit in one or more parallel batches (multiple replicas per task)
    n_replicas_per_task, rem = divmod(batch_size, n_tasks)
    schedule_flat = []
    for i, tid in enumerate(task_ids):
        task = suite.get_task(tid)
        available = _n_available(tid)
        n_ep = min(n_episodes_per_task, available) if n_episodes_per_task else available
        n_replicas = n_replicas_per_task + (1 if i < rem else 0)
        n_replicas = min(n_replicas, n_ep)

        n_per_replica, ep_rem = divmod(n_ep, n_replicas)
        start_ep = 0
        for r in range(n_replicas):
            end_ep = start_ep + n_per_replica + (1 if r < ep_rem else 0)
            env_fns = [_make_env_fn(suite, suite_name, tid, ep, max_episode_steps, obs_type)
                       for ep in range(start_ep, end_ep)]
            schedule_flat.append((tid, task.language, env_fns))
            start_ep = end_ep

    return [schedule_flat]


# ---------------------------------------------------------------------------
# Core rollout — no SM
# ---------------------------------------------------------------------------

def rollout_chunked(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    libero_proc,
    task_descs: list[str],
    seeds=None,
    return_observations: bool = False,
    async_delay: int = 0,
    execution_horizon: int = 10,
    rtc_enabled: bool = False,
) -> dict:
    """Chunked rollout for Baseline and RTC conditions (no state machine).

    All env.step() calls happen synchronously.  Inference delay is simulated
    by feeding the policy an observation from `async_delay` steps ago.

    RTC guidance (rtc_enabled=True)
    --------------------------------
    Follows the reference (eval_flow.py + model.py from Kinetix):
      1. prev_chunk is shifted by effective_horizon; position [i] aligns
         1:1 with position [i] of the new chunk (model-space).
      2. old_actions[:delay] + new_actions[delay:horizon] are executed.
    """
    device = get_device_from_parameters(policy)
    policy.reset()
    observation, info = env.reset(seed=list(seeds) if seeds is not None else None)

    n_envs = env.num_envs
    obs_buffer = []
    all_actions, all_rewards, all_successes, all_dones = [], [], [], []
    if return_observations:
        all_observations = []

    done = np.zeros(n_envs, dtype=bool)

    # prev_chunk: model-space (normalized) actions, (n_envs, chunk_size, action_dim)
    prev_chunk_model: Tensor | None = None
    prev_chunk_for_exec: Tensor | None = None   # same, kept for mixing
    action_queue: list[Tensor] = []             # list of (n_envs, action_dim) env-space tensors

    while not np.all(done):
        obs_proc = _preprocess_obs_for_policy(observation, libero_proc, preprocessor,
                                              task_descs, device)
        if return_observations:
            all_observations.append(deepcopy(obs_proc))
        obs_buffer.append(deepcopy(obs_proc))

        if len(action_queue) == 0:
            # Select (possibly delayed) observation
            if async_delay > 0 and len(obs_buffer) > async_delay:
                policy_obs = obs_buffer[-(async_delay + 1)]
            else:
                policy_obs = obs_buffer[-1]

            # effective_horizon: how many actions to execute before replanning.
            # RTC:      max(d, s)  — paper constraint s >= d
            # Baseline: s          — explicit replanning frequency control
            #           s=0 or s>=chunk_size means "use full chunk" (legacy behavior)
            with torch.no_grad():
                if rtc_enabled and prev_chunk_model is not None:
                    effective_horizon = max(async_delay, execution_horizon)
                    new_chunk_model = policy.predict_action_chunk(
                        policy_obs,
                        inference_delay=async_delay,
                        prev_chunk_left_over=prev_chunk_model.to(device),
                        execution_horizon=effective_horizon,
                    )
                else:
                    effective_horizon = None  # resolved after chunk is generated
                    new_chunk_model = policy.predict_action_chunk(policy_obs)

            raw_new_model = new_chunk_model.detach().cpu()    # (n_envs, chunk, dim)
            chunk_size = raw_new_model.shape[1]

            if effective_horizon is None:
                # Baseline: use execution_horizon if it is a meaningful sub-chunk value,
                # otherwise fall back to the full chunk (standard chunked-execution baseline).
                if 0 < execution_horizon < chunk_size:
                    effective_horizon = execution_horizon
                else:
                    effective_horizon = chunk_size

            action_dim = raw_new_model.shape[2]

            # Shift prev_chunk for the NEXT planning cycle (model space).
            # When effective_horizon >= chunk_size the entire chunk is consumed
            # before replanning → there is NO leftover prefix.  Feeding an all-zero
            # prefix to RTCProcessor would compute err=(0 - x1_t)*weights and pull
            # every action toward zero (catastrophic SR collapse).  Disable guidance
            # for the next cycle instead (prev_chunk_model=None → behaves like baseline).
            prev_chunk_for_exec = prev_chunk_model  # save before overwriting
            if effective_horizon >= chunk_size:
                prev_chunk_model = None
            else:
                prev_chunk_model = torch.cat([
                    raw_new_model[:, effective_horizon:, :],
                    torch.zeros(n_envs, effective_horizon, action_dim),
                ], dim=1)

            # Build execution chunk (model space), then unnormalize
            if rtc_enabled and prev_chunk_for_exec is not None and async_delay > 0:
                old_model = prev_chunk_for_exec[:, :async_delay, :]
                new_model = raw_new_model[:, async_delay:effective_horizon, :]
                exec_chunk_model = torch.cat([old_model, new_model], dim=1)
            else:
                exec_chunk_model = raw_new_model[:, :effective_horizon, :]

            # Unnormalize to env action space
            exec_chunk_env = _postprocess_actions(exec_chunk_model, postprocessor)

            for i in range(exec_chunk_env.shape[1]):
                action_queue.append(exec_chunk_env[:, i, :])

        action = action_queue.pop(0)
        observation, reward, terminated, truncated, info = env.step(action.numpy())
        successes = [info[i].get("is_success", False) if isinstance(info, list)
                     else bool(info.get("is_success", np.zeros(n_envs))[i])
                     for i in range(n_envs)]
        done = np.logical_or(done, np.logical_or(terminated, truncated))

        all_actions.append(action)
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes, dtype=torch.float32))

    ret = {
        "action":  torch.stack(all_actions, dim=1),
        "reward":  torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done":    torch.stack(all_dones, dim=1),
    }
    if return_observations:
        ret["observation"] = {
            k: torch.stack([o[k] for o in all_observations], dim=1)
            for k in all_observations[0]
            if isinstance(all_observations[0][k], torch.Tensor)
        }
    return ret


# ---------------------------------------------------------------------------
# Rollout with SM (single-env, sequential)
# ---------------------------------------------------------------------------

def rollout_chunked_with_sm(
    env: gym.vector.SyncVectorEnv,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    libero_proc,
    task_desc: str,
    sm: OfflineGripperSM,
    seeds=None,
    async_delay: int = 0,
    execution_horizon: int = 10,
    rtc_enabled: bool = False,
) -> dict:
    """Single-env chunked rollout with gripper SM rewind support.

    Requires SyncVectorEnv (n_envs=1) for direct MuJoCo state access.
    """
    device = get_device_from_parameters(policy)
    policy.reset()
    sm.reset()

    seed = [seeds[0]] if seeds is not None else None
    observation, info = env.reset(seed=seed)
    sm.save_initial_snapshot(env)

    obs_buffer = []
    all_actions, all_rewards, all_successes, all_dones = [], [], [], []
    done = np.zeros(1, dtype=bool)

    prev_chunk_model: Tensor | None = None
    prev_chunk_for_exec: Tensor | None = None
    action_queue: list[Tensor] = []

    # Stats for result
    retries = 0

    while not np.all(done):
        # ── SM warmup phase ────────────────────────────────────────────────
        if sm.warmup_remaining > 0:
            # Hold action during warmup (open gripper, no EEF delta)
            hold_action = np.zeros((1, 7), dtype=np.float32)
            hold_action[0, -1] = -1.0
            observation, reward, terminated, truncated, info = env.step(hold_action)
            sm.warmup_remaining -= 1

            # Save snapshot and continue (no policy inference yet)
            sm.save_snapshot(env)
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done))
            all_successes.append(torch.tensor([False]))
            all_actions.append(torch.from_numpy(hold_action))

            if sm.warmup_remaining == 0:
                # Clear action queue: request fresh inference
                action_queue.clear()
                obs_buffer.clear()
                prev_chunk_model = None
                prev_chunk_for_exec = None
                policy.reset()
            continue

        # ── Normal step ────────────────────────────────────────────────────
        obs_proc = _preprocess_obs_for_policy(observation, libero_proc, preprocessor,
                                              [task_desc], device)
        obs_buffer.append(deepcopy(obs_proc))

        if len(action_queue) == 0:
            if async_delay > 0 and len(obs_buffer) > async_delay:
                policy_obs = obs_buffer[-(async_delay + 1)]
            else:
                policy_obs = obs_buffer[-1]

            with torch.no_grad():
                if rtc_enabled and prev_chunk_model is not None:
                    effective_horizon = max(async_delay, execution_horizon)
                    new_chunk_model = policy.predict_action_chunk(
                        policy_obs,
                        inference_delay=async_delay,
                        prev_chunk_left_over=prev_chunk_model.to(device),
                        execution_horizon=effective_horizon,
                    )
                else:
                    effective_horizon = None
                    new_chunk_model = policy.predict_action_chunk(policy_obs)

            raw_new_model = new_chunk_model.detach().cpu()
            chunk_size = raw_new_model.shape[1]
            if effective_horizon is None:
                effective_horizon = execution_horizon if 0 < execution_horizon < chunk_size else chunk_size

            action_dim = raw_new_model.shape[2]
            prev_chunk_for_exec = prev_chunk_model
            # effective_horizon >= chunk_size → no leftover; disable next-cycle
            # guidance (None) instead of feeding an all-zero prefix (see rollout_chunked).
            if effective_horizon >= chunk_size:
                prev_chunk_model = None
            else:
                prev_chunk_model = torch.cat([
                    raw_new_model[:, effective_horizon:, :],
                    torch.zeros(1, effective_horizon, action_dim),
                ], dim=1)

            if rtc_enabled and prev_chunk_for_exec is not None and async_delay > 0:
                exec_chunk_model = torch.cat([
                    prev_chunk_for_exec[:, :async_delay, :],
                    raw_new_model[:, async_delay:effective_horizon, :],
                ], dim=1)
            else:
                exec_chunk_model = raw_new_model[:, :effective_horizon, :]

            exec_chunk_env = _postprocess_actions(exec_chunk_model, postprocessor)
            for i in range(exec_chunk_env.shape[1]):
                action_queue.append(exec_chunk_env[:, i, :])

        action = action_queue.pop(0)
        action_np = action.numpy()

        # Save SM snapshot before step
        sm.save_snapshot(env)

        observation, reward, terminated, truncated, info = env.step(action_np)

        # Extract gripper qpos for SM from robot_state in raw observation
        try:
            gripper_qpos = observation["robot_state"]["gripper"]["qpos"]  # (1, 2)
        except (KeyError, TypeError):
            gripper_qpos = np.zeros((1, 2))

        success = bool(info[0].get("is_success", False)) if isinstance(info, list) else \
                  bool(info.get("is_success", [False])[0])
        done_step = np.logical_or(terminated, truncated)
        done = np.logical_or(done, done_step)

        all_actions.append(action)
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor([success], dtype=torch.float32))

        # ── SM update ──────────────────────────────────────────────────────
        if not np.all(done) and sm.is_active:
            action_gripper = float(action_np[0, -1])
            # Pass env so the SM can query MuJoCo contacts for robust
            # empty-grasp / slip detection (matches SimSmartClient).
            should_rewind = sm.update(action_gripper, gripper_qpos, env)
            if should_rewind:
                retries += 1
                ok = sm.execute_rewind(env)
                if ok:
                    # Warmup loop (sm.warmup_remaining > 0) will step env and
                    # refresh `observation`; clear queue so no stale actions execute.
                    action_queue.clear()
                    obs_buffer.clear()

    ret = {
        "action":  torch.stack(all_actions, dim=1),
        "reward":  torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done":    torch.stack(all_dones, dim=1),
        "retries": retries,
    }
    return ret


# ---------------------------------------------------------------------------
# Eval policy
# ---------------------------------------------------------------------------

def eval_policy(
    env_cfg,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    libero_proc,
    schedule,
    suite_name: str,
    suite,
    start_seed: int | None = None,
    max_steps: int = 520,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    async_delay: int = 0,
    execution_horizon: int = 10,
    rtc_enabled: bool = False,
    enable_sm: bool = False,
    sm_config: OfflineSMConfig | None = None,
    obs_type: str = "pixels_agent_pos",
) -> dict:
    start = time.time()
    policy.eval()

    sm = OfflineGripperSM(sm_config) if (enable_sm and sm_config is not None) else None

    total_episodes = sum(
        sum(len(env_fns) for _, _, env_fns in batch) for batch in schedule
    )
    progbar = trange(len(schedule), desc="Eval batches", dynamic_ncols=True)

    info: dict = {"overall": {
        "avg_sum_rewards": [], "avg_max_rewards": [],
        "pc_successes": [], "avg_episode_length": [],
    }}
    global_ep = 0
    rendered_count_by_task: dict[str, int] = {}
    video_paths_by_task: dict[str, list[str]] = {}

    for batch_idx in progbar:
        batch = schedule[batch_idx]

        for task_id, task_language, env_fns in batch:
            task = suite.get_task(task_id)
            task_name = task.name

            if task_name not in info:
                info[task_name] = {
                    "avg_sum_rewards": [], "avg_max_rewards": [],
                    "pc_successes": [], "avg_episode_length": [],
                }
            rendered_count_by_task.setdefault(task_name, 0)
            video_paths_by_task.setdefault(task_name, [])

            n_ep = len(env_fns)
            task_descs = [task_language] * (1 if enable_sm else len(env_fns))

            if enable_sm:
                # Sequential single-env rollout
                for ep_idx, env_fn in enumerate(env_fns):
                    seeds_ep = [start_seed + global_ep] if start_seed is not None else None
                    single_env = gym.vector.SyncVectorEnv([env_fn])
                    try:
                        rollout_data = rollout_chunked_with_sm(
                            env=single_env,
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            libero_proc=libero_proc,
                            task_desc=task_language,
                            sm=sm,
                            seeds=seeds_ep,
                            async_delay=async_delay,
                            execution_horizon=execution_horizon,
                            rtc_enabled=rtc_enabled,
                        )
                    finally:
                        single_env.close()

                    n_steps = rollout_data["action"].shape[1]
                    done_idx = torch.argmax(rollout_data["done"].int(), dim=1)
                    mask = (torch.arange(n_steps) <= (done_idx + 1).unsqueeze(1)).int()

                    sum_rew = (rollout_data["reward"] * mask).sum(dim=1).item()
                    max_rew = (rollout_data["reward"] * mask).max(dim=1).values.item()
                    success  = bool((rollout_data["success"] * mask).any(dim=1).item())
                    ep_len   = int(done_idx.item()) + 1

                    for k in (task_name, "overall"):
                        info[k]["avg_sum_rewards"].append(sum_rew)
                        info[k]["avg_max_rewards"].append(max_rew)
                        info[k]["pc_successes"].append(float(success))
                        info[k]["avg_episode_length"].append(ep_len)

                    global_ep += 1

            else:
                # Vectorized rollout (all envs in batch at once)
                vec_env = gym.vector.SyncVectorEnv(env_fns)
                seeds_batch = (
                    list(range(start_seed + global_ep, start_seed + global_ep + n_ep))
                    if start_seed is not None else None
                )
                ep_frames: list[np.ndarray] = []
                render_cb = None
                if max_episodes_rendered > 0 and videos_dir is not None:
                    def render_cb(e, _f=ep_frames):
                        _f.append(np.stack(e.call("render")))

                try:
                    rollout_data = rollout_chunked(
                        env=vec_env,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        libero_proc=libero_proc,
                        task_descs=task_descs,  # already [task_language]*n_ep
                        seeds=seeds_batch,
                        async_delay=async_delay,
                        execution_horizon=execution_horizon,
                        rtc_enabled=rtc_enabled,
                    )
                finally:
                    vec_env.close()

                n_steps = rollout_data["action"].shape[1]
                done_indices = torch.argmax(rollout_data["done"].int(), dim=1)
                mask = (
                    torch.arange(n_steps)
                    <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)
                ).int()

                batch_sum_rew = (rollout_data["reward"] * mask).sum(dim=1)
                batch_max_rew = (rollout_data["reward"] * mask).max(dim=1).values
                batch_success = (rollout_data["success"] * mask).any(dim=1).float()

                for env_idx in range(n_ep):
                    ep_len = int(done_indices[env_idx].item()) + 1
                    for k in (task_name, "overall"):
                        info[k]["avg_sum_rewards"].append(batch_sum_rew[env_idx].item())
                        info[k]["avg_max_rewards"].append(batch_max_rew[env_idx].item())
                        info[k]["pc_successes"].append(batch_success[env_idx].item())
                        info[k]["avg_episode_length"].append(ep_len)

                global_ep += n_ep

        # Update progress bar
        if info["overall"]["pc_successes"]:
            acc = np.mean(info["overall"]["pc_successes"]) * 100
            n_done = len(info["overall"]["pc_successes"])
            progbar.set_description(
                f"Eval batches (Acc: {acc:.1f}%, {n_done}/{total_episodes} eps)"
            )

    for key in info:
        for metric in ("avg_sum_rewards", "avg_max_rewards", "pc_successes", "avg_episode_length"):
            if info[key][metric]:
                info[key][metric] = float(np.nanmean(info[key][metric]))
                if metric == "pc_successes":
                    info[key][metric] *= 100
            else:
                info[key][metric] = float("nan")

    info["eval_s"] = time.time() - start
    return info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@parser.wrap()
def main(cfg: EvalPipelineConfig):
    _lazy_libero_imports()   # load LIBERO/robosuite/GL-dependent symbols
    init_logging()
    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # ── Load policy ────────────────────────────────────────────────────────
    logging.info("Loading policy...")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()

    # Override norm stats if requested
    norm_stats_path = getattr(cfg.dataset, "norm_stats_path", None)
    if norm_stats_path:
        from safetensors.torch import load_file
        import os
        model_file = os.path.join(norm_stats_path, "model.safetensors")
        donor_sd = load_file(model_file)
        norm_prefixes = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")
        norm_keys = {k: v for k, v in donor_sd.items() if any(k.startswith(p) for p in norm_prefixes)}
        policy.load_state_dict(norm_keys, strict=False)
        logging.info(f"Loaded norm stats from {norm_stats_path}")

    # ── State-dependency check ─────────────────────────────────────────────
    # Pi05 encodes observation.state into the language prompt via
    # Pi05PrepareStateTokenizerProcessorStep: "Task: {t}, State: {bins};\nAction: "
    # SmolVLA passes state as a separate tensor to embed_prefix.
    # Both policies require obs_type='pixels_agent_pos' to get robot state.
    obs_type = getattr(cfg.env, "obs_type", "pixels_agent_pos")
    if obs_type != "pixels_agent_pos":
        policy_type_str = getattr(cfg.policy, "type", "")
        logging.warning(
            f"obs_type='{obs_type}' — robot state (gripper_qpos, eef_pos, …) will be missing. "
            f"Policy '{policy_type_str}' requires 'pixels_agent_pos' for state input. "
            f"SM gripper detection will also be disabled if state is absent."
        )
        if cfg.eval.enable_sm:
            raise ValueError(
                "enable_sm=True requires obs_type='pixels_agent_pos' for gripper qpos access."
            )

    # ── Load preprocessor / postprocessor ──────────────────────────────────
    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            cfg.policy,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
        )
    except Exception as e:
        logging.warning(f"Could not load preprocessor: {e}. Using pass-through.")
        preprocessor = None
        postprocessor = None

    libero_proc = _make_libero_processor()

    # ── Configure RTC ──────────────────────────────────────────────────────
    method_type = cfg.eval.method_type
    rtc_enabled = method_type == "rtc"

    if rtc_enabled:
        # RTC sanity check: it needs leftover actions to anchor guidance, which
        # only exist when execution_horizon < chunk_size. With s >= chunk_size the
        # whole chunk is executed before replanning, the leftover prefix is empty,
        # and RTC degrades to baseline (guidance auto-disabled in rollout_chunked).
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is not None and cfg.eval.execution_horizon >= chunk_size:
            logging.warning(
                colored("RTC WARNING: ", "red", attrs=["bold"])
                + f"execution_horizon ({cfg.eval.execution_horizon}) >= chunk_size ({chunk_size}). "
                f"There is NO leftover prefix to guide on, so RTC will behave like baseline. "
                f"For RTC to take effect use execution_horizon < chunk_size "
                f"(e.g. {chunk_size // 2} or {chunk_size // 4}), and async_delay < execution_horizon."
            )
        rtc_method = RTCMethodConfig(
            max_guidance_weight=cfg.eval.max_guidance_weight,
            prefix_attention_schedule=cfg.eval.prefix_attention_schedule,
        )
        enable_rtc_on_policy(policy, rtc_method, execution_horizon=cfg.eval.execution_horizon)

    enable_sm = cfg.eval.enable_sm
    sm_config = cfg.eval.make_sm_config() if enable_sm else None

    logging.info(
        colored(f"Method: {method_type.upper()}", "cyan", attrs=["bold"])
        + f"  |  async_delay={cfg.eval.async_delay}"
        + f"  |  execution_horizon={cfg.eval.execution_horizon}"
        + f"  |  enable_sm={enable_sm}"
        + (f"  |  guidance={cfg.eval.max_guidance_weight}  schedule={cfg.eval.prefix_attention_schedule}"
           if rtc_enabled else "")
    )

    # ── Build task schedule ────────────────────────────────────────────────
    suite_name = cfg.env.task
    suite = _get_suite(suite_name)
    task_ids = list(range(suite.n_tasks))
    max_steps = MAX_STEP_BY_SUITE.get(suite_name, 530)

    schedule = schedule_envs(
        suite=suite,
        suite_name=suite_name,
        task_ids=task_ids,
        batch_size=cfg.eval.batch_size,
        n_episodes_per_task=cfg.eval.n_episodes,
        max_episode_steps=max_steps,
        obs_type=getattr(cfg.env, "obs_type", "pixels_agent_pos"),
    )

    logging.info(f"Evaluating {len(task_ids)} tasks, {cfg.eval.n_episodes} ep/task")

    # ── Run evaluation ─────────────────────────────────────────────────────
    with torch.no_grad():
        info = eval_policy(
            env_cfg=cfg.env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            libero_proc=libero_proc,
            schedule=schedule,
            suite_name=suite_name,
            suite=suite,
            start_seed=cfg.seed,
            max_steps=max_steps,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            async_delay=cfg.eval.async_delay,
            execution_horizon=cfg.eval.execution_horizon,
            rtc_enabled=rtc_enabled,
            enable_sm=enable_sm,
            sm_config=sm_config,
            obs_type=getattr(cfg.env, "obs_type", "pixels_agent_pos"),
        )

    # ── Save results ───────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_info: dict = {}
    if "overall" in info:
        ordered_info["overall"] = info["overall"]
    for key in sorted(info.keys()):
        if key not in ("overall", "eval_s", "video_paths"):
            ordered_info[key] = info[key]
    for meta_key in ("eval_s", "video_paths"):
        if meta_key in info:
            ordered_info[meta_key] = info[meta_key]

    ordered_info["config"] = {
        "method":             method_type,
        "async_delay":        cfg.eval.async_delay,
        "execution_horizon":  cfg.eval.execution_horizon,
        "max_guidance_weight":     cfg.eval.max_guidance_weight if rtc_enabled else None,
        "prefix_attention_schedule": cfg.eval.prefix_attention_schedule if rtc_enabled else None,
        "enable_sm":          enable_sm,
        "n_episodes":         cfg.eval.n_episodes,
        "seed":               cfg.seed,
        "policy_path":        str(pretrained_path) if pretrained_path else None,
        "suite":              suite_name,
    }

    results_file = output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(ordered_info, f, indent=2)

    logging.info(colored("Results saved to:", "yellow", attrs=["bold"]) + f" {results_file}")

    # Pretty-print summary
    overall = ordered_info.get("overall", {})
    sm_tag = " [+SM]" if enable_sm else ""
    print(colored("=" * 60, "cyan", attrs=["bold"]))
    print(colored(
        f"  {method_type.upper()}{sm_tag}  delay={cfg.eval.async_delay}"
        f"  horizon={cfg.eval.execution_horizon}", "cyan", attrs=["bold"]
    ))
    print(colored("=" * 60, "cyan", attrs=["bold"]))
    sr_pct = f"{overall.get('pc_successes', float('nan')):.1f}%"
    print(f"  Success rate : {colored(sr_pct, 'green', attrs=['bold'])}")
    print(f"  Avg reward   : {overall.get('avg_sum_rewards', float('nan')):.3f}")
    print(f"  Avg ep length: {overall.get('avg_episode_length', float('nan')):.1f} steps")
    print(f"  Eval time    : {info.get('eval_s', 0):.0f}s")
    print(colored("=" * 60, "cyan", attrs=["bold"]))

    logging.info("Done.")
    return info


if __name__ == "__main__":
    main()
