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

"""Async-inference LIBERO evaluation with gripper SM + set_state() rewind.

Identical to run_libero_test.py but uses SimSmartClient and SimSmartClientConfig,
which add empty_grasp detection and MuJoCo set_state() rewind.

Usage::

    # 1. Start policy server:
    python -m lerobot.async_inference.policy_server \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=<path> \\
        --host=localhost --port=8080

    # 2. Run evaluation with SM:
    python -m lerobot.async_inference.sim_test.run_libero_smart_test \\
        --env_task=libero_object \\
        --obs_type=pixels_agent_pos \\
        --policy_type=smolvla \\
        --pretrained_name_or_path=<path> \\
        --server_address=localhost:8080 \\
        --enable_gripper_sm=true \\
        --rewind_buffer_steps=60 \\
        --rewind_warmup_steps=10 \\
        --max_empty_grasp_retries=3 \\
        --episodes_per_task=10 \\
        --results_dir=./libero_smart_results

    # 3. Save evaluation videos (inherited from LiberoSimConfig):
    python -m lerobot.async_inference.sim_test.run_libero_smart_test \\
        ...same flags... \\
        --save_video=true \\
        --video_dir=./libero_videos   # default: <results_dir>/videos

Results are saved to:
  <results_dir>/episodes.json     — per-episode record (includes retries field)
  <results_dir>/aggregate.json    — per-task + overall success rates
  <results_dir>/summary.txt       — human-readable aggregate + retry stats
  <video_dir>/ep_<N>_<status>.mp4 — episode videos (when --save_video=true)
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

import draccus
from lerobot.utils.utils import init_logging

from .run_libero_test import _build_libero_env_and_preprocessor, _read_timing_tables, _save_results
from .sim_client import EpisodeResult, _get_task_description
from .sim_smart_client import SimSmartClient, SimSmartClientConfig, SmartEpisodeResult


def _save_smart_summary(
    all_results: list,
    smart_results: list,
    cfg: "SimSmartClientConfig",
    total_t: float,
    timing_output_dir: str | None = None,
) -> None:
    """Write aggregate + retry stats (+ optional timing tables) to <results_dir>/summary.txt."""
    if not all_results:
        return

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_sr = sum(r.success for r in all_results) / len(all_results)
    avg_steps  = sum(r.steps   for r in all_results) / len(all_results)

    # Per-task breakdown
    by_task: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_task[r.task_description].append(r)

    nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731

    lines: list[str] = [
        "=" * 72,
        "  LIBERO Smart Evaluation — Summary",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "── Config ──────────────────────────────────────────────────────────",
        f"  suite                : {cfg.env_task}",
        f"  model                : {cfg.pretrained_name_or_path}",
        f"  obs_type             : {cfg.obs_type}",
        f"  fps                  : {cfg.fps}",
        f"  episodes_per_task    : {cfg.episodes_per_task}",
        f"  enable_gripper_sm    : {cfg.enable_gripper_sm}",
    ]
    if cfg.enable_gripper_sm:
        lines += [
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
    ]

    if smart_results:
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
        # Rescue rate: fraction of retried episodes that ultimately succeeded
        rescue_rate = (
            success_after_retry / len(eps_with_retry)
            if eps_with_retry else float("nan")
        )
        # SR lift: how much the SM raised overall_sr vs. a no-SM baseline
        # (baseline = same run but success_after_retry episodes would have failed)
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
        if smart_eps:
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
    logging.info(f"[LiberoSmartTest] Summary saved to {summary_path}")


@draccus.wrap()
def run_libero_smart_test(cfg: SimSmartClientConfig):
    init_logging(console_level=cfg.log_level.upper())
    logging.info("[LiberoSmartTest] Config:\n" + pformat(asdict(cfg)))

    logging.info(f"[LiberoSmartTest] Building LIBERO envs for suite '{cfg.env_task}' ...")
    envs_dict, env_preprocessor, lerobot_features = _build_libero_env_and_preprocessor(cfg)

    task_list = [
        (suite, tid, env)
        for suite, task_envs in envs_dict.items()
        for tid, env in sorted(task_envs.items())
    ]
    logging.info(f"[LiberoSmartTest] Built {len(task_list)} task env(s)")

    if not task_list:
        logging.error("[LiberoSmartTest] No task environments created. Aborting.")
        return

    first_suite, first_tid, first_env = task_list[0]
    first_task_desc = _get_task_description(first_env)

    client = SimSmartClient(
        config=cfg,
        env=first_env,
        env_preprocessor=env_preprocessor,
        lerobot_features=lerobot_features,
        task_description=first_task_desc,
    )

    all_results: list[EpisodeResult] = []

    if not client.start():
        logging.error("[LiberoSmartTest] Could not connect to policy server. Aborting.")
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

    receiver = threading.Thread(
        target=client.receive_actions, daemon=True, name="action-receiver"
    )
    receiver.start()

    t_all_start = time.perf_counter()
    global_ep = 0

    try:
        for task_idx, (suite_name, task_id, task_env) in enumerate(task_list):
            task_desc = _get_task_description(task_env)
            logging.info(
                f"[LiberoSmartTest] ══ Task {task_idx + 1}/{len(task_list)} "
                f"| suite={suite_name} | task_id={task_id} | desc='{task_desc}' ══"
            )

            client.env = task_env

            task_results: list[EpisodeResult] = []
            for ep_local in range(cfg.episodes_per_task):
                result = client.run_episode(
                    episode_id=global_ep,
                    max_steps=cfg.max_episode_steps or 500,
                    first_episode=(global_ep == 0),
                    task_description=task_desc,
                )
                task_results.append(result)
                all_results.append(result)
                global_ep += 1

                logging.info(
                    f"[LiberoSmartTest] task={task_id} ep={ep_local}/{cfg.episodes_per_task - 1} "
                    f"success={result.success}  steps={result.steps}  duration={result.duration_s:.2f}s"
                )

            task_sr = sum(r.success for r in task_results) / len(task_results) if task_results else 0.0
            logging.info(
                f"[LiberoSmartTest] Task {task_id}: "
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

    total_t = time.perf_counter() - t_all_start
    if all_results:
        overall_sr = sum(r.success for r in all_results) / len(all_results)
        avg_steps = sum(r.steps for r in all_results) / len(all_results)

        # ── Retry stats (only meaningful when SM is enabled) ──────────────
        smart_results = [r for r in all_results if isinstance(r, SmartEpisodeResult)]
        total_retries = sum(r.retries for r in smart_results)
        eps_with_retry = [r for r in smart_results if r.retries > 0]
        eps_no_retry   = [r for r in smart_results if r.retries == 0]
        sr_with_retry  = (
            sum(r.success for r in eps_with_retry) / len(eps_with_retry)
            if eps_with_retry else float("nan")
        )
        sr_no_retry    = (
            sum(r.success for r in eps_no_retry) / len(eps_no_retry)
            if eps_no_retry else float("nan")
        )
        success_after_retry = sum(r.success_after_retry for r in smart_results)

        retry_lines = ""
        if smart_results:
            nan_fmt = lambda v: f"{v:.1%}" if v == v else "n/a"  # noqa: E731
            rescue_rate = (
                success_after_retry / len(eps_with_retry) if eps_with_retry else float("nan")
            )
            sr_lift = success_after_retry / len(all_results) if all_results else 0.0
            retry_lines = (
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
            f"[LiberoSmartTest] ═══ Final summary ═══\n"
            f"  suite           : {cfg.env_task}\n"
            f"  total_episodes  : {len(all_results)}\n"
            f"  overall_sr      : {overall_sr:.1%}\n"
            f"  avg_steps       : {avg_steps:.1f}\n"
            f"  total_time      : {total_t:.2f}s"
            f"{retry_lines}"
        )
        if cfg.save_results:
            _save_results(all_results, cfg)
            _save_smart_summary(all_results, smart_results, cfg, total_t, cfg.timing_output_dir)
    else:
        logging.warning("[LiberoSmartTest] No episodes completed.")


if __name__ == "__main__":
    run_libero_smart_test()
