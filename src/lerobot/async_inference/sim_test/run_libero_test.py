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

"""Async-inference LIBERO evaluation — connects to a running policy_server.py.

Drives real LIBERO gym environments via SimRobotClient.  The policy server must
already be running before this script is launched.

Usage:
```shell
# 1. Start the policy server (in another terminal):
python -m lerobot.async_inference.policy_server \\
    --policy_type=smolvla \\
    --pretrained_name_or_path=lerobot/smolvla_base \\
    --host=localhost --port=8080

# 2. Run LIBERO evaluation:
python -m lerobot.async_inference.sim_test.run_libero_test \\
    --env_task=libero_10 \\
    --policy_type=smolvla \\
    --pretrained_name_or_path=lerobot/smolvla_base \\
    --server_address=localhost:8080 \\
    --actions_per_chunk=16 \\
    --episodes_per_task=10 \\
    --fps=30 \\
    --results_dir=./libero_results \\
    --timing_output_dir=./libero_timing
```

Results are saved to:
  <results_dir>/episodes.json     — per-episode record (task, success, steps, duration)
  <results_dir>/aggregate.json    — success rates and averages per task + overall
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
from lerobot.utils.utils import init_logging

from .configs import LiberoSimConfig
from .sim_client import EpisodeResult, SimRobotClient, _get_task_description


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_timing_tables(timing_dir) -> list:
    """Read *_summary.json files from timing_dir and return formatted table lines."""
    if timing_dir is None:
        return []
    import json as _json
    from pathlib import Path as _Path
    output_dir = _Path(timing_dir)
    lines = []

    def _fmt_table(json_path):
        if not json_path.exists():
            return []
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        n = data.get("n_records", "?")
        prefix = json_path.stem.replace("_summary", "")
        rows = [
            f"[TimingRecorder] {prefix}  ({n} records)",
            f"  {'field':<30} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}",
            "  " + "-" * 72,
        ]
        for key, stats in data.items():
            if key == "n_records" or not isinstance(stats, dict):
                continue
            if "mean" not in stats or stats["mean"] is None:
                continue
            nan_tag = f"  [{stats['n_nan']} NaN]" if stats.get("n_nan", 0) > 0 else ""
            rows.append(
                f"  {key:<30} {stats['mean']:>7.2f}  "
                f"{stats['p50']:>7.2f}  "
                f"{stats['p95']:>7.2f}  "
                f"{stats['p99']:>7.2f}  "
                f"{stats['max']:>7.2f}{nan_tag}"
            )
        return rows

    for prefix in ("client_obs_sent", "client_chunk_recv", "client_chunk_action", "client_aggregate"):
        tbl = _fmt_table(output_dir / f"{prefix}_summary.json")
        if tbl:
            lines.extend(tbl)
            lines.append("")
    return lines


def _write_timing_summary_txt(
    results: list,
    cfg: "LiberoSimConfig",
) -> None:
    """Write episode success stats + timing tables to <results_dir>/timing_summary.txt."""
    from datetime import datetime as _dt
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_sr = sum(r.success for r in results) / len(results) if results else 0.0
    avg_steps  = sum(r.steps   for r in results) / len(results) if results else 0.0
    avg_dur    = sum(r.duration_s for r in results) / len(results) if results else 0.0

    lines: list[str] = [
        "=" * 72,
        "  LIBERO Evaluation — Timing Summary",
        f"  Generated : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "── Config ──────────────────────────────────────────────────────────",
        f"  suite                : {cfg.env_task}",
        f"  model                : {cfg.pretrained_name_or_path}",
        f"  fps                  : {cfg.fps}",
        f"  episodes_per_task    : {cfg.episodes_per_task}",
        f"  actions_per_chunk    : {cfg.actions_per_chunk}",
        "",
        "── Episode Stats ────────────────────────────────────────────────────",
        f"  total_episodes       : {len(results)}",
        f"  overall_sr           : {overall_sr:.1%}",
        f"  avg_steps            : {avg_steps:.1f}",
        f"  avg_duration_s       : {avg_dur:.2f}",
    ]

    timing_lines = _read_timing_tables(cfg.timing_output_dir)
    if timing_lines:
        lines += ["", "── Timing Tables ────────────────────────────────────────────────────", ""]
        lines.extend(timing_lines)

    lines += ["", "=" * 72, ""]

    txt_path = out_dir / "timing_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"[LiberoTest] Timing summary saved → {txt_path}")


def _build_libero_env_and_preprocessor(cfg: LiberoSimConfig):
    """Create LIBERO envs, preprocessor, and lerobot feature spec from config.

    Returns:
        (envs_dict, env_preprocessor, lerobot_features)
        where envs_dict = {suite_name: {task_id: SyncVectorEnv}}
    """
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

    envs_dict = make_env(env_cfg, n_envs=1)

    # Build env preprocessor: LiberoProcessorStep (flip images + flatten robot_state)
    env_preprocessor, _ = env_cfg.get_env_processors()

    # Build lerobot feature spec for RemotePolicyConfig
    try:
        lerobot_features = env_to_policy_features(env_cfg)
    except Exception as exc:
        logging.warning(f"[LiberoTest] Could not build lerobot features: {exc}. Using empty dict.")
        lerobot_features = {}

    return envs_dict, env_preprocessor, lerobot_features


def _save_results(results: list[EpisodeResult], cfg: LiberoSimConfig) -> dict:
    """Save per-episode results and per-task / overall aggregate stats to JSON."""
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
            "obs_type": cfg.obs_type,
            "actions_per_chunk": cfg.actions_per_chunk,
            "fps": cfg.fps,
            "episodes_per_task": cfg.episodes_per_task,
            "aggregate_fn": cfg.aggregate_fn_name,
        },
    }

    (out_dir / "episodes.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    logging.info(f"[LiberoTest] Results saved to {out_dir}")
    return aggregate


# ── Main ──────────────────────────────────────────────────────────────────────

@draccus.wrap()
def run_libero_test(cfg: LiberoSimConfig):
    init_logging(console_level=cfg.log_level.upper())
    logging.info("[LiberoTest] Config:\n" + pformat(asdict(cfg)))

    # ── Build envs + preprocessor ────────────────────────────────────────────
    logging.info(f"[LiberoTest] Building LIBERO envs for suite '{cfg.env_task}' ...")
    envs_dict, env_preprocessor, lerobot_features = _build_libero_env_and_preprocessor(cfg)

    # Flatten into a list of (suite_name, task_id, vec_env) triples
    task_list = [
        (suite, tid, env)
        for suite, task_envs in envs_dict.items()
        for tid, env in sorted(task_envs.items())
    ]
    logging.info(f"[LiberoTest] Built {len(task_list)} task env(s)")

    if not task_list:
        logging.error("[LiberoTest] No task environments created. Aborting.")
        return

    # ── Create client with the first env (will be swapped per task) ──────────
    first_suite, first_tid, first_env = task_list[0]
    first_task_desc = _get_task_description(first_env)

    client = SimRobotClient(
        config=cfg,
        env=first_env,
        env_preprocessor=env_preprocessor,
        lerobot_features=lerobot_features,
        task_description=first_task_desc,
    )

    all_results: list[EpisodeResult] = []

    if not client.start():
        logging.error("[LiberoTest] Could not connect to policy server. Aborting.")
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
            f"Queue size monitor started — saving PNG every "
            f"{cfg.queue_size_monitor_interval}s to {cfg.queue_size_monitor_path}"
        )

    # ── Start persistent receiver thread ─────────────────────────────────────
    receiver = threading.Thread(target=client.receive_actions, daemon=True, name="action-receiver")
    receiver.start()

    # ── Task × episode loop ───────────────────────────────────────────────────
    t_all_start = time.perf_counter()
    global_ep = 0   # monotonically increasing episode counter across tasks

    try:
        for task_idx, (suite_name, task_id, task_env) in enumerate(task_list):
            task_desc = _get_task_description(task_env)
            logging.info(
                f"[LiberoTest] ══ Task {task_idx + 1}/{len(task_list)} "
                f"| suite={suite_name} | task_id={task_id} | desc='{task_desc}' ══"
            )

            # Swap the environment for this task
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
                    f"[LiberoTest] task={task_id} ep={ep_local}/{cfg.episodes_per_task - 1} "
                    f"success={result.success}  steps={result.steps}  duration={result.duration_s:.2f}s"
                )

            task_sr = sum(r.success for r in task_results) / len(task_results) if task_results else 0.0
            logging.info(
                f"[LiberoTest] Task {task_id} summary: "
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
        logging.info(
            f"[LiberoTest] ═══ Final summary ═══\n"
            f"  suite           : {cfg.env_task}\n"
            f"  total_episodes  : {len(all_results)}\n"
            f"  overall_sr      : {overall_sr:.1%}\n"
            f"  avg_steps       : {avg_steps:.1f}\n"
            f"  total_time      : {total_t:.2f}s"
        )
        if cfg.save_results:
            _save_results(all_results, cfg)
            _write_timing_summary_txt(all_results, cfg)
    else:
        logging.warning("[LiberoTest] No episodes completed.")


if __name__ == "__main__":
    run_libero_test()
