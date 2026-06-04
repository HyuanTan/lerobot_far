#!/usr/bin/env python3
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

"""Cross-method timing and performance comparison for SO-101 real-robot eval runs.

Reads the merged eval_thesis/so101 directory tree (produced after running
copy_so101_client_to_thesis.sh), loads per-method timing summaries and JSONL
records, and generates:

  Console output:
    - Per-method latency summary table
    - RTC health summary table
    - SM (gripper state machine) stats table (SM methods only)

  Figures:
    fig1_latency_bars.png       — round-trip / server-infer / deser p50+p95 grouped bars
    fig2_infer_pipeline.png     — server pipeline stages: infer_ms / queue_wait / total
    fig3_rtc_overlap.png        — RTC overlap ratio, diff_l2_mean, leftover_steps, infer_delay
    fig4_latency_cdf.png        — CDF of round_trip_ms per method
    fig5_latency_violin.png     — violin plots for round_trip / server_infer across methods
    fig6_sm_stats.png           — SM-only: sr, retries, rescue rate, gripper event breakdown
    fig7_queue_starvation.png   — action-queue starvation rate per method

Directory structure expected::

    <eval_root>/
      <method>/<policy>/<param>/
        client_timing/
          client_chunk_recv_summary.json
          client_chunk_recv_records.jsonl
          client_chunk_action_summary.json
          client_aggregate_summary.json
          client_obs_sent_records.jsonl
          sm_summary.txt                   (SM methods only)
          gripper_sm_events_summary.json   (SM methods only)
        server_timing/
          server_infer_summary.json
          server_infer_records.jsonl
          server_recv_summary.json

Usage::

    # Default: outputs/eval_thesis/so101
    python -m lerobot.async_inference.analyze_so101_comparison

    # Custom root
    python -m lerobot.async_inference.analyze_so101_comparison \\
        --eval_root outputs/eval_thesis/so101 --policy pi05 --param H15 \\
        --out_dir outputs/eval_thesis/so101/comparison
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 130)


# ── Colour palette (consistent with analyze_sweep.py) ─────────────────────────

_METHOD_COLORS: dict[str, str] = {
    "async_rtc":              "#EF4444",  # red
    "async_rtc_inter":        "#F97316",  # orange
    "async_rtc_no_imgcrop":   "#FBBF24",  # amber
    "async_rtc_sm":           "#DC2626",  # dark-red
    "async_rtc_sm_inter":     "#991B1B",  # deeper-red
}
_DEFAULT_COLOR = "#64748B"  # slate


def _method_color(method: str) -> str:
    return _METHOD_COLORS.get(method, _DEFAULT_COLOR)


def _is_sm(method: str) -> bool:
    return "_sm" in method


def _is_multicand(method: str) -> bool:
    return "_sm_multicand" in method or "_multicand" in method


def _method_hatch(method: str) -> str:
    """Bar hatch: multicand=dense crosses, SM=dots, base=none."""
    if _is_multicand(method):
        return "xx"
    if _is_sm(method):
        return ".."
    return ""


def _method_alpha(method: str) -> float:
    """Bar fill alpha: base=0.75 → SM=0.88 → multicand=1.0."""
    if _is_multicand(method):
        return 1.0
    if _is_sm(method):
        return 0.88
    return 0.75


def _method_linestyle(method: str) -> str:
    """Line/CDF style: SM/multicand → solid, base → dashed."""
    return "-" if _is_sm(method) else "--"


def _method_linewidth(method: str) -> float:
    """Line width hierarchy: base=1.4, SM=1.8, multicand=2.2."""
    if _is_multicand(method):
        return 2.2
    if _is_sm(method):
        return 1.8
    return 1.4


def _method_edgecolor(method: str) -> str:
    """Bar edge: SM methods get a dark border for extra pop."""
    return "#1a1a1a" if _is_sm(method) else "white"


def _method_short_label(method: str) -> str:
    """Compact x-tick label: append '(SM)' badge for SM methods."""
    if _is_multicand(method):
        base = method.replace("_sm_multicand", "")
        return f"{base}\n(SM+MC)"
    if _is_sm(method):
        base = method.replace("_sm", "")
        return f"{base}\n(SM)"
    return method


def _add_sm_separator(ax, methods: list[dict], x: np.ndarray):
    """Draw a vertical dashed line between the last base and first SM method."""
    transitions = [
        i for i in range(1, len(methods))
        if _is_sm(methods[i]["method"]) and not _is_sm(methods[i - 1]["method"])
    ]
    for t in transitions:
        ax.axvline(x[t] - 0.5, color="#555", linewidth=1.0, linestyle="--", alpha=0.45)


def _sm_legend_patches() -> list[mpatches.Patch]:
    """Return two legend proxies explaining the hatch/alpha convention."""
    return [
        mpatches.Patch(facecolor="gray", alpha=0.75, hatch="",   label="Base method"),
        mpatches.Patch(facecolor="gray", alpha=0.88, hatch="..", label="SM variant"),
        mpatches.Patch(facecolor="gray", alpha=1.00, hatch="xx", label="SM + multicand"),
    ]


# ── Data discovery ─────────────────────────────────────────────────────────────

def discover_methods(
    eval_root: Path,
    filter_policy: str | None = None,
    filter_param: str | None = None,
) -> list[dict[str, Any]]:
    """Walk eval_root and return one record per (method, policy, param) leaf."""
    entries = []
    for method_dir in sorted(eval_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for policy_dir in sorted(method_dir.iterdir()):
            if not policy_dir.is_dir():
                continue
            policy = policy_dir.name
            if filter_policy and policy != filter_policy:
                continue
            for param_dir in sorted(policy_dir.iterdir()):
                if not param_dir.is_dir():
                    continue
                param = param_dir.name
                if filter_param and param != filter_param:
                    continue
                entries.append({
                    "method": method,
                    "policy": policy,
                    "param":  param,
                    "path":   param_dir,
                    "client_timing": param_dir / "client_timing",
                    "server_timing": param_dir / "server_timing",
                })
    return entries


# ── Summary JSON / text loaders ───────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _load_jsonl(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_json(path, lines=True, convert_dates=False)
            return df if not df.empty else None
        except Exception:
            return None
    return None


def _parse_sm_summary(sm_txt: Path) -> dict[str, float]:
    """Extract key numbers from sm_summary.txt via regex."""
    stats: dict[str, float] = {}
    if not sm_txt.exists():
        return stats
    text = sm_txt.read_text()
    patterns = {
        "total_episodes":  r"total_episodes\s*:\s*(\d+)",
        "overall_sr":      r"overall_sr\s*:\s*([\d.]+)%",
        "total_retries":   r"total_retries\s*:\s*(\d+)",
        "eps_with_retry":  r"eps_with_retry\s*:\s*(\d+)",
        "eps_no_retry":    r"eps_no_retry\s*:\s*(\d+)",
        "rescue_rate":     r"rescue_rate\s*:\s*([\d.]+)%",
        "sr_lift":         r"sr_lift \(SM→no-SM\)\s*:\s*\+([\d.]+)%",
        "grasp_success":   r"grasp_success\s*:\s*(\d+)",
        "empty_grasp":     r"empty_grasp\s*:\s*(\d+)",
        "slip":            r"slip\s*:\s*(\d+)",
        "recovery":        r"recovery\s*:\s*(\d+)",
        "lift_retry":      r"lift_retry\s*:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            stats[key] = float(m.group(1))
    return stats


def _get_field(summary: dict | None, field: str, stat: str = "mean") -> float:
    if summary is None:
        return float("nan")
    f = summary.get(field, {})
    if not isinstance(f, dict):
        return float("nan")
    return float(f.get(stat, float("nan")))


# ── Per-method data loading ───────────────────────────────────────────────────

def load_method_data(entry: dict[str, Any]) -> dict[str, Any]:
    """Load all summary + JSONL data for one (method, policy, param) leaf."""
    ct = entry["client_timing"]
    st = entry["server_timing"]

    chunk_recv_sum   = _load_json(ct / "client_chunk_recv_summary.json")
    chunk_action_sum = _load_json(ct / "client_chunk_action_summary.json")
    aggregate_sum    = _load_json(ct / "client_aggregate_summary.json")
    server_infer_sum = _load_json(st / "server_infer_summary.json")
    sm_stats         = _parse_sm_summary(ct / "sm_summary.txt")

    # Event type counts from raw JSONL (more reliable than text parsing)
    sm_events_raw = _load_jsonl(ct / "gripper_sm_events_records.jsonl")
    if sm_events_raw is not None and "event_type" in sm_events_raw.columns:
        counts = sm_events_raw["event_type"].value_counts().to_dict()
        for event_field in ("grasp_success", "empty_grasp", "slip",
                            "recovery", "lift_retry", "stop"):
            if event_field in counts:
                sm_stats[event_field] = float(counts[event_field])

    # Raw JSONL for CDF / violin
    chunk_recv_raw   = _load_jsonl(ct / "client_chunk_recv_records.jsonl")
    server_infer_raw = _load_jsonl(st / "server_infer_records.jsonl")
    sent_raw         = _load_jsonl(ct / "client_obs_sent_records.jsonl")

    return {
        **entry,
        "chunk_recv_sum":   chunk_recv_sum,
        "chunk_action_sum": chunk_action_sum,
        "aggregate_sum":    aggregate_sum,
        "server_infer_sum": server_infer_sum,
        "sm_stats":         sm_stats,
        "sm_events_raw":    sm_events_raw,
        "chunk_recv_raw":   chunk_recv_raw,
        "server_infer_raw": server_infer_raw,
        "sent_raw":         sent_raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Console tables
# ══════════════════════════════════════════════════════════════════════════════

def _divider(title: str = "", width: int = 88):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


def print_latency_table(methods: list[dict]) -> None:
    """Side-by-side latency summary for all methods."""
    _divider("LATENCY SUMMARY  (ms; from summary JSONs)")

    metrics = [
        # (label, source, field, stat)
        ("round_trip  p50",  "chunk_recv_sum",   "round_trip_ms",               "p50"),
        ("round_trip  p95",  "chunk_recv_sum",   "round_trip_ms",               "p95"),
        ("srv_infer   p50",  "chunk_recv_sum",   "server_infer_ms",             "p50"),
        ("srv_infer   p95",  "chunk_recv_sum",   "server_infer_ms",             "p95"),
        ("deser       p50",  "chunk_recv_sum",   "deser_ms",                    "p50"),
        ("exec_lag    p50",  "chunk_recv_sum",   "estimated_first_exec_lag_ms", "p50"),
        ("queue_wait  p50",  "server_infer_sum", "queue_wait_ms",               "p50"),
        ("queue_wait  p95",  "server_infer_sum", "queue_wait_ms",               "p95"),
        ("infer_ms    p50",  "server_infer_sum", "infer_ms",                    "p50"),
        ("infer_ms    p95",  "server_infer_sum", "infer_ms",                    "p95"),
        ("pipeline    p50",  "server_infer_sum", "total_pipeline_ms",           "p50"),
        ("n_records",        "chunk_recv_sum",   "round_trip_ms",               "n"),
    ]

    header = f"  {'metric':<25}" + "".join(f"  {m['method']:<25}" for m in methods)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for label, src, field, stat in metrics:
        row = f"  {label:<25}"
        for m in methods:
            val = _get_field(m[src], field, stat)
            if stat == "n":
                cell = f"{int(val)}" if not np.isnan(val) else "—"
            else:
                cell = f"{val:.1f}" if not np.isnan(val) else "—"
            row += f"  {cell:<25}"
        print(row)


def print_rtc_table(methods: list[dict]) -> None:
    """RTC-specific metrics: overlap, diff_l2, leftover_steps, infer_delay."""
    _divider("RTC HEALTH SUMMARY")

    metrics = [
        ("n_overlap    p50",  "aggregate_sum",    "n_overlap",        "p50"),
        ("n_overlap    mean", "aggregate_sum",    "n_overlap",        "mean"),
        ("n_new        mean", "aggregate_sum",    "n_new",            "mean"),
        ("diff_l2_mean p50",  "aggregate_sum",    "diff_l2_mean",     "p50"),
        ("diff_l2_mean p95",  "aggregate_sum",    "diff_l2_mean",     "p95"),
        ("leftover_stp p50",  "chunk_action_sum", "leftover_steps",   "p50"),
        ("leftover_stp mean", "chunk_action_sum", "leftover_steps",   "mean"),
        ("infer_delay  p50",  "chunk_action_sum", "infer_delay_used", "p50"),
        ("infer_delay  mean", "chunk_action_sum", "infer_delay_used", "mean"),
    ]

    header = f"  {'metric':<25}" + "".join(f"  {m['method']:<25}" for m in methods)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for label, src, field, stat in metrics:
        row = f"  {label:<25}"
        for m in methods:
            val = _get_field(m[src], field, stat)
            cell = f"{val:.2f}" if not np.isnan(val) else "—"
            row += f"  {cell:<25}"
        print(row)


def print_sm_table(methods: list[dict]) -> None:
    """SM stats: success rate, retries, rescue rate, gripper events."""
    sm_methods = [m for m in methods if m["sm_stats"]]
    if not sm_methods:
        return

    _divider("GRIPPER STATE MACHINE STATS  (SM methods only)")

    fields = [
        ("total_episodes", "{:.0f}"),
        ("overall_sr",     "{:.1f}%"),
        ("total_retries",  "{:.0f}"),
        ("eps_with_retry", "{:.0f}"),
        ("rescue_rate",    "{:.1f}%"),
        ("sr_lift",        "+{:.1f}%"),
        ("grasp_success",  "{:.0f}"),
        ("empty_grasp",    "{:.0f}"),
        ("slip",           "{:.0f}"),
        ("recovery",       "{:.0f}"),
    ]

    header = f"  {'stat':<22}" + "".join(f"  {m['method']:<25}" for m in sm_methods)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for field, fmt in fields:
        row = f"  {field:<22}"
        for m in sm_methods:
            val = m["sm_stats"].get(field, float("nan"))
            if np.isnan(val):
                cell = "—"
            elif field == "sr_lift":
                cell = f"+{val:.1f}%"
            elif "%" in fmt:
                cell = f"{val:.1f}%"
            else:
                cell = f"{val:.0f}"
            row += f"  {cell:<25}"
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, path: Path, label: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}  [{label}]")


def _method_labels(methods: list[dict]) -> list[str]:
    return [m["method"] for m in methods]


def _bar_group(ax, group_data: list[list[float]], method_labels: list[str],
               group_labels: list[str], colors: list[str],
               ylabel: str, title: str,
               hatches: list[str] | None = None,
               alphas: list[float] | None = None,
               edgecolors: list[str] | None = None):
    """Draw a grouped-bar chart.

    group_data[i][j] = value for group_labels[i] (x-group), method j (color).
    """
    n_groups  = len(group_labels)
    n_methods = len(method_labels)
    x = np.arange(n_groups)
    width = 0.8 / n_methods

    for j, (method, color) in enumerate(zip(method_labels, colors)):
        vals = [group_data[i][j] for i in range(n_groups)]
        offset = (j - n_methods / 2 + 0.5) * width
        hatch = hatches[j]    if hatches    else ""
        alpha = alphas[j]     if alphas     else 0.85
        ec    = edgecolors[j] if edgecolors else "white"
        bars = ax.bar(x + offset, vals, width * 0.92,
                      label=method, color=color,
                      hatch=hatch, alpha=alpha, edgecolor=ec)
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ax.get_ylim()[1] * 0.008,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_latency_bars(methods: list[dict], out_dir: Path) -> None:
    """Fig 1: Grouped bar chart — round_trip, srv_infer, deser for p50 and p95."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SO-101 Cross-Method Latency Comparison", fontsize=12, fontweight="bold")

    colors     = [_method_color(m["method"])    for m in methods]
    hatches    = [_method_hatch(m["method"])    for m in methods]
    alphas     = [_method_alpha(m["method"])    for m in methods]
    edgecolors = [_method_edgecolor(m["method"]) for m in methods]
    mlabels    = _method_labels(methods)
    kw = dict(hatches=hatches, alphas=alphas, edgecolors=edgecolors)

    # Panel 1: round_trip p50 + p95
    groups = ["p50", "p95"]
    data_rt = [
        [_get_field(m["chunk_recv_sum"], "round_trip_ms", g) for m in methods]
        for g in groups
    ]
    _bar_group(axes[0], data_rt, mlabels, groups, colors,
               "Round-trip (ms)", "Round-trip latency", **kw)

    # Panel 2: server_infer p50 + p95
    data_si = [
        [_get_field(m["chunk_recv_sum"], "server_infer_ms", g) for m in methods]
        for g in groups
    ]
    _bar_group(axes[1], data_si, mlabels, groups, colors,
               "Server infer (ms)", "Server inference (client-observed)", **kw)

    # Panel 3: deser p50 + exec_lag p50
    groups3 = ["deser p50", "exec_lag p50"]
    data3 = [
        [_get_field(m["chunk_recv_sum"], "deser_ms", "p50") for m in methods],
        [_get_field(m["chunk_recv_sum"], "estimated_first_exec_lag_ms", "p50") for m in methods],
    ]
    _bar_group(axes[2], data3, mlabels, groups3, colors,
               "ms", "Deser latency  &  First-exec lag", **kw)

    conv_patches = _sm_legend_patches()
    fig.legend(handles=conv_patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, title="Method type", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_dir / "fig1_latency_bars.png", "latency_bars")


def plot_infer_pipeline(methods: list[dict], out_dir: Path) -> None:
    """Fig 2: Server pipeline breakdown — infer_ms / queue_wait / total by p50 and p95."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SO-101 Server Pipeline Comparison (server_timing)", fontsize=12, fontweight="bold")

    colors     = [_method_color(m["method"])    for m in methods]
    hatches    = [_method_hatch(m["method"])    for m in methods]
    alphas     = [_method_alpha(m["method"])    for m in methods]
    edgecolors = [_method_edgecolor(m["method"]) for m in methods]
    mlabels    = _method_labels(methods)
    groups     = ["p50", "p95"]
    kw = dict(hatches=hatches, alphas=alphas, edgecolors=edgecolors)

    for ax, (field, title) in zip(axes, [
        ("infer_ms",          "Model inference (infer_ms)"),
        ("queue_wait_ms",     "Server queue wait"),
        ("total_pipeline_ms", "Total pipeline"),
    ]):
        data = [
            [_get_field(m["server_infer_sum"], field, g) for m in methods]
            for g in groups
        ]
        _bar_group(ax, data, mlabels, groups, colors, "ms", title, **kw)

    conv_patches = _sm_legend_patches()
    fig.legend(handles=conv_patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, title="Method type", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_dir / "fig2_infer_pipeline.png", "infer_pipeline")


def plot_rtc_health(methods: list[dict], out_dir: Path) -> None:
    """Fig 3: RTC health — overlap ratio, diff_l2, leftover_steps, infer_delay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("RTC Health: Overlap / Continuity / Delay Calibration",
                 fontsize=12, fontweight="bold")

    colors     = [_method_color(m["method"])    for m in methods]
    hatches    = [_method_hatch(m["method"])    for m in methods]
    alphas     = [_method_alpha(m["method"])    for m in methods]
    edgecolors = [_method_edgecolor(m["method"]) for m in methods]
    mlabels    = [m["method"] for m in methods]
    x = np.arange(len(mlabels))

    # (0,0) n_new + n_overlap stacked per method
    ax = axes[0, 0]
    n_new = [_get_field(m["aggregate_sum"], "n_new",    "mean") for m in methods]
    n_ol  = [_get_field(m["aggregate_sum"], "n_overlap","mean") for m in methods]
    for i, (nv, nov, hatch, ec) in enumerate(zip(n_new, n_ol, hatches, edgecolors)):
        ax.bar(i, nv,  color="#4C9BE8", alpha=0.85, edgecolor=ec, hatch=hatch,
               label="n_new (truly new)"    if i == 0 else "_")
        ax.bar(i, nov, bottom=nv, color="#F59E0B", alpha=0.85, edgecolor=ec, hatch=hatch,
               label="n_overlap (RTC prefix)" if i == 0 else "_")
    ax.set_xticks(x); ax.set_xticklabels(mlabels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Actions per chunk")
    ax.set_title("RTC chunk composition (mean)", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(facecolor="#4C9BE8", label="n_new (truly new)"),
        mpatches.Patch(facecolor="#F59E0B", label="n_overlap (RTC prefix)"),
    ], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _add_sm_separator(ax, methods, x)

    # (0,1) diff_l2_mean p50 + p95
    ax = axes[0, 1]
    d50 = [_get_field(m["aggregate_sum"], "diff_l2_mean", "p50") for m in methods]
    d95 = [_get_field(m["aggregate_sum"], "diff_l2_mean", "p95") for m in methods]
    width = 0.35
    for i, (v50, v95, color, hatch, alpha, ec) in enumerate(
        zip(d50, d95, colors, hatches, alphas, edgecolors)
    ):
        ax.bar(i - width / 2, v50, width, color=color, hatch=hatch, alpha=alpha,    edgecolor=ec)
        ax.bar(i + width / 2, v95, width, color=color, hatch=hatch, alpha=alpha * 0.55, edgecolor=ec)
    ax.set_xticks(x); ax.set_xticklabels(mlabels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("L2 norm")
    ax.set_title("Chunk discontinuity (diff_l2_mean)", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(facecolor="gray", alpha=0.85, label="p50"),
        mpatches.Patch(facecolor="gray", alpha=0.45, label="p95"),
    ], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _add_sm_separator(ax, methods, x)

    # (1,0) leftover_steps p50
    ax = axes[1, 0]
    lft = [_get_field(m["chunk_action_sum"], "leftover_steps", "p50") for m in methods]
    for i, (v, color, hatch, alpha, ec) in enumerate(
        zip(lft, colors, hatches, alphas, edgecolors)
    ):
        ax.bar(i, v, color=color, hatch=hatch, alpha=alpha, edgecolor=ec)
        if not np.isnan(v):
            ax.text(i, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mlabels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Steps")
    ax.set_title("leftover_steps p50  (RTC buffer size)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _add_sm_separator(ax, methods, x)

    # (1,1) infer_delay p50 + p95
    ax = axes[1, 1]
    id50 = [_get_field(m["chunk_action_sum"], "infer_delay_used", "p50") for m in methods]
    id95 = [_get_field(m["chunk_action_sum"], "infer_delay_used", "p95") for m in methods]
    width = 0.35
    for i, (v50, v95, color, hatch, alpha, ec) in enumerate(
        zip(id50, id95, colors, hatches, alphas, edgecolors)
    ):
        ax.bar(i - width / 2, v50, width, color=color, hatch=hatch, alpha=alpha,    edgecolor=ec)
        ax.bar(i + width / 2, v95, width, color=color, hatch=hatch, alpha=alpha * 0.55, edgecolor=ec)
    ax.set_xticks(x); ax.set_xticklabels(mlabels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Steps")
    ax.set_title("infer_delay_used  (RTC prediction horizon)", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(facecolor="gray", alpha=0.85, label="p50"),
        mpatches.Patch(facecolor="gray", alpha=0.45, label="p95"),
    ], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _add_sm_separator(ax, methods, x)

    conv_patches = _sm_legend_patches()
    fig.legend(handles=conv_patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, title="Method type", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_dir / "fig3_rtc_health.png", "rtc_health")


def plot_latency_cdf(methods: list[dict], out_dir: Path) -> None:
    """Fig 4: CDF of round_trip_ms per method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Latency CDF Comparison", fontsize=12, fontweight="bold")

    for field, ax, title in [
        ("round_trip_ms",  axes[0], "Round-trip latency CDF"),
        ("server_infer_ms", axes[1], "Server inference latency CDF"),
    ]:
        for m in methods:
            df = m["chunk_recv_raw"]
            if df is None or field not in df.columns:
                continue
            vals = df[field].dropna().values
            if len(vals) == 0:
                continue
            vals_s = np.sort(vals)
            cdf = np.arange(1, len(vals_s) + 1) / len(vals_s)
            method = m["method"]
            ax.plot(vals_s, cdf,
                    color=_method_color(method),
                    linestyle=_method_linestyle(method),
                    linewidth=_method_linewidth(method),
                    label=method)
            # p95 marker — dotted to avoid clashing with base method dashes
            p95 = np.percentile(vals_s, 95)
            ax.axvline(p95, color=_method_color(method), linewidth=0.8,
                       linestyle=":", alpha=0.5)

        ax.set_xlabel("Latency (ms)", fontsize=9)
        ax.set_ylabel("CDF", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.axhline(0.95, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)

    fig.tight_layout()
    _savefig(fig, out_dir / "fig4_latency_cdf.png", "latency_cdf")


def plot_latency_violin(methods: list[dict], out_dir: Path) -> None:
    """Fig 5: Violin plots for round_trip / server_infer across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Latency Distribution — Violin Plots", fontsize=12, fontweight="bold")

    for field, ax, title in [
        ("round_trip_ms",   axes[0], "Round-trip (ms)"),
        ("server_infer_ms", axes[1], "Server inference (ms)"),
    ]:
        data_list, labels, colors, edgecolors_v, alphas_v = [], [], [], [], []
        for m in methods:
            df = m["chunk_recv_raw"]
            if df is None or field not in df.columns:
                continue
            vals = df[field].dropna().values
            if len(vals) < 5:
                continue
            data_list.append(vals)
            labels.append(m["method"])
            colors.append(_method_color(m["method"]))
            edgecolors_v.append(_method_edgecolor(m["method"]))
            alphas_v.append(_method_alpha(m["method"]))

        if not data_list:
            continue

        parts = ax.violinplot(data_list, positions=range(len(data_list)),
                              showmedians=True, showextrema=False)
        for body, color, ec, alpha in zip(
            parts["bodies"], colors, edgecolors_v, alphas_v
        ):
            body.set_facecolor(color)
            body.set_alpha(alpha)
            body.set_edgecolor(ec)
            body.set_linewidth(1.4 if ec != "white" else 0.5)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        for i, vals in enumerate(data_list):
            p95 = np.percentile(vals, 95)
            ax.scatter([i], [p95], color=colors[i], zorder=5, s=40,
                       marker="^", edgecolors="white", linewidths=0.8)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("ms", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    conv_patches = _sm_legend_patches()
    fig.legend(handles=conv_patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, title="Method type", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_dir / "fig5_latency_violin.png", "latency_violin")


def plot_sm_stats(methods: list[dict], out_dir: Path) -> None:
    """Fig 6: SM-only stats — success rate, retries, rescue rate + event breakdown."""
    sm_methods = [m for m in methods if m["sm_stats"]]
    if not sm_methods:
        print("  (skipping fig6: no SM methods found)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Gripper State Machine Performance (SM methods)", fontsize=12, fontweight="bold")

    labels = [m["method"] for m in sm_methods]
    colors = [_method_color(m["method"]) for m in sm_methods]
    x = np.arange(len(sm_methods))

    # (0) Overall SR + rescue_rate + sr_lift
    ax = axes[0]
    sr_vals    = [m["sm_stats"].get("overall_sr", float("nan")) for m in sm_methods]
    rescue_vals = [m["sm_stats"].get("rescue_rate", float("nan")) for m in sm_methods]
    sl_vals    = [m["sm_stats"].get("sr_lift", float("nan")) for m in sm_methods]
    width = 0.25
    ax.bar(x - width, sr_vals,     width, label="overall_sr (%)",  color="#22C55E", alpha=0.85, edgecolor="white")
    ax.bar(x,         rescue_vals, width, label="rescue_rate (%)", color="#4C9BE8", alpha=0.85, edgecolor="white")
    ax.bar(x + width, sl_vals,     width, label="sr_lift (%)",     color="#F59E0B", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("%"); ax.set_ylim(0, 120)
    ax.set_title("Success & rescue metrics", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)

    # (1) Episode counts: total, with_retry, no_retry
    ax = axes[1]
    total = [m["sm_stats"].get("total_episodes", 0) for m in sm_methods]
    wretr = [m["sm_stats"].get("eps_with_retry", 0) for m in sm_methods]
    nretr = [m["sm_stats"].get("eps_no_retry",   0) for m in sm_methods]
    ax.bar(x, nretr, label="clean (no retry)",  color="#22C55E", alpha=0.85, edgecolor="white")
    ax.bar(x, wretr, bottom=nretr, label="needed retry", color="#F59E0B", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Episodes"); ax.set_title("Episode breakdown", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (2) Gripper event counts per method (stacked bar)
    ax = axes[2]
    event_fields = [
        ("grasp_success", "#22C55E"),
        ("empty_grasp",   "#EF4444"),
        ("slip",          "#F59E0B"),
        ("recovery",      "#7C3AED"),
    ]
    bottoms = np.zeros(len(sm_methods))
    for field, color in event_fields:
        vals = np.array([m["sm_stats"].get(field, 0) for m in sm_methods])
        ax.bar(x, vals, bottom=bottoms, label=field, color=color, alpha=0.85, edgecolor="white")
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Event count"); ax.set_title("Gripper SM event breakdown", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / "fig6_sm_stats.png", "sm_stats")


def plot_queue_starvation(methods: list[dict], out_dir: Path, fps: float = 10.0) -> None:
    """Fig 7: Action-queue starvation rate per method (from chunk records).

    A starvation event = next chunk arrives AFTER the current chunk is exhausted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Action-Queue Starvation Analysis  (fps={fps:.0f})",
                 fontsize=12, fontweight="bold")

    labels, starve_rates, starve_totals = [], [], []
    must_go_rates = []

    for m in methods:
        df = m["chunk_recv_raw"]
        if df is None or "chunk_size" not in df.columns or "wall_time" not in df.columns:
            labels.append(m["method"])
            starve_rates.append(float("nan"))
            starve_totals.append(float("nan"))
            must_go_rates.append(float("nan"))
            continue

        dt = 1.0 / fps
        cs = df.sort_values("wall_time").copy()
        cs["exhaust_time"] = cs["wall_time"] + cs["chunk_size"] * dt
        cs["next_arrival"] = cs["wall_time"].shift(-1)
        if "episode" in cs.columns:
            same_ep = cs["episode"] == cs["episode"].shift(-1)
            cs.loc[~same_ep, "next_arrival"] = np.nan
        cs["gap_s"] = (cs["next_arrival"] - cs["exhaust_time"]).clip(lower=0)
        n_intervals = int((~cs["next_arrival"].isna()).sum())
        n_starved   = int((cs["gap_s"] > 0).sum())

        labels.append(m["method"])
        starve_rates.append(100.0 * n_starved / n_intervals if n_intervals > 0 else float("nan"))
        starve_totals.append(float(cs["gap_s"].sum()))

        # must_go rate from sent records
        sent = m["sent_raw"]
        if sent is not None and "must_go" in sent.columns:
            must_go_rates.append(100.0 * float(sent["must_go"].mean()))
        else:
            must_go_rates.append(float("nan"))

    x = np.arange(len(labels))
    colors     = [_method_color(m["method"])    for m in methods]
    hatches    = [_method_hatch(m["method"])    for m in methods]
    alphas     = [_method_alpha(m["method"])    for m in methods]
    edgecolors = [_method_edgecolor(m["method"]) for m in methods]

    ax = axes[0]
    for i, (v, color, hatch, alpha, ec) in enumerate(
        zip(starve_rates, colors, hatches, alphas, edgecolors)
    ):
        ax.bar(i, v, color=color, hatch=hatch, alpha=alpha, edgecolor=ec)
        if not np.isnan(v):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(5, color="red", linewidth=1.2, linestyle="--", alpha=0.6, label="5% threshold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Starvation rate (%)")
    ax.set_title("Action-queue starvation rate", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _add_sm_separator(ax, methods, x)

    ax = axes[1]
    for i, (v, color, hatch, alpha, ec) in enumerate(
        zip(must_go_rates, colors, hatches, alphas, edgecolors)
    ):
        ax.bar(i, v, color=color, hatch=hatch, alpha=alpha, edgecolor=ec)
        if not np.isnan(v):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(30, color="orange", linewidth=1.2, linestyle="--", alpha=0.6, label="30% warning")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("must_go rate (%)")
    ax.set_title("Client must_go rate\n(queue-empty at obs send)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _add_sm_separator(ax, methods, x)

    conv_patches = _sm_legend_patches()
    fig.legend(handles=conv_patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, title="Method type", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_dir / "fig7_queue_starvation.png", "queue_starvation")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Cross-method timing comparison for SO-101 real-robot eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "eval_root", nargs="?",
        default="outputs/eval_thesis/so101",
        help="Root of the merged eval tree (default: outputs/eval_thesis/so101)",
    )
    ap.add_argument("--policy", default=None,
                    help="Filter to a single policy name (e.g. pi05)")
    ap.add_argument("--param", default=None,
                    help="Filter to a single param tag (e.g. H15)")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for figures (default: <eval_root>/comparison)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Control-loop fps for starvation analysis (default: 10)")
    ap.add_argument("--no_figs", action="store_true",
                    help="Skip figure generation (console tables only)")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.is_dir():
        ap.error(f"eval_root not found: {eval_root}")

    out_dir = Path(args.out_dir) if args.out_dir else eval_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 88)
    print("  analyze_so101_comparison — Cross-Method Timing & Performance Analysis")
    print("═" * 88)
    print(f"  eval_root : {eval_root}")
    print(f"  out_dir   : {out_dir}")
    print(f"  fps       : {args.fps}")

    entries = discover_methods(eval_root, args.policy, args.param)
    if not entries:
        print("\nERROR: No (method/policy/param) leaves found under eval_root.")
        sys.exit(1)

    print(f"\n  Found {len(entries)} run(s):")
    for e in entries:
        print(f"    {e['method']:30s} / {e['policy']} / {e['param']}")

    print("\nLoading data...")
    methods = []
    for entry in entries:
        print(f"\n  [{entry['method']}]")
        m = load_method_data(entry)
        methods.append(m)

    # ── Console tables ─────────────────────────────────────────────────────────
    print_latency_table(methods)
    print_rtc_table(methods)
    print_sm_table(methods)

    if args.no_figs:
        print("\n  (--no_figs: skipping figure generation)")
        return

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_latency_bars(methods, out_dir)
    plot_infer_pipeline(methods, out_dir)
    plot_rtc_health(methods, out_dir)
    plot_latency_cdf(methods, out_dir)
    plot_latency_violin(methods, out_dir)
    plot_sm_stats(methods, out_dir)
    plot_queue_starvation(methods, out_dir, fps=args.fps)

    print(f"\n  All figures saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
