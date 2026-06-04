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

"""Sweep-level analysis: solve rate vs. delay and solve rate vs. horizon.

Reads aggregate.json files from a sweep output root and generates paper-quality
comparison plots across methods, models, and suites.

Expected directory layout (produced by eval-scripts/*eval*.sh)::

    <eval_root>/
      <suite>/<method>/<model>/latency_d<L>/K<K>/results/aggregate.json

      suite  : libero_object | libero_spatial | libero_goal | libero_10
      method : sync_nortc | sync_rtc | async_nortc | async_rtc | async_rtc_sm | ...
      model  : smolvla | pi05 | ...

Line-style convention (for paper figures):
    Color     — method family (same family = similar hue; see _PALETTE)
    Linestyle — method variant: SM/multicand → solid (─), base → dashed (╌)
    Marker    — method variant: multicand → triangle (▲), SM → circle (●), base → square (■)

Usage::

    # Single method, single model
    python -m lerobot.async_inference.analyze_sweep \\
        outputs/eval_thesis/libero --method sync_nortc --model smolvla

    # Compare multiple methods, single model
    python -m lerobot.async_inference.analyze_sweep \\
        outputs/eval_thesis/libero \\
        --method sync_nortc async_nortc async_rtc async_rtc_sm --model smolvla

    # Compare smolvla vs pi05 on one method
    python -m lerobot.async_inference.analyze_sweep \\
        outputs/eval_thesis/libero \\
        --method async_rtc --model smolvla pi05

    # Full comparison (all methods × models found under eval_root)
    python -m lerobot.async_inference.analyze_sweep outputs/eval_thesis/libero

Output files (saved to <out_dir>/):
    sweep_data.csv
    solve_rate_vs_delay_K<K>.png          — per K; lines = methods/models
    solve_rate_vs_delay_combined.png      — all K levels in one grid figure
    solve_rate_vs_horizon_d<L>.png        — per latency; lines = methods/models
    solve_rate_vs_horizon_combined.png    — all d levels in one grid figure
    solve_rate_heatmap_<suite>_<method>_<model>.png   — L × K grid (single method)
    method_comparison_s<d>_<P><K>.png    — bar chart per (d, K) combo (multi-method only)
    method_comparison_combined.png        — all (d, K) combos in one grid (multi-method only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Stable style maps (consistent across all figures) ─────────────────────────

# Color: indexed by method in a fixed canonical order.
# SM variants sit immediately after their base for visual grouping.
_METHOD_ORDER = [
    "sync_nortc",
    "sync_nortc_sm",
    "sync_nortc_sm_multicand",
    "sync_rtc",
    "sync_rtc_sm",
    "sync_rtc_sm_multicand",
    "async_nortc",
    "async_nortc_sm",
    "async_nortc_sm_multicand",
    "async_rtc",
    "async_rtc_sm",
    "async_rtc_sm_multicand",
    # extras get colours from the tail
]
_PALETTE = [
    "#4C9BE8",  # blue          — sync_nortc
    "#2563EB",  # dark-blue     — sync_nortc_sm
    "#1E40AF",  # deeper-blue   — sync_nortc_sm_multicand
    "#22C55E",  # green         — sync_rtc
    "#16A34A",  # dark-green    — sync_rtc_sm
    "#15803D",  # deeper-green  — sync_rtc_sm_multicand
    "#F59E0B",  # amber         — async_nortc
    "#D97706",  # dark-amber    — async_nortc_sm
    "#B45309",  # deeper-amber  — async_nortc_sm_multicand
    "#EF4444",  # red            — async_rtc
    "#DC2626",  # dark-red      — async_rtc_sm
    "#991B1B",  # deeper-red    — async_rtc_sm_multicand
    "#D946EF",  # pink
    "#F97316",  # orange
    "#64748B",  # slate
]

# Base method families (without _sm / _sm_multicand variants) in canonical order.
# Used by _method_sort_key to group variants adjacently in bar charts.
_BASE_FAMILIES = ["sync_nortc", "sync_rtc", "async_nortc", "async_rtc"]


def _method_sort_key(method: str) -> tuple[int, int]:
    """Sort key: (base_family_rank, variant_rank).

    Groups _sm and _sm_multicand variants immediately after their base so they
    appear adjacent in bar charts for easy visual comparison.
      variant 0 = base, 1 = _sm, 2 = _sm_multicand
    """
    if method.endswith("_sm_multicand"):
        base, variant = method[: -len("_sm_multicand")], 2
    elif method.endswith("_sm"):
        base, variant = method[: -len("_sm")], 1
    else:
        base, variant = method, 0
    try:
        rank = _BASE_FAMILIES.index(base)
    except ValueError:
        rank = len(_BASE_FAMILIES) + abs(hash(base)) % 100
    return rank, variant

# Marker: indexed by model in a fixed canonical order
_MODEL_ORDER = ["smolvla", "pi05"]
_MARKERS     = ["o", "s", "^", "D", "v", "P"]
# Linestyle: by model
_LINESTYLES  = ["-", "--", "-.", ":"]


def _method_color(method: str) -> str:
    try:
        idx = _METHOD_ORDER.index(method)
    except ValueError:
        idx = len(_METHOD_ORDER) + hash(method) % (len(_PALETTE) - len(_METHOD_ORDER))
    return _PALETTE[idx % len(_PALETTE)]


def _model_marker(model: str) -> str:
    try:
        idx = _MODEL_ORDER.index(model)
    except ValueError:
        idx = len(_MODEL_ORDER) + hash(model) % (len(_MARKERS) - len(_MODEL_ORDER))
    return _MARKERS[idx % len(_MARKERS)]


def _model_linestyle(model: str) -> str:
    try:
        idx = _MODEL_ORDER.index(model)
    except ValueError:
        idx = len(_MODEL_ORDER) + hash(model) % (len(_LINESTYLES) - len(_MODEL_ORDER))
    return _LINESTYLES[idx % len(_LINESTYLES)]


def _method_linestyle(method: str) -> str:
    """Solid for SM/multicand variants; dashed for base methods."""
    return "-" if "_sm" in method else "--"


def _method_marker(method: str) -> str:
    """Triangle for multicand, circle for SM, square for base."""
    if "_sm_multicand" in method:
        return "^"
    if "_sm" in method:
        return "o"
    return "s"


def _method_linewidth(method: str) -> float:
    """Thicker lines for more-capable variants to create visual hierarchy."""
    if "_sm_multicand" in method:
        return 2.2
    if "_sm" in method:
        return 1.8
    return 1.4


def _method_hatch(method: str) -> str:
    """Bar hatch pattern by variant: multicand=dense crosses, SM=dots, base=none."""
    if "_sm_multicand" in method:
        return "xx"
    if "_sm" in method:
        return ".."
    return ""


def _line_label(method: str, model: str, multi_model: bool) -> str:
    return f"{method} ({model})" if multi_model else method


def _wilson_ci(
    p: np.ndarray, n: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score 95% confidence interval for binomial proportions.

    Numerically stable for p near 0 or 1 and small n. Returns (lower, upper)
    arrays clipped to [0, 1].
    """
    n = np.where(n > 0, n, 1)  # guard div-by-zero; CI → [0,1] for n=0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return np.clip(center - margin, 0.0, 1.0), np.clip(center + margin, 0.0, 1.0)


def _pool_avg(grp_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Weighted-pool success_rate and n_episodes across suites.

    Uses episode counts as weights so the pooled rate equals
    total_successes / total_episodes rather than a naive mean of rates.
    """
    def _pool(g: pd.DataFrame) -> pd.Series:
        total_n = g["n_episodes"].sum()
        pooled_sr = (
            (g["success_rate"] * g["n_episodes"]).sum() / total_n
            if total_n > 0 else float("nan")
        )
        return pd.Series({"success_rate": pooled_sr, "n_episodes": int(total_n)})
    return grp_df.groupby(group_cols, sort=False).apply(_pool).reset_index()


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_latency_tag(tag: str) -> tuple[int | None, float | None]:
    """Parse directory latency tag → (delay_steps, latency_s).

    New format : 's2'      → (2, None)     — steps-based (paper-aligned)
    Legacy format: 'd0.050' → (None, 0.05)  — seconds-based (backward compat)
    """
    m = re.fullmatch(r"s([0-9]+)", tag)
    if m:
        return int(m.group(1)), None
    m = re.fullmatch(r"d([0-9]+\.[0-9]+)", tag)
    if m:
        return None, float(m.group(1))
    return None, None


def _parse_param_tag(tag: str) -> tuple[str, int]:
    """Parse sweep-parameter directory tag → (param_name, value).

    'K10' → ('K', 10)   sync_nortc   (actions_per_chunk)
    'T10' → ('T', 10)   async_nortc  (chunk_size_threshold steps)
    'H20' → ('H', 20)   async_rtc    (rtc_execution_horizon steps)
    Returns ('', -1) on parse failure.
    """
    m = re.fullmatch(r"([A-Za-z]+)([0-9]+)", tag)
    if m:
        return m.group(1).upper(), int(m.group(2))
    return "", -1


_PARAM_XLABELS: dict[str, str] = {
    "K": "K (actions_per_chunk)",
    "T": "T (threshold steps)",
    "H": "H (rtc_horizon steps)",
}


def _param_pname(df: pd.DataFrame) -> str:
    """Return short param name(s) for titles and filenames, e.g. 'K', 'T', 'K/T/H'."""
    if "param_name" not in df.columns or df.empty:
        return "K"
    names = sorted(df["param_name"].dropna().unique())
    return "/".join(names) if names else "K"


def _param_xlabel(df: pd.DataFrame) -> str:
    """Return x-axis label for the sweep parameter inferred from df['param_name'].

    Single param: 'K (actions_per_chunk)'
    Multiple:     'K (actions_per_chunk) / T (threshold steps) / H (rtc_horizon steps)'
    """
    if "param_name" not in df.columns or df.empty:
        return "K"
    names = df["param_name"].dropna().unique()
    if len(names) == 1:
        return _PARAM_XLABELS.get(names[0], names[0])
    return " / ".join(_PARAM_XLABELS.get(n, n) for n in sorted(names))


def collect_results(
    eval_root: Path,
    methods: list[str],
    models: list[str],
) -> pd.DataFrame:
    """Walk eval_root and collect all matching aggregate.json records.

    Returns DataFrame with columns:
        suite, method, model, delay_steps (int), latency_s (float), fps (int),
        K (int), success_rate, n_episodes
    delay_steps is the primary delay axis (paper notation: d).
    latency_s   is derived: delay_steps / fps.
    """
    records = []

    # Pattern: <suite>/<method>/<model>/latency_<tag>/<param_tag>/results/aggregate.json
    # param_tag: K<K> (sync_nortc), T<T> (async_nortc), H<H> (async_rtc)
    for agg_path in sorted(eval_root.rglob("results/aggregate.json")):
        parts = agg_path.relative_to(eval_root).parts
        if len(parts) != 7:
            continue
        suite, method, model, latency_tag, param_tag, _, _ = parts

        if not latency_tag.startswith("latency_"):
            continue
        tag = latency_tag[len("latency_"):]   # strip prefix

        if methods and method not in methods:
            continue
        if models and model not in models:
            continue

        delay_steps, latency_s = _parse_latency_tag(tag)
        param_name, param_val = _parse_param_tag(param_tag)
        if delay_steps is None and latency_s is None:
            continue
        if param_val < 0:
            continue

        try:
            data = json.loads(agg_path.read_text())
            sr   = data.get("overall_success_rate", float("nan"))
            n_ep = data.get("total_episodes", 0)
            fps  = int(data.get("config", {}).get("fps", 0)) or None
        except Exception:
            continue

        # Derive missing field from the other + fps
        if delay_steps is None and fps and latency_s is not None:
            delay_steps = int(round(latency_s * fps))
        if latency_s is None and fps and delay_steps is not None:
            latency_s = delay_steps / fps

        records.append({
            "suite":        suite,
            "method":       method,
            "model":        model,
            "delay_steps":  delay_steps if delay_steps is not None else -1,
            "latency_s":    latency_s   if latency_s   is not None else float("nan"),
            "fps":          fps or 0,
            "K":            param_val,    # sweep param value (K/T/H depending on method)
            "param_name":   param_name,  # sweep param letter: "K", "T", or "H"
            "success_rate": sr,
            "n_episodes":   n_ep,
        })

    if not records:
        return pd.DataFrame(columns=["suite", "method", "model", "delay_steps",
                                     "latency_s", "fps", "K", "param_name",
                                     "success_rate", "n_episodes"])

    df = pd.DataFrame(records)
    df = df.sort_values(["method", "model", "suite", "delay_steps", "K"]).reset_index(drop=True)
    return df


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, label: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}  [{label}]")


def _suite_display(suite: str) -> str:
    return suite.replace("libero_", "")


def _pct_formatter():
    return plt.FuncFormatter(lambda y, _: f"{y:.0%}")


def _setup_ax(ax, xlabel: str, ylabel: str | None, title: str):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(_pct_formatter())
    ax.grid(True, alpha=0.3)


def _add_secondary_seconds_axis(ax, fps: int):
    """Add a secondary x-axis on top showing latency in milliseconds.

    Only called when x-axis is delay_steps and all data share the same fps.
    """
    if fps <= 0:
        return
    dt_ms = 1000.0 / fps
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Place ticks at the same positions as primary x-axis
    primary_ticks = ax.get_xticks()
    ax2.set_xticks(primary_ticks)
    ax2.set_xticklabels([f"{t * dt_ms:.0f}" for t in primary_ticks], fontsize=7)
    ax2.set_xlabel("latency (ms)", fontsize=7, labelpad=2)


def _draw_lines(ax, grp_df: pd.DataFrame, x_col: str,
                multi_model: bool, single_suite_mode: bool = False):
    """Draw one line per (method, model) pair into ax.

    Returns list of (handle, label) for shared legend.
    """
    handles, labels = [], []
    pairs = grp_df.groupby(["method", "model"], sort=False)
    for (method, model), sub in pairs:
        sub = sub.sort_values(x_col)
        if sub.empty:
            continue
        color = _method_color(method)
        # 95% Wilson score CI shaded band (requires n_episodes column)
        if "n_episodes" in sub.columns:
            p = sub["success_rate"].to_numpy(dtype=float)
            n = sub["n_episodes"].to_numpy(dtype=float)
            lo, hi = _wilson_ci(p, n)
            ax.fill_between(sub[x_col], lo, hi, color=color, alpha=0.13, linewidth=0)
        line, = ax.plot(
            sub[x_col], sub["success_rate"],
            color=color,
            linestyle=_method_linestyle(method),
            marker=_method_marker(method),
            linewidth=_method_linewidth(method),
            markersize=5,
        )
        lbl = _line_label(method, model, multi_model)
        handles.append(line)
        labels.append(lbl)
    return handles, labels


# ── Figure: solve rate vs. latency (one per K) ───────────────────────────────

def plot_vs_delay(df: pd.DataFrame, out_dir: Path):
    """For each K, plot solve_rate vs. d (delay in control steps).

    X-axis: delay_steps (paper notation: d).
    Secondary x-axis: latency in ms (when all records share the same fps).
    Layout: one subplot per suite + one 'average (all suites)' panel.
    Lines: one per (method, model).
    """
    suites = sorted(df["suite"].unique())
    K_values = sorted(df["K"].unique())
    multi_model = df["model"].nunique() > 1
    n_panels = len(suites) + (1 if len(suites) > 1 else 0)

    # Shared fps for secondary axis (None if inconsistent)
    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0

    for K in K_values:
        sub = df[df["K"] == K]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.8), sharey=True)
        if n_panels == 1:
            axes = [axes]
        fps_note = f"  [{shared_fps} Hz → 1 step = {1000//shared_fps} ms]" if shared_fps else ""
        pname = _param_pname(sub)
        pname_file = pname.replace("/", "_")
        fig.suptitle(f"Solve rate vs. delay d  ({pname} = {K}){fps_note}", fontsize=11, y=1.01)

        all_handles, all_labels = [], []

        for ax_i, suite in enumerate(suites):
            ax = axes[ax_i]
            _setup_ax(ax,
                      xlabel="d (control steps)",
                      ylabel="Solve rate" if ax_i == 0 else None,
                      title=_suite_display(suite))
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            grp = sub[sub["suite"] == suite]
            h, l = _draw_lines(ax, grp, "delay_steps", multi_model)
            if not all_handles:
                all_handles, all_labels = h, l
            if shared_fps:
                _add_secondary_seconds_axis(ax, shared_fps)

        # Average panel
        if len(suites) > 1:
            ax = axes[-1]
            _setup_ax(ax, xlabel="d (control steps)", ylabel=None,
                      title="avg (all suites)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            avg = _pool_avg(sub, ["method", "model", "delay_steps"])
            _draw_lines(ax, avg, "delay_steps", multi_model)
            if shared_fps:
                _add_secondary_seconds_axis(ax, shared_fps)

        if all_handles:
            fig.legend(all_handles, all_labels,
                       loc="lower center", ncol=min(len(all_labels), 4),
                       bbox_to_anchor=(0.5, -0.10), fontsize=8, framealpha=0.9)
        fig.tight_layout()
        _savefig(fig, out_dir / f"solve_rate_vs_delay_{pname_file}{K}.png", f"vs_delay_{pname_file}{K}")


# ── Figure: solve rate vs. K/horizon (one per latency) ───────────────────────

def plot_vs_horizon(df: pd.DataFrame, out_dir: Path):
    """For each delay d, plot solve_rate vs. K (actions_per_chunk / execution horizon)."""
    suites = sorted(df["suite"].unique())
    delay_vals = sorted(df["delay_steps"].unique())
    multi_model = df["model"].nunique() > 1
    n_panels = len(suites) + (1 if len(suites) > 1 else 0)

    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0

    for d in delay_vals:
        sub = df[df["delay_steps"] == d]
        if sub.empty:
            continue

        d_tag = f"s{d}"
        lat_note = f" = {d / shared_fps * 1000:.0f} ms" if shared_fps else ""
        xlabel = _param_xlabel(sub)
        pname = _param_pname(sub)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.8), sharey=True)
        if n_panels == 1:
            axes = [axes]
        fig.suptitle(f"Solve rate vs. {pname}  (d = {d} steps{lat_note})", fontsize=11, y=1.01)

        all_handles, all_labels = [], []

        for ax_i, suite in enumerate(suites):
            ax = axes[ax_i]
            _setup_ax(ax,
                      xlabel=xlabel,
                      ylabel="Solve rate" if ax_i == 0 else None,
                      title=_suite_display(suite))
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            grp = sub[sub["suite"] == suite]
            h, l = _draw_lines(ax, grp, "K", multi_model)
            if not all_handles:
                all_handles, all_labels = h, l

        if len(suites) > 1:
            ax = axes[-1]
            _setup_ax(ax, xlabel=xlabel, ylabel=None,
                      title="avg (all suites)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            avg = _pool_avg(sub, ["method", "model", "K"])
            _draw_lines(ax, avg, "K", multi_model)

        if all_handles:
            fig.legend(all_handles, all_labels,
                       loc="lower center", ncol=min(len(all_labels), 4),
                       bbox_to_anchor=(0.5, -0.10), fontsize=8, framealpha=0.9)
        fig.tight_layout()
        _savefig(fig, out_dir / f"solve_rate_vs_horizon_{d_tag}.png", f"vs_horizon_{d_tag}")


# ── Figure: combined grid — solve rate vs. delay (all K in one figure) ───────

def plot_combined_vs_delay(df: pd.DataFrame, out_dir: Path):
    """All K levels in one figure: rows = K values, columns = suites + avg.

    Equivalent to stacking all individual plot_vs_delay figures into a grid.
    Output: solve_rate_vs_delay_combined.png
    """
    suites = sorted(df["suite"].unique())
    K_values = sorted(df["K"].unique())
    multi_model = df["model"].nunique() > 1
    has_avg = len(suites) > 1
    n_rows = len(K_values)
    n_cols = len(suites) + (1 if has_avg else 0)
    if n_rows == 0 or n_cols == 0:
        return

    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0
    pname = _param_pname(df)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )
    fps_note = f"  [{shared_fps} Hz]" if shared_fps else ""
    fig.suptitle(f"Solve rate vs. delay d — all {pname} levels{fps_note}",
                 fontsize=12, y=1.01)

    # Column titles on top row
    col_titles = [_suite_display(s) for s in suites] + (["avg (all suites)"] if has_avg else [])
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=10, fontweight="bold")

    all_handles, all_labels = [], []

    for ri, K in enumerate(K_values):
        sub = df[df["K"] == K]
        row_axes = axes[ri]

        # Row label on leftmost axis
        row_axes[0].set_ylabel(f"{pname} = {K}\nSolve rate", fontsize=9)

        for ci, suite in enumerate(suites):
            ax = row_axes[ci]
            _setup_ax(ax, xlabel="d (steps)",
                      ylabel=None,
                      title="" if ri > 0 else col_titles[ci])
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            grp = sub[sub["suite"] == suite]
            h, l = _draw_lines(ax, grp, "delay_steps", multi_model)
            if not all_handles and h:
                all_handles, all_labels = h, l
            if shared_fps:
                _add_secondary_seconds_axis(ax, shared_fps)

        if has_avg:
            ax = row_axes[n_cols - 1]
            _setup_ax(ax, xlabel="d (steps)", ylabel=None,
                      title="" if ri > 0 else "avg (all suites)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            avg = _pool_avg(sub, ["method", "model", "delay_steps"])
            _draw_lines(ax, avg, "delay_steps", multi_model)
            if shared_fps:
                _add_secondary_seconds_axis(ax, shared_fps)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    if all_handles:
        fig.legend(all_handles, all_labels,
                   loc="lower center", ncol=min(len(all_handles), 5),
                   bbox_to_anchor=(0.5, 0.01), fontsize=8, framealpha=0.9)
    _savefig(fig, out_dir / "solve_rate_vs_delay_combined.png", "vs_delay_combined")


# ── Figure: combined grid — solve rate vs. horizon (all d in one figure) ─────

def plot_combined_vs_horizon(df: pd.DataFrame, out_dir: Path):
    """All delay levels in one figure: rows = d values, columns = suites + avg.

    Equivalent to stacking all individual plot_vs_horizon figures into a grid.
    Output: solve_rate_vs_horizon_combined.png
    """
    suites = sorted(df["suite"].unique())
    delay_vals = sorted(df["delay_steps"].unique())
    multi_model = df["model"].nunique() > 1
    has_avg = len(suites) > 1
    n_rows = len(delay_vals)
    n_cols = len(suites) + (1 if has_avg else 0)
    if n_rows == 0 or n_cols == 0:
        return

    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0
    xlabel = _param_xlabel(df)
    pname = _param_pname(df)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )
    fig.suptitle(f"Solve rate vs. {pname} — all delay levels",
                 fontsize=12, y=1.01)

    col_titles = [_suite_display(s) for s in suites] + (["avg (all suites)"] if has_avg else [])
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=10, fontweight="bold")

    all_handles, all_labels = [], []

    for ri, d in enumerate(delay_vals):
        sub = df[df["delay_steps"] == d]
        row_axes = axes[ri]

        lat_note = f" ({d * 1000 // shared_fps} ms)" if shared_fps else ""
        row_axes[0].set_ylabel(f"d = {d}{lat_note}\nSolve rate", fontsize=9)

        for ci, suite in enumerate(suites):
            ax = row_axes[ci]
            _setup_ax(ax, xlabel=xlabel, ylabel=None,
                      title="" if ri > 0 else col_titles[ci])
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            grp = sub[sub["suite"] == suite]
            h, l = _draw_lines(ax, grp, "K", multi_model)
            if not all_handles and h:
                all_handles, all_labels = h, l

        if has_avg:
            ax = row_axes[n_cols - 1]
            _setup_ax(ax, xlabel=xlabel, ylabel=None,
                      title="" if ri > 0 else "avg (all suites)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            avg = _pool_avg(sub, ["method", "model", "K"])
            _draw_lines(ax, avg, "K", multi_model)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    if all_handles:
        fig.legend(all_handles, all_labels,
                   loc="lower center", ncol=min(len(all_handles), 5),
                   bbox_to_anchor=(0.5, 0.01), fontsize=8, framealpha=0.9)
    _savefig(fig, out_dir / "solve_rate_vs_horizon_combined.png", "vs_horizon_combined")


# ── Figure: solve rate heatmap L × K (single method, per suite) ──────────────

def plot_heatmap(df: pd.DataFrame, out_dir: Path):
    """d×K heatmap of solve_rate for each (suite, method, model) combination."""
    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0

    for (suite, method, model), grp in df.groupby(["suite", "method", "model"]):
        pivot = grp.pivot_table(index="delay_steps", columns="K",
                                values="success_rate", aggfunc="mean")
        if pivot.empty:
            continue

        nrows, ncols = len(pivot.index), len(pivot.columns)
        fig, ax = plt.subplots(figsize=(max(4, ncols * 0.9 + 1.5),
                                        max(3, nrows * 0.8 + 1.2)))
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1,
                       cmap="RdYlGn", origin="upper")
        pxlabel = _param_xlabel(grp)
        pcol = _param_pname(grp)
        ax.set_xticks(range(ncols))
        ax.set_xticklabels([f"{pcol}={c}" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(nrows))
        # Y-axis labels: "d=N (Xms)"
        if shared_fps:
            ylabels = [f"d={int(d)} ({int(d)*1000//shared_fps}ms)" for d in pivot.index]
        else:
            ylabels = [f"d={int(d)}" for d in pivot.index]
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel(pxlabel, fontsize=9)
        ax.set_ylabel("delay d (control steps)", fontsize=9)
        ax.set_title(f"Solve rate — {_suite_display(str(suite))} | {method} | {model}",
                     fontsize=10)
        for i in range(nrows):
            for j in range(ncols):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=9, color="black")
        plt.colorbar(im, ax=ax, label="Solve rate")
        fig.tight_layout()
        fname = f"solve_rate_heatmap_{suite}_{method}_{model}.png"
        _savefig(fig, out_dir / fname, f"heatmap_{suite}")


# ── Bar-chart drawing helper ──────────────────────────────────────────────────

def _draw_bar_ax(ax, sub: pd.DataFrame, suites: list[str],
                 multi_model: bool, title: str = "",
                 label_fontsize: int = 7, tick_fontsize: int = 8) -> tuple[list, list]:
    """Draw grouped bars (x=suites, groups=method×model) into ax.

    Returns (handles, labels) for a shared legend.
    """
    pairs = sorted(
        sub.groupby(["method", "model"]).groups.keys(),
        key=lambda p: (_method_sort_key(p[0]), p[1]),
    ) if not sub.empty else []
    models = sorted(sub["model"].unique()) if not sub.empty else []
    n_groups = len(pairs)
    x = np.arange(len(suites))
    width = 0.8 / max(n_groups, 1)

    handles, labels = [], []
    for gi, (method, model) in enumerate(pairs):
        vals = []
        for suite in suites:
            mask = (sub["suite"] == suite) & (sub["method"] == method) & (sub["model"] == model)
            v = sub[mask]["success_rate"].mean()
            vals.append(float(v) if not np.isnan(v) else 0.0)
        offset = (gi - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=_line_label(method, model, multi_model),
                      color=_method_color(method),
                      hatch=_method_hatch(method),
                      alpha=0.75 if "_sm" not in method else (0.88 if "_sm_multicand" not in method else 1.0))
        for bar, v in zip(bars, vals, strict=False):
            if v > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.0%}", ha="center", va="bottom",
                        fontsize=label_fontsize)
        h, l = ax.get_legend_handles_labels()
        handles, labels = h, l

    ax.set_xticks(x)
    ax.set_xticklabels([_suite_display(s) for s in suites], fontsize=tick_fontsize)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(_pct_formatter())
    ax.grid(axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=tick_fontsize)
    return handles, labels


# ── Figure: method comparison bar charts — all (d, K) combos ─────────────────

def plot_method_comparison(df: pd.DataFrame, out_dir: Path):
    """One bar chart per (d, K) combo + one combined grid.

    Separate:  method_comparison_s{d}_{pname}{K}.png — one per combo
    Combined:  method_comparison_combined.png         — rows=K, cols=d
    """
    if df["method"].nunique() < 2:
        return

    suites = sorted(df["suite"].unique())
    d_vals = sorted(df["delay_steps"].unique())
    K_vals = sorted(df["K"].unique())
    multi_model = df["model"].nunique() > 1
    pname = _param_pname(df)
    pname_file = pname.replace("/", "_")  # safe for filesystem paths

    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0

    # Build legend proxies upfront — reliable regardless of which axes data lands on
    pairs_all = sorted(
        df.groupby(["method", "model"]).groups.keys(),
        key=lambda p: (_method_sort_key(p[0]), p[1]),
    )
    models_all = sorted(df["model"].unique())
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=_method_color(m),
                       hatch=_method_hatch(m),
                       alpha=0.75 if "_sm" not in m else (0.88 if "_sm_multicand" not in m else 1.0))
        for m, mdl in pairs_all
    ]
    legend_labels = [_line_label(m, mdl, multi_model) for m, mdl in pairs_all]
    n_legend_cols = min(len(legend_handles), 3)

    # ── Per-combo figures ─────────────────────────────────────────────────────
    for d in d_vals:
        ms_note = f" = {d * 1000 // shared_fps} ms" if shared_fps else ""
        for K in K_vals:
            sub = df[(df["delay_steps"] == d) & (df["K"] == K)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(5, len(suites) * 2.0 + 2), 4.5))
            _draw_bar_ax(ax, sub, suites, multi_model,
                         title=f"Method comparison — d={d} steps{ms_note}  {pname}={K}",
                         label_fontsize=8, tick_fontsize=9)
            ax.set_ylabel("Solve rate", fontsize=9)
            fig.tight_layout(rect=[0, 0.10, 1, 1])
            fig.legend(legend_handles, legend_labels,
                       loc="lower center", ncol=n_legend_cols,
                       bbox_to_anchor=(0.5, 0.01), fontsize=7, framealpha=0.9)
            _savefig(fig, out_dir / f"method_comparison_s{d}_{pname_file}{K}.png",
                     f"bar_s{d}_{pname_file}{K}")

    # ── Combined grid: rows = K values, cols = d values ───────────────────────
    n_rows, n_cols = len(K_vals), len(d_vals)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )
    fig.suptitle(f"Method comparison — all {pname} × delay combinations", fontsize=12, y=1.01)

    for ri, K in enumerate(K_vals):
        axes[ri, 0].set_ylabel(f"{pname}={K}\nSolve rate", fontsize=9)
        for ci, d in enumerate(d_vals):
            ax = axes[ri, ci]
            sub = df[(df["delay_steps"] == d) & (df["K"] == K)]
            ms = f"\n({d * 1000 // shared_fps} ms)" if shared_fps else ""
            col_title = f"d={d}{ms}" if ri == 0 else ""
            _draw_bar_ax(ax, sub, suites, multi_model,
                         title=col_title, label_fontsize=6, tick_fontsize=7)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(legend_handles, legend_labels,
               loc="lower center", ncol=min(len(legend_handles), 5),
               bbox_to_anchor=(0.5, 0.01), fontsize=8, framealpha=0.9)
    _savefig(fig, out_dir / "method_comparison_combined.png", "bar_combined")


# ── Figure: 2×2 paper summary (avg across suites, both axes) ─────────────────

def plot_paper_summary(df: pd.DataFrame, out_dir: Path,
                       summary_K: int | None = None,
                       summary_delay: int | None = None):
    """2-panel figure for paper: (A) vs delay d at fixed K, (B) vs K at fixed d.

    Uses the average across all suites.
    """
    multi_model = df["model"].nunique() > 1

    K_vals = sorted(df["K"].unique())
    d_vals = sorted(df["delay_steps"].unique())
    if not K_vals or not d_vals:
        return

    fix_K = summary_K if summary_K is not None else K_vals[len(K_vals) // 2]
    fix_d = summary_delay if summary_delay is not None else (d_vals[1] if len(d_vals) > 1 else d_vals[0])

    fps_vals = df["fps"][df["fps"] > 0].unique()
    shared_fps = int(fps_vals[0]) if len(fps_vals) == 1 else 0

    fig, (ax_delay, ax_horiz) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Solve rate — average across suites", fontsize=11)

    all_handles, all_labels = [], []

    # Panel A: vs delay at fixed K
    sub_K = df[df["K"] == fix_K]
    if not sub_K.empty:
        _setup_ax(ax_delay, xlabel="d (control steps)",
                  ylabel="Solve rate", title=f"(A)  vs. delay  [K={fix_K}]")
        ax_delay.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        avg = _pool_avg(sub_K, ["method", "model", "delay_steps"])
        h, l = _draw_lines(ax_delay, avg, "delay_steps", multi_model)
        all_handles, all_labels = h, l
        if shared_fps:
            _add_secondary_seconds_axis(ax_delay, shared_fps)

    # Panel B: vs horizon at fixed d
    sub_d = df[df["delay_steps"] == fix_d]
    if not sub_d.empty:
        d_ms_note = f" = {fix_d * 1000 // shared_fps} ms" if shared_fps else ""
        pxlabel = _param_xlabel(sub_d)
        pname_b = _param_pname(sub_d)
        _setup_ax(ax_horiz, xlabel=pxlabel,
                  ylabel=None, title=f"(B)  vs. {pname_b}  [d={fix_d}{d_ms_note}]")
        ax_horiz.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        avg = _pool_avg(sub_d, ["method", "model", "K"])
        _draw_lines(ax_horiz, avg, "K", multi_model)

    if all_handles:
        fig.legend(all_handles, all_labels,
                   loc="lower center", ncol=min(len(all_labels), 6),
                   bbox_to_anchor=(0.5, -0.10), fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fname = f"paper_summary_s{fix_d}_K{fix_K}.png"
    _savefig(fig, out_dir / fname, "paper_summary")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sweep-level analysis: solve rate vs. delay / horizon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "eval_root",
        help="Root of sweep output (contains <suite>/<method>/<model>/... subdirs)",
    )
    ap.add_argument(
        "--method", nargs="+", default=None,
        help=(
            "Method(s) to include (default: all found). "
            "E.g. --method sync_nortc async_rtc async_rtc_sm"
        ),
    )
    ap.add_argument(
        "--model", nargs="+", default=None,
        help="Model(s) to include (default: all found). E.g. --model smolvla pi05",
    )
    ap.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: <eval_root>/analysis/<methods>/<models>)",
    )
    ap.add_argument(
        "--no_combined", action="store_true",
        help="Skip combined grid figures (solve_rate_vs_delay_combined.png etc.)",
    )
    ap.add_argument(
        "--no_heatmap", action="store_true",
        help="Skip L×K heatmap figures",
    )
    ap.add_argument(
        "--no_summary", action="store_true",
        help="Skip 2-panel paper summary figure",
    )
    ap.add_argument(
        "--summary_K", type=int, default=None,
        help="Fixed K for paper summary panel A (default: median K)",
    )
    ap.add_argument(
        "--summary_delay", type=int, default=None,
        help="Fixed d (steps) for paper summary panel B (default: second-lowest d)",
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.is_dir():
        ap.error(f"eval_root not found: {eval_root}")

    methods = args.method or []
    models  = args.model  or []

    print("\n" + "═" * 72)
    print("  analyze_sweep — Solve Rate Sweep Analysis")
    print("═" * 72)
    print(f"  eval_root : {eval_root}")
    print(f"  methods   : {methods or '(all)'}")
    print(f"  models    : {models  or '(all)'}")

    df = collect_results(eval_root, methods, models)
    if df.empty:
        print("\nERROR: No aggregate.json files found matching the specified criteria.")
        print("  Expected: <eval_root>/<suite>/<method>/<model>/latency_s<d>/<param><val>/results/aggregate.json")
        print("  param: K (sync_nortc), T (async_nortc), H (async_rtc)")
        sys.exit(1)

    found_methods = sorted(df["method"].unique())
    found_models  = sorted(df["model"].unique())
    print(f"\n  Found {len(df)} records  |  methods: {found_methods}  |  models: {found_models}")
    print(df.groupby(["method", "model", "suite"])["success_rate"].agg(
        n="count", mean="mean", min="min", max="max"
    ).to_string())

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        method_tag = "_vs_".join(found_methods) if len(found_methods) <= 3 else "multi_method"
        model_tag  = "_vs_".join(found_models)  if len(found_models)  <= 2 else "multi_model"
        out_dir = eval_root / "analysis" / method_tag / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  out_dir   : {out_dir}")

    # Save raw data
    csv_path = out_dir / "sweep_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved : {csv_path}")

    print("\nGenerating figures...")

    plot_vs_delay(df, out_dir)
    plot_vs_horizon(df, out_dir)

    if not args.no_combined:
        plot_combined_vs_delay(df, out_dir)
        plot_combined_vs_horizon(df, out_dir)

    if not args.no_heatmap:
        plot_heatmap(df, out_dir)

    if df["method"].nunique() > 1:
        plot_method_comparison(df, out_dir)

    if not args.no_summary:
        plot_paper_summary(df, out_dir, args.summary_K, args.summary_delay)

    print(f"\n  All figures saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
