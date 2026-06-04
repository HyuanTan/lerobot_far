#!/usr/bin/env python3
"""Parse and plot LIBERO eval results from eval_results.json files.

Four plot generators:
  1. generate_delay_plots          — SR vs. delay, one figure per horizon value
                                     subplots = 4 LIBERO suites, curves = models
  2. generate_horizon_plots        — SR vs. horizon, one figure per delay value
                                     same layout
  3. generate_per_combination_plots— one figure per method combo (baseline/rtc/+SM)
                                     layout: 2 rows × 4 cols (delay-row, horizon-row)
                                     curves = checkpoint variants + Wilson 95% CI
  4. generate_combined_overview    — single overview figure
                                     layout: 4 rows (suites) × 2 cols (delay, horizon)
                                     curves = 4 method combos + Wilson 95% CI

Usage::

    python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \\
        outputs/eval/pi05                   \\
        [--plot-delay]                       # SR vs delay (default on)
        [--plot-horizon]                     # SR vs horizon
        [--plot-combo]                       # per-combination figures
        [--plot-overview]                    # combined overview
        [--all-plots]                        # all four
        [--autoscale] [--ci]                 # y-axis + confidence interval
        [--output-dir plots/]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Discovery & parsing
# ---------------------------------------------------------------------------

RESULT_FILENAMES = ("eval_results.json", "merged_results.json")


def find_eval_results(root_dir: str) -> list[str]:
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for candidate in RESULT_FILENAMES:
            if candidate in filenames:
                results.append(os.path.join(dirpath, candidate))
                break
    return sorted(results)


def parse_config_params(folder_name: str) -> dict:
    tokens = folder_name.split("_")
    params: dict = {}
    i = len(tokens) - 1
    while i >= 1:
        tok = tokens[i]
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            j = i - 1
            while j >= 0 and not re.fullmatch(r"\d+(?:\.\d+)?", tokens[j]):
                j -= 1
            key = "_".join(tokens[j + 1: i])
            if key:
                params[key] = float(tok) if "." in tok else int(tok)
            i = j
        else:
            i -= 1
    return params


def extract_path_info(eval_result_path: str, root_dir: str) -> dict:
    rel = os.path.relpath(eval_result_path, root_dir)
    parts = list(Path(rel).parts[:-1])
    info: dict = {"path": eval_result_path, "rel_path": rel}
    if len(parts) >= 1:
        info["policy"] = parts[-1]
    if len(parts) >= 2:
        info["config"] = parts[-2]
        info["params"] = parse_config_params(parts[-2])
    if len(parts) >= 3:
        info["suite"] = parts[-3]
    if len(parts) >= 4:
        info["model"] = parts[-4]
    return info


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def adapt_it_rtc_entry(entry: Dict[str, Any]) -> None:
    """Populate suite / model / method_combo / params from the JSON's 'config' dict."""
    data = entry.get("data", {})
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return

    suite = cfg.get("suite")
    if suite:
        entry["suite"] = suite

    method  = cfg.get("method", "")
    sm_flag = bool(cfg.get("enable_sm", False))
    sm_tag  = "/SM" if sm_flag else ""
    label   = f"{method}{sm_tag}"
    entry["method"]  = method
    entry["sm"]      = sm_flag
    entry["combo"]   = (method, sm_flag)        # e.g. ("baseline", False)
    entry["combo_label"] = COMBO_LABELS.get((method, sm_flag), label)

    original_model = entry.get("model", "")
    entry["model"] = f"{original_model}/{label}" if original_model else label

    params = dict(entry.get("params") or {})
    for key in ("async_delay", "execution_horizon"):
        val = cfg.get(key)
        if isinstance(val, (int, float)):
            params[key] = val
    if "action" not in params:
        exec_h = cfg.get("execution_horizon")
        if isinstance(exec_h, (int, float)):
            params["action"] = exec_h
    entry["params"] = params

    # N total episodes for CI
    n_ep   = cfg.get("n_episodes", 0)
    n_task = sum(1 for k in data
                 if k not in ("overall", "eval_s", "video_paths", "config"))
    entry["n_total"] = max(int(n_ep) * max(int(n_task), 1), 1)


# ---------------------------------------------------------------------------
# Method-combination style constants  (fixed colours so plots are consistent)
# ---------------------------------------------------------------------------

COMBO_LABELS: dict[Tuple, str] = {
    ("baseline", False): "Baseline",
    ("rtc",      False): "RTC",
    ("baseline", True):  "Baseline+SM",
    ("rtc",      True):  "RTC+SM",
}
COMBO_ORDER = [("baseline", False), ("rtc", False), ("baseline", True), ("rtc", True)]

# Colour design:
#   No-SM variants → lighter shade, dashed line  (less prominent)
#   SM  variants   → darker shade, solid line    (more prominent)
#   Baseline family: blue hues  |  RTC family: orange/red hues
COMBO_COLORS: dict[Tuple, str] = {
    ("baseline", False): "#aec7e8",   # light blue   (no SM, dashed)
    ("rtc",      False): "#ffbb78",   # light orange (no SM, dashed)
    ("baseline", True):  "#1f77b4",   # dark  blue   (SM,    solid)
    ("rtc",      True):  "#d62728",   # dark  red    (SM,    solid)
}
COMBO_LINESTYLES: dict[Tuple, str] = {
    ("baseline", False): "--",
    ("rtc",      False): "--",
    ("baseline", True):  "-",
    ("rtc",      True):  "-",
}
COMBO_MARKERS: dict[Tuple, str] = {
    ("baseline", False): "o",
    ("rtc",      False): "s",
    ("baseline", True):  "^",
    ("rtc",      True):  "D",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "pc_successes": "Success %",
    "avg_sum_rewards": "Sum Rew",
    "avg_max_rewards": "Max Rew",
    "avg_episode_length": "Avg Ep Len",
}

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_DISPLAY = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object":  "LIBERO-Object",
    "libero_goal":    "LIBERO-Goal",
    "libero_10":      "LIBERO-10",
}

def _col_width(header, cells):
    return max(len(str(header)), *(len(str(c)) for c in cells))


def print_table(headers, rows, col_align=None, indent=0):
    if not rows:
        return
    if col_align is None:
        col_align = ["left"] * len(headers)
    widths = [_col_width(h, [r[i] for r in rows]) for i, h in enumerate(headers)]
    fmts = [f"{{:>{w}}}" if a == "right" else f"{{:<{w}}}" for w, a in zip(widths, col_align)]
    line_fmt = " | ".join(fmts)
    sep      = "-+-".join("-" * w for w in widths)
    pad      = " " * indent
    print(pad + line_fmt.format(*headers))
    print(pad + sep)
    for row in rows:
        print(pad + line_fmt.format(*[str(c) for c in row]))


def shorten_task(name: str) -> str:
    s = name
    for prefix in ("pick_up_the_", "open_the_", "close_the_", "put_the_", "turn_on_the_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.replace("_and_place_it_in_the_basket", " → basket")
    s = s.replace("_and_place_it_", " → ")
    return s.replace("_", " ")


def shorten_model_name(name: str) -> str:
    if "/" in name:
        model_part, _, method_part = name.rpartition("/")
        return shorten_model_name(model_part) + " / " + method_part
    parts = name.split("_")
    keep  = [p for p in parts if p.lower() not in {"pi05lb", "libero"}]
    s     = "_".join(keep) if keep else name
    return s[:47] + "..." if len(s) > 50 else s


# ---------------------------------------------------------------------------
# matplotlib / CI helpers
# ---------------------------------------------------------------------------

def _init_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        return plt, Line2D
    except ImportError:
        print("Error: matplotlib required — pip install matplotlib")
        return None, None


def _build_style_maps(active_models: list, plt):
    """Dynamic colour/marker map for arbitrary model names."""
    colors = []
    for name in ("tab20", "tab20b", "tab20c"):
        try:
            colors.extend(list(plt.colormaps[name].colors))
        except (AttributeError, KeyError):
            colors.extend(list(getattr(plt.cm, name).colors))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "d", "P", "X"]
    return (
        {m: colors[i % len(colors)] for i, m in enumerate(active_models)},
        {m: markers[i % len(markers)] for i, m in enumerate(active_models)},
    )


def _build_combo_style_maps(active_models: list, entries: list, plt):
    """Style maps that use COMBO_* constants for known method combos.

    When a model string is associated with a known (method, sm) combo via
    entry["combo"], use COMBO_COLORS / COMBO_MARKERS / COMBO_LINESTYLES so
    colours are consistent across all four plot generators.  Unrecognised models
    fall back to the dynamic tab20 palette with solid lines.

    Returns (model_colors, model_markers, model_linestyles, model_labels) where
    model_labels replaces long checkpoint paths with the short combo name.
    """
    # Build model → combo lookup from entries
    model_to_combo: dict[str, tuple] = {}
    for e in entries:
        combo = e.get("combo")
        model = e.get("model", "")
        if combo is not None and model:
            model_to_combo[model] = combo

    # Dynamic fallback pool
    dyn_colors: list = []
    for name in ("tab20", "tab20b", "tab20c"):
        try:
            dyn_colors.extend(list(plt.colormaps[name].colors))
        except (AttributeError, KeyError):
            dyn_colors.extend(list(getattr(plt.cm, name).colors))
    dyn_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "d", "P", "X"]

    model_colors: dict[str, object] = {}
    model_markers: dict[str, str] = {}
    model_linestyles: dict[str, str] = {}
    model_labels: dict[str, str] = {}
    fallback_idx = 0

    for m in active_models:
        combo = model_to_combo.get(m)
        if combo is not None and combo in COMBO_COLORS:
            model_colors[m]     = COMBO_COLORS[combo]
            model_markers[m]    = COMBO_MARKERS[combo]
            model_linestyles[m] = COMBO_LINESTYLES[combo]
            model_labels[m]     = COMBO_LABELS[combo]
        else:
            model_colors[m]     = dyn_colors[fallback_idx % len(dyn_colors)]
            model_markers[m]    = dyn_markers[fallback_idx % len(dyn_markers)]
            model_linestyles[m] = "-"
            model_labels[m]     = shorten_model_name(m)
            fallback_idx += 1

    return model_colors, model_markers, model_linestyles, model_labels


def _wilson_ci(sr_pct: float, n: int, z: float = 1.96) -> float:
    """95 % Wilson score interval half-width (in percentage points)."""
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, sr_pct / 100.0))
    ci = z * math.sqrt(p * (1.0 - p) / n)
    return ci * 100.0


def _add_bottom_legend(fig, handles, plt, ncols: int = 4):
    n_items = len(handles)
    ncols   = min(ncols, n_items)
    n_rows  = (n_items + ncols - 1) // ncols
    legend_h = 0.22 * n_rows
    fig_h    = fig.get_size_inches()[1] + legend_h
    fig.set_size_inches(fig.get_size_inches()[0], fig_h)
    bottom = (legend_h + 0.15) / fig_h
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    leg = fig.legend(
        handles=handles, loc="lower center", ncol=ncols,
        fontsize=9, framealpha=1.0, bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.2, handletextpad=0.4, edgecolor="0.3",
    )
    leg.get_frame().set_linewidth(0.8)


def _style_ax(ax, xlabel: str, ylabel: str = "Solve Rate (%)",
              xlim=None, xticks=None, ylim=(-2, 102)):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
        if len(xticks) > 10:
            ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, plt, path: str, saved: list[str]):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Helper: collect (x, sr, ci) series from entries
# ---------------------------------------------------------------------------

def _collect_series(entries, x_param: str, filter_fn=None, show_ci: bool = True):
    """Return {suite: {model: [(x, sr, ci), ...]}} for a given x-axis parameter."""
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if filter_fn and not filter_fn(e):
            continue
        x_val = e.get("params", {}).get(x_param)
        sr    = e["data"].get("overall", {}).get("pc_successes")
        if x_val is None or sr is None:
            continue
        suite = e.get("suite", "unknown")
        model = e.get("model", "unknown")
        ci    = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0
        data[suite][model].append((x_val, sr, ci))
    return data


def _collect_combo_series(entries, x_param: str, fixed_param: str,
                          fixed_val: Any, show_ci: bool = True):
    """Return {suite: {combo: [(x, sr, ci)]}} filtering on fixed_param==fixed_val."""
    data: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fp = e.get("params", {}).get(fixed_param)
        if fp is None or fp != fixed_val:
            continue
        x_val = e.get("params", {}).get(x_param)
        sr    = e["data"].get("overall", {}).get("pc_successes")
        if x_val is None or sr is None:
            continue
        suite = e.get("suite", "unknown")
        combo = e.get("combo")
        if combo is None:
            continue
        ci = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0
        data[suite][combo].append((x_val, sr, ci))
    return data


# ---------------------------------------------------------------------------
# Plot 1: SR vs. delay (one figure per execution_horizon)
# ---------------------------------------------------------------------------

def generate_delay_plots(entries: List[Dict], output_dir: str,
                         autoscale: bool = False, show_ci: bool = True) -> List[str]:
    """SR vs async_delay; one 2×2 figure per execution_horizon value."""
    plt, Line2D = _init_mpl()
    if plt is None:
        return []

    plot_data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in entries:
        params  = e.get("params", {})
        horizon = params.get("action", params.get("execution_horizon", "?"))
        delay   = params.get("async_delay")
        sr      = e["data"].get("overall", {}).get("pc_successes")
        if delay is None or sr is None:
            continue
        ci = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0
        plot_data[horizon][e.get("suite", "?")][e.get("model", "?")].append((delay, sr, ci))

    if not plot_data:
        print("Warning: no data for delay plots.")
        return []

    global_models = sorted({m for hd in plot_data.values() for sd in hd.values() for m in sd})
    active        = [m for m in global_models if not m.startswith("x")]
    col, mkr, ls, lbl = _build_combo_style_maps(active, entries, plt)

    saved: list[str] = []
    for h_val in sorted(plot_data):
        all_d = sorted({d for sd in plot_data[h_val].values()
                        for pts in sd.values() for d, _, _ in pts})
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Solve Rate vs. Inference Delay  (horizon = {h_val})", fontsize=14, fontweight="bold")

        for idx, sk in enumerate(SUITE_ORDER):
            ax = axes[idx // 2][idx % 2]
            ax.set_title(SUITE_DISPLAY.get(sk, sk), fontsize=12, fontweight="bold")
            xlim = (min(all_d) - 0.4, max(all_d) + 0.4) if all_d else None
            _style_ax(ax, "Inference Delay  d", xticks=all_d, xlim=xlim,
                      ylim=None if autoscale else (-2, 102))

            # Draw SM variants last so they render on top of dashed no-SM lines.
            suite_dict = plot_data[h_val].get(sk, {})
            all_ys = []
            for m in sorted(active, key=lambda m: ls[m] == "-"):
                pts = suite_dict.get(m)
                if not pts:
                    continue
                pts.sort()
                xs, ys, cis = zip(*pts)
                all_ys.extend(ys)
                ax.errorbar(xs, ys, yerr=cis if show_ci else None,
                            marker=mkr[m], color=col[m], linestyle=ls[m],
                            linewidth=1.4, markersize=5, capsize=3,
                            label=lbl[m], zorder=2, alpha=0.9)

            if autoscale and all_ys:
                pad = max((max(all_ys) - min(all_ys)) * 0.15, 4.0)
                ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

        handles = [Line2D([0], [0], color=col[m], lw=1.4, marker=mkr[m],
                          linestyle=ls[m], ms=5, label=lbl[m]) for m in active]
        _add_bottom_legend(fig, handles, plt)
        _save(fig, plt, os.path.join(output_dir, f"sr_vs_delay_horizon_{h_val}.png"), saved)
    return saved


# ---------------------------------------------------------------------------
# Plot 2: SR vs. horizon (one figure per async_delay)
# ---------------------------------------------------------------------------

def generate_horizon_plots(entries: List[Dict], output_dir: str,
                            autoscale: bool = False, show_ci: bool = True) -> List[str]:
    """SR vs execution_horizon; one 2×2 figure per async_delay value."""
    plt, Line2D = _init_mpl()
    if plt is None:
        return []

    plot_data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in entries:
        params  = e.get("params", {})
        delay   = params.get("async_delay")
        horizon = params.get("execution_horizon", params.get("action"))
        sr      = e["data"].get("overall", {}).get("pc_successes")
        if delay is None or horizon is None or sr is None:
            continue
        ci = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0
        plot_data[delay][e.get("suite", "?")][e.get("model", "?")].append((horizon, sr, ci))

    if not plot_data:
        print("Warning: no data for horizon plots.")
        return []

    global_models = sorted({m for dd in plot_data.values() for sd in dd.values() for m in sd})
    active        = [m for m in global_models if not m.startswith("x")]
    col, mkr, ls, lbl = _build_combo_style_maps(active, entries, plt)

    saved: list[str] = []
    for d_val in sorted(plot_data):
        all_h = sorted({h for sd in plot_data[d_val].values()
                        for pts in sd.values() for h, _, _ in pts})
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Solve Rate vs. Execution Horizon  (delay = {d_val})", fontsize=14, fontweight="bold")

        for idx, sk in enumerate(SUITE_ORDER):
            ax = axes[idx // 2][idx % 2]
            ax.set_title(SUITE_DISPLAY.get(sk, sk), fontsize=12, fontweight="bold")
            xlim = (min(all_h) - 0.4, max(all_h) + 0.4) if all_h else None
            _style_ax(ax, "Execution Horizon  s", xticks=all_h, xlim=xlim,
                      ylim=None if autoscale else (-2, 102))

            suite_dict = plot_data[d_val].get(sk, {})
            all_ys: list = []
            for m in sorted(active, key=lambda m: ls[m] == "-"):
                pts = suite_dict.get(m)
                if not pts:
                    continue
                pts.sort()
                xs, ys, cis = zip(*pts)
                all_ys.extend(ys)
                ax.errorbar(xs, ys, yerr=cis if show_ci else None,
                            marker=mkr[m], color=col[m], linestyle=ls[m],
                            linewidth=1.4, markersize=5, capsize=3,
                            label=lbl[m], zorder=2, alpha=0.9)

            if autoscale and all_ys:
                pad = max((max(all_ys) - min(all_ys)) * 0.15, 4.0)
                ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

        handles = [Line2D([0], [0], color=col[m], lw=1.4, marker=mkr[m],
                          linestyle=ls[m], ms=5, label=lbl[m]) for m in active]
        _add_bottom_legend(fig, handles, plt)
        _save(fig, plt, os.path.join(output_dir, f"sr_vs_horizon_delay_{d_val}.png"), saved)
    return saved


# ---------------------------------------------------------------------------
# Plot 3: Per-combination — one figure per method combo
#   Layout: 2 rows × 4 cols
#     Row 0: SR vs Delay  (fixed s, sweep d)
#     Row 1: SR vs Horizon (fixed d=1, sweep s)
#   Curves: checkpoint variants
# ---------------------------------------------------------------------------

def generate_per_combination_plots(entries: List[Dict], output_dir: str,
                                    autoscale: bool = False, show_ci: bool = True,
                                    delay_fixed_horizon: int = 15,
                                    horizon_fixed_delay: int = 1) -> List[str]:
    """One figure per method combination (baseline/rtc × no-SM/SM).

    Each figure: 2 rows × 4 cols.
      Row 0: SR vs. Inference Delay (with execution_horizon fixed at delay_fixed_horizon)
      Row 1: SR vs. Execution Horizon (with async_delay fixed at horizon_fixed_delay)
    Curves: different checkpoint variants (or single curve if only one ckpt).
    Error bars: Wilson 95% CI.
    """
    plt, Line2D = _init_mpl()
    if plt is None:
        return []

    # Group entries by combo
    combo_entries: dict[tuple, list] = defaultdict(list)
    for e in entries:
        combo = e.get("combo")
        if combo is not None:
            combo_entries[combo].append(e)

    if not combo_entries:
        print("Warning: no combo-tagged entries for per-combination plots.")
        return []

    saved: list[str] = []

    for combo in COMBO_ORDER:
        if combo not in combo_entries:
            continue
        c_entries = combo_entries[combo]
        combo_name = COMBO_LABELS.get(combo, str(combo))
        method, sm_flag = combo

        # Collect checkpoint variants
        ckpt_labels = sorted({e.get("model", "?") for e in c_entries})
        ckpt_col, ckpt_mkr = _build_style_maps(ckpt_labels, plt)

        # ── Row 0: delay sweep (filter: execution_horizon == delay_fixed_horizon)
        delay_data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for e in c_entries:
            h = e.get("params", {}).get("execution_horizon",
                      e.get("params", {}).get("action", None))
            d = e.get("params", {}).get("async_delay")
            sr = e["data"].get("overall", {}).get("pc_successes")
            if h != delay_fixed_horizon or d is None or sr is None:
                continue
            ci = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0
            delay_data[e.get("suite", "?")][e.get("model", "?")].append((d, sr, ci))

        # ── Row 1: horizon sweep (filter: async_delay == horizon_fixed_delay)
        horiz_data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for e in c_entries:
            d  = e.get("params", {}).get("async_delay")
            h  = e.get("params", {}).get("execution_horizon",
                       e.get("params", {}).get("action", None))
            sr = e["data"].get("overall", {}).get("pc_successes")
            if d != horizon_fixed_delay or h is None or sr is None:
                continue
            ci = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0
            horiz_data[e.get("suite", "?")][e.get("model", "?")].append((h, sr, ci))

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        sm_str = " + SM" if sm_flag else ""
        fig.suptitle(
            f"{method.upper()}{sm_str}  —  Solve Rate vs. Delay  |  vs. Horizon",
            fontsize=14, fontweight="bold",
        )

        # Row labels (text on left side)
        row_labels = [
            f"vs. Delay\n(s = {delay_fixed_horizon})",
            f"vs. Horizon\n(d = {horizon_fixed_delay})",
        ]
        for row_i, row_lbl in enumerate(row_labels):
            axes[row_i][0].annotate(
                row_lbl, xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-0.32, 0.5), textcoords="axes fraction",
                ha="center", va="center", fontsize=9, style="italic",
                rotation=0,
                arrowprops=None,
            )

        for col_i, sk in enumerate(SUITE_ORDER):
            suite_label = SUITE_DISPLAY.get(sk, sk)

            # ── Row 0: delay ──────────────────────────────────────────────
            ax = axes[0][col_i]
            ax.set_title(suite_label, fontsize=11, fontweight="bold")
            all_d = sorted({d for mdl in delay_data.get(sk, {}).values()
                            for d, _, _ in mdl})
            xlim = (min(all_d) - 0.4, max(all_d) + 0.4) if all_d else None
            _style_ax(ax, "Delay  d", xticks=all_d, xlim=xlim,
                      ylim=None if autoscale else (-2, 102))

            all_ys: list = []
            for ckpt in ckpt_labels:
                pts = delay_data.get(sk, {}).get(ckpt)
                if not pts:
                    continue
                pts.sort()
                xs, ys, cis = zip(*pts)
                all_ys.extend(ys)
                ax.errorbar(xs, ys, yerr=cis if show_ci else None,
                            marker=ckpt_mkr[ckpt], color=ckpt_col[ckpt],
                            linestyle="-", lw=1.4, ms=5, capsize=3,
                            label=shorten_model_name(ckpt), zorder=2, alpha=0.9)

            if autoscale and all_ys:
                pad = max((max(all_ys) - min(all_ys)) * 0.15, 4.0)
                ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

            # ── Row 1: horizon ────────────────────────────────────────────
            ax = axes[1][col_i]
            all_h = sorted({h for mdl in horiz_data.get(sk, {}).values()
                            for h, _, _ in mdl})
            xlim = (min(all_h) - 0.4, max(all_h) + 0.4) if all_h else None
            _style_ax(ax, "Horizon  s", xticks=all_h, xlim=xlim,
                      ylim=None if autoscale else (-2, 102))

            all_ys = []
            for ckpt in ckpt_labels:
                pts = horiz_data.get(sk, {}).get(ckpt)
                if not pts:
                    continue
                pts.sort()
                xs, ys, cis = zip(*pts)
                all_ys.extend(ys)
                ax.errorbar(xs, ys, yerr=cis if show_ci else None,
                            marker=ckpt_mkr[ckpt], color=ckpt_col[ckpt],
                            linestyle="-", lw=1.4, ms=5, capsize=3,
                            label=shorten_model_name(ckpt), zorder=2, alpha=0.9)

            if autoscale and all_ys:
                pad = max((max(all_ys) - min(all_ys)) * 0.15, 4.0)
                ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

        # Legend
        handles = []
        for ckpt in ckpt_labels:
            handles.append(Line2D([0], [0], color=ckpt_col[ckpt], lw=1.4,
                                  marker=ckpt_mkr[ckpt], ms=5,
                                  label=shorten_model_name(ckpt)))
        _add_bottom_legend(fig, handles, plt, ncols=min(5, len(handles)))

        sm_suffix = "_sm" if sm_flag else ""
        fname = f"per_combo_{method}{sm_suffix}.png"
        _save(fig, plt, os.path.join(output_dir, fname), saved)

    return saved


# ---------------------------------------------------------------------------
# Plot 4: Combined overview — one figure, all combos + all suites
#   Layout: 4 rows (suites) × 2 cols (delay sweep | horizon sweep)
#   Curves: 4 method combos (fixed colours) + Wilson 95% CI
# ---------------------------------------------------------------------------

def generate_combined_overview(entries: List[Dict], output_dir: str,
                                autoscale: bool = False, show_ci: bool = True,
                                delay_fixed_horizon: int = 15,
                                horizon_fixed_delay: int = 1) -> List[str]:
    """Single overview figure: 4 suites × 2 plot types, all 4 method curves.

    Left column  : SR vs. Inference Delay  (s = delay_fixed_horizon)
    Right column : SR vs. Execution Horizon (d = horizon_fixed_delay)
    Curves       : Baseline / RTC / Baseline+SM / RTC+SM  (fixed colours)
    Error bars   : Wilson 95% CI shaded region + cap
    """
    plt, Line2D = _init_mpl()
    if plt is None:
        return []

    # ── Collect data ─────────────────────────────────────────────────────────
    # delay_series[suite][combo] = [(d, sr, ci)]
    delay_series: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    horiz_series: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))

    for e in entries:
        params = e.get("params", {})
        combo  = e.get("combo")
        sr     = e["data"].get("overall", {}).get("pc_successes")
        suite  = e.get("suite", "?")
        if combo is None or sr is None:
            continue
        ci = _wilson_ci(sr, e.get("n_total", 0)) if show_ci else 0.0

        d = params.get("async_delay")
        h = params.get("execution_horizon", params.get("action"))

        # Delay sweep: filter by fixed horizon
        if d is not None and h == delay_fixed_horizon:
            delay_series[suite][combo].append((d, sr, ci))

        # Horizon sweep: filter by fixed delay
        if h is not None and d == horizon_fixed_delay:
            horiz_series[suite][combo].append((h, sr, ci))

    has_delay  = any(bool(cd) for cd in delay_series.values())
    has_horiz  = any(bool(ch) for ch in horiz_series.values())
    if not has_delay and not has_horiz:
        print("Warning: no data for combined overview.")
        return []

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        f"Solve Rate — Overview\n"
        f"Left: vs. Delay (s={delay_fixed_horizon})  |  "
        f"Right: vs. Horizon (d={horizon_fixed_delay})",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # Column headers
    for col_i, col_title in enumerate([
        f"SR vs. Inference Delay  (s = {delay_fixed_horizon})",
        f"SR vs. Execution Horizon  (d = {horizon_fixed_delay})",
    ]):
        axes[0][col_i].set_title(col_title, fontsize=11, fontweight="bold",
                                 pad=12, loc="center")

    for row_i, sk in enumerate(SUITE_ORDER):
        suite_label = SUITE_DISPLAY.get(sk, sk)

        # Row label on left axis
        axes[row_i][0].set_ylabel(f"{suite_label}\n\nSolve Rate (%)", fontsize=10)

        # ── Left: delay sweep ─────────────────────────────────────────────
        ax_d = axes[row_i][0]
        all_d = sorted({d for combo_data in delay_series.get(sk, {}).values()
                        for d, _, _ in combo_data})
        xlim_d = (min(all_d) - 0.5, max(all_d) + 0.5) if all_d else None
        _style_ax(ax_d, "Inference Delay  d", xticks=all_d, xlim=xlim_d,
                  ylim=None if autoscale else (-2, 102))

        all_ys_d: list = []
        for combo in COMBO_ORDER:
            pts = delay_series.get(sk, {}).get(combo)
            if not pts:
                continue
            pts.sort()
            xs, ys, cis = zip(*pts)
            all_ys_d.extend(ys)
            color  = COMBO_COLORS[combo]
            marker = COMBO_MARKERS[combo]
            ls     = COMBO_LINESTYLES[combo]
            label  = COMBO_LABELS[combo]
            if show_ci and any(c > 0 for c in cis):
                ax_d.fill_between(xs,
                                  [y - c for y, c in zip(ys, cis)],
                                  [y + c for y, c in zip(ys, cis)],
                                  color=color, alpha=0.12, zorder=1)
            ax_d.errorbar(xs, ys, yerr=cis if show_ci else None,
                          marker=marker, color=color, linestyle=ls,
                          lw=1.6, ms=5.5, capsize=3,
                          label=label, zorder=3, alpha=0.92)

        if autoscale and all_ys_d:
            pad = max((max(all_ys_d) - min(all_ys_d)) * 0.15, 4.0)
            ax_d.set_ylim(min(all_ys_d) - pad, max(all_ys_d) + pad)

        # ── Right: horizon sweep ──────────────────────────────────────────
        ax_h = axes[row_i][1]
        all_h = sorted({h for combo_data in horiz_series.get(sk, {}).values()
                        for h, _, _ in combo_data})
        xlim_h = (min(all_h) - 0.5, max(all_h) + 0.5) if all_h else None
        _style_ax(ax_h, "Execution Horizon  s", xticks=all_h, xlim=xlim_h,
                  ylim=None if autoscale else (-2, 102))
        ax_h.set_ylabel("")  # shared row label from left ax

        all_ys_h: list = []
        for combo in COMBO_ORDER:
            pts = horiz_series.get(sk, {}).get(combo)
            if not pts:
                continue
            pts.sort()
            xs, ys, cis = zip(*pts)
            all_ys_h.extend(ys)
            color  = COMBO_COLORS[combo]
            marker = COMBO_MARKERS[combo]
            ls     = COMBO_LINESTYLES[combo]
            if show_ci and any(c > 0 for c in cis):
                ax_h.fill_between(xs,
                                  [y - c for y, c in zip(ys, cis)],
                                  [y + c for y, c in zip(ys, cis)],
                                  color=color, alpha=0.12, zorder=1)
            ax_h.errorbar(xs, ys, yerr=cis if show_ci else None,
                          marker=marker, color=color, linestyle=ls,
                          lw=1.6, ms=5.5, capsize=3,
                          label=COMBO_LABELS[combo], zorder=3, alpha=0.92)

        if autoscale and all_ys_h:
            pad = max((max(all_ys_h) - min(all_ys_h)) * 0.15, 4.0)
            ax_h.set_ylim(min(all_ys_h) - pad, max(all_ys_h) + pad)

    # ── Global legend ─────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color=COMBO_COLORS[c], lw=1.6,
               linestyle=COMBO_LINESTYLES[c],
               marker=COMBO_MARKERS[c], ms=5.5, label=COMBO_LABELS[c])
        for c in COMBO_ORDER
    ]
    _add_bottom_legend(fig, handles, plt, ncols=4)

    saved: list[str] = []
    _save(fig, plt, os.path.join(output_dir, "overview_combined.png"), saved)
    return saved


# ---------------------------------------------------------------------------
# Table printing (unchanged)
# ---------------------------------------------------------------------------

def shorten_task(name: str) -> str:  # re-defined with correct import scope
    s = name
    for prefix in ("pick_up_the_", "open_the_", "close_the_", "put_the_", "turn_on_the_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.replace("_and_place_it_in_the_basket", " → basket")
    s = s.replace("_and_place_it_", " → ")
    return s.replace("_", " ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parse and plot LIBERO eval results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", nargs="?", default="outputs/eval",
                    help="Root directory to scan for eval_results.json")
    ap.add_argument("--tasks",        action="store_true")
    ap.add_argument("--sort-by",      type=str, default=None)
    ap.add_argument("--metric",       type=str, default="pc_successes",
                    choices=list(METRIC_LABELS))
    ap.add_argument("--csv",          action="store_true")

    # ── Plot flags ────────────────────────────────────────────────────────────
    ap.add_argument("--plot-delay",   action="store_true", default=True,
                    help="SR vs async_delay (one figure per horizon)")
    ap.add_argument("--no-plot-delay", action="store_false", dest="plot_delay")
    ap.add_argument("--plot-horizon", action="store_true", default=False,
                    help="SR vs horizon (one figure per delay)")
    ap.add_argument("--plot-combo",   action="store_true", default=False,
                    help="Per-combination figures (2-row × 4-col: delay + horizon rows)")
    ap.add_argument("--plot-overview", action="store_true", default=False,
                    help="Single combined overview figure (4-suite × 2-plot-type)")
    ap.add_argument("--all-plots",    action="store_true", default=False,
                    help="Enable all four plot types")

    # ── Style flags ───────────────────────────────────────────────────────────
    ap.add_argument("--autoscale",    action="store_true",
                    help="Autoscale y-axis per subplot")
    ap.add_argument("--ci",           action="store_true", default=True,
                    help="Show Wilson 95%% CI error bars (default on)")
    ap.add_argument("--no-ci",        action="store_false", dest="ci")

    # ── Fixed values for combo/overview ──────────────────────────────────────
    ap.add_argument("--delay-sweep-horizon", type=int, default=15,
                    help="Fixed horizon for delay sweep plots (default 15)")
    ap.add_argument("--horizon-sweep-delay", type=int, default=1,
                    help="Fixed delay for horizon sweep plots (default 1)")

    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    if args.all_plots:
        args.plot_delay   = True
        args.plot_horizon = True
        args.plot_combo   = True
        args.plot_overview = True

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory")
        return 1

    plot_dir = os.path.abspath(args.output_dir) if args.output_dir else root
    os.makedirs(plot_dir, exist_ok=True)

    eval_files = find_eval_results(root)
    if not eval_files:
        print(f"No eval_results.json found under {root}")
        return 1

    entries = []
    for ef in eval_files:
        info = extract_path_info(ef, root)
        try:
            info["data"] = load_results(ef)
            adapt_it_rtc_entry(info)
            entries.append(info)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: could not load {ef}: {e}")

    if not entries:
        print("No valid results loaded.")
        return 1

    metric       = args.metric
    metric_label = METRIC_LABELS[metric]
    show_ci      = args.ci

    # ── Text tables ───────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for e in entries:
        groups[(e.get("model", ""), e.get("suite", ""))].append(e)

    all_params = sorted({p for e in entries for p in e.get("params", {})})
    sort_key   = args.sort_by if args.sort_by in all_params else None

    print(f"Found {len(entries)} result(s) under: {root}\n")

    for (model, suite), grp in sorted(groups.items()):
        grp.sort(key=lambda e: tuple(e.get("params", {}).get(p, 0) for p in all_params)
                 if not sort_key else lambda e: e.get("params", {}).get(sort_key, 0))
        parts = []
        if model:
            parts.append(f"Model: {model}")
        if suite:
            parts.append(f"Suite: {suite}")
        title = "  |  ".join(parts) or "Results"
        if not args.csv:
            print("=" * max(60, len(title)))
            print(title)
            print("=" * max(60, len(title)))

        headers = (all_params + [metric_label]
                   + [METRIC_LABELS[m] for m in METRIC_LABELS if m != metric]
                   + ["Eval s"])
        other   = [m for m in METRIC_LABELS if m != metric]
        rows    = []
        for e in grp:
            row = [e.get("params", {}).get(p, "-") for p in all_params]
            ov  = e["data"].get("overall", {})
            row.append(ov.get(metric, "N/A"))
            for m in other:
                row.append(ov.get(m, "N/A"))
            es = e["data"].get("eval_s")
            row.append(f"{es:.0f}" if isinstance(es, (int, float)) else "N/A")
            rows.append(row)

        align = (["right"] * len(all_params)
                 + ["right"] * (1 + len(other))
                 + ["right"])
        if args.csv:
            print(",".join(map(str, headers)))
            for row in rows:
                print(",".join(map(str, row)))
        else:
            print_table(headers, rows, align)
        print()

        if args.tasks:
            task_names = sorted({k for e in grp for k in e["data"]
                                  if k not in ("overall", "eval_s", "video_paths", "config")})
            if task_names:
                cfg_labels = [e.get("config", "?") for e in grp]
                t_headers  = ["Task"] + cfg_labels
                t_rows     = [[shorten_task(t)] +
                               [e["data"].get(t, {}).get(metric, "-") for e in grp]
                               for t in task_names]
                t_rows.append(["OVERALL"] +
                               [e["data"].get("overall", {}).get(metric, "-") for e in grp])
                if not args.csv:
                    print(f"  Per-task {metric_label}:\n")
                    print_table(t_headers, t_rows,
                                ["left"] + ["right"] * len(grp), indent=2)
                print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    kw = dict(autoscale=args.autoscale, show_ci=show_ci)

    if args.plot_delay:
        print(f"\nGenerating SR-vs-delay plots → {plot_dir}")
        generate_delay_plots(entries, plot_dir, **kw)

    if args.plot_horizon:
        print(f"\nGenerating SR-vs-horizon plots → {plot_dir}")
        generate_horizon_plots(entries, plot_dir, **kw)

    if args.plot_combo:
        print(f"\nGenerating per-combination plots → {plot_dir}")
        generate_per_combination_plots(
            entries, plot_dir,
            delay_fixed_horizon=args.delay_sweep_horizon,
            horizon_fixed_delay=args.horizon_sweep_delay,
            **kw,
        )

    if args.plot_overview:
        print(f"\nGenerating combined overview → {plot_dir}")
        generate_combined_overview(
            entries, plot_dir,
            delay_fixed_horizon=args.delay_sweep_horizon,
            horizon_fixed_delay=args.horizon_sweep_delay,
            **kw,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
