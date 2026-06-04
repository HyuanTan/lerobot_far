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

"""RTC-specific timing analysis for async-inference client/server records.

Loads the RTC-augmented JSONL files produced by base_client.py when timing is
enabled and generates analyses 4–7:

  Analysis 4 — Leftover pathway health
      Are leftover_steps sent with obs?  Does leftover_steps track _orig_buf state?
      Large or zero leftover_steps both indicate problems (over/under-RTC).

  Analysis 5 — Chunk boundary continuity
      Original action L2 norms per chunk — sudden jumps indicate discontinuities
      that RTC should prevent.

  Analysis 6 — Aggregate function corruption
      When n_overlap > 0, diff_l2_mean reveals how much blending distorts the
      chunk.  High diff_l2_mean with weighted_average = corrupted RTC guidance.

  Analysis 7 — (Placeholder) Robot tracking error
      Requires robot_client to log commanded vs feedback (not yet instrumented).
      Prints a notice directing the user to enable that logging.

Usage::

    # Auto-discover timing dirs under an eval output directory
    python -m lerobot.async_inference.analyze_rtc \\
        ~/outputs/eval/libero/pi05/sim/libero_spatial

    # Explicit directory
    python -m lerobot.async_inference.analyze_rtc \\
        --client_dir ./client_timing \\
        --out_dir    ./rtc_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 120)

# ── Colour palette (consistent with analyze_timing.py) ────────────────────────

_C = {
    "obs_prep":    "#4C9BE8",
    "infer":       "#D946EF",
    "postprocess": "#F472B6",
    "queue_wait":  "#E87D4C",
    "net_c2s":     "#F5A623",
    "other":       "#9CA3AF",
    "green":       "#22C55E",
    "red":         "#EF4444",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_json(path, lines=True, convert_dates=False)
            return df if len(df) > 0 else None
        except Exception:
            return None
    return None


def _auto_discover(eval_dir: str) -> Path | None:
    """Return the client_timing sub-dir under eval_dir if it exists."""
    ed = Path(eval_dir)
    cd = ed / "client_timing"
    return cd if cd.is_dir() else None


def load_data(client_dir: str | Path | None) -> dict[str, pd.DataFrame | None]:
    cd = Path(client_dir) if client_dir else None
    names = {
        "chunk_action": "client_chunk_action_records.jsonl",
        "aggregate":    "client_aggregate_records.jsonl",
        "sent":         "client_obs_sent_records.jsonl",
        "chunk_recv":   "client_chunk_recv_records.jsonl",
    }
    data: dict[str, pd.DataFrame | None] = {}
    for key, fname in names.items():
        df = _load(cd / fname) if cd else None
        data[key] = df
        status = f"{len(df):5d} records" if df is not None else "(not found)"
        print(f"  Loaded {key:<14s}: {status}")
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(title: str = "", width: int = 80):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


def _savefig(fig: plt.Figure, path: Path, label: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _infer_fps(data: dict, fallback: float = 30.0) -> float:
    sent = data.get("sent")
    if sent is not None and "wall_time" in sent.columns and len(sent) > 10:
        diffs = np.diff(np.sort(sent["wall_time"].values))
        diffs = diffs[(diffs > 0.005) & (diffs < 0.5)]
        if len(diffs) > 5:
            return float(1.0 / np.median(diffs))
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4 — Leftover pathway health
# ══════════════════════════════════════════════════════════════════════════════

def print_leftover_health(data: dict):
    """Analysis 4: Are leftover actions being sent? Is leftover_steps reasonable?"""
    _divider("ANALYSIS 4: LEFTOVER PATHWAY HEALTH", 80)

    ca = data.get("chunk_action")
    sent = data.get("sent")

    if ca is None:
        print("  (no client_chunk_action records — enable timing and ensure base_client is updated)")
        return

    n_total = len(ca)
    n_rtc_active = int(ca["has_original_actions"].sum()) if "has_original_actions" in ca.columns else 0
    rtc_frac = n_rtc_active / n_total if n_total > 0 else 0.0

    print(f"  total chunks recorded  : {n_total}")
    print(f"  RTC active (orig acts) : {n_rtc_active}  ({rtc_frac:.1%})")

    if "leftover_steps" in ca.columns:
        ls = ca["leftover_steps"]
        print(f"\n  leftover_steps stats (pre-chunk _orig_buf size):")
        print(f"    mean={ls.mean():.1f}  p50={np.percentile(ls, 50):.0f}  "
              f"p95={np.percentile(ls, 95):.0f}  max={ls.max():.0f}")
        print(f"    zero-leftover chunks: {(ls == 0).sum()}  ({(ls == 0).mean():.1%})")

        if (ls == 0).mean() > 0.8 and rtc_frac > 0.5:
            print("\n  ⚠  >80% of chunks have leftover_steps=0 despite RTC being active.")
            print("     This means the client is not bundling leftover actions in obs.")
            print("     Check: _orig_buf population in receive_actions() and control_loop_observation().")
        elif (ls == 0).mean() < 0.1:
            print("\n  △  Almost all chunks carry leftover — verify chunk_size_threshold is ≥0.5.")

    # Cross-check with sent records
    if sent is not None and "leftover_steps" in sent.columns:
        ls_sent = sent["leftover_steps"]
        print(f"\n  sent obs leftover_steps (from ClientObsSentRecord):")
        print(f"    mean={ls_sent.mean():.1f}  p50={np.percentile(ls_sent, 50):.0f}  "
              f"p95={np.percentile(ls_sent, 95):.0f}  max={ls_sent.max():.0f}")
        print(f"    zero-leftover obs: {(ls_sent == 0).sum()}  ({(ls_sent == 0).mean():.1%})")

    # Show action execution lag from chunk_recv records (how long before first action executes)
    chunk_recv = data.get("chunk_recv")
    if chunk_recv is not None and "estimated_first_exec_lag_ms" in chunk_recv.columns:
        lag = chunk_recv["estimated_first_exec_lag_ms"]
        print(f"\n  estimated_first_exec_lag_ms (queue_depth × dt at receipt):")
        print(f"    mean={lag.mean():.1f}  p50={np.percentile(lag, 50):.1f}  "
              f"p95={np.percentile(lag, 95):.1f}  max={lag.max():.1f}  ms")
        zero_frac = (lag == 0).mean()
        if zero_frac > 0.5:
            print(f"    ✓  {zero_frac:.0%} of chunks arrive to an empty queue → low exec lag.")
        elif lag.mean() > 200:
            print(f"    ⚠  Mean exec lag {lag.mean():.0f}ms — actions sit in queue a long time before "
                  "execution.  Consider using latest_only aggregate_fn or increasing fps.")

    if "orig_action_l2_mean" in ca.columns and n_rtc_active > 0:
        rtc_rows = ca[ca["has_original_actions"]]
        l2m = rtc_rows["orig_action_l2_mean"]
        print(f"\n  original_actions L2 norm (when RTC active):")
        print(f"    mean={l2m.mean():.3f}  p50={np.percentile(l2m, 50):.3f}  "
              f"p95={np.percentile(l2m, 95):.3f}  max={l2m.max():.3f}")

        # Infer whether original_actions are in model-space (normalized, L2≈small) or
        # absolute joint-position space (L2≈larger, Bug-1 fix for relative-action policies).
        # Heuristic: absolute coords for a typical 6-DOF robot at ~0.3–1.0 rad/joint → L2≈1–3.
        _coord_space = "absolute (relative-action policy)" if l2m.mean() > 0.8 else "model-space (normalized)"
        print(f"    inferred coord space  : {_coord_space}")
        if _coord_space.startswith("absolute"):
            print("    Note: L2 norms reflect robot joint positions, not normalized deltas.")
            print("    High variance here is expected (robot moves during task).")
            print("    Use Analysis 5 jump detection for continuity checks — it is self-calibrating.")
        elif l2m.std() / (l2m.mean() + 1e-6) > 0.5:
            print("  △  High variance in model-space L2 — chunk continuity may be inconsistent.")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5 — Chunk boundary continuity
# ══════════════════════════════════════════════════════════════════════════════

def print_chunk_continuity(data: dict):
    """Analysis 5: Original action L2 norms per chunk — discontinuities show RTC failure."""
    _divider("ANALYSIS 5: CHUNK BOUNDARY CONTINUITY", 80)

    ca = data.get("chunk_action")
    if ca is None or "orig_action_l2_mean" not in ca.columns:
        print("  (no chunk_action records with orig_action_l2_mean)")
        return

    rtc_rows = ca[ca.get("has_original_actions", pd.Series([False] * len(ca)))]
    if len(rtc_rows) == 0:
        print("  (no chunks with original_actions — RTC may be disabled on server)")
        return

    l2m = rtc_rows["orig_action_l2_mean"]
    l2x = rtc_rows["orig_action_l2_max"] if "orig_action_l2_max" in rtc_rows.columns else l2m

    # Detect jumps: consecutive chunks where L2 changes >2σ
    diffs = np.abs(np.diff(l2m.values))
    threshold = 2 * float(diffs.std()) + float(diffs.mean())
    n_jumps = int((diffs > threshold).sum())

    print(f"  RTC-active chunks     : {len(rtc_rows)}")
    print(f"  orig_action_l2_mean   : mean={l2m.mean():.3f}  std={l2m.std():.3f}  "
          f"p50={np.percentile(l2m, 50):.3f}  max={l2m.max():.3f}")
    print(f"  orig_action_l2_max    : mean={l2x.mean():.3f}  max={l2x.max():.3f}")
    print(f"  boundary jump events  : {n_jumps}  (|ΔL2| > {threshold:.3f} = mean+2σ of diffs)")

    if n_jumps > len(rtc_rows) * 0.1:
        print(f"\n  ⚠  >10% of chunk boundaries show L2 jumps → discontinuous action sequences.")
        print(f"     RTC should produce smooth transitions — jumps suggest misaligned leftover or")
        print(f"     infer_delay pointing to a timestep far from the current action chunk window.")
    elif n_jumps == 0:
        print(f"\n  ✓  No boundary jumps — original_actions are smooth across chunks.")
    else:
        print(f"\n  △  A few boundary jumps — investigate if they correlate with starvation events.")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 6 — Aggregate function corruption
# ══════════════════════════════════════════════════════════════════════════════

def print_aggregate_corruption(data: dict):
    """Analysis 6: Overlap region corruption from weighted-average blending."""
    _divider("ANALYSIS 6: AGGREGATE FUNCTION CORRUPTION", 80)

    agg = data.get("aggregate")
    if agg is None:
        print("  (no client_aggregate records — enable timing and ensure base_client is updated)")
        return

    n_total   = len(agg)
    n_overlap = int((agg["n_overlap"] > 0).sum()) if "n_overlap" in agg.columns else 0
    n_new_only = n_total - n_overlap

    print(f"  total merge events    : {n_total}")
    print(f"  with overlap          : {n_overlap}  ({n_overlap/n_total:.1%})")
    print(f"  no overlap (new-only) : {n_new_only}  ({n_new_only/n_total:.1%})")

    if "aggregate_fn_name" in agg.columns:
        fn_counts = agg["aggregate_fn_name"].value_counts()
        print(f"\n  aggregate_fn used:")
        for fn, cnt in fn_counts.items():
            print(f"    {fn}: {cnt} ({cnt/n_total:.1%})")

    if n_overlap > 0 and "diff_l2_mean" in agg.columns:
        overlap_rows = agg[agg["n_overlap"] > 0]
        dl2 = overlap_rows["diff_l2_mean"]
        ol2 = overlap_rows["old_l2_mean"]  if "old_l2_mean"  in agg.columns else dl2
        nl2 = overlap_rows["new_l2_mean"]  if "new_l2_mean"  in agg.columns else dl2
        overlap_sz = overlap_rows["n_overlap"] if "n_overlap" in agg.columns else None

        print(f"\n  Overlap region stats (n={len(overlap_rows)}):")
        print(f"    old_l2_mean  : {ol2.mean():.3f}  (L2 of existing queued actions)")
        print(f"    new_l2_mean  : {nl2.mean():.3f}  (L2 of incoming chunk actions)")
        print(f"    diff_l2_mean : {dl2.mean():.3f}  (||old - new|| = blend corruption proxy)")
        if overlap_sz is not None:
            print(f"    overlap size : mean={overlap_sz.mean():.1f}  max={overlap_sz.max():.0f}")

        # Corruption severity: diff normalized by new (how much the blend moves the action)
        corr_frac = dl2 / (nl2 + 1e-6)
        print(f"\n  Corruption fraction   : mean={corr_frac.mean():.3f}  "
              f"p95={np.percentile(corr_frac, 95):.3f}")

        fn_name = agg["aggregate_fn_name"].iloc[0] if "aggregate_fn_name" in agg.columns else "unknown"
        if corr_frac.mean() > 0.1 and fn_name != "latest_only":
            print(f"\n  ⚠  Mean corruption fraction {corr_frac.mean():.2f} with '{fn_name}'.")
            print(f"     weighted_average blends 30% old + 70% new in overlap — this corrupts")
            print(f"     RTC's designed prefix continuity.  Use latest_only to disable blending.")
        elif fn_name == "latest_only":
            print(f"\n  ✓  latest_only: no blending corruption (diff_l2 is 0 for overlap region).")
        elif corr_frac.mean() < 0.05:
            print(f"\n  ✓  Corruption fraction low (<5%) — blending minimal impact.")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 7 — Robot tracking error (placeholder)
# ══════════════════════════════════════════════════════════════════════════════

def print_tracking_error_notice():
    """Analysis 7: Robot feedback tracking error — explains how to enable."""
    _divider("ANALYSIS 7: ROBOT TRACKING ERROR", 80)
    print("  Tracking error analysis (commanded vs feedback motor positions) requires")
    print("  additional logging in robot_client.py::control_loop_action().")
    print()
    print("  To enable:")
    print("    1. Add RobotFeedbackRecord to timing.py:")
    print("         wall_time, episode, timestep")
    print("         commanded_action:  list[float]  (motor positions sent to robot)")
    print("         feedback_action:   list[float]  (motor positions read from robot)")
    print("         tracking_error_l2: float        (||commanded - feedback||_2)")
    print()
    print("    2. In robot_client.py::control_loop_action(), after reading robot feedback,")
    print("       add:  self._feedback_recorder.add(RobotFeedbackRecord(...))")
    print()
    print("    3. Load 'client_robot_feedback_records.jsonl' here to plot per-joint")
    print("       tracking error over time and detect action-delay artifacts.")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_leftover_health(data: dict, fps: float, out_dir: Path, chunk_size: int = 50):
    """Fig 1: leftover_steps per chunk + orig_action_l2 per chunk over time.

    Added Panel 3: RTC chunk region decomposition — old / transition / freed.

      old        = infer_delay_used steps  (guided region, matching prev chunk)
      overlap    = n_overlap steps from aggregate records (queue merge perspective)
      freed      = chunk_size - infer_delay_used  (transition + freed, new policy output)

    Ideal: n_overlap ≈ infer_delay_used (buffer depth ≈ latency estimate).
    If n_overlap >> infer_delay_used: chunk arrived much earlier than expected.
    If n_overlap << infer_delay_used: chunk arrived late, buffer partly exhausted.

    Note: split between 'transition' and 'freed' within the non-old region requires
    the execution_horizon config value, which is not logged in client records.
    Pass ``chunk_size`` to match actions_per_chunk (default 50).
    """
    ca = data.get("chunk_action")
    if ca is None or "wall_time" not in ca.columns:
        return

    t0 = float(ca["wall_time"].min())
    ca = ca.copy()
    ca["t"] = ca["wall_time"] - t0

    has_l2 = "orig_action_l2_mean" in ca.columns
    agg = data.get("aggregate")
    has_agg = (agg is not None and "wall_time" in agg.columns
               and "n_overlap" in agg.columns and "n_new" in agg.columns)
    has_infer_delay = "infer_delay_used" in ca.columns
    # Panel 3 requires both agg overlap and infer_delay_used from chunk_action records.
    has_decomp = has_infer_delay and has_agg

    n_panels = (2 + int(has_l2) + int(has_decomp))
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3 * n_panels), sharex=True)
    axes = list(axes)

    # Panel 0: leftover_steps per chunk
    ax = axes[0]
    if "leftover_steps" in ca.columns:
        rtc_mask = ca.get("has_original_actions", pd.Series([False] * len(ca))).values
        colors = [_C["obs_prep"] if m else _C["other"] for m in rtc_mask]
        ax.bar(ca["t"], ca["leftover_steps"], width=0.2, color=colors, alpha=0.8, edgecolor="none")
        ax.set_ylabel("leftover_steps")
        legend_elems = [
            mpatches.Patch(color=_C["obs_prep"], label="RTC active (has_original_actions=True)"),
            mpatches.Patch(color=_C["other"],    label="RTC inactive"),
        ]
        ax.legend(handles=legend_elems, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No leftover_steps data", transform=ax.transAxes, ha="center")
    ax.set_title("Analysis 4: Leftover Steps per Chunk\n"
                 "(should increase as chunk ages; 0 = leftover pathway broken)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 1: n_overlap (from aggregate records)
    ax = axes[1]
    agg = data.get("aggregate")
    if agg is not None and "wall_time" in agg.columns and "n_overlap" in agg.columns:
        agg = agg.copy()
        agg["t"] = agg["wall_time"] - t0
        ax.bar(agg["t"], agg["n_overlap"], width=0.2, color=_C["queue_wait"], alpha=0.8,
               label="n_overlap (steps blended)")
        ax.bar(agg["t"], agg["n_new"], bottom=agg["n_overlap"], width=0.2,
               color=_C["obs_prep"], alpha=0.6, label="n_new (steps appended)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No aggregate data", transform=ax.transAxes, ha="center")
    ax.set_title("Overlap vs new steps per merge (analysis 6 overview)", fontweight="bold")
    ax.set_ylabel("Steps")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: original_actions L2 over time
    panel_idx = 2
    if has_l2:
        ax = axes[panel_idx]
        panel_idx += 1
        rtc_rows = ca[ca.get("has_original_actions", pd.Series([False] * len(ca))).values]
        if len(rtc_rows) > 0:
            ax.plot(rtc_rows["t"], rtc_rows["orig_action_l2_mean"], color=_C["infer"],
                    linewidth=1.0, alpha=0.85, label="orig_action_l2_mean")
            if "orig_action_l2_max" in rtc_rows.columns:
                ax.fill_between(rtc_rows["t"], rtc_rows["orig_action_l2_mean"],
                                rtc_rows["orig_action_l2_max"],
                                alpha=0.2, color=_C["infer"], label="L2 range [mean, max]")
            ax.legend(fontsize=8)
        ax.set_title("Analysis 5: Original Action L2 Norm per Chunk\n"
                     "(large jumps = boundary discontinuities)", fontweight="bold")
        ax.set_ylabel("L2 norm")
        ax.grid(alpha=0.3)

    # Panel 3: RTC chunk region decomposition — old / freed + n_overlap alignment
    if has_decomp:
        ax = axes[panel_idx]

        # Align chunk_action (infer_delay_used) with aggregate (n_overlap, n_new)
        # via wall_time proximity (both record per-chunk events).
        agg2 = agg.copy()
        agg2["t"] = agg2["wall_time"] - t0
        # Sort both by time and merge-asof so each chunk_action row gets the
        # nearest aggregate row (they fire at the same wall_time per chunk).
        ca_sorted  = ca.sort_values("t").reset_index(drop=True)
        agg_sorted = agg2.sort_values("t").reset_index(drop=True)
        merged = pd.merge_asof(
            ca_sorted[["t", "infer_delay_used"]],
            agg_sorted[["t", "n_overlap", "n_new"]],
            on="t", direction="nearest", tolerance=1.0,
        ).dropna(subset=["n_overlap", "n_new"])

        if len(merged) > 0:
            t_vals = merged["t"].values
            old_steps    = merged["infer_delay_used"].values.clip(0, chunk_size)
            freed_steps  = (chunk_size - old_steps).clip(0)
            n_ov         = merged["n_overlap"].values.clip(0)
            n_nw         = merged["n_new"].values.clip(0)

            bar_w = max(0.15, float(np.diff(t_vals).mean()) * 0.4) if len(t_vals) > 1 else 0.3

            # Stacked bar: old (guided) + freed (new policy output)
            ax.bar(t_vals, old_steps,   width=bar_w, color=_C["queue_wait"],
                   alpha=0.85, label=f"old (infer_delay_used, guided prefix)")
            ax.bar(t_vals, freed_steps, width=bar_w, bottom=old_steps,
                   color=_C["obs_prep"], alpha=0.6, label="freed (new policy output)")

            # Overlay n_overlap dots to show queue-merge alignment
            ax.scatter(t_vals, n_ov, s=14, color=_C["infer"], alpha=0.8, zorder=5,
                       linewidths=0, label="n_overlap (queue merge, ≈ old if calibrated)")

            mean_old = float(np.mean(old_steps))
            mean_ov  = float(np.mean(n_ov))
            ax.axhline(mean_old, linestyle="--", color=_C["queue_wait"], linewidth=1,
                       alpha=0.7, label=f"mean old={mean_old:.1f} steps")
            ax.axhline(mean_ov,  linestyle=":",  color=_C["infer"], linewidth=1,
                       alpha=0.7, label=f"mean n_overlap={mean_ov:.1f} steps")

            # Annotation: calibration quality (ideal: n_overlap ≈ old)
            delta = mean_ov - mean_old
            sign  = "+" if delta >= 0 else ""
            ax.text(0.02, 0.95,
                    f"n_overlap − old = {sign}{delta:.1f} steps\n"
                    f"  ≈0 → calibrated  |  >0 → chunk early  |  <0 → chunk late/starvation",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8})
            ax.legend(fontsize=8, ncol=2)
        else:
            ax.text(0.5, 0.5, "No aligned data (chunk_action / aggregate mismatch)",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_title(
            "Analysis 4b: RTC Chunk Region Decomposition — old / freed + queue-merge alignment\n"
            "(old = guided prefix = infer_delay_used;  freed = chunk_size − old;\n"
            " n_overlap dots show actual queue-merge depth — should ≈ old when calibrated)",
            fontweight="bold")
        ax.set_ylabel("Steps")
        ax.set_ylim(0, chunk_size + 2)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Wall-clock time (s from start)")
    plt.tight_layout()
    _savefig(fig, out_dir / "rtc_fig1_leftover_health.png", "leftover_health")


def plot_aggregate_corruption(data: dict, out_dir: Path):
    """Fig 2: Overlap diff_l2_mean over time — shows blending corruption per chunk."""
    agg = data.get("aggregate")
    if agg is None or "diff_l2_mean" not in agg.columns or "wall_time" not in agg.columns:
        return

    agg = agg.copy()
    t0 = float(agg["wall_time"].min())
    agg["t"] = agg["wall_time"] - t0
    overlap_rows = agg[agg["n_overlap"] > 0] if "n_overlap" in agg.columns else agg

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

    # Panel 0: diff_l2_mean over time
    ax = axes[0]
    if len(overlap_rows) > 0:
        ax.scatter(overlap_rows["t"], overlap_rows["diff_l2_mean"],
                   s=20, color=_C["queue_wait"], alpha=0.7, linewidths=0,
                   label="diff_l2_mean (||old - new||, overlap region)")
        ax.axhline(float(overlap_rows["diff_l2_mean"].mean()), linestyle="--", color="gray",
                   linewidth=1, label=f"mean={overlap_rows['diff_l2_mean'].mean():.3f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No overlap events", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("diff_l2_mean")
    ax.set_title("Analysis 6: Blend Corruption Over Time\n"
                 "(high = old/new actions differ greatly; with weighted_avg this distorts RTC guidance)",
                 fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel 1: histogram of diff_l2_mean
    ax = axes[1]
    if len(overlap_rows) > 0:
        vals = overlap_rows["diff_l2_mean"].dropna().values
        ax.hist(vals, bins=40, color=_C["queue_wait"], alpha=0.8, edgecolor="white")
        ax.axvline(float(np.percentile(vals, 95)), linestyle="--", color="red",
                   label=f"p95={np.percentile(vals, 95):.3f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("diff_l2_mean per chunk (||old - new|| in overlap region)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of blend diff L2", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, out_dir / "rtc_fig2_aggregate_corruption.png", "aggregate_corruption")


def plot_rtc_timeline(data: dict, fps: float, out_dir: Path):
    """Fig 3: 4-panel RTC health timeline combining all RTC-specific signals."""
    ca  = data.get("chunk_action")
    agg = data.get("aggregate")
    sent = data.get("sent")

    all_wt = []
    for df in [ca, agg, sent]:
        if df is not None and "wall_time" in df.columns:
            all_wt.extend(df["wall_time"].values.tolist())
    if not all_wt:
        return
    t0 = min(all_wt)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes = list(axes)

    # Panel 0: RTC active flag per chunk
    ax = axes[0]
    if ca is not None and "has_original_actions" in ca.columns and "wall_time" in ca.columns:
        t = ca["wall_time"].values - t0
        rtc = ca["has_original_actions"].values.astype(bool)
        ax.scatter(t[rtc],  np.ones(int(rtc.sum())),   c=_C["obs_prep"], s=20, alpha=0.8,
                   linewidths=0, label="RTC active")
        ax.scatter(t[~rtc], np.zeros(int((~rtc).sum())), c=_C["other"],    s=10, alpha=0.4,
                   linewidths=0, label="RTC inactive")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["inactive", "active"], fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("RTC\nactive", fontsize=9, rotation=0, labelpad=40)
    ax.set_title("RTC Health Timeline", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    # Panel 1: leftover_steps per chunk
    ax = axes[1]
    if ca is not None and "leftover_steps" in ca.columns and "wall_time" in ca.columns:
        t  = ca["wall_time"].values - t0
        ls = ca["leftover_steps"].values
        ax.bar(t, ls, width=0.3, color=_C["obs_prep"], alpha=0.75, edgecolor="none")
        ax.axhline(float(ls.mean()), linestyle="--", color="gray", linewidth=1,
                   label=f"mean={ls.mean():.1f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("leftover\nsteps", fontsize=9, rotation=0, labelpad=40)
    ax.set_title("Leftover steps per chunk (analysis 4)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: n_overlap + diff_l2 from aggregate
    ax = axes[2]
    if agg is not None and "wall_time" in agg.columns:
        t_agg = agg["wall_time"].values - t0
        if "diff_l2_mean" in agg.columns:
            ax.bar(t_agg, agg["diff_l2_mean"].values, width=0.3,
                   color=_C["queue_wait"], alpha=0.8, edgecolor="none", label="diff_l2_mean")
        if "n_overlap" in agg.columns:
            ax2 = ax.twinx()
            ax2.plot(t_agg, agg["n_overlap"].values, color=_C["infer"],
                     linewidth=1.0, alpha=0.7, label="n_overlap")
            ax2.set_ylabel("n_overlap", fontsize=8, color=_C["infer"])
        ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("diff_l2\n(corruption)", fontsize=9, rotation=0, labelpad=40)
    ax.set_title("Blend corruption + overlap steps (analysis 6)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: orig_action_l2_mean per chunk (continuity signal)
    ax = axes[3]
    if ca is not None and "orig_action_l2_mean" in ca.columns and "wall_time" in ca.columns:
        rtc_rows = ca[ca.get("has_original_actions", pd.Series([False] * len(ca))).values]
        if len(rtc_rows) > 0:
            t_rtc = rtc_rows["wall_time"].values - t0
            ax.plot(t_rtc, rtc_rows["orig_action_l2_mean"].values, color=_C["postprocess"],
                    linewidth=1.0, alpha=0.9, label="orig_action_l2_mean")
            ax.set_ylabel("orig L2\n(continuity)", fontsize=9, rotation=0, labelpad=40)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No RTC-active chunks", transform=ax.transAxes, ha="center")
    ax.set_title("Original action L2 per chunk (analysis 5 — jumps = discontinuities)", fontweight="bold")
    ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Wall-clock time (s from start)")
    plt.tight_layout()
    _savefig(fig, out_dir / "rtc_fig3_timeline.png", "rtc_timeline")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="RTC-specific timing analysis (analyses 4–7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "eval_dir", nargs="?", type=str, default=None,
        help="Evaluation output directory (auto-discovers client_timing/ subdir)",
    )
    ap.add_argument("--client_dir", type=str, default=None,
                    help="Directory with client timing JSONL files (overrides eval_dir)")
    ap.add_argument("--out_dir",    type=str, default=None,
                    help="Output directory for PNG figures (default: <eval_dir>/rtc_analysis)")
    ap.add_argument("--fps",        type=float, default=None,
                    help="Control-loop fps (default: auto-detected from obs send intervals)")
    ap.add_argument("--no_plots",   action="store_true",
                    help="Skip figure generation (tables only)")
    args = ap.parse_args()

    client_dir = args.client_dir
    if args.eval_dir and not client_dir:
        cd = _auto_discover(args.eval_dir)
        client_dir = str(cd) if cd else None
        if client_dir is None:
            # Fall back to eval_dir itself if it directly contains the files
            client_dir = args.eval_dir

    if client_dir is None:
        ap.error("Provide eval_dir (positional) or --client_dir")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.eval_dir:
        out_dir = Path(args.eval_dir) / "rtc_analysis"
    else:
        out_dir = Path("./rtc_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 80)
    print("  RTC Timing Analysis (analyses 4–7)")
    print("═" * 80)
    print(f"  client_dir : {client_dir}")
    print(f"  out_dir    : {out_dir}")

    print("\nLoading records:")
    data = load_data(client_dir)

    if all(v is None for v in data.values()):
        print("ERROR: No RTC timing files found.")
        print("       Ensure timing is enabled (--timing_output_dir) and base_client is updated.")
        sys.exit(1)

    fps = args.fps if args.fps is not None else _infer_fps(data)
    print(f"\n  fps (for analyses): {fps:.1f}"
          + ("  (auto-detected)" if args.fps is None else "  (from --fps)"))

    # ── Console analyses ───────────────────────────────────────────────────────
    print_leftover_health(data)
    print_chunk_continuity(data)
    print_aggregate_corruption(data)
    print_tracking_error_notice()

    if args.no_plots:
        return

    # ── Figures ────────────────────────────────────────────────────────────────
    _divider("GENERATING FIGURES", 80)
    plot_leftover_health(data, fps, out_dir)
    plot_aggregate_corruption(data, out_dir)
    plot_rtc_timeline(data, fps, out_dir)

    print(f"\n  All figures saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
