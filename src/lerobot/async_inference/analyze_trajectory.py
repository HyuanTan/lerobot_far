#!/usr/bin/env python

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
Visualise SO-101 trajectory JSON files recorded by RobotClient / SmartRobotClient.

Produces four PNG figures (always auto-saved, suffixed _ee3d / _ee_deriv /
_joint_pos / _joint_deriv):

  Fig 1 — 3-D EE trajectory
      blue gradient = received action chunks
      red           = executed actions (raw policy)
      orange dashed = executed interpolated sub-steps
      green         = robot feedback state (actual joint positions)

  Fig 2 — EE speed & acceleration magnitude vs timestep

  Fig 3 — Joint-space position vs timestep  (one subplot per joint)

  Fig 4 — Joint-space velocity & acceleration vs timestep

Usage
-----
Single file:
    python src/lerobot/async_inference/analyze_trajectory.py \\
        trajectories/episode_0000_20260511_120000.json

Directory (all episodes):
    python src/lerobot/async_inference/analyze_trajectory.py trajectories/

Multiple files / dirs:
    python src/lerobot/async_inference/analyze_trajectory.py \\
        trajectories/episode_0000*.json trajectories_run2/

Options:
    --urdf PATH     Path to SO-101 kinematics URDF (auto-detected by default)
    --out  PATH     Base path for saved figures (no extension; _ee3d.png etc. appended)
    --no-show       Skip interactive display (only auto-save)
    --no-interp     Exclude interpolated sub-step actions from all plots
    --no-feedback   Exclude the robot feedback state trajectory from all plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Joint names fed to RobotKinematics.forward_kinematics() (excludes gripper).
# Must match the order of the q vector accepted by the URDF solver.
_FK_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

_EE_FRAME = "gripper_frame_link"

# Default URDF relative to this script's location
_DEFAULT_URDF = (
    Path(__file__).parent.parent / "robots" / "so_follower" / "so101_kinematics.urdf"
)

# Colour palettes — one per episode loaded
_CHUNK_CMAPS    = ["Blues",   "Greens",   "Purples",  "Oranges", "YlOrBr", "PuRd"]
_EXEC_COLOURS   = ["red",     "crimson",  "firebrick","darkred", "tomato", "orangered"]
_IEXEC_COLOURS  = ["orange",  "gold",     "darkorange","peru",   "sienna", "chocolate"]
_FB_COLOURS     = ["green",   "limegreen","seagreen", "darkgreen","teal",  "olive"]

# All joints for joint-space plots (5 FK arm joints + gripper)
_ALL_JOINT_NAMES = _FK_MOTOR_NAMES + ["gripper"]
_JOINT_LABELS    = ["Shldr Pan", "Shldr Lift", "Elbow", "Wrist Flex", "Wrist Roll", "Gripper"]


# ── FK helpers ────────────────────────────────────────────────────────────────

def _build_kinematics(urdf: str):
    """Instantiate RobotKinematics; raises SystemExit if placo is unavailable."""
    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError as exc:
        sys.exit(
            f"Cannot import RobotKinematics: {exc}\n"
            "Install the placo dependency:  pip install lerobot[placo-dep]"
        )
    return RobotKinematics(urdf, target_frame_name=_EE_FRAME, joint_names=_FK_MOTOR_NAMES)


def _action_to_ee(action: list[float], action_keys: list[str], kinematics) -> np.ndarray | None:
    """Convert one action vector to an (x, y, z) EE position via FK.

    Returns None if any required motor key is missing.
    """
    key_idx = {k: i for i, k in enumerate(action_keys)}
    q = []
    for motor in _FK_MOTOR_NAMES:
        key = f"{motor}.pos"
        if key not in key_idx:
            return None
        q.append(action[key_idx[key]])
    T = kinematics.forward_kinematics(np.array(q, dtype=float))
    return T[:3, 3]


def _state_to_ee(state: dict[str, float], kinematics) -> np.ndarray | None:
    """Convert a feedback_state dict (motor_name.pos → float) to (x, y, z) via FK."""
    q = []
    for motor in _FK_MOTOR_NAMES:
        key = f"{motor}.pos"
        if key not in state:
            return None
        q.append(state[key])
    T = kinematics.forward_kinematics(np.array(q, dtype=float))
    return T[:3, 3]


def _actions_to_ee_array(
    actions: list[list[float]],
    action_keys: list[str],
    kinematics,
) -> np.ndarray:
    """Batch-convert a list of action vectors; drops any that fail FK."""
    pts = [_action_to_ee(a, action_keys, kinematics) for a in actions]
    pts = [p for p in pts if p is not None]
    return np.array(pts) if pts else np.empty((0, 3))


def _states_to_ee_array(
    states: list[dict[str, float] | None],
    kinematics,
) -> np.ndarray:
    """Batch-convert feedback_state dicts; skips None / missing entries."""
    pts = []
    for s in states:
        if s is None:
            continue
        ee = _state_to_ee(s, kinematics)
        if ee is not None:
            pts.append(ee)
    return np.array(pts) if pts else np.empty((0, 3))


# ── Joint-space & derivative helpers ─────────────────────────────────────────

def _extract_joint_array(
    actions: list[list[float]],
    action_keys: list[str],
    joint_names: list[str] | None = None,
) -> np.ndarray:
    """Extract (N, len(joint_names)) array from action vectors; rows with missing keys dropped."""
    if joint_names is None:
        joint_names = _ALL_JOINT_NAMES
    key_idx = {k: i for i, k in enumerate(action_keys)}
    rows = []
    for action in actions:
        row, ok = [], True
        for jn in joint_names:
            idx = key_idx.get(f"{jn}.pos")
            if idx is None:
                ok = False
                break
            row.append(action[idx])
        if ok:
            rows.append(row)
    return np.array(rows, dtype=float) if rows else np.empty((0, len(joint_names)))


def _extract_joint_from_states_arr(
    states: list[dict[str, float] | None],
    joint_names: list[str] | None = None,
) -> np.ndarray:
    """Extract (N, len(joint_names)) from feedback_state dicts; None / missing skipped."""
    if joint_names is None:
        joint_names = _ALL_JOINT_NAMES
    rows = []
    for s in states:
        if s is None:
            continue
        row = [s.get(f"{jn}.pos") for jn in joint_names]
        if any(v is None for v in row):
            continue
        rows.append(row)
    return np.array(rows, dtype=float) if rows else np.empty((0, len(joint_names)))


def _finite_diff(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second finite differences along axis 0 (dt = 1 step).

    Returns (velocity, acceleration). Either may have shape (0, ...) if arr is too short.
    """
    if arr.shape[0] < 2:
        e = np.empty((0,) + arr.shape[1:])
        return e, e
    vel = np.diff(arr, axis=0)
    if vel.shape[0] < 2:
        return vel, np.empty((0,) + vel.shape[1:])
    return vel, np.diff(vel, axis=0)


# ── Per-episode stats ─────────────────────────────────────────────────────────

def _print_stats(data: dict, fk_ok: bool) -> None:
    ep      = data["episode"]
    task    = data["task"]
    chunks  = data["chunks"]
    execs   = data["executed"]

    n_chunk_actions = sum(len(c["actions"]) for c in chunks)
    n_raw    = sum(1 for e in execs if not e.get("interpolated", False))
    n_interp = sum(1 for e in execs if e.get("interpolated", False))
    n_fb     = sum(1 for e in execs if e.get("feedback_state") is not None)

    print(
        f"\n── Episode {ep:04d} ─────────────────────────────────────────────\n"
        f"  Task         : {task!r}\n"
        f"  Chunks rcvd  : {len(chunks)}  (total chunk actions: {n_chunk_actions})\n"
        f"  Executed     : {len(execs)}  (raw: {n_raw}, interpolated: {n_interp})\n"
        f"  Feedback obs : {n_fb} / {len(execs)}  (steps with joint feedback recorded)\n"
        f"  FK ok        : {fk_ok}"
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_episode(
    ax,
    data: dict,
    kinematics,
    ep_idx: int,
    show_interp: bool,
    show_feedback: bool,
) -> bool:
    """Plot one episode onto *ax*.  Returns True if FK produced any points."""
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS [ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS [ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS   [ep_idx % len(_FB_COLOURS)]

    any_points = False
    ep_label   = f"ep{ep:04d}"

    # Build timestep → chunk_index mapping for boundary markers
    ts_to_ci: dict[int, int] = {}
    for ci, chunk in enumerate(chunks):
        for ts in chunk.get("timesteps", []):
            ts_to_ci[ts] = ci
    ci_to_colour = {
        ci: chunk_cmap(0.35 + 0.60 * (ci + 1) / max(len(chunks), 1))
        for ci in range(len(chunks))
    }

    # ── Received chunks (blue-gradient lines) ─────────────────────────────────
    for ci, chunk in enumerate(chunks):
        pts = _actions_to_ee_array(chunk["actions"], action_keys, kinematics)
        if pts.shape[0] < 2:
            continue
        any_points = True

        t = (ci + 1) / max(len(chunks), 1)
        colour = chunk_cmap(0.35 + 0.60 * t)

        label = f"{ep_label} chunks" if ci == 0 else None
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=colour, linewidth=1.4, alpha=0.92, zorder=5, label=label)
        ax.scatter(*pts[0],  color=colour, s=12,  zorder=4,  alpha=0.9)
        ax.scatter(*pts[-1], color=colour, s=80,  zorder=8,  alpha=0.95,
                   marker="*", edgecolors="white", linewidths=0.4)

    # ── Executed — raw policy actions (red solid) ─────────────────────────────
    raw_execs = [e for e in execs if not e.get("interpolated", False)]
    if raw_execs:
        pts = _actions_to_ee_array([e["action"] for e in raw_execs], action_keys, kinematics)
        if pts.shape[0] >= 1:
            any_points = True
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=exec_col, linewidth=2.8, alpha=0.55, zorder=6,
                        label=f"{ep_label} executed (raw)")
            ax.scatter(*pts[0],  color="limegreen", s=60,  marker="o", zorder=10)
            ax.scatter(*pts[-1], color="red",        s=120, marker="X", zorder=11,
                       linewidths=2.5, edgecolors="darkred")

    # ── Executed — interpolated sub-steps (orange dashed) ────────────────────
    if show_interp:
        interp_execs = [e for e in execs if e.get("interpolated", False)]
        if interp_execs:
            pts = _actions_to_ee_array([e["action"] for e in interp_execs], action_keys, kinematics)
            if pts.shape[0] >= 2:
                any_points = True
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=iexec_col, linewidth=1.0, linestyle="--",
                        alpha=0.75, zorder=5,
                        label=f"{ep_label} executed (interpolated)")

    # ── Actual robot state — feedback (green solid) ───────────────────────────
    if show_feedback:
        fb_states = [e.get("feedback_state") for e in execs]
        pts = _states_to_ee_array(fb_states, kinematics)
        if pts.shape[0] >= 2:
            any_points = True
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=fb_col, linewidth=1.4, zorder=7, alpha=0.85,
                    label=f"{ep_label} feedback (actual state)")
        elif pts.shape[0] == 1:
            ax.scatter(*pts[0], color=fb_col, s=20, zorder=7)

    # ── Chunk boundaries on executed / feedback trajectories ──────────────────
    # Group raw executed entries by which received chunk they came from.
    # ▲ = chunk start on this trajectory   ▼ = chunk end on this trajectory
    if ts_to_ci and raw_execs:
        exec_groups: list[tuple[int, list]] = []
        for e in raw_execs:
            ci = ts_to_ci.get(e["timestep"], -1)
            if ci < 0:
                continue
            if exec_groups and exec_groups[-1][0] == ci:
                exec_groups[-1][1].append(e)
            else:
                exec_groups.append((ci, [e]))

        for ci, group in exec_groups:
            col = ci_to_colour.get(ci, "gray")

            # Markers on the executed trajectory (red, same as line colour)
            ee_s = _action_to_ee(group[0]["action"],  action_keys, kinematics)
            ee_e = _action_to_ee(group[-1]["action"], action_keys, kinematics)
            if ee_s is not None:
                ax.scatter(*ee_s, color=exec_col, s=45, marker="o", zorder=9,
                           edgecolors="black", linewidths=1.0)
            if ee_e is not None:
                ax.scatter(*ee_e, color=exec_col, s=90, marker="*", zorder=9,
                           edgecolors="black", linewidths=0.6)

            # Markers on the feedback trajectory (green, same as line colour)
            if show_feedback:
                fb_group = [e for e in group if e.get("feedback_state") is not None]
                if fb_group:
                    fb_s = _state_to_ee(fb_group[0]["feedback_state"],  kinematics)
                    fb_e = _state_to_ee(fb_group[-1]["feedback_state"], kinematics)
                    if fb_s is not None:
                        ax.scatter(*fb_s, color=fb_col, s=45, marker="o", zorder=9,
                                   edgecolors="black", linewidths=1.0)
                    if fb_e is not None:
                        ax.scatter(*fb_e, color=fb_col, s=90, marker="*", zorder=9,
                                   edgecolors="black", linewidths=0.6)

    return any_points


# ── Shared legend helper ──────────────────────────────────────────────────────

def _add_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=7, framealpha=0.7)


# ── Figure 1f/g/h: single-trajectory 3-D plots ───────────────────────────────

def _plot_episode_3d_single(
    ax,
    data: dict,
    kinematics,
    ep_idx: int,
    what: str,          # "chunks" | "executed" | "feedback"
    show_interp: bool = True,
) -> bool:
    """Plot one trajectory type in 3-D. Returns True if any data."""
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS[ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    ts_to_ci: dict[int, int] = {}
    for ci, chunk in enumerate(chunks):
        for ts in chunk.get("timesteps", []):
            ts_to_ci[ts] = ci

    any_points = False

    if what == "chunks":
        for ci, chunk in enumerate(chunks):
            pts = _actions_to_ee_array(chunk["actions"], action_keys, kinematics)
            if pts.shape[0] < 2:
                continue
            any_points = True
            t      = (ci + 1) / max(len(chunks), 1)
            colour = chunk_cmap(0.35 + 0.60 * t)
            label  = f"{ep_label} chunks" if ci == 0 else None
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=colour, lw=1.4, alpha=0.92, zorder=5, label=label)
            ax.scatter(*pts[0],  color=colour, s=25, zorder=4, alpha=0.9)
            ax.scatter(*pts[-1], color=colour, s=90, zorder=8, alpha=0.95,
                       marker="*", edgecolors="white", linewidths=0.5)

    elif what == "executed":
        raw_execs = [e for e in execs if not e.get("interpolated", False)]
        if raw_execs:
            pts = _actions_to_ee_array([e["action"] for e in raw_execs],
                                       action_keys, kinematics)
            if pts.shape[0] >= 1:
                any_points = True
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            color=exec_col, lw=2.8, alpha=0.55, zorder=6,
                            label=f"{ep_label} executed (raw)")
                ax.scatter(*pts[0],  color="limegreen", s=80,  marker="o", zorder=10)
                ax.scatter(*pts[-1], color="red",       s=140, marker="X", zorder=11,
                           linewidths=2.5, edgecolors="darkred")

        if show_interp:
            interp_execs = [e for e in execs if e.get("interpolated", False)]
            if interp_execs:
                pts = _actions_to_ee_array([e["action"] for e in interp_execs],
                                           action_keys, kinematics)
                if pts.shape[0] >= 2:
                    any_points = True
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            color=iexec_col, lw=1.0, ls="--", alpha=0.75, zorder=5,
                            label=f"{ep_label} executed (interpolated)")

        # Chunk boundary markers
        if ts_to_ci and raw_execs:
            exec_groups: list[tuple[int, list]] = []
            for e in raw_execs:
                ci = ts_to_ci.get(e["timestep"], -1)
                if ci < 0:
                    continue
                if exec_groups and exec_groups[-1][0] == ci:
                    exec_groups[-1][1].append(e)
                else:
                    exec_groups.append((ci, [e]))
            for ci, group in exec_groups:
                ee_s = _action_to_ee(group[0]["action"],  action_keys, kinematics)
                ee_e = _action_to_ee(group[-1]["action"], action_keys, kinematics)
                if ee_s is not None:
                    ax.scatter(*ee_s, color=exec_col, s=50, marker="o",
                               zorder=9, edgecolors="black", linewidths=1.0)
                if ee_e is not None:
                    ax.scatter(*ee_e, color=exec_col, s=100, marker="*",
                               zorder=9, edgecolors="black", linewidths=0.6)

    elif what == "feedback":
        fb_states = [e.get("feedback_state") for e in execs]
        pts = _states_to_ee_array(fb_states, kinematics)
        if pts.shape[0] >= 2:
            any_points = True
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=fb_col, lw=1.8, zorder=7, alpha=0.9,
                    label=f"{ep_label} feedback (actual state)")
            ax.scatter(*pts[0],  color="limegreen", s=80,  marker="o", zorder=10)
            ax.scatter(*pts[-1], color=fb_col,      s=140, marker="X", zorder=11,
                       linewidths=2.5, edgecolors="darkgreen")

        raw_execs = [e for e in execs if not e.get("interpolated", False)]
        if ts_to_ci and raw_execs:
            exec_groups = []
            for e in raw_execs:
                ci = ts_to_ci.get(e["timestep"], -1)
                if ci < 0:
                    continue
                if exec_groups and exec_groups[-1][0] == ci:
                    exec_groups[-1][1].append(e)
                else:
                    exec_groups.append((ci, [e]))
            for ci, group in exec_groups:
                fb_group = [e for e in group if e.get("feedback_state") is not None]
                if fb_group:
                    fb_s = _state_to_ee(fb_group[0]["feedback_state"],  kinematics)
                    fb_e = _state_to_ee(fb_group[-1]["feedback_state"], kinematics)
                    if fb_s is not None:
                        ax.scatter(*fb_s, color=fb_col, s=50, marker="o",
                                   zorder=9, edgecolors="black", linewidths=1.0)
                    if fb_e is not None:
                        ax.scatter(*fb_e, color=fb_col, s=100, marker="*",
                                   zorder=9, edgecolors="black", linewidths=0.6)

    return any_points


# ── Figure 1c/d/e: single-trajectory 2-D projections ─────────────────────────

def _plot_episode_2d_single(
    axes,
    data: dict,
    kinematics,
    ep_idx: int,
    what: str,          # "chunks" | "executed" | "feedback"
    show_interp: bool = True,
) -> bool:
    """Plot one trajectory type (chunks / executed / feedback) on three 2-D axes.

    Returns True if at least one point was produced.
    """
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS[ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    # Chunk-index mapping for boundary markers (needed for executed / feedback)
    ts_to_ci: dict[int, int] = {}
    for ci, chunk in enumerate(chunks):
        for ts in chunk.get("timesteps", []):
            ts_to_ci[ts] = ci

    any_points = False

    def _s2(ax, pt, xi, yi, **kw):
        ax.scatter(pt[xi], pt[yi], **kw)

    for pi, ax in enumerate(axes):
        xi, yi = _PROJ_IDX[pi]

        if what == "chunks":
            # ── Received chunks ───────────────────────────────────────────────
            for ci, chunk in enumerate(chunks):
                pts = _actions_to_ee_array(chunk["actions"], action_keys, kinematics)
                if pts.shape[0] < 2:
                    continue
                any_points = True
                t      = (ci + 1) / max(len(chunks), 1)
                colour = chunk_cmap(0.35 + 0.60 * t)
                label  = f"{ep_label} chunks" if ci == 0 else None
                ax.plot(pts[:, xi], pts[:, yi],
                        color=colour, lw=1.2, alpha=0.75, label=label)
                _s2(ax, pts[0],  xi, yi, color=colour, s=25, zorder=4, alpha=0.9)
                _s2(ax, pts[-1], xi, yi, color=colour, s=90, zorder=8, alpha=0.95,
                    marker="*", edgecolors="white", linewidths=0.5)

        elif what == "executed":
            # ── Executed raw ──────────────────────────────────────────────────
            raw_execs = [e for e in execs if not e.get("interpolated", False)]
            if raw_execs:
                pts = _actions_to_ee_array([e["action"] for e in raw_execs],
                                           action_keys, kinematics)
                if pts.shape[0] >= 1:
                    any_points = True
                    if pts.shape[0] >= 2:
                        ax.plot(pts[:, xi], pts[:, yi],
                                color=exec_col, lw=1.8, zorder=6,
                                label=f"{ep_label} executed (raw)")
                    # Episode start / end
                    _s2(ax, pts[0],  xi, yi,
                        color="limegreen", s=80, marker="o", zorder=10)
                    _s2(ax, pts[-1], xi, yi,
                        color="red", s=140, marker="X", zorder=11,
                        linewidths=2.5, edgecolors="darkred")

            # Interpolated sub-steps
            if show_interp:
                interp_execs = [e for e in execs if e.get("interpolated", False)]
                if interp_execs:
                    pts = _actions_to_ee_array([e["action"] for e in interp_execs],
                                               action_keys, kinematics)
                    if pts.shape[0] >= 2:
                        any_points = True
                        ax.plot(pts[:, xi], pts[:, yi],
                                color=iexec_col, lw=1.0, ls="--", alpha=0.75,
                                zorder=5,
                                label=f"{ep_label} executed (interpolated)")

            # Chunk boundary markers on executed
            if ts_to_ci and raw_execs:
                exec_groups: list[tuple[int, list]] = []
                for e in raw_execs:
                    ci = ts_to_ci.get(e["timestep"], -1)
                    if ci < 0:
                        continue
                    if exec_groups and exec_groups[-1][0] == ci:
                        exec_groups[-1][1].append(e)
                    else:
                        exec_groups.append((ci, [e]))
                for ci, group in exec_groups:
                    ee_s = _action_to_ee(group[0]["action"],  action_keys, kinematics)
                    ee_e = _action_to_ee(group[-1]["action"], action_keys, kinematics)
                    if ee_s is not None:
                        _s2(ax, ee_s, xi, yi, color=exec_col, s=50, marker="o",
                            zorder=9, edgecolors="black", linewidths=1.0)
                    if ee_e is not None:
                        _s2(ax, ee_e, xi, yi, color=exec_col, s=100, marker="*",
                            zorder=9, edgecolors="black", linewidths=0.6)

        elif what == "feedback":
            # ── Feedback ─────────────────────────────────────────────────────
            fb_states = [e.get("feedback_state") for e in execs]
            pts = _states_to_ee_array(fb_states, kinematics)
            if pts.shape[0] >= 2:
                any_points = True
                ax.plot(pts[:, xi], pts[:, yi],
                        color=fb_col, lw=1.8, zorder=7, alpha=0.9,
                        label=f"{ep_label} feedback (actual state)")
                # Episode start / end
                _s2(ax, pts[0],  xi, yi,
                    color="limegreen", s=80, marker="o", zorder=10)
                _s2(ax, pts[-1], xi, yi,
                    color=fb_col, s=140, marker="X", zorder=11,
                    linewidths=2.5, edgecolors="darkgreen")

            # Chunk boundary markers on feedback
            raw_execs = [e for e in execs if not e.get("interpolated", False)]
            if ts_to_ci and raw_execs:
                exec_groups = []
                for e in raw_execs:
                    ci = ts_to_ci.get(e["timestep"], -1)
                    if ci < 0:
                        continue
                    if exec_groups and exec_groups[-1][0] == ci:
                        exec_groups[-1][1].append(e)
                    else:
                        exec_groups.append((ci, [e]))
                for ci, group in exec_groups:
                    fb_group = [e for e in group
                                if e.get("feedback_state") is not None]
                    if fb_group:
                        fb_s = _state_to_ee(fb_group[0]["feedback_state"],  kinematics)
                        fb_e = _state_to_ee(fb_group[-1]["feedback_state"], kinematics)
                        if fb_s is not None:
                            _s2(ax, fb_s, xi, yi, color=fb_col, s=50, marker="o",
                                zorder=9, edgecolors="black", linewidths=1.0)
                        if fb_e is not None:
                            _s2(ax, fb_e, xi, yi, color=fb_col, s=100, marker="*",
                                zorder=9, edgecolors="black", linewidths=0.6)

    return any_points


# ── Figure 1b: 2-D EE projections (XY / XZ / YZ) ────────────────────────────

# (xi, yi) index into the (x, y, z) EE point; axis labels; subplot title
_PROJ_IDX    = [(0, 1), (0, 2), (1, 2)]
_PROJ_XLABEL = ["X (m)", "X (m)", "Y (m)"]
_PROJ_YLABEL = ["Y (m)", "Z (m)", "Z (m)"]
_PROJ_TITLE  = ["XY  (top view)", "XZ  (front view)", "YZ  (side view)"]


def _plot_episode_2d(
    axes,            # list/array of 3 Axes
    data: dict,
    kinematics,
    ep_idx: int,
    show_interp: bool,
    show_feedback: bool,
) -> bool:
    """Plot one episode onto three 2-D projection axes. Returns True if any data."""
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS[ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    # Build timestep → chunk-index mapping (same logic as _plot_episode)
    ts_to_ci: dict[int, int] = {}
    for ci, chunk in enumerate(chunks):
        for ts in chunk.get("timesteps", []):
            ts_to_ci[ts] = ci
    ci_to_colour = {
        ci: chunk_cmap(0.35 + 0.60 * (ci + 1) / max(len(chunks), 1))
        for ci in range(len(chunks))
    }

    any_points = False

    def _p(pt, xi, yi):
        return pt[xi], pt[yi]

    def _scatter2(ax, pt, xi, yi, **kw):
        ax.scatter(pt[xi], pt[yi], **kw)

    for pi, ax in enumerate(axes):
        xi, yi = _PROJ_IDX[pi]

        # ── Received chunks ───────────────────────────────────────────────────
        for ci, chunk in enumerate(chunks):
            pts = _actions_to_ee_array(chunk["actions"], action_keys, kinematics)
            if pts.shape[0] < 2:
                continue
            any_points = True
            t      = (ci + 1) / max(len(chunks), 1)
            colour = chunk_cmap(0.35 + 0.60 * t)
            label  = f"{ep_label} chunks" if ci == 0 else None
            ax.plot(pts[:, xi], pts[:, yi],
                    color=colour, lw=0.9, alpha=0.65, label=label)
            _scatter2(ax, pts[0],  xi, yi, color=colour, s=12, zorder=4, alpha=0.8)
            _scatter2(ax, pts[-1], xi, yi, color=colour, s=70, zorder=8, alpha=0.9,
                      marker="*", edgecolors="white", linewidths=0.4)

        # ── Executed raw ──────────────────────────────────────────────────────
        raw_execs = [e for e in execs if not e.get("interpolated", False)]
        if raw_execs:
            pts = _actions_to_ee_array([e["action"] for e in raw_execs],
                                       action_keys, kinematics)
            if pts.shape[0] >= 1:
                any_points = True
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, xi], pts[:, yi],
                            color=exec_col, lw=1.8, zorder=6,
                            label=f"{ep_label} executed (raw)")
                _scatter2(ax, pts[0],  xi, yi,
                          color="limegreen", s=60, marker="o", zorder=10)
                _scatter2(ax, pts[-1], xi, yi,
                          color="red", s=120, marker="X", zorder=11,
                          linewidths=2.5, edgecolors="darkred")

        # ── Executed interpolated ─────────────────────────────────────────────
        if show_interp:
            interp_execs = [e for e in execs if e.get("interpolated", False)]
            if interp_execs:
                pts = _actions_to_ee_array([e["action"] for e in interp_execs],
                                           action_keys, kinematics)
                if pts.shape[0] >= 2:
                    any_points = True
                    ax.plot(pts[:, xi], pts[:, yi],
                            color=iexec_col, lw=1.0, ls="--", alpha=0.75, zorder=5,
                            label=f"{ep_label} executed (interpolated)")

        # ── Feedback ─────────────────────────────────────────────────────────
        if show_feedback:
            fb_states = [e.get("feedback_state") for e in execs]
            pts = _states_to_ee_array(fb_states, kinematics)
            if pts.shape[0] >= 2:
                any_points = True
                ax.plot(pts[:, xi], pts[:, yi],
                        color=fb_col, lw=1.4, zorder=7, alpha=0.85,
                        label=f"{ep_label} feedback (actual state)")

        # ── Chunk boundaries on executed / feedback ───────────────────────────
        if ts_to_ci and raw_execs:
            exec_groups: list[tuple[int, list]] = []
            for e in raw_execs:
                ci = ts_to_ci.get(e["timestep"], -1)
                if ci < 0:
                    continue
                if exec_groups and exec_groups[-1][0] == ci:
                    exec_groups[-1][1].append(e)
                else:
                    exec_groups.append((ci, [e]))

            for ci, group in exec_groups:
                # Executed boundary markers
                ee_s = _action_to_ee(group[0]["action"],  action_keys, kinematics)
                ee_e = _action_to_ee(group[-1]["action"], action_keys, kinematics)
                if ee_s is not None:
                    _scatter2(ax, ee_s, xi, yi, color=exec_col, s=45, marker="o",
                              zorder=9, edgecolors="black", linewidths=1.0)
                if ee_e is not None:
                    _scatter2(ax, ee_e, xi, yi, color=exec_col, s=90, marker="*",
                              zorder=9, edgecolors="black", linewidths=0.6)

                # Feedback boundary markers
                if show_feedback:
                    fb_group = [e for e in group
                                if e.get("feedback_state") is not None]
                    if fb_group:
                        fb_s = _state_to_ee(fb_group[0]["feedback_state"],  kinematics)
                        fb_e = _state_to_ee(fb_group[-1]["feedback_state"], kinematics)
                        if fb_s is not None:
                            _scatter2(ax, fb_s, xi, yi, color=fb_col, s=45,
                                      marker="o", zorder=9,
                                      edgecolors="black", linewidths=1.0)
                        if fb_e is not None:
                            _scatter2(ax, fb_e, xi, yi, color=fb_col, s=90,
                                      marker="*", zorder=9,
                                      edgecolors="black", linewidths=0.6)

    return any_points


# ── Figure 2: EE speed & acceleration ────────────────────────────────────────

def _plot_ee_derivatives(
    ax_vel,
    ax_acc,
    data: dict,
    kinematics,
    ep_idx: int,
    show_interp: bool,
    show_feedback: bool,
) -> bool:
    """Plot EE speed and acceleration magnitude vs timestep for one episode."""
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS[ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    min_ts   = min((e["timestep"] for e in execs), default=0)
    any_data = False

    def _plot_mag(pts, xs, color, lw, ls, alpha, label, zorder=5):
        nonlocal any_data
        if pts.shape[0] < 3:
            return
        vel, acc = _finite_diff(pts)
        ax_vel.plot(xs[: len(vel)], np.linalg.norm(vel, axis=1),
                    color=color, lw=lw, ls=ls, alpha=alpha, label=label, zorder=zorder)
        if acc.shape[0] > 0:
            ax_acc.plot(xs[: len(acc)], np.linalg.norm(acc, axis=1),
                        color=color, lw=lw, ls=ls, alpha=alpha, label=label, zorder=zorder)
        any_data = True

    # Received chunks
    for ci, chunk in enumerate(chunks):
        pts = _actions_to_ee_array(chunk["actions"], action_keys, kinematics)
        ts  = np.array(chunk.get("timesteps", list(range(len(chunk["actions"])))),
                       dtype=float) - min_ts
        t   = (ci + 1) / max(len(chunks), 1)
        col = chunk_cmap(0.35 + 0.60 * t)
        _plot_mag(pts, ts, color=col, lw=0.8, ls="-", alpha=0.6,
                  label=(f"{ep_label} chunks" if ci == 0 else None))

    # Executed raw
    raw_execs = [e for e in execs if not e.get("interpolated", False)]
    if raw_execs:
        pts = _actions_to_ee_array([e["action"] for e in raw_execs], action_keys, kinematics)
        xs  = np.array([e["timestep"] - min_ts for e in raw_execs], dtype=float)
        _plot_mag(pts, xs, color=exec_col, lw=1.8, ls="-", alpha=1.0,
                  label=f"{ep_label} executed (raw)", zorder=6)

    # Executed interpolated
    if show_interp:
        interp_execs = [e for e in execs if e.get("interpolated", False)]
        if interp_execs:
            pts = _actions_to_ee_array([e["action"] for e in interp_execs], action_keys, kinematics)
            xs  = np.arange(len(interp_execs), dtype=float)
            _plot_mag(pts, xs, color=iexec_col, lw=1.0, ls="--", alpha=0.75,
                      label=f"{ep_label} executed (interpolated)")

    # Feedback
    if show_feedback:
        fb_pairs = [(e["timestep"] - min_ts, e["feedback_state"])
                    for e in execs if e.get("feedback_state") is not None]
        if len(fb_pairs) >= 3:
            xs_fb = np.array([p[0] for p in fb_pairs], dtype=float)
            pts   = _states_to_ee_array([p[1] for p in fb_pairs], kinematics)
            _plot_mag(pts, xs_fb, color=fb_col, lw=1.4, ls="-", alpha=0.85,
                      label=f"{ep_label} feedback", zorder=7)

    return any_data


# ── Figure 3: Joint positions ─────────────────────────────────────────────────

def _plot_joint_positions(
    axes: list,
    data: dict,
    ep_idx: int,
    show_interp: bool,
    show_feedback: bool,
) -> bool:
    """Plot per-joint angle time series for one episode (one subplot per joint)."""
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS[ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    n      = len(axes)
    min_ts = min((e["timestep"] for e in execs), default=0)
    any_data = False

    def _plot_lines(arr, xs, color, lw, ls, alpha, base_label, zorder=5):
        nonlocal any_data
        if arr.shape[0] < 2:
            return
        any_data = True
        for j in range(min(n, arr.shape[1])):
            axes[j].plot(xs, arr[:, j], color=color, lw=lw, ls=ls, alpha=alpha,
                         label=(base_label if j == 0 else None), zorder=zorder)

    # Received chunks
    for ci, chunk in enumerate(chunks):
        arr = _extract_joint_array(chunk["actions"], action_keys)
        if arr.shape[0] < 2:
            continue
        ts  = np.array(chunk.get("timesteps", list(range(len(chunk["actions"])))),
                       dtype=float) - min_ts
        t   = (ci + 1) / max(len(chunks), 1)
        col = chunk_cmap(0.35 + 0.60 * t)
        _plot_lines(arr, ts, color=col, lw=0.9, ls="-", alpha=0.65,
                    base_label=(f"{ep_label} chunks" if ci == 0 else None))

    # Executed raw
    raw_execs = [e for e in execs if not e.get("interpolated", False)]
    if raw_execs:
        arr = _extract_joint_array([e["action"] for e in raw_execs], action_keys)
        xs  = np.array([e["timestep"] - min_ts for e in raw_execs], dtype=float)
        _plot_lines(arr, xs, color=exec_col, lw=1.8, ls="-", alpha=1.0,
                    base_label=f"{ep_label} executed (raw)", zorder=6)

    # Executed interpolated
    if show_interp:
        interp_execs = [e for e in execs if e.get("interpolated", False)]
        if interp_execs:
            arr = _extract_joint_array([e["action"] for e in interp_execs], action_keys)
            xs  = np.arange(len(interp_execs), dtype=float)
            _plot_lines(arr, xs, color=iexec_col, lw=1.0, ls="--", alpha=0.75,
                        base_label=f"{ep_label} executed (interpolated)")

    # Feedback
    if show_feedback:
        fb_pairs = [(e["timestep"] - min_ts, e["feedback_state"])
                    for e in execs if e.get("feedback_state") is not None]
        if fb_pairs:
            xs_fb = np.array([p[0] for p in fb_pairs], dtype=float)
            arr   = _extract_joint_from_states_arr([p[1] for p in fb_pairs])
            _plot_lines(arr, xs_fb, color=fb_col, lw=1.4, ls="-", alpha=0.85,
                        base_label=f"{ep_label} feedback", zorder=7)

    return any_data


# ── Figure 4: Joint velocity & acceleration ───────────────────────────────────

def _plot_joint_derivatives(
    axes,          # ndarray shape (n_joints, 2): col 0 = vel, col 1 = acc
    data: dict,
    ep_idx: int,
    show_interp: bool,
    show_feedback: bool,
) -> bool:
    """Plot per-joint velocity and acceleration for one episode."""
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    chunks      = data["chunks"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    chunk_cmap = plt.get_cmap(_CHUNK_CMAPS[ep_idx % len(_CHUNK_CMAPS)])
    exec_col   = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    iexec_col  = _IEXEC_COLOURS[ep_idx % len(_IEXEC_COLOURS)]
    fb_col     = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    n      = axes.shape[0]
    min_ts = min((e["timestep"] for e in execs), default=0)
    any_data = False

    def _plot_der(arr, xs, color, lw, ls, alpha, base_label, zorder=5):
        nonlocal any_data
        if arr.shape[0] < 3:
            return
        vel, acc = _finite_diff(arr)
        any_data = True
        for j in range(min(n, arr.shape[1])):
            axes[j, 0].plot(xs[: len(vel)], vel[:, j],
                            color=color, lw=lw, ls=ls, alpha=alpha,
                            label=(base_label if j == 0 else None), zorder=zorder)
            if acc.shape[0] > 0:
                axes[j, 1].plot(xs[: len(acc)], acc[:, j],
                                color=color, lw=lw, ls=ls, alpha=alpha,
                                label=(base_label if j == 0 else None), zorder=zorder)

    # Received chunks
    for ci, chunk in enumerate(chunks):
        arr = _extract_joint_array(chunk["actions"], action_keys)
        ts  = np.array(chunk.get("timesteps", list(range(len(chunk["actions"])))),
                       dtype=float) - min_ts
        t   = (ci + 1) / max(len(chunks), 1)
        col = chunk_cmap(0.35 + 0.60 * t)
        _plot_der(arr, ts, color=col, lw=0.8, ls="-", alpha=0.6,
                  base_label=(f"{ep_label} chunks" if ci == 0 else None))

    # Executed raw
    raw_execs = [e for e in execs if not e.get("interpolated", False)]
    if raw_execs:
        arr = _extract_joint_array([e["action"] for e in raw_execs], action_keys)
        xs  = np.array([e["timestep"] - min_ts for e in raw_execs], dtype=float)
        _plot_der(arr, xs, color=exec_col, lw=1.8, ls="-", alpha=1.0,
                  base_label=f"{ep_label} executed (raw)", zorder=6)

    # Executed interpolated
    if show_interp:
        interp_execs = [e for e in execs if e.get("interpolated", False)]
        if interp_execs:
            arr = _extract_joint_array([e["action"] for e in interp_execs], action_keys)
            xs  = np.arange(len(interp_execs), dtype=float)
            _plot_der(arr, xs, color=iexec_col, lw=1.0, ls="--", alpha=0.75,
                      base_label=f"{ep_label} executed (interpolated)")

    # Feedback
    if show_feedback:
        fb_pairs = [(e["timestep"] - min_ts, e["feedback_state"])
                    for e in execs if e.get("feedback_state") is not None]
        if fb_pairs:
            xs_fb = np.array([p[0] for p in fb_pairs], dtype=float)
            arr   = _extract_joint_from_states_arr([p[1] for p in fb_pairs])
            _plot_der(arr, xs_fb, color=fb_col, lw=1.4, ls="-", alpha=0.85,
                      base_label=f"{ep_label} feedback", zorder=7)

    return any_data


# ── Figure 5: Tracking error (executed action vs feedback state) ──────────────

# Distinct colours for per-joint L1 diff lines
_JOINT_DIFF_COLORS = ["#E74C3C", "#E67E22", "#2ECC71", "#3498DB", "#9B59B6", "#1ABC9C"]


def _plot_tracking_error(
    ax_joint: "Axes",
    ax_ee: "Axes",
    data: dict,
    kinematics,
    ep_idx: int,
) -> bool:
    """Plot joint L1 and EE L2 tracking error (executed action vs feedback state).

    Only uses raw (non-interpolated) executed steps that have a non-None feedback_state,
    since these are the pairs where the commanded action and the actual robot state are
    recorded at the same control step.

    Returns True if at least one pair was found.
    """
    import matplotlib.pyplot as plt

    action_keys = data["action_keys"]
    execs       = data["executed"]
    ep          = data["episode"]
    ep_label    = f"ep{ep:04d}"

    exec_col = _EXEC_COLOURS[ep_idx % len(_EXEC_COLOURS)]
    fb_col   = _FB_COLOURS[ep_idx % len(_FB_COLOURS)]

    # Collect aligned (timestep, action, feedback_state) triples — raw steps only
    min_ts = min((e["timestep"] for e in execs), default=0)
    pairs = [
        (e["timestep"] - min_ts, e["action"], e["feedback_state"])
        for e in execs
        if not e.get("interpolated", False) and e.get("feedback_state") is not None
    ]
    if not pairs:
        return False

    xs          = np.array([p[0] for p in pairs], dtype=float)
    act_actions = [p[1] for p in pairs]
    fb_states   = [p[2] for p in pairs]

    # ── Joint signed diff (action − feedback) ────────────────────────────────
    # Signed, not absolute:
    #   > 0  commanded > actual  →  robot lagging  (under-executed)
    #   < 0  commanded < actual  →  robot overshot (over-executed)
    act_arr = _extract_joint_array(act_actions, action_keys, _ALL_JOINT_NAMES)
    fb_arr  = _extract_joint_from_states_arr(fb_states, _ALL_JOINT_NAMES)

    n_pairs = min(act_arr.shape[0], fb_arr.shape[0])
    if n_pairs > 0:
        diff = act_arr[:n_pairs] - fb_arr[:n_pairs]   # signed, (N, n_joints), degrees
        xs_j = xs[:n_pairs]

        ax_joint.axhline(0, color="gray", lw=0.8, ls="--", zorder=2)

        for j, (jlabel, col) in enumerate(zip(_JOINT_LABELS, _JOINT_DIFF_COLORS)):
            if j >= diff.shape[1]:
                break
            d = diff[:, j]
            ax_joint.plot(xs_j, d, color=col, lw=0.9, alpha=0.8,
                          label=jlabel if ep_idx == 0 else None)

    # ── EE signed XYZ component diff ─────────────────────────────────────────
    # dX/dY/dZ = EE_action − EE_feedback (mm).  Same sign convention as joints:
    #   > 0  commanded > actual  →  lagging in that axis
    #   < 0  commanded < actual  →  overshooting in that axis
    _EE_AXES   = ["dX", "dY", "dZ"]
    _EE_COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]

    ee_triples: list[tuple[float, np.ndarray]] = []   # (x_ts, dXYZ in mm)
    for x_ts, act, fb in zip(xs, act_actions, fb_states):
        ee_act = _action_to_ee(act, action_keys, kinematics)
        ee_fb  = _state_to_ee(fb, kinematics)
        if ee_act is not None and ee_fb is not None:
            ee_triples.append((x_ts, (ee_act - ee_fb) * 1000.0))

    if ee_triples:
        xs_ee  = np.array([t[0] for t in ee_triples], dtype=float)
        dxyz   = np.array([t[1] for t in ee_triples])   # (N, 3) mm

        ax_ee.axhline(0, color="gray", lw=0.8, ls="--", zorder=2)
        for ci, (axis_lbl, col) in enumerate(zip(_EE_AXES, _EE_COLORS)):
            d = dxyz[:, ci]
            ax_ee.plot(xs_ee, d, color=col, lw=1.2,
                       label=f"{axis_lbl}  {ep_label}" if ep_idx == 0 else axis_lbl)

    return True


# ── Save-path derivation ──────────────────────────────────────────────────────

def _derive_figure_save_paths(json_file: Path, out_override: str | None) -> dict[str, Path]:
    """Return {figure_key → Path} for all 12 figures for one episode."""
    if out_override:
        base = Path(out_override) / json_file.stem
    else:
        base = json_file.parent / json_file.stem
    base.parent.mkdir(parents=True, exist_ok=True)
    return {
        "ee3d":        base.parent / (base.name + "_ee3d.png"),
        "ee2d":        base.parent / (base.name + "_ee2d.png"),
        "ee2d_chunks": base.parent / (base.name + "_ee2d_chunks.png"),
        "ee2d_exec":   base.parent / (base.name + "_ee2d_exec.png"),
        "ee2d_fb":     base.parent / (base.name + "_ee2d_fb.png"),
        "ee3d_chunks": base.parent / (base.name + "_ee3d_chunks.png"),
        "ee3d_exec":   base.parent / (base.name + "_ee3d_exec.png"),
        "ee3d_fb":     base.parent / (base.name + "_ee3d_fb.png"),
        "ee_deriv":    base.parent / (base.name + "_ee_deriv.png"),
        "joint_pos":   base.parent / (base.name + "_joint_pos.png"),
        "joint_deriv": base.parent / (base.name + "_joint_deriv.png"),
        "track_err":   base.parent / (base.name + "_track_err.png"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _collect_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("episode_*.json")))
        elif p.suffix == ".json" and p.exists():
            files.append(p)
        else:
            print(f"[warn] skipping {p!r} (not a .json file or directory)", file=sys.stderr)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot SO-101 trajectories from RobotClient trajectory JSON files."
    )
    ap.add_argument(
        "paths", nargs="+",
        help="JSON file(s) or director(ies) containing episode_NNNN_*.json files",
    )
    ap.add_argument(
        "--urdf", default=str(_DEFAULT_URDF),
        help=f"Path to SO-101 kinematics URDF (default: {_DEFAULT_URDF})",
    )
    ap.add_argument(
        "--out", default=None,
        help=(
            "Output directory for saved figures (multi-episode mode) or base path prefix "
            "(single-file mode). 12 PNGs per episode are saved with the episode stem as filename base."
        ),
    )
    ap.add_argument("--no-show",     action="store_true", help="Skip interactive display")
    ap.add_argument("--no-interp",   action="store_true",
                    help="Exclude interpolated sub-step actions from all plots")
    ap.add_argument("--no-feedback", action="store_true",
                    help="Exclude the robot feedback state trajectory from all plots")
    args = ap.parse_args()

    json_files = _collect_json_files(args.paths)
    if not json_files:
        sys.exit("No JSON files found. Check the paths provided.")

    print(f"Loading {len(json_files)} episode file(s) ...")

    kinematics = _build_kinematics(args.urdf)

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    show_interp   = not args.no_interp
    show_feedback = not args.no_feedback
    n_joints      = len(_ALL_JOINT_NAMES)

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        save_paths = _derive_figure_save_paths(jf, args.out)

        # ── Create all 12 figures fresh for this episode ──────────────────────
        fig1 = plt.figure(figsize=(13, 9))
        ax1  = fig1.add_subplot(111, projection="3d")

        fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
        fig7, axes7 = plt.subplots(1, 3, figsize=(18, 6))
        fig8, axes8 = plt.subplots(1, 3, figsize=(18, 6))
        fig9, axes9 = plt.subplots(1, 3, figsize=(18, 6))

        fig10 = plt.figure(figsize=(10, 8)); ax10 = fig10.add_subplot(111, projection="3d")
        fig11 = plt.figure(figsize=(10, 8)); ax11 = fig11.add_subplot(111, projection="3d")
        fig12 = plt.figure(figsize=(10, 8)); ax12 = fig12.add_subplot(111, projection="3d")

        fig2, (ax2_vel, ax2_acc) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

        fig3, _axes3 = plt.subplots(n_joints, 1, figsize=(13, 2 * n_joints + 1), sharex=True)
        axes3: list = list(_axes3) if n_joints > 1 else [_axes3]

        fig4, _axes4 = plt.subplots(n_joints, 2, figsize=(14, 2 * n_joints + 1), sharex=True)
        axes4 = _axes4 if n_joints > 1 else _axes4[np.newaxis, :]

        fig5, (ax5_joint, ax5_ee) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

        # ── Plot (ep_idx=0: each figure holds exactly one episode) ────────────
        ok = _plot_episode(ax1, data, kinematics, 0,
                           show_interp=show_interp, show_feedback=show_feedback)
        _plot_episode_2d(axes6, data, kinematics, 0,
                         show_interp=show_interp, show_feedback=show_feedback)
        _plot_episode_2d_single(axes7, data, kinematics, 0, what="chunks")
        _plot_episode_2d_single(axes8, data, kinematics, 0, what="executed",
                                show_interp=show_interp)
        _plot_episode_2d_single(axes9, data, kinematics, 0, what="feedback")
        _plot_episode_3d_single(ax10, data, kinematics, 0, what="chunks")
        _plot_episode_3d_single(ax11, data, kinematics, 0, what="executed",
                                show_interp=show_interp)
        _plot_episode_3d_single(ax12, data, kinematics, 0, what="feedback")
        _plot_ee_derivatives(ax2_vel, ax2_acc, data, kinematics, 0,
                             show_interp=show_interp, show_feedback=show_feedback)
        _plot_joint_positions(axes3, data, 0,
                              show_interp=show_interp, show_feedback=show_feedback)
        _plot_joint_derivatives(axes4, data, 0,
                                show_interp=show_interp, show_feedback=show_feedback)
        _plot_tracking_error(ax5_joint, ax5_ee, data, kinematics, 0)

        _print_stats(data, fk_ok=ok)
        if not ok:
            print("[warn] No FK points computed for this episode. Check URDF path and action_keys.")

        # ── Decorations ───────────────────────────────────────────────────────
        for pi, ax in enumerate(axes6):
            ax.set_xlabel(_PROJ_XLABEL[pi])
            ax.set_ylabel(_PROJ_YLABEL[pi])
            ax.set_title(_PROJ_TITLE[pi])
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linewidth=0.4, alpha=0.5)
            if pi == 0:
                _add_legend(ax)
        fig6.suptitle(
            "EE Trajectory 2-D Projections  |  ●=chunk start  ★=chunk end  "
            "│  blue=chunks  red=executed  green=feedback",
            fontsize=9,
        )
        fig6.tight_layout()

        _split_meta = [
            (fig7, axes7, "Received Action Chunks",  "●=chunk start  ★=chunk end"),
            (fig8, axes8, "Executed Actions",         "●=chunk start  ★=chunk end  ●=ep.start  ✕=ep.end"),
            (fig9, axes9, "Feedback State (actual)",  "●=chunk start  ★=chunk end  ●=ep.start  ✕=ep.end"),
        ]
        for fig_s, axes_s, title, legend_note in _split_meta:
            for pi, ax in enumerate(axes_s):
                ax.set_xlabel(_PROJ_XLABEL[pi])
                ax.set_ylabel(_PROJ_YLABEL[pi])
                ax.set_title(_PROJ_TITLE[pi])
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(True, linewidth=0.4, alpha=0.5)
                if pi == 0:
                    _add_legend(ax)
            fig_s.suptitle(f"EE Trajectory — {title}  |  {legend_note}", fontsize=9)
            fig_s.tight_layout()

        _3d_meta = [
            (fig10, ax10, "Received Action Chunks",  "blue=chunks"),
            (fig11, ax11, "Executed Actions",         "red=executed  ●=ep.start  ✕=ep.end  ○=chunk.start  ★=chunk.end"),
            (fig12, ax12, "Feedback State (actual)",  "green=feedback  ●=ep.start  ✕=ep.end  ○=chunk.start  ★=chunk.end"),
        ]
        for fig_s, ax_s, title, note in _3d_meta:
            ax_s.set_xlabel("X (m)")
            ax_s.set_ylabel("Y (m)")
            ax_s.set_zlabel("Z (m)")
            ax_s.set_title(f"{title}\n{note}", fontsize=8)
            _add_legend(ax_s)
            fig_s.tight_layout()

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title(
            "EE Trajectory  |  ●=start  ✕=end  │  blue=chunks  red=executed  green=feedback",
            fontsize=9,
        )
        _add_legend(ax1)
        fig1.tight_layout()

        ax2_vel.set_ylabel("|v|  (m / step)")
        ax2_vel.set_title("EE Speed")
        _add_legend(ax2_vel)
        ax2_acc.set_xlabel("Timestep")
        ax2_acc.set_ylabel("|a|  (m / step²)")
        ax2_acc.set_title("EE Acceleration Magnitude")
        _add_legend(ax2_acc)
        fig2.suptitle("EE Velocity & Acceleration", fontsize=10)
        fig2.tight_layout()

        for j, (ax, lbl) in enumerate(zip(axes3, _JOINT_LABELS)):
            ax.set_ylabel(f"{lbl}\n(deg)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            if j == 0:
                _add_legend(ax)
        axes3[-1].set_xlabel("Timestep")
        fig3.suptitle("Joint Positions", fontsize=10)
        fig3.tight_layout()

        axes4[0, 0].set_title("Velocity  (deg / step)")
        axes4[0, 1].set_title("Acceleration  (deg / step²)")
        for j, jlabel in enumerate(_JOINT_LABELS):
            for c in range(2):
                axes4[j, c].set_ylabel(jlabel, fontsize=7)
                axes4[j, c].tick_params(labelsize=7)
                axes4[j, c].grid(True, linewidth=0.4, alpha=0.5)
            if j == 0:
                _add_legend(axes4[0, 0])
        axes4[-1, 0].set_xlabel("Timestep")
        axes4[-1, 1].set_xlabel("Timestep")
        fig4.suptitle("Joint Velocity & Acceleration", fontsize=10)
        fig4.tight_layout()

        ax5_joint.set_ylabel("action − feedback  (deg)")
        ax5_joint.set_title("Joint Tracking Error  (per joint, signed)  |  >0 lag  <0 overshoot")
        ax5_joint.grid(True, linewidth=0.4, alpha=0.5)
        _add_legend(ax5_joint)
        ax5_ee.set_xlabel("Timestep")
        ax5_ee.set_ylabel("EE_action − EE_feedback  (mm)\n>0 lag  <0 overshoot")
        ax5_ee.set_title("EE Tracking Error per Axis  (dX / dY / dZ, signed)")
        ax5_ee.grid(True, linewidth=0.4, alpha=0.5)
        _add_legend(ax5_ee)
        fig5.suptitle("Tracking Error  (commanded − actual)", fontsize=10)
        fig5.tight_layout()

        # ── Save ──────────────────────────────────────────────────────────────
        print()
        for key, fig, sp in [
            ("ee3d",        fig1,  save_paths["ee3d"]),
            ("ee2d",        fig6,  save_paths["ee2d"]),
            ("ee2d_chunks", fig7,  save_paths["ee2d_chunks"]),
            ("ee2d_exec",   fig8,  save_paths["ee2d_exec"]),
            ("ee2d_fb",     fig9,  save_paths["ee2d_fb"]),
            ("ee3d_chunks", fig10, save_paths["ee3d_chunks"]),
            ("ee3d_exec",   fig11, save_paths["ee3d_exec"]),
            ("ee3d_fb",     fig12, save_paths["ee3d_fb"]),
            ("ee_deriv",    fig2,  save_paths["ee_deriv"]),
            ("joint_pos",   fig3,  save_paths["joint_pos"]),
            ("joint_deriv", fig4,  save_paths["joint_deriv"]),
            ("track_err",   fig5,  save_paths["track_err"]),
        ]:
            fig.savefig(sp, dpi=150)
            print(f"  {key:12s} → {sp}")

        if not args.no_show and len(json_files) == 1:
            try:
                plt.show()
            except Exception:
                pass
        else:
            plt.close("all")


if __name__ == "__main__":
    main()
