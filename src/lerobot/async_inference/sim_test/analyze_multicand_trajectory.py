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

"""Visualize multi-candidate trajectory data (LIBERO Franka or SO-101 real robot).

Reads per-episode JSON files written by MultiCandSimClient / MultiCandSO101Client
(--record_trajectory=true) and produces a set of PNG figures per episode.

All trajectory figures use EE Cartesian coordinates (metres):
  LIBERO  : robot_state[:3] is already EE XYZ; action[:3] is EE XYZ.
  SO-101  : robot_state and action are joint angles (degrees) — FK is applied
            via RobotKinematics (placo) using so101_kinematics.urdf.

  Fig 1 (ee_traj):    EE position projections (XY / XZ / YZ)
  Fig 2 (scores):     Score timeline across chunks (server_score + spread_l2)
  Fig 3 (cand_fan):   Candidate action band per action dimension (min/max ± selected)
  Fig 4 (selection):  Selection analysis (override rate + delay/noise distribution)

Usage::

    # LIBERO (default):
    python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \\
        --traj_dir=./mc_trajectories \\
        --out_dir=./mc_viz

    # SO-101 real robot:
    python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \\
        --traj_dir=./mc_trajectories \\
        --out_dir=./mc_viz \\
        --robot_type=so101

    # SO-101 with custom URDF:
    python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \\
        --traj_dir=./mc_trajectories/ep0003.json \\
        --out_dir=./mc_viz \\
        --robot_type=so101 \\
        --urdf_path=/path/to/so101_kinematics.urdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# SM phase display constants
# ---------------------------------------------------------------------------

# Per-phase line/marker colors (used in EE traj + SM timeline)
_PHASE_COL: dict[str, str] = {
    "NORMAL":  "#3498db",  # blue
    "CLOSING": "#f39c12",  # orange
    "HOLDING": "#2ecc71",  # green
    "REWIND":  "#e74c3c",  # red
}
# Per-phase background band colors (axvspan fill; subset of above)
_PHASE_BG: dict[str, str] = {
    "CLOSING": "#f39c12",
    "HOLDING": "#2ecc71",
}


# ---------------------------------------------------------------------------
# Config (simple namespace; argparse fills it)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Visualize multi-candidate LIBERO trajectory data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--traj_dir", default="./mc_trajectories",
                    help="Directory of ep*.json files, or path to a single episode JSON.")
    ap.add_argument("--out_dir", default="./mc_viz",
                    help="Output directory for PNG figures.")
    ap.add_argument("--max_episodes", type=int, default=0,
                    help="Max episodes to process (0 = all).")
    ap.add_argument("--action_dims", type=int, default=7,
                    help="Number of action dimensions to show in candidate fan plot.")
    ap.add_argument("--action_dim_names", default="",
                    help="Comma-separated action dimension names (e.g. 'j0,j1,j2,j3,j4,j5,grip').")
    ap.add_argument("--robot_type", default="libero", choices=["libero", "so101"],
                    help="Robot type: 'libero' (Franka, state/action[:3]=EE Cartesian) "
                         "or 'so101' (SO-101, joint angles in degrees).")
    ap.add_argument("--viz_mode", default="ee", choices=["ee", "joint"],
                    help="SO-101 visualization mode: 'ee' uses FK to show EE Cartesian "
                         "(auto-falls back to joint-space if FK unavailable); "
                         "'joint' always shows raw joint angles without FK. "
                         "Ignored for LIBERO (always EE).")
    ap.add_argument("--urdf_path", default="",
                    help="Path to SO-101 URDF for FK. "
                         "Defaults to robots/so_follower/so101_kinematics.urdf.")
    ap.add_argument("--dpi", type=int, default=120,
                    help="DPI for saved PNG figures.")
    ap.add_argument("--show", action="store_true",
                    help="Also call plt.show() after saving.")
    return ap


# ---------------------------------------------------------------------------
# SO-101 FK helpers
# ---------------------------------------------------------------------------

# SO-101 joint names passed to RobotKinematics (must match URDF + smart_robot_client.py)
_SO101_FK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_SO101_FK_FRAME  = "gripper_frame_link"


def _init_fk_solver(cfg: argparse.Namespace):
    """Initialise a RobotKinematics FK solver for SO-101.

    Returns the solver on success, or None if placo/URDF is unavailable (the
    calling plots will fall back to raw joint-space with a warning label).
    """
    urdf = cfg.urdf_path or str(
        Path(__file__).parent.parent.parent / "robots" / "so_follower" / "so101_kinematics.urdf"
    )
    if not Path(urdf).exists():
        logging.warning(f"[so101 FK] URDF not found at '{urdf}' — falling back to joint-space plots")
        return None
    try:
        from lerobot.model.kinematics import RobotKinematics
        solver = RobotKinematics(
            urdf_path=urdf,
            target_frame_name=_SO101_FK_FRAME,
            joint_names=_SO101_FK_JOINTS,
        )
        logging.info(f"[so101 FK] Loaded kinematics: urdf='{urdf}' joints={_SO101_FK_JOINTS}")
        return solver
    except Exception as exc:
        logging.warning(f"[so101 FK] kinematics init failed: {exc} — falling back to joint-space plots")
        return None


def _to_ee_xyz(pts: np.ndarray, fk_solver) -> tuple[np.ndarray, bool]:
    """Convert (N, D) joint-angle rows (degrees) to (N, 3) EE XYZ via FK.

    Returns (xyz_array, used_fk) where used_fk=False means FK was unavailable
    and the first 3 columns are returned unchanged as a fallback.
    """
    if fk_solver is None:
        return pts[:, :3].copy(), False
    xyz = np.zeros((len(pts), 3), dtype=np.float64)
    for i, row in enumerate(pts):
        T = fk_solver.forward_kinematics(row)
        xyz[i] = T[:3, 3]
    return xyz, True


def _ee_labels(robot_type: str, fk_ok: bool) -> tuple[str, str, str]:
    """Return (x_label, y_label, z_label) for axis annotations.

    When FK is unavailable/bypassed for SO-101, use actual joint names from
    _SO101_FK_JOINTS so the plot axes are unambiguously labelled.
    """
    if robot_type != "so101" or fk_ok:
        return "EE X (m)", "EE Y (m)", "EE Z (m)"
    # SO-101 joint-space (intentional joint mode or FK fallback)
    j0 = _SO101_FK_JOINTS[0] if len(_SO101_FK_JOINTS) > 0 else "j0"
    j1 = _SO101_FK_JOINTS[1] if len(_SO101_FK_JOINTS) > 1 else "j1"
    j2 = _SO101_FK_JOINTS[2] if len(_SO101_FK_JOINTS) > 2 else "j2"
    return f"{j0} (deg)", f"{j1} (deg)", f"{j2} (deg)"


def _robot_tag(cfg: argparse.Namespace, fk_ok: bool) -> str:
    """Build a short title tag indicating robot type and effective viz mode."""
    if cfg.robot_type != "so101":
        return f"[{cfg.robot_type}]"
    if fk_ok:
        return "[so101  EE]"
    if getattr(cfg, "viz_mode", "ee") == "joint":
        return "[so101  joint-space]"
    return "[so101  joint-space (FK fallback)]"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_episodes(cfg: argparse.Namespace) -> list[dict]:
    """Load one or more episode JSON files from cfg.traj_dir."""
    p = Path(cfg.traj_dir)
    if p.is_file():
        paths = [p]
    else:
        paths = sorted(p.glob("ep*.json"))
    if not paths:
        logging.error(f"No ep*.json files found in {p}")
        return []
    if cfg.max_episodes > 0:
        paths = paths[: cfg.max_episodes]
    episodes = []
    for jp in paths:
        try:
            episodes.append(json.loads(jp.read_text(encoding="utf-8")))
        except Exception as exc:
            logging.warning(f"Skipping {jp}: {exc}")
    logging.info(f"Loaded {len(episodes)} episode(s) from {p}")
    return episodes


def _has_candidates(ep: dict) -> bool:
    """True if any chunk has Phase-2 candidates."""
    return any(len(c.get("candidates", [])) > 0 for c in ep.get("chunks", []))


def _dim_names(cfg: argparse.Namespace) -> list[str]:
    if cfg.action_dim_names:
        return cfg.action_dim_names.split(",")
    return [f"d{i}" for i in range(cfg.action_dims)]


def _has_sm_phases(ep: dict) -> bool:
    """True when this episode has SM phase annotations with any non-NORMAL phase."""
    for s in ep.get("executed", []):
        if s.get("grasp_phase", "NORMAL") not in ("NORMAL", ""):
            return True
    for c in ep.get("chunks", []):
        if c.get("grasp_phase", "NORMAL") not in ("NORMAL", ""):
            return True
    return False


# ---------------------------------------------------------------------------
# Fig 1: EE Trajectory
# ---------------------------------------------------------------------------


def _plot_ee_trajectory(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """2D projections of EE position (XY / XZ / YZ).

    LIBERO : robot_state[:3] = EE Cartesian XYZ (metres) — used directly.
    SO-101 : robot_state = joint angles (degrees) — FK applied to get EE XYZ.
    """
    import matplotlib.pyplot as plt

    executed = ep.get("executed", [])
    states = [s.get("robot_state") for s in executed if s.get("robot_state")]
    if not states:
        logging.warning(f"ep{ep['episode']:04d}: no robot_state in executed steps; skipping ee_traj")
        return

    states_arr = np.array(states)
    pos, fk_ok = _actions_to_ee(states_arr, cfg)

    xl, yl, zl = _ee_labels(cfg.robot_type, fk_ok)
    phases = np.array([s.get("episode_phase", 0.0) for s in executed if s.get("robot_state")])
    success = ep.get("success")
    title_suffix = " ✓ success" if success else (" ✗ failed" if success is False else "")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    projs = [((0, 1), xl, yl), ((0, 2), xl, zl), ((1, 2), yl, zl)]

    for ax, ((ix, iy), x_lbl, y_lbl) in zip(axes, projs):
        sc = ax.scatter(pos[:, ix], pos[:, iy], c=phases, cmap="viridis", s=6, alpha=0.7)
        ax.plot(pos[:, ix], pos[:, iy], lw=0.5, alpha=0.3, color="gray")
        ax.scatter(pos[0, ix], pos[0, iy], marker="o", color="lime", s=80, zorder=5, label="start")
        ax.scatter(pos[-1, ix], pos[-1, iy], marker="*", color="red", s=120, zorder=5, label="end")
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="phase")

    ep_id = ep["episode"]
    task_short = ep.get("task", "")[:60]
    fig.suptitle(
        f"EE Trajectory {_robot_tag(cfg, fk_ok)}  ep{ep_id:04d}{title_suffix}\n{task_short}",
        fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_ee_traj.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


def _plot_ee_trajectory_3d(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """3D EE trajectory colored by SM grasp phase (when SM active) or episode phase (fallback).

    When SM phase data is present (any non-NORMAL step), segments are colored:
      blue=NORMAL, orange=CLOSING, green=HOLDING, red=REWIND.
    Otherwise falls back to viridis colormap over episode phase (0→1).

    LIBERO : robot_state[:3] = EE Cartesian XYZ (metres) — used directly.
    SO-101 : robot_state = joint angles (degrees) — FK applied to get EE XYZ.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    executed = ep.get("executed", [])
    states_with_phase = [(s.get("robot_state"), s.get("episode_phase", 0.0),
                          s.get("grasp_phase", "NORMAL"))
                         for s in executed if s.get("robot_state")]
    if not states_with_phase:
        logging.warning(f"ep{ep['episode']:04d}: no robot_state in executed steps; skipping ee_traj_3d")
        return

    states, ep_phases, sm_phases = zip(*states_with_phase)
    pos, fk_ok = _actions_to_ee(np.array(states), cfg)
    xl, yl, zl = _ee_labels(cfg.robot_type, fk_ok)
    use_sm = any(p not in ("NORMAL", "") for p in sm_phases)

    success = ep.get("success")
    title_suffix = " ✓" if success else (" ✗" if success is False else "")
    ep_id = ep["episode"]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if use_sm:
        # ── Color segments by SM grasp phase ─────────────────────────────────
        for i in range(len(pos) - 1):
            seg_color = _PHASE_COL.get(sm_phases[i], "#888888")
            ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                    color=seg_color, lw=1.2, alpha=0.85)
        # Phase legend
        present = list(dict.fromkeys(sm_phases))  # deduplicate, preserve order
        legend_handles = [
            mpatches.Patch(color=_PHASE_COL.get(p, "#888888"), label=p)
            for p in present if p in _PHASE_COL
        ]
        color_note = "SM phase"
    else:
        # ── Fallback: color by episode phase (viridis) ────────────────────────
        cmap = plt.get_cmap("viridis")
        for i in range(len(pos) - 1):
            ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                    color=cmap(ep_phases[i]), lw=1.2, alpha=0.8)
        sm_ep = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm_ep.set_array([])
        fig.colorbar(sm_ep, ax=ax, shrink=0.6, label="episode phase")
        legend_handles = []
        color_note = "episode phase"

    ax.scatter(*pos[0], color="lime", s=80, zorder=5, label="start", depthshade=False)
    ax.scatter(*pos[-1], color="red", s=120, marker="*", zorder=5, label="end", depthshade=False)

    if use_sm:
        legend_handles += [
            mpatches.Patch(color="lime", label="start"),
            mpatches.Patch(color="red", label="end"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    else:
        ax.legend(fontsize=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_zlabel(zl)

    fig.suptitle(
        f"EE Trajectory 3D {_robot_tag(cfg, fk_ok)} ({color_note})  ep{ep_id:04d}{title_suffix}"
        f"\n{ep.get('task', '')[:60]}",
        fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_ee_traj_3d.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


def _assign_executed_to_chunks(
    chunks: list[dict], executed: list[dict]
) -> dict[int, list[dict]]:
    """Map executed steps to their chunk by timestep range.

    Chunk i covers [chunks[i].first_timestep, chunks[i+1].first_timestep).
    The last chunk extends to infinity.
    """
    if not chunks or not executed:
        return {i: [] for i in range(len(chunks))}
    sorted_exec = sorted(
        [s for s in executed if s.get("timestep") is not None],
        key=lambda s: s["timestep"],
    )
    result: dict[int, list[dict]] = {i: [] for i in range(len(chunks))}
    for i, chunk in enumerate(chunks):
        t_start = chunk.get("first_timestep", 0)
        t_end = (
            chunks[i + 1].get("first_timestep", float("inf"))
            if i + 1 < len(chunks)
            else float("inf")
        )
        for step in sorted_exec:
            ts = step.get("timestep", -1)
            if t_start <= ts < t_end:
                result[i].append(step)
    return result


def _get_chunk_selected_actions(chunk: dict) -> list[list[float]]:
    """Return the selected candidate's action list for a chunk record.

    Multi-cand chunks: look for candidate with selected=True.
    Phase-1 / top_k=1: fall back to chunk["selected_actions"].
    """
    for c in chunk.get("candidates", []):
        if c.get("selected", False):
            return c.get("actions", [])
    return chunk.get("selected_actions", [])


def _actions_to_ee(actions: np.ndarray, cfg: argparse.Namespace) -> tuple[np.ndarray, bool]:
    """Convert (N, D) state/action array to (N, 3) EE XYZ (or joint-space fallback).

    LIBERO          : [:3] is already EE Cartesian → returned as-is, fk_ok=True.
    SO-101 ee mode  : FK applied to joint angles (degrees); fk_ok=False if unavailable.
    SO-101 joint mode: [:3] returned directly (shoulder_pan/lift/elbow_flex), fk_ok=False.
    """
    if actions.ndim != 2 or actions.shape[1] < 3:
        return (actions[:, :3].copy() if actions.ndim == 2 else np.zeros((0, 3))), False
    if cfg.robot_type != "so101":
        return actions[:, :3].copy(), True
    viz_mode = getattr(cfg, "viz_mode", "ee")
    if viz_mode == "joint":
        return actions[:, :3].copy(), False
    return _to_ee_xyz(actions, cfg.fk_solver)


def _plot_cand_3d(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """3D candidate action trajectories vs selected — one subplot per chunk.

    All trajectories are shown in EE Cartesian space:
      LIBERO : action[:3] = EE XYZ directly.
      SO-101 : FK applied to joint angles (degrees) to obtain EE XYZ.

    Multi-candidate chunks  → unselected candidates use a tab10 colormap,
                               selected = green, client_override selected = red.
    Single-candidate chunks → selected_actions plotted in gray ("no candidate").
    """
    import matplotlib.pyplot as plt

    chunks = ep.get("chunks", [])
    if not chunks:
        logging.warning(f"ep{ep['episode']:04d}: no chunks; skipping cand_3d")
        return

    ep_id = ep["episode"]
    success = ep.get("success")
    title_suffix = " ✓" if success else (" ✗" if success is False else "")

    # Include ALL chunks: multi-cand and single-cand (phase-1 / top_k=1)
    n = len(chunks)
    ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(max(4 * ncols, 6), max(4 * nrows, 4)))

    # Probe FK availability once using the first available action
    _fk_ok = True
    for _c in chunks:
        _probe = _c.get("candidates", [{}])[0].get("actions") or _c.get("selected_actions")
        if _probe:
            _, _fk_ok = _actions_to_ee(np.array(_probe), cfg)
            break
    xl, yl, zl = _ee_labels(cfg.robot_type, _fk_ok)

    # Palette for unselected candidates — skip green/red reserved for selected/override
    _UNSEL_PALETTE = [
        "#3498db", "#9b59b6", "#f39c12", "#1abc9c",
        "#e67e22", "#2980b9", "#8e44ad", "#16a085",
    ]

    for plot_i, chunk in enumerate(chunks):
        ax = fig.add_subplot(nrows, ncols, plot_i + 1, projection="3d")
        cands = chunk.get("candidates", [])
        sel_idx = chunk.get("selected_candidate_idx", -1)
        override = chunk.get("client_override", False)

        if cands:
            # ── Multi-candidate chunk ────────────────────────────────────────
            # Draw non-top_k (all_candidates) first so top_k draws on top
            all_cands = chunk.get("all_candidates", [])
            for ac in all_cands:
                if ac.get("in_top_k"):
                    continue  # drawn in top_k loop below
                raw = np.array(ac.get("actions", []))
                if raw.ndim != 2 or raw.shape[1] < 3:
                    continue
                pts, _ = _actions_to_ee(raw, cfg)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color="#bbbbbb", lw=0.7, alpha=0.45, linestyle="--",
                        label=f"rank{ac.get('rank','?')} (not top_k)")
                ax.scatter(*pts[0], color="#bbbbbb", s=10, alpha=0.4, depthshade=False)

            unsel_ci = 0
            for ci, cand in enumerate(cands):
                raw = np.array(cand.get("actions", []))  # (T, D)
                if raw.ndim != 2 or raw.shape[1] < 3:
                    continue
                pts, _ = _actions_to_ee(raw, cfg)
                is_sel = cand.get("selected", False) or (ci == sel_idx)
                if is_sel:
                    color = "#e74c3c" if override else "#2ecc71"
                    label = "selected (override)" if override else "selected"
                    lw, alpha = 2.2, 1.0
                else:
                    color = _UNSEL_PALETTE[unsel_ci % len(_UNSEL_PALETTE)]
                    unsel_ci += 1
                    label = f"cand {ci}"
                    lw, alpha = 1.0, 0.75
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=color, lw=lw, alpha=alpha, label=label)
                ax.scatter(*pts[0], color=color, s=20, alpha=alpha, depthshade=False)
            n_all = len(all_cands)
            mode_tag = (f"n={len(cands)}" + (f"/{n_all}" if n_all > len(cands) else "")
                        + ("  override" if override else ""))
        else:
            # ── Single-candidate / Phase-1 chunk (no multi-cand) ────────────
            sel_actions = chunk.get("selected_actions", [])
            if sel_actions:
                raw = np.array(sel_actions)
                if raw.ndim == 2 and raw.shape[1] >= 3:
                    pts, _ = _actions_to_ee(raw, cfg)
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            color="#888888", lw=1.5, alpha=0.7, label="no candidate")
                    ax.scatter(*pts[0], color="#888888", s=20, depthshade=False)
            mode_tag = "no cand"

        ax.set_xlabel(xl, fontsize=6)
        ax.set_ylabel(yl, fontsize=6)
        ax.set_zlabel(zl, fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(
            f"chunk {chunk.get('chunk_idx', plot_i)}  t={chunk.get('first_timestep', '?')}"
            f"  {mode_tag}",
            fontsize=7,
        )
        ax.legend(fontsize=5, loc="upper left")

    n_mc = sum(1 for c in chunks if c.get("candidates"))
    fig.suptitle(
        f"Candidate vs Selected EE 3D {_robot_tag(cfg, _fk_ok)}  ep{ep_id:04d}{title_suffix}"
        f"  mc={n_mc}/{n} chunks\n{ep.get('task','')[:60]}",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_cand_3d.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig ee3d: Episode-level 3D overview (all chunks + exec cmd + feedback)
# ---------------------------------------------------------------------------


def _plot_all_chunks_ee3d(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """Episode-level 3D overview: all selected chunks + executed cmd in EE space.

    LIBERO : action[:3] = EE XYZ directly.
    SO-101 : FK applied to joint angles (degrees) to obtain EE XYZ.

      blue gradient : each chunk's selected action → EE XYZ (planned, one line per chunk)
      red dashed    : all executed["action"] → EE XYZ (exec cmd, continuous)
    Chunk boundary ● / ★ markers on exec cmd line per chunk group.
    Chunk index annotations at the start of each planned trajectory.
    Saved as ep{N:04d}_ee3d.png.
    """
    import matplotlib.pyplot as plt

    chunks = ep.get("chunks", [])
    executed = ep.get("executed", [])
    if not chunks:
        return

    ep_id = ep.get("episode", 0)
    success = ep.get("success")
    title_suffix = " ✓" if success else (" ✗" if success is False else "")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ── Blue gradient: one line per chunk (selected action → EE) ──────────────
    chunk_cmap = plt.get_cmap("Blues")
    has_chunks = False
    _fk_ok = True  # track FK availability for title

    for ci, chunk in enumerate(chunks):
        t = (ci + 1) / max(len(chunks), 1)
        colour = chunk_cmap(0.35 + 0.60 * t)

        sel_actions = _get_chunk_selected_actions(chunk)
        if not sel_actions:
            continue
        raw = np.array(sel_actions)
        if raw.ndim != 2 or raw.shape[1] < 3:
            continue
        pts, _fk_ok = _actions_to_ee(raw, cfg)

        label = "chunks (planned EE)" if not has_chunks else None
        has_chunks = True
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=colour, lw=1.4, alpha=0.92, zorder=5, label=label)
        ax.scatter(*pts[0], color=colour, s=18, zorder=4, alpha=0.9, depthshade=False)
        ax.scatter(*pts[-1], color=colour, s=70, zorder=8, alpha=0.95,
                   marker="*", edgecolors="white", linewidths=0.5, depthshade=False)

        chunk_idx = chunk.get("chunk_idx", ci)
        ax.text(pts[0, 0], pts[0, 1], pts[0, 2], f" c{chunk_idx}",
                fontsize=5, color=colour, zorder=12)

    # ── Red: exec cmd action → EE (continuous + chunk boundary markers) ────────
    exec_col = "#e74c3c"
    exec_steps = [s for s in executed
                  if s.get("action") is not None and len(s.get("action", [])) >= 3]
    if exec_steps:
        raw_exec = np.array([s["action"] for s in exec_steps])
        exec_pts, _fk_ok = _actions_to_ee(raw_exec, cfg)
        if exec_pts.shape[0] >= 2:
            ax.plot(exec_pts[:, 0], exec_pts[:, 1], exec_pts[:, 2],
                    color=exec_col, lw=2.5, alpha=0.85, zorder=6,
                    linestyle="--", label="executed cmd (EE)")
            ax.scatter(*exec_pts[0], color="limegreen", s=60, marker="o",
                       zorder=10, depthshade=False)
            ax.scatter(*exec_pts[-1], color=exec_col, s=120, marker="X",
                       zorder=11, linewidths=2.0, edgecolors="darkred", depthshade=False)

        chunk_exec = _assign_executed_to_chunks(chunks, exec_steps)
        for ci, group in chunk_exec.items():
            valid = [s for s in group if len(s.get("action", [])) >= 3]
            if not valid:
                continue
            raw_g = np.array([s["action"] for s in valid])
            g_pts, _ = _actions_to_ee(raw_g, cfg)
            ax.scatter(*g_pts[0], color=exec_col, s=35, marker="o", zorder=9,
                       edgecolors="black", linewidths=0.8, depthshade=False)
            ax.scatter(*g_pts[-1], color=exec_col, s=70, marker="*", zorder=9,
                       edgecolors="black", linewidths=0.5, depthshade=False)

    xl, yl, zl = _ee_labels(cfg.robot_type, _fk_ok)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_zlabel(zl)
    ax.legend(fontsize=8, loc="upper left")

    n_mc = sum(1 for c in chunks if c.get("candidates"))
    fig.suptitle(
        f"Episode EE 3D Overview {_robot_tag(cfg, _fk_ok)}  ep{ep_id:04d}{title_suffix}"
        f"  mc={n_mc}/{len(chunks)} chunks"
        f"\nblue=planned  red=exec cmd"
        f"\n{ep.get('task', '')[:60]}",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_ee3d.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig SM: SM Phase Timeline (only when enable_gripper_sm=True)
# ---------------------------------------------------------------------------


def _plot_sm_phase_timeline(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """Gripper command + SM grasp phase step-function over executed timesteps.

    Skipped when no non-NORMAL phase is present (SM was off or never triggered).
    Background bands: orange = CLOSING, green = HOLDING.
    Saved as ep{N:04d}_sm_phase.png.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    executed = ep.get("executed", [])
    if not executed:
        return
    if not any(s.get("grasp_phase", "NORMAL") not in ("NORMAL", "") for s in executed):
        return  # SM was off or never transitioned — skip silently

    ep_id = ep["episode"]
    success = ep.get("success")
    title_suf = " ✓" if success else (" ✗" if success is False else "")

    timesteps = [s.get("timestep", i) for i, s in enumerate(executed)]
    gripper_actions = [float(s.get("action", [0.0])[-1]) for s in executed]
    phases = [s.get("grasp_phase", "NORMAL") for s in executed]
    _PHASE_INT = {"NORMAL": 0, "CLOSING": 1, "HOLDING": 2, "REWIND": -1}
    phase_vals = [_PHASE_INT.get(p, 0) for p in phases]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # ── Background phase bands ────────────────────────────────────────────────
    for ax in (ax1, ax2):
        prev = phases[0]
        span_start = timesteps[0]
        for i in range(1, len(timesteps)):
            if phases[i] != prev:
                if prev in _PHASE_BG:
                    ax.axvspan(span_start, timesteps[i], alpha=0.18,
                               color=_PHASE_BG[prev], zorder=0)
                prev = phases[i]
                span_start = timesteps[i]
        if prev in _PHASE_BG:
            ax.axvspan(span_start, timesteps[-1] + 1, alpha=0.18,
                       color=_PHASE_BG[prev], zorder=0)

    # ── ax1: gripper action over time ─────────────────────────────────────────
    ax1.plot(timesteps, gripper_actions, color="#2c3e50", lw=1.5, label="gripper action")
    ax1.axhline(0, color="gray", lw=0.5, linestyle="--", alpha=0.6)
    ax1.set_ylabel("action[-1]\n(>0 close, <0 open)")
    ax1.set_title("Gripper Command")
    ax1.legend(fontsize=8, loc="upper right")

    # ── ax2: phase step function ──────────────────────────────────────────────
    ax2.step(timesteps, phase_vals, where="post", color="#2c3e50", lw=1.8)
    ax2.set_yticks([-1, 0, 1, 2])
    ax2.set_yticklabels(["REWIND", "NORMAL", "CLOSING", "HOLDING"])
    ax2.set_ylabel("SM Phase")
    ax2.set_xlabel("timestep")
    ax2.set_title("SM Grasp Phase")
    ax2.set_ylim(-1.5, 2.5)

    legend_patches = [
        mpatches.Patch(color=_PHASE_BG["CLOSING"], alpha=0.5, label="CLOSING"),
        mpatches.Patch(color=_PHASE_BG["HOLDING"], alpha=0.5, label="HOLDING"),
    ]
    ax2.legend(handles=legend_patches, fontsize=8, loc="upper right")

    # ── Chunk boundary ticks on ax1 ───────────────────────────────────────────
    chunk_ts = [c.get("first_timestep") for c in ep.get("chunks", [])
                if c.get("first_timestep") is not None]
    for ct in chunk_ts:
        ax1.axvline(ct, color="#bdc3c7", lw=0.6, linestyle=":", alpha=0.8)

    fig.suptitle(
        f"SM Phase Timeline  ep{ep_id:04d}{title_suf}\n{ep.get('task', '')[:60]}",
        fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_sm_phase.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2: Score Timeline
# ---------------------------------------------------------------------------


def _plot_score_timeline(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """Server score + spread_l2 per chunk, with SM phase bands and alpha_effective subplot."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    chunks = ep.get("chunks", [])
    if not chunks:
        return

    chunk_idxs   = [c["chunk_idx"] for c in chunks]
    server_scores = [c.get("server_score", 0.0) for c in chunks]
    spread_l2     = [c.get("spread_l2", 0.0) for c in chunks]
    overrides     = [c.get("client_override", False) for c in chunks]
    phases        = [c.get("grasp_phase", "NORMAL") for c in chunks]
    alphas_raw    = [c.get("alpha_effective") for c in chunks]

    has_sm    = any(p not in ("NORMAL", "") for p in phases)
    has_alpha = any(a is not None for a in alphas_raw)

    n_rows = 3 if has_alpha else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_alpha else None

    # ── Phase background bands (all axes) ────────────────────────────────────
    if has_sm:
        for ci, ph in zip(chunk_idxs, phases):
            if ph in _PHASE_BG:
                for ax in filter(None, [ax1, ax2, ax3]):
                    ax.axvspan(ci - 0.5, ci + 0.5, alpha=0.18,
                               color=_PHASE_BG[ph], zorder=0)

    # ── ax1: server score bars ────────────────────────────────────────────────
    colors = ["#e74c3c" if ov else "#3498db" for ov in overrides]
    ax1.bar(chunk_idxs, server_scores, color=colors, alpha=0.8, width=0.7)
    ax1.set_ylabel("server_score")
    ax1.set_title("Server Score per Chunk")
    ax1.axhline(0, color="gray", lw=0.5)
    legend_patches = [
        mpatches.Patch(color="#3498db", label="server-rank-0"),
        mpatches.Patch(color="#e74c3c", label="client_override"),
    ]
    if has_sm:
        legend_patches += [
            mpatches.Patch(color=_PHASE_BG["CLOSING"], alpha=0.5, label="CLOSING"),
            mpatches.Patch(color=_PHASE_BG["HOLDING"], alpha=0.5, label="HOLDING"),
        ]
    ax1.legend(handles=legend_patches, fontsize=7)

    # ── ax2: spread_l2 ───────────────────────────────────────────────────────
    ax2.plot(chunk_idxs, spread_l2, color="#2ecc71", marker="o", markersize=3,
             lw=1.5, label="spread_l2")
    ax2.fill_between(chunk_idxs, spread_l2, alpha=0.15, color="#2ecc71")
    ax2.set_ylabel("spread_l2")
    if ax3 is None:
        ax2.set_xlabel("chunk_idx")
    ax2.set_title("Candidate Spread (model uncertainty)")
    ax2.legend(fontsize=8)

    # ── ax3: alpha_effective ─────────────────────────────────────────────────
    if ax3 is not None:
        alpha_xs   = [ci for ci, a in zip(chunk_idxs, alphas_raw) if a is not None]
        alpha_vals = [a for a in alphas_raw if a is not None]
        ax3.plot(alpha_xs, alpha_vals, color="#8e44ad", marker="o",
                 markersize=3, lw=1.5, label="alpha_effective")
        ax3.fill_between(alpha_xs, alpha_vals, alpha=0.12, color="#8e44ad")
        ax3.set_ylabel("alpha_effective")
        ax3.set_xlabel("chunk_idx")
        ax3.set_title("Effective Alpha (phase + latency adapted)")
        ax3.set_ylim(0, max(0.8, max(alpha_vals) * 1.1) if alpha_vals else 0.8)
        ax3.legend(fontsize=8)

    ep_id = ep["episode"]
    success = ep.get("success")
    title_suf = " ✓" if success else (" ✗" if success is False else "")
    fig.suptitle(f"Score Timeline  ep{ep_id:04d}{title_suf}", fontsize=10)
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_scores.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Candidate Fan (action space)
# ---------------------------------------------------------------------------


def _plot_candidate_fan(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """Per-action-dim min/max band across candidates + selected line."""
    import matplotlib.pyplot as plt

    chunks = ep.get("chunks", [])
    has_cands = _has_candidates(ep)
    if not chunks:
        return

    dim_names = _dim_names(cfg)
    n_dims = min(cfg.action_dims, len(dim_names))

    # Collect all "selected" first actions and candidate fan per chunk
    chunk_ts: list[int] = []
    sel_first: list[list[float]] = []   # (num_chunks, D)
    cand_min_first: list[list[float]] = []
    cand_max_first: list[list[float]] = []

    for c in chunks:
        sel_acts = c.get("selected_actions", [])
        if not sel_acts:
            continue
        chunk_ts.append(c["first_timestep"])
        sel_first.append(sel_acts[0])

        # fan from candidates if Phase-2
        cands = c.get("candidates", [])
        if cands:
            cand_firsts = [cand["actions"][0] for cand in cands if cand.get("actions")]
        else:
            cand_firsts = [sel_acts[0]]
        arr = np.array(cand_firsts)   # (K, D)
        cand_min_first.append(arr.min(axis=0).tolist())
        cand_max_first.append(arr.max(axis=0).tolist())

    if not chunk_ts:
        return

    ts = np.array(chunk_ts)
    sel = np.array(sel_first)        # (N_chunks, D)
    cmin = np.array(cand_min_first)  # (N_chunks, D)
    cmax = np.array(cand_max_first)  # (N_chunks, D)

    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)

    for dim in range(n_dims):
        r, col = divmod(dim, n_cols)
        ax = axes[r][col]
        ax.fill_between(ts, cmin[:, dim], cmax[:, dim], alpha=0.25, color="#9b59b6", label="cand range")
        ax.plot(ts, sel[:, dim], lw=1.5, color="#9b59b6", label="selected")
        ax.set_title(dim_names[dim], fontsize=9)
        ax.set_xlabel("timestep")
        if dim == 0:
            ax.legend(fontsize=7)

    # hide unused subplots
    for dim in range(n_dims, n_rows * n_cols):
        r, col = divmod(dim, n_cols)
        axes[r][col].set_visible(False)

    ep_id = ep["episode"]
    success = ep.get("success")
    title_suf = " ✓" if success else (" ✗" if success is False else "")
    fig.suptitle(
        f"Candidate Action Fan (first action of chunk)  ep{ep_id:04d}{title_suf}",
        fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_cand_fan.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4: Selection Analysis
# ---------------------------------------------------------------------------


def _plot_selection_analysis(ep: dict, out_dir: Path, cfg: argparse.Namespace) -> None:
    """Override rate + score scatter + delay distribution + per-SM-phase override rate."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import Counter, defaultdict

    chunks = ep.get("chunks", [])
    if not chunks:
        return

    has_cands   = _has_candidates(ep)
    has_sm      = _has_sm_phases(ep)
    ep_id       = ep["episode"]
    success     = ep.get("success")
    title_suf   = " ✓" if success else (" ✗" if success is False else "")

    # Layout: base subplots + 1 extra when SM phases present and multi-cand
    base_n = 3 if has_cands else 2
    extra  = 1 if (has_sm and has_cands) else 0
    n_sub  = base_n + extra
    fig, axes = plt.subplots(1, n_sub, figsize=(4.5 * n_sub, 4))
    if n_sub == 1:
        axes = [axes]

    chunk_idxs = [c["chunk_idx"] for c in chunks]
    overrides  = [int(c.get("client_override", False)) for c in chunks]
    phases_ch  = [c.get("grasp_phase", "NORMAL") for c in chunks]

    # ── Subplot 1: cumulative override rate with SM phase bands ───────────────
    ax_ov = axes[0]
    if has_sm:
        for ci, ph in zip(chunk_idxs, phases_ch):
            if ph in _PHASE_BG:
                ax_ov.axvspan(ci - 0.5, ci + 0.5, alpha=0.18,
                              color=_PHASE_BG[ph], zorder=0)
    cumulative_rate = np.cumsum(overrides) / (np.arange(len(overrides)) + 1)
    ax_ov.bar(chunk_idxs, overrides, alpha=0.4, color="#e74c3c", label="override (binary)")
    ax_ov.plot(chunk_idxs, cumulative_rate, color="#c0392b", lw=1.5, label="cumulative rate")
    ax_ov.set_xlabel("chunk_idx")
    ax_ov.set_ylabel("override / rate")
    ax_ov.set_ylim(-0.05, 1.1)
    ax_ov.set_title(f"Client Override (total={sum(overrides)}/{len(overrides)})")
    ax_ov.legend(fontsize=8)

    # ── Subplot 2: server score vs continuity score scatter ───────────────────
    ax_sc = axes[1]
    if has_cands:
        for c in chunks:
            ph = c.get("grasp_phase", "NORMAL")
            ph_color = _PHASE_COL.get(ph, "#888888")
            for cand in c.get("candidates", []):
                selected = cand.get("selected", False)
                ax_sc.scatter(
                    cand.get("server_score", 0.0),
                    cand.get("continuity_score", 0.0),
                    color="#e74c3c" if selected else ph_color,
                    s=20 if selected else 8,
                    alpha=0.75,
                    zorder=3 if selected else 1,
                )
        ax_sc.set_xlabel("server_score")
        ax_sc.set_ylabel("continuity_score")
        ax_sc.set_title("Score Space (colored by SM phase)")
        sel_patch = mpatches.Patch(color="#e74c3c", label="selected")
        phase_patches = [mpatches.Patch(color=_PHASE_COL.get(p, "#888888"), label=p)
                         for p in ["NORMAL", "CLOSING", "HOLDING"] if p in set(phases_ch)]
        ax_sc.legend(handles=[sel_patch] + phase_patches, fontsize=7)
    else:
        scores = [c.get("server_score", 0.0) for c in chunks]
        ax_sc.plot(chunk_idxs, scores, color="#3498db", lw=1.5)
        ax_sc.set_xlabel("chunk_idx")
        ax_sc.set_ylabel("server_score")
        ax_sc.set_title("Server Score (Phase-1)")

    # ── Subplot 3: delay distribution of selected candidates ──────────────────
    if has_cands:
        ax_dl = axes[2]
        delays = [
            cand.get("delay")
            for c in chunks
            for cand in c.get("candidates", [])
            if cand.get("selected", False) and cand.get("delay") is not None
        ]
        if delays:
            cnt = Counter(delays)
            ax_dl.bar(list(cnt.keys()), list(cnt.values()), color="#1abc9c", alpha=0.8)
            ax_dl.set_xlabel("inference_delay (selected)")
            ax_dl.set_ylabel("count")
            ax_dl.set_title("Selected Candidate Delay Distribution")
        else:
            ax_dl.set_visible(False)

    # ── Subplot 4 (optional): per-SM-phase override rate ─────────────────────
    if has_sm and has_cands:
        ax_ph = axes[base_n]
        phase_data: dict[str, list[int]] = defaultdict(list)
        for ph, ov in zip(phases_ch, overrides):
            phase_data[ph].append(ov)
        phase_order = [p for p in ["NORMAL", "CLOSING", "HOLDING"] if p in phase_data]
        rates  = [sum(phase_data[p]) / len(phase_data[p]) for p in phase_order]
        counts = [len(phase_data[p]) for p in phase_order]
        bar_colors = [_PHASE_COL.get(p, "#888888") for p in phase_order]
        ax_ph.bar(phase_order, rates, color=bar_colors, alpha=0.85, width=0.5)
        for i, (r, n) in enumerate(zip(rates, counts)):
            ax_ph.text(i, r + 0.03, f"{r:.0%}\n(n={n})",
                       ha="center", va="bottom", fontsize=9)
        ax_ph.set_ylabel("override rate")
        ax_ph.set_ylim(0, 1.2)
        ax_ph.set_title("Override Rate by SM Phase")

    fig.suptitle(f"Selection Analysis  ep{ep_id:04d}{title_suf}", fontsize=10)
    fig.tight_layout()
    out = out_dir / f"ep{ep_id:04d}_selection.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Aggregate summary plot
# ---------------------------------------------------------------------------


def _plot_aggregate(episodes: list[dict], out_dir: Path, cfg: argparse.Namespace) -> None:
    """Multi-episode summary: success rate, override rate, spread_l2, SM phase distribution."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_ep = len(episodes)
    if n_ep < 2:
        return

    ep_ids = [ep["episode"] for ep in episodes]
    successes = [1 if ep.get("success") else 0 for ep in episodes]
    override_rates = []
    mean_spreads = []
    for ep in episodes:
        chunks = ep.get("chunks", [])
        if chunks:
            overrides = [int(c.get("client_override", False)) for c in chunks]
            override_rates.append(sum(overrides) / len(overrides))
            spreads = [c.get("spread_l2", 0.0) for c in chunks if c.get("spread_l2", 0.0) > 0]
            mean_spreads.append(np.mean(spreads) if spreads else 0.0)
        else:
            override_rates.append(0.0)
            mean_spreads.append(0.0)

    # Collect per-episode SM phase fractions (fraction of executed steps in each phase)
    has_agg_sm = any(_has_sm_phases(ep) for ep in episodes)
    phase_order = ["NORMAL", "CLOSING", "HOLDING"]
    phase_fracs: dict[str, list[float]] = {p: [] for p in phase_order}
    if has_agg_sm:
        for ep in episodes:
            steps = ep.get("executed", [])
            total = len(steps) or 1
            counts = {p: 0 for p in phase_order}
            for s in steps:
                ph = s.get("grasp_phase", "NORMAL") or "NORMAL"
                if ph in counts:
                    counts[ph] += 1
                else:
                    counts["NORMAL"] += 1
            for p in phase_order:
                phase_fracs[p].append(counts[p] / total)

    n_sub = 4 if has_agg_sm else 3
    fig_w = 5 * n_sub
    fig, axes = plt.subplots(1, n_sub, figsize=(fig_w, 4))

    # Success by episode
    colors = ["#2ecc71" if s else "#e74c3c" for s in successes]
    axes[0].bar(ep_ids, successes, color=colors, alpha=0.8)
    sr = np.mean(successes)
    axes[0].axhline(sr, color="black", lw=1.5, linestyle="--", label=f"SR={sr:.1%}")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("success")
    axes[0].set_title("Success per Episode")
    axes[0].legend()

    # Override rate per episode
    axes[1].bar(ep_ids, override_rates, color="#9b59b6", alpha=0.8)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("override rate")
    axes[1].set_title("Client Override Rate per Episode")

    # Mean spread_l2 per episode (model uncertainty)
    axes[2].bar(ep_ids, mean_spreads, color="#1abc9c", alpha=0.8)
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("mean spread_l2")
    axes[2].set_title("Mean Candidate Spread per Episode")

    # SM phase distribution per episode (stacked bars, fraction of executed steps)
    if has_agg_sm:
        ax_ph = axes[3]
        bottoms = np.zeros(n_ep)
        x = np.arange(n_ep)
        for p in phase_order:
            fracs = np.array(phase_fracs[p])
            ax_ph.bar(x, fracs, bottom=bottoms,
                      color=_PHASE_COL.get(p, "#888888"), alpha=0.85,
                      label=p, width=0.6)
            bottoms += fracs
        ax_ph.set_xticks(x)
        ax_ph.set_xticklabels(ep_ids, rotation=45 if n_ep > 8 else 0, ha="right")
        ax_ph.set_xlabel("episode")
        ax_ph.set_ylabel("fraction of steps")
        ax_ph.set_ylim(0, 1.05)
        ax_ph.set_title("SM Phase Distribution per Episode")
        patches = [mpatches.Patch(color=_PHASE_COL.get(p, "#888888"), label=p)
                   for p in phase_order]
        ax_ph.legend(handles=patches, fontsize=8, loc="upper right")

    fig.suptitle(
        f"Aggregate Summary  ({n_ep} episodes, SR={sr:.1%})",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "aggregate_summary.png"
    fig.savefig(out, dpi=cfg.dpi)
    if cfg.show:
        plt.show()
    plt.close(fig)
    logging.info(f"Saved aggregate summary → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze_multicand_trajectory(cfg: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import matplotlib
        matplotlib.use("Agg" if not cfg.show else "TkAgg")
    except Exception:
        pass

    # Initialise FK solver once (reused across all episodes/plots).
    # Only attempted for SO-101 in ee mode; joint mode skips FK entirely.
    cfg.fk_solver = (
        _init_fk_solver(cfg)
        if cfg.robot_type == "so101" and getattr(cfg, "viz_mode", "ee") == "ee"
        else None
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_episodes(cfg)
    if not episodes:
        logging.error("No episodes loaded. Exiting.")
        sys.exit(1)

    for ep in episodes:
        ep_id = ep.get("episode", "?")
        n_chunks = len(ep.get("chunks", []))
        n_steps = len(ep.get("executed", []))
        logging.info(
            f"Processing ep{ep_id:04d}  chunks={n_chunks}  executed_steps={n_steps}"
            f"  success={ep.get('success')}  task='{ep.get('task', '')[:40]}'"
        )
        _plot_ee_trajectory(ep, out_dir, cfg)
        _plot_ee_trajectory_3d(ep, out_dir, cfg)
        _plot_cand_3d(ep, out_dir, cfg)
        _plot_all_chunks_ee3d(ep, out_dir, cfg)
        _plot_score_timeline(ep, out_dir, cfg)
        _plot_candidate_fan(ep, out_dir, cfg)
        _plot_selection_analysis(ep, out_dir, cfg)
        _plot_sm_phase_timeline(ep, out_dir, cfg)

    if len(episodes) >= 2:
        _plot_aggregate(episodes, out_dir, cfg)

    logging.info(f"Done. Figures written to {out_dir}/")


if __name__ == "__main__":
    analyze_multicand_trajectory(_build_parser().parse_args())
