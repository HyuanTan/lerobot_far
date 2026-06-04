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

"""Thread-safe per-episode trajectory recorder for multi-candidate LIBERO evaluation.

JSON schema written per episode::

    {
      "episode":   int,
      "task":      str,
      "success":   bool | null,
      "total_steps": int,

      "chunks": [
        {
          "chunk_idx":              int,       # 0-based sequence within episode
          "first_timestep":         int,       # timestep of selected[0]
          "n_candidates":           int,
          "selected_candidate_idx": int,       # -1 = Phase-1 (server-only)
          "client_override":        bool,
          "server_score":           float,     # bundle.selected_score
          "spread_l2":              float,     # mean pairwise L2 across candidates
          "grasp_phase":            str,       # SM phase at selection time: NORMAL|CLOSING|HOLDING
          "alpha_effective":        float,     # actual alpha after phase + latency adaptation
          "selected_actions":       [[float]], # T × D (selected chunk)
          "candidates": [                      # empty in Phase-1; top_k only
            {
              "delay":              int | null,
              "noise_idx":          int,
              "jerk":               float,
              "vel_peak":           float,
              "server_score":       float,
              "continuity_score":   float,
              "combined_score":     float,
              "selected":           bool,
              "actions":            [[float]]  # T × D
            }
          ],
          "all_candidates": [                  # all N server candidates, ranked by server score
            {                                  # empty unless --record_all_candidates=true
              "rank":         int,             # 0 = best server score
              "in_top_k":     bool,            # True if also in candidates[]
              "delay":        int | null,
              "noise_idx":    int,
              "jerk":         float,
              "vel_peak":     float,
              "server_score": float,
              "actions":      [[float]]        # T × D
            }
          ]
        }
      ],

      "executed": [
        {
          "timestep":     int,
          "action":       [float],        # D-dim, post-processed
          "robot_state":  [float] | null, # 9-dim: EE pos(3)+quat(4)+gripper_qpos(2)
          "episode_phase": float,         # timestep / max_steps
          "grasp_phase":  str             # SM phase: NORMAL|CLOSING|HOLDING (absent when SM off)
        }
      ]
    }
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class TrajectoryRecorder:
    """Thread-safe recorder that serialises one JSON file per episode.

    Thread model
    ─────────────
    - ``record_chunk()``  → receiver thread (one at a time)
    - ``record_step()``   → control-loop thread (serialised by control loop)
    - ``reset()``         → main / test-loop thread (between episodes)
    - ``save()``          → main / test-loop thread (after episode ends)

    The lock guards the shared list objects; individual list.append() calls
    are themselves atomic under CPython GIL, but we hold the lock anyway for
    correctness across ``reset()`` / ``save()`` boundaries.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Initialised by reset(); safe to call record_* before first reset()
        # (records just accumulate under episode_id=0).
        self._episode_id: int = 0
        self._task: str = ""
        self._chunks: list[dict] = []
        self._executed: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, episode_id: int, task: str) -> None:
        """Clear per-episode buffers.  Call from main thread before each episode."""
        with self._lock:
            self._episode_id = episode_id
            self._task = task
            self._chunks = []
            self._executed = []

    def save(self, success: bool | None, total_steps: int) -> Path:
        """Flush episode data to ``<output_dir>/ep{N:04d}.json``.

        Call from main thread after episode ends.  Thread-safe: acquires lock
        to snapshot lists, then writes without holding the lock.
        """
        with self._lock:
            data = {
                "episode": self._episode_id,
                "task": self._task,
                "success": success,
                "total_steps": total_steps,
                "chunks": list(self._chunks),
                "executed": list(self._executed),
            }
        out = self._output_dir / f"ep{self._episode_id:04d}.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Per-chunk and per-step recording
    # ------------------------------------------------------------------

    def record_chunk(self, chunk_data: dict) -> None:
        """Append a chunk record.  Called from receiver thread."""
        with self._lock:
            self._chunks.append(chunk_data)

    def record_step(self, step_data: dict) -> None:
        """Append a step record.  Called from control-loop thread."""
        with self._lock:
            self._executed.append(step_data)
