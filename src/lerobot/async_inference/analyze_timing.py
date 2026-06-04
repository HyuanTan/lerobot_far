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

"""Timing analysis for async-inference client/server records.

Loads JSONL timing files produced by TimingRecorder and generates:
  - Console tables (warmup detection, latency budget, tail latency, per-episode, diagnosis)
  - PNG figures (budget bar, time series, pipeline breakdown,
                 tail latency, control-loop health, per-episode stats, wall-clock timeline)

Usage::

    # Convenience: pass the eval output directory (auto-discovers sub-dirs)
    python -m lerobot.async_inference.analyze_timing \\
        ~/outputs/eval/libero/pi05/sim/libero_goal

    # Explicit directories
    python -m lerobot.async_inference.analyze_timing \\
        --client_dir ./client_timing \\
        --server_dir ./server_timing \\
        --out_dir ./timing_analysis

    # Client-only (no server data)
    python -m lerobot.async_inference.analyze_timing \\
        --client_dir ./client_timing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

# ── Colour palette ────────────────────────────────────────────────────────────

_C = {
    "obs_prep":       "#4C9BE8",
    "serialize":      "#7DB3E8",
    "net_c2s":        "#F5A623",
    "recv_deser_srv": "#F7C266",   # server obs-deser: hidden inside grpc_send_ms; split out for clarity
    "server_recv":    "#F7C266",
    "queue_wait":     "#E87D4C",
    "prepare":        "#A855F7",
    "preprocess":     "#C084FC",
    "infer":          "#D946EF",
    "postprocess":    "#F472B6",
    "srv_serialize":  "#BE185D",
    "net_s2c":        "#34D399",
    "deser":          "#7DB3E8",
    "other":          "#9CA3AF",
}


# SM event colour palette and canonical ordering (mirrors GripperDecision names)
_SM_COLORS: dict[str, str] = {
    "grasp_success":        "#22C55E",   # green
    "empty_grasp":          "#EF4444",   # red
    "slip":                 "#F5A623",   # orange
    "stop":                 "#7C3AED",   # purple
    "recovery_home_ready":  "#60A5FA",   # blue  — arm back at home, obs-send resumes
    "lift_position_ready":  "#FCD34D",   # yellow — arm at lift position, obs-send resumes
}
_SM_ORDER = ("grasp_success", "empty_grasp", "slip", "stop",
             "recovery_home_ready", "lift_position_ready")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_json(path, lines=True, convert_dates=False)
    return None


def _auto_discover(eval_dir: str) -> tuple[Path | None, Path | None, Path | None]:
    """Search eval_dir for standard timing sub-directories."""
    ed = Path(eval_dir)
    client_dir  = ed / "client_timing"  if (ed / "client_timing").is_dir()  else None
    server_dir  = ed / "server_timing"  if (ed / "server_timing").is_dir()  else None
    # Accept "sim_test_results" (legacy) or "results" (current eval scripts)
    if (ed / "sim_test_results").is_dir():
        results_dir = ed / "sim_test_results"
    elif (ed / "results").is_dir():
        results_dir = ed / "results"
    else:
        results_dir = None
    return client_dir, server_dir, results_dir


def load_data(
    client_dir: str | Path | None,
    server_dir: str | Path | None,
) -> dict[str, pd.DataFrame | None]:
    cd = Path(client_dir) if client_dir else None
    sd = Path(server_dir) if server_dir else None
    data: dict[str, pd.DataFrame | None] = {
        "sent":         _load(cd / "client_obs_sent_records.jsonl")     if cd else None,
        "chunk":        _load(cd / "client_chunk_recv_records.jsonl")    if cd else None,
        "recv":         _load(sd / "server_recv_records.jsonl")          if sd else None,
        "infer":        _load(sd / "server_infer_records.jsonl")         if sd else None,
        "control_step": _load(cd / "client_control_step_records.jsonl")  if cd else None,
    }
    for name, df in data.items():
        if df is not None:
            print(f"  Loaded {name:16s}: {len(df):6d} records")
        else:
            print(f"  Loaded {name:16s}: (not found)")
    return data


def load_results(results_dir: Path | str | None) -> dict | None:
    if results_dir is None:
        return None
    rd = Path(results_dir)
    agg = rd / "aggregate.json"
    eps = rd / "episodes.json"
    out: dict = {}
    if agg.exists():
        out["aggregate"] = json.loads(agg.read_text())
    if eps.exists():
        out["episodes"] = json.loads(eps.read_text())
    return out if out else None


# ── Warmup handling ───────────────────────────────────────────────────────────

def _detect_warmup_n(infer: pd.DataFrame | None, threshold: float = 5.0) -> int:
    """Return the number of leading warmup inferences.

    A record is warmup if its infer_ms > threshold × median(infer_ms[3:]).
    """
    if infer is None or len(infer) < 4:
        return 0
    tail_median = float(np.median(infer["infer_ms"].values[3:]))
    n = 0
    for val in infer["infer_ms"].values:
        if val > threshold * tail_median:
            n += 1
        else:
            break
    return n


def _filter_warmup(data: dict, n_warmup: int) -> dict:
    """Return a copy of data with the first n_warmup server infer records removed."""
    if n_warmup == 0:
        return data
    out = dict(data)
    if data["infer"] is not None and n_warmup > 0:
        out["infer"] = data["infer"].iloc[n_warmup:].copy()
    # Also filter the matching chunk records by wall_time (first n_warmup chunks
    # correspond to the warmup inferences by wall_time order).
    if data["chunk"] is not None and n_warmup > 0:
        out["chunk"] = data["chunk"].iloc[n_warmup:].copy()
    return out


def _assign_episodes(data: dict) -> dict:
    """Add 'episode' column to server records using wall_time join with client chunks.

    Server records (recv, infer) lack episode info. We assign episode by finding
    the client chunk record with the closest wall_time.
    """
    chunk = data.get("chunk")
    if chunk is None or "wall_time" not in chunk.columns or "episode" not in chunk.columns:
        return data

    chunk_sorted = chunk.sort_values("wall_time")
    chunk_times = chunk_sorted["wall_time"].values
    chunk_eps   = chunk_sorted["episode"].values.astype(int)

    out = dict(data)
    for key in ("recv", "infer"):
        df = data.get(key)
        if df is None or "wall_time" not in df.columns or "episode" in df.columns:
            continue
        wt = df["wall_time"].values
        idx = np.searchsorted(chunk_times, wt).clip(0, len(chunk_times) - 1)
        prev = np.maximum(idx - 1, 0)
        use_prev = np.abs(wt - chunk_times[prev]) < np.abs(wt - chunk_times[idx])
        idx = np.where(use_prev, prev, idx)
        df2 = df.copy()
        df2["episode"] = chunk_eps[idx]
        out[key] = df2

    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(arr, p):
    valid = arr.dropna()
    return float(np.percentile(valid, p)) if len(valid) else float("nan")


def _stats(s: pd.Series) -> dict:
    v = s.dropna()
    if len(v) == 0:
        return dict(mean=np.nan, std=np.nan, p50=np.nan, p95=np.nan, p99=np.nan, max=np.nan, n=0)
    return dict(
        mean=float(v.mean()),
        std =float(v.std()),
        p50 =float(np.percentile(v, 50)),
        p95 =float(np.percentile(v, 95)),
        p99 =float(np.percentile(v, 99)),
        max =float(v.max()),
        n   =int(len(v)),
    )


def _divider(title: str = "", width: int = 80):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


def _build_budget(data: dict) -> pd.Series | None:
    """Assemble mean latency budget (ms) using the provided data dict.

    Pass warmup-filtered data to get an accurate budget.
    """
    sent  = data["sent"]
    chunk = data["chunk"]
    infer = data["infer"]

    if sent is None or chunk is None:
        return None

    recv  = data.get("recv")

    budget = {}
    budget["obs_prep"]  = sent["total_prep_ms"].mean()
    # jpeg_encode_ms sits between obs_prep and serialize in the client timeline
    if "jpeg_encode_ms" in sent.columns:
        budget["jpeg_encode"] = sent["jpeg_encode_ms"].mean()
    budget["serialize"] = sent["serialize_ms"].mean()

    # grpc_send_ms on the client includes: true_net_c2s + server recv_deser + ACK return.
    # When server recv records are available, subtract recv_deser_ms to isolate the network
    # transfer portion and expose server obs-deser as its own budget stage.
    _grpc_mean = sent["grpc_send_ms"].mean()
    if recv is not None and "recv_deser_ms" in recv.columns:
        _recv_deser_mean = recv["recv_deser_ms"].mean()
        budget["net_c2s"]        = max(0.0, _grpc_mean - _recv_deser_mean)
        budget["recv_deser_srv"] = _recv_deser_mean
    else:
        budget["net_c2s"] = _grpc_mean

    if infer is not None:
        budget["queue_wait"]    = infer["queue_wait_ms"].mean()
        budget["prepare"]       = infer["prepare_ms"].mean()
        budget["preprocess"]    = infer["preprocess_ms"].mean()
        budget["infer"]         = infer["infer_ms"].mean()
        budget["postprocess"]   = infer["postprocess_ms"].mean()
        budget["srv_serialize"] = infer["serialize_ms"].mean()
        # throttle_sleep is outside total_pipeline_ms; include it in budget decomposition
        if "throttle_sleep_ms" in infer.columns:
            budget["throttle_sleep"] = infer["throttle_sleep_ms"].mean()
        rt   = chunk["round_trip_ms"].dropna().mean()
        used = (budget["queue_wait"] + infer["total_pipeline_ms"].mean()
                + budget["srv_serialize"]
                + budget.get("throttle_sleep", 0.0))
        budget["net_s2c"] = max(0.0, rt - used)
    else:
        rt  = chunk["round_trip_ms"].dropna().mean()
        si  = chunk["server_infer_ms"].dropna().mean()
        budget["server_infer"] = si
        budget["net_s2c"]      = max(0.0, rt - si)

    budget["deser"] = chunk["deser_ms"].mean()
    return pd.Series(budget)


# ══════════════════════════════════════════════════════════════════════════════
# Console tables
# ══════════════════════════════════════════════════════════════════════════════

def print_warmup_info(data_raw: dict, n_warmup: int):
    """Report detected warmup inferences and their timing."""
    _divider("WARMUP DETECTION", 80)
    infer = data_raw["infer"]
    if infer is None:
        print("  (no server inference data)")
        return

    if n_warmup == 0:
        print("  No warmup inferences detected (all infer_ms within 5× of steady-state).")
        return

    print(f"  Detected {n_warmup} warmup inference(s) — excluded from budget and episode plots.\n")
    hdr = f"  {'idx':>4}  {'infer_ms':>12}  {'total_pipeline_ms':>18}  {'queue_wait_ms':>14}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i in range(min(n_warmup, len(infer))):
        r = infer.iloc[i]
        print(f"  {i:>4}  {r['infer_ms']:>12.1f}  {r['total_pipeline_ms']:>18.1f}  "
              f"{r['queue_wait_ms']:>14.2f}")

    inf_filt = data_raw["infer"].iloc[n_warmup:]
    if len(inf_filt) > 0:
        print(f"\n  Steady-state infer_ms (n={len(inf_filt)}):  "
              f"mean={inf_filt['infer_ms'].mean():.1f}  "
              f"p50={np.percentile(inf_filt['infer_ms'], 50):.1f}  "
              f"p95={np.percentile(inf_filt['infer_ms'], 95):.1f}  "
              f"max={inf_filt['infer_ms'].max():.1f}  ms")


def print_stats_table(data: dict, data_filtered: dict, n_warmup: int):
    """Full percentile table for every timing field, with warmup-excluded column for key server metrics."""
    _divider("FULL STATS TABLE (ms unless noted)", 80)

    sections = [
        ("Client — Obs Sent",     data["sent"],  [
            "obs_capture_ms", "infer_delay_calc_ms", "leftover_collect_ms",
            "obs_build_ms", "total_prep_ms",
            "jpeg_encode_ms",          # Issue 1: was missing; 0.0 when JPEG disabled
            "serialize_ms", "grpc_send_ms", "payload_kb",
        ]),
        ("Client — Chunk Recv",   data["chunk"], [
            "round_trip_ms", "server_infer_ms", "deser_ms", "chunk_size",
            "estimated_first_exec_lag_ms",  # Issue 5: queue_depth×dt before first action executes
        ]),
        ("Server — Obs Received", data["recv"],  [
            "recv_deser_ms", "one_way_ms",
            "adj_one_way_ms",          # Issue 6: one_way_ms minus client JPEG overhead
        ]),
        ("Server — Inference",    data["infer"], [
            "queue_wait_ms", "prepare_ms", "preprocess_ms", "infer_ms",
            "postprocess_ms", "total_pipeline_ms", "serialize_ms",
            "throttle_sleep_ms",       # Issue 2: inference_latency throttle sleep duration
        ]),
    ]
    server_sections = {"Server — Inference", "Client — Chunk Recv"}

    for title, df, cols in sections:
        if df is None:
            continue
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        rows = []
        for c in available:
            s = _stats(df[c])
            n_nan = int(df[c].isna().sum())
            nan_tag = f"({n_nan} NaN)" if n_nan else ""

            # Add warmup-excluded column for server inference metrics
            wex = ""
            if n_warmup > 0 and title in server_sections:
                key = "infer" if title == "Server — Inference" else "chunk"
                df_f = data_filtered.get(key)
                if df_f is not None and c in df_f.columns:
                    sf = _stats(df_f[c])
                    wex = f"{sf['mean']:7.2f}"

            rows.append({
                "field":       c,
                "mean":        f"{s['mean']:7.2f}",
                "std":         f"{s['std']:7.2f}",
                "p50":         f"{s['p50']:7.2f}",
                "p95":         f"{s['p95']:7.2f}",
                "p99":         f"{s['p99']:7.2f}",
                "max":         f"{s['max']:7.2f}",
                "n":           f"{s['n']}",
                "*mean_no_wup": wex,
                "notes":       nan_tag,
            })
        tdf = pd.DataFrame(rows).set_index("field")
        if n_warmup == 0 or title not in server_sections:
            tdf = tdf.drop(columns=["*mean_no_wup"])
        print(f"\n  ── {title} ({len(df)} records)")
        print(tdf.to_string())
    if n_warmup > 0:
        print(f"\n  * *mean_no_wup: mean excluding first {n_warmup} warmup inference(s).")


def print_budget_table(data_raw: dict, data_filtered: dict, n_warmup: int):
    """Latency budget: absolute ms + % of round-trip."""
    _divider("LATENCY BUDGET DECOMPOSITION", 80)

    budget_raw = _build_budget(data_raw)
    budget_flt = _build_budget(data_filtered) if n_warmup > 0 else None

    if budget_raw is None:
        print("  (insufficient data — need both sent and chunk records)")
        return

    chunk_raw = data_raw["chunk"]
    rt_raw = chunk_raw["round_trip_ms"].dropna().mean() if chunk_raw is not None else budget_raw.sum()

    rows = []
    for stage, ms in budget_raw.items():
        row: dict = {
            "stage":               stage,
            "mean_ms (all)":       f"{ms:.2f}",
            "% of rt (all)":       f"{100 * ms / rt_raw:.1f}%" if rt_raw else "—",
        }
        if budget_flt is not None and stage in budget_flt:
            chunk_flt = data_filtered["chunk"]
            rt_flt = chunk_flt["round_trip_ms"].dropna().mean() if chunk_flt is not None else budget_flt.sum()
            ms_f = budget_flt[stage]
            row["mean_ms (no wup)"] = f"{ms_f:.2f}"
            row["% of rt (no wup)"] = f"{100 * ms_f / rt_flt:.1f}%" if rt_flt else "—"
        rows.append(row)

    rt_note = f"(warmup-excluded rt: {data_filtered['chunk']['round_trip_ms'].dropna().mean():.1f} ms)" if budget_flt is not None else ""
    rows.append({
        "stage": "─── round_trip_total ───",
        "mean_ms (all)": f"{rt_raw:.2f}",
        "% of rt (all)": "100.0%",
        "mean_ms (no wup)": rt_note,
        "% of rt (no wup)": "" if budget_flt is None else "100.0%",
    })

    tdf = pd.DataFrame(rows)
    print(tdf.to_string(index=False))
    if n_warmup > 0:
        print(f"\n  Note: 'no wup' columns exclude the first {n_warmup} warmup inference(s).")


def print_tail_table(data: dict):
    """Tail-latency table: p99/p50 ratio to identify jitter sources."""
    _divider("TAIL LATENCY  (p50 / p95 / p99 / p99÷p50)", 80)

    pairs = [
        ("round_trip_ms",               data["chunk"]),
        ("server_infer_ms",             data["chunk"]),
        ("estimated_first_exec_lag_ms", data["chunk"]),  # Issue 5
        ("grpc_send_ms",                data["sent"]),
        ("jpeg_encode_ms",              data["sent"]),   # Issue 1
        ("serialize_ms",                data["sent"]),
        ("deser_ms",                    data["chunk"]),
        ("queue_wait_ms",               data["infer"]),
        ("total_pipeline_ms",           data["infer"]),
        ("infer_ms",                    data["infer"]),
        ("preprocess_ms",               data["infer"]),
        ("throttle_sleep_ms",           data["infer"]),  # Issue 2
        ("recv_deser_ms",               data["recv"]),
        ("one_way_ms",                  data["recv"]),
        ("adj_one_way_ms",              data["recv"]),   # Issue 6
    ]

    rows = []
    for field, df in pairs:
        if df is None or field not in df.columns:
            continue
        s = _stats(df[field])
        ratio = s["p99"] / s["p50"] if s["p50"] > 0 else float("nan")
        flag = "⚠ " if ratio > 3 else ("△ " if ratio > 1.5 else "  ")
        rows.append({
            "field":   field,
            "p50":     f"{s['p50']:7.2f}",
            "p95":     f"{s['p95']:7.2f}",
            "p99":     f"{s['p99']:7.2f}",
            "p99/p50": f"{ratio:5.2f}x",
            "status":  flag,
        })

    tdf = pd.DataFrame(rows).set_index("field")
    print(tdf.to_string())
    print("\n  Legend:  ⚠ p99/p50 > 3x (severe jitter)  △ 1.5–3x  (moderate)  blank: stable")


def print_per_episode_table(data: dict, results: dict | None):
    """Per-episode breakdown from chunk and server records."""
    _divider("PER-EPISODE STATS", 80)

    chunk = data.get("chunk")
    infer = data.get("infer")

    if chunk is None or "episode" not in chunk.columns:
        print("  (no episode info in chunk records)")
        return

    episodes = sorted(chunk["episode"].unique())

    # Build per-episode stats from chunk records
    rows = []
    for ep in episodes:
        mask_c = chunk["episode"] == ep
        ep_chunk = chunk[mask_c]
        rt_p50 = np.percentile(ep_chunk["round_trip_ms"].dropna(), 50) if len(ep_chunk) > 0 else np.nan
        si_p50 = np.percentile(ep_chunk["server_infer_ms"].dropna(), 50) if len(ep_chunk) > 0 else np.nan
        n_chunks = len(ep_chunk)

        # Server infer stats for this episode (if available and episode was assigned)
        qw_p50 = np.nan
        if infer is not None and "episode" in infer.columns:
            mask_i = infer["episode"] == ep
            ep_infer = infer[mask_i]
            if len(ep_infer) > 0:
                qw_p50 = np.percentile(ep_infer["queue_wait_ms"].dropna(), 50)

        row: dict = {
            "ep":         int(ep),
            "n_chunks":   n_chunks,
            "rt_p50(ms)": f"{rt_p50:.1f}",
            "si_p50(ms)": f"{si_p50:.1f}",
            "qw_p50(ms)": f"{qw_p50:.1f}" if not np.isnan(qw_p50) else "—",
        }

        # Add success if results available
        if results and "episodes" in results:
            ep_results = results["episodes"]
            ep_r = next((r for r in ep_results if r.get("episode_id") == int(ep)), None)
            if ep_r:
                row["success"] = "✓" if ep_r.get("success") else "✗"
                row["steps"] = ep_r.get("steps", "—")

        rows.append(row)

    if rows:
        tdf = pd.DataFrame(rows).set_index("ep")
        print(tdf.to_string())

    # Summary from results
    if results and "aggregate" in results:
        agg = results["aggregate"]
        print(f"\n  Overall success rate : {agg.get('overall_success_rate', '?'):.1%}  "
              f"({agg.get('total_episodes', '?')} episodes)")
        per_task = agg.get("per_task", [])
        if per_task:
            print(f"\n  Per-task breakdown:")
            for t in per_task:
                desc = t.get("task_description", "?")[:60]
                sr = t.get("success_rate", 0)
                n = t.get("episodes", 0)
                avg_steps = t.get("avg_steps", 0)
                print(f"    [{sr:.0%}] {desc} (n={n}, avg_steps={avg_steps:.0f})")


def print_obs_funnel(data: dict):
    """Show obs sent → server received → enqueued → inferred funnel."""
    sent  = data.get("sent")
    recv  = data.get("recv")
    infer = data.get("infer")

    if recv is None:
        return

    _divider("OBS PIPELINE FUNNEL", 80)

    n_sent   = len(sent)  if sent  is not None else "?"
    n_recv   = len(recv)
    n_enq    = int(recv["enqueued"].sum()) if "enqueued" in recv.columns else "?"
    n_infer  = len(infer) if infer is not None else "?"

    def _pct_str(num, den):
        if isinstance(num, int) and isinstance(den, int) and den > 0:
            return f"({num/den:.1%})"
        return ""

    print(f"  Client obs sent       : {n_sent}")
    print(f"  Server obs received   : {n_recv}  {_pct_str(n_recv if isinstance(n_recv,int) else 0, n_sent if isinstance(n_sent,int) else 1)}")
    print(f"  Server obs enqueued   : {n_enq}   {_pct_str(n_enq if isinstance(n_enq,int) else 0, n_recv)}")
    print(f"  Server inferences run : {n_infer}  {_pct_str(n_infer if isinstance(n_infer,int) else 0, n_recv)}")
    if isinstance(n_recv, int) and isinstance(n_enq, int) and n_recv > 0:
        filtered = n_recv - n_enq
        print(f"  Filtered (similarity) : {filtered} ({filtered/n_recv:.1%})")
    if isinstance(n_enq, int) and isinstance(n_infer, int) and n_enq > 0:
        replaced = n_enq - n_infer
        print(f"  Queue-replaced        : {replaced} ({replaced/n_enq:.1%}) "
              "(enqueued but superseded before inference)")


def print_diagnosis(data: dict, n_warmup: int):
    """Rule-based quick diagnosis."""
    _divider("QUICK DIAGNOSIS", 80)

    sent  = data["sent"]
    chunk = data["chunk"]
    recv  = data["recv"]
    infer = data["infer"]
    issues: list[tuple[str, str]] = []

    if sent is not None:
        must_go_rate = sent["must_go"].mean()
        print(f"  must_go rate          : {must_go_rate:.1%}"
              + ("  (note: episode-start obs are sent outside timing scope)" if must_go_rate == 0.0 else ""))
        if must_go_rate > 0.3:
            issues.append(("HIGH must_go rate",
                           "Action queue runs empty often → fps too high or inference too slow. "
                           "Try reducing fps or increasing actions_per_chunk."))

        mean_payload = sent["payload_kb"].mean()
        print(f"  mean payload          : {mean_payload:.1f} KB")
        if mean_payload > 200:
            issues.append(("LARGE obs payload",
                           f"{mean_payload:.0f} KB per obs → high serialise/network cost. "
                           "Consider reducing image resolution or enabling obs_pre_mapped."))

    if chunk is not None:
        rt = chunk["round_trip_ms"].dropna()
        print(f"  round_trip p50/p99    : {rt.median():.1f} / {_pct(rt, 99):.1f} ms")
        if rt.std() / rt.mean() > 0.5:
            if n_warmup > 0:
                issues.append(("HIGH round-trip jitter (may include warmup)",
                               "CV > 0.5 — partially explained by warmup inferences. "
                               "Check steady-state jitter in per-episode plot."))
            else:
                issues.append(("HIGH round-trip jitter",
                               "CV > 0.5 → unstable network or sporadic server delays. "
                               "Check server GPU usage and network bandwidth."))

    if recv is not None and "enqueued" in recv.columns:
        filter_rate = 1 - recv["enqueued"].mean()
        print(f"  obs filter rate       : {filter_rate:.1%}")
        if filter_rate > 0.9:
            issues.append(("OBS OVER-FILTERED",
                           f"{filter_rate:.0%} of obs dropped by similarity check → "
                           "robot may be nearly stationary or atol threshold too tight."))
        if filter_rate < 0.1:
            issues.append(("OBS UNDER-FILTERED",
                           f"Only {filter_rate:.0%} filtered → every obs triggers inference. "
                           "Check if obs_similar() is working correctly."))

    if infer is not None:
        qw = infer["queue_wait_ms"]
        print(f"  queue_wait p50/p95    : {qw.median():.1f} / {_pct(qw, 95):.1f} ms")
        if qw.median() > 5:
            issues.append(("SERVER QUEUE BUILDUP",
                           "queue_wait > 5 ms median → obs arrive faster than inference. "
                           "Server is the bottleneck; consider batching or faster GPU."))
        if qw.median() < 0.5:
            issues.append(("SERVER IDLE",
                           "queue_wait ≈ 0 → server is waiting for obs. "
                           "fps can safely be increased."))

        total = infer["total_pipeline_ms"].mean()
        infer_frac  = infer["infer_ms"].mean() / total if total > 0 else 0
        pre_frac    = infer["preprocess_ms"].mean() / total if total > 0 else 0
        print(f"  infer / preprocess    : {infer_frac:.0%} / {pre_frac:.0%} of pipeline")
        if pre_frac > 0.2:
            issues.append(("PREPROCESS BOTTLENECK",
                           f"preprocess = {pre_frac:.0%} of pipeline → "
                           "image resizing/normalization is expensive. "
                           "Move preprocessing to client (obs_pre_mapped=True)."))

    if n_warmup > 0:
        issues.append(("WARMUP DETECTED",
                       f"First {n_warmup} inference(s) are JIT/GPU warmup and heavily skew statistics. "
                       "Budget and episode plots exclude them. Use --warmup_n 0 to include all."))

    print()
    if not issues:
        print("  ✓  No obvious issues detected.")
    else:
        print(f"  {len(issues)} issue(s) found:\n")
        for i, (title, detail) in enumerate(issues, 1):
            print(f"  [{i}] {title}")
            print(f"      {detail}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, path: Path, title: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_frequency_stats(data: dict, out_dir: Path) -> None:
    """Fig 13 – Actual loop frequencies: obs_send / inference / chunk_recv / control_step.

    Two rows per signal:
      Row 0 – instantaneous Hz over wall-clock time (time series)
      Row 1 – interval distribution in ms (violin + p50/p95 text)

    control_step column is added only when data["control_step"] is available.
    """
    signals = [
        ("sent",         "obs send",    "#F59E0B"),  # amber
        ("infer",        "inference",   "#8B5CF6"),  # purple
        ("chunk",        "chunk recv",  "#10B981"),  # green
    ]
    if data.get("control_step") is not None:
        signals.append(("control_step", "control step", "#3B82F6"))  # blue

    n_cols = len(signals)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    fig.suptitle("Analysis 13: Actual Loop Frequencies", fontsize=13, fontweight="bold")

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (key, label, color) in enumerate(signals):
        df = data.get(key)
        ax_ts   = axes[0, col]   # top row: time series
        ax_dist = axes[1, col]   # bottom row: distribution

        ax_ts.set_title(label, fontsize=10, fontweight="bold")
        ax_ts.set_xlabel("Wall-clock time (s)")
        ax_ts.set_ylabel("Hz")
        ax_dist.set_xlabel("Interval (ms)")
        ax_dist.set_ylabel("Count")

        if df is None or "wall_time" not in df.columns or len(df) < 2:
            for ax in (ax_ts, ax_dist):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            continue

        wt = df["wall_time"].sort_values().values
        t0 = wt[0]
        intervals_ms = np.diff(wt) * 1000.0          # ms
        hz_inst      = 1000.0 / np.where(intervals_ms > 0, intervals_ms, np.nan)
        t_mid        = (wt[:-1] + wt[1:]) / 2 - t0  # midpoint wall-clock

        # ── Time series ───────────────────────────────────────────────────────
        ax_ts.scatter(t_mid, hz_inst, s=4, alpha=0.5, color=color, linewidths=0)
        # rolling median for trend line
        if len(hz_inst) >= 10:
            df_tmp = pd.Series(hz_inst, index=t_mid)
            roll = df_tmp.rolling(10, center=True, min_periods=1).median()
            ax_ts.plot(roll.index, roll.values, color=color, linewidth=1.5, alpha=0.9)

        p50_hz = float(np.nanmedian(hz_inst))
        ax_ts.axhline(p50_hz, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax_ts.text(0.98, 0.96, f"p50={p50_hz:.1f} Hz", ha="right", va="top",
                   transform=ax_ts.transAxes, fontsize=8, color=color)
        ax_ts.set_ylim(bottom=0)

        # ── Distribution ──────────────────────────────────────────────────────
        valid = intervals_ms[np.isfinite(intervals_ms) & (intervals_ms > 0)]
        # clip extremes for readability (p99 cap)
        cap = float(np.percentile(valid, 99)) if len(valid) > 0 else 1.0
        valid_clipped = valid[valid <= cap]

        ax_dist.hist(valid_clipped, bins=40, color=color, alpha=0.75, edgecolor="none")

        p50_ms = float(np.percentile(valid, 50)) if len(valid) > 0 else float("nan")
        p95_ms = float(np.percentile(valid, 95)) if len(valid) > 0 else float("nan")
        for pct, val, ls in [(50, p50_ms, "--"), (95, p95_ms, ":")]:
            if np.isfinite(val):
                ax_dist.axvline(val, color=color, linestyle=ls, linewidth=1.2)
        stats_txt = (
            f"p50={p50_ms:.0f} ms ({1000/p50_ms:.1f} Hz)\n"
            f"p95={p95_ms:.0f} ms ({1000/p95_ms:.1f} Hz)\n"
            f"n={len(valid)}"
        ) if np.isfinite(p50_ms) and p50_ms > 0 else "n/a"
        ax_dist.text(0.97, 0.97, stats_txt, ha="right", va="top",
                     transform=ax_dist.transAxes, fontsize=8,
                     bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8})

    plt.tight_layout()
    _savefig(fig, out_dir / "fig13_frequency_stats.png", "frequency_stats")


def plot_budget(data_raw: dict, data_filtered: dict, n_warmup: int, out_dir: Path):
    """Horizontal stacked bar: mean latency budget breakdown.

    Shows warmup-excluded budget when warmup inferences are detected.
    """
    data_use = data_filtered if n_warmup > 0 else data_raw
    budget = _build_budget(data_use)
    if budget is None:
        return
    budget = budget[budget > 0]

    fig, axes = plt.subplots(1 if n_warmup == 0 else 2, 1,
                             figsize=(11, 2.8 if n_warmup == 0 else 5.2))
    if n_warmup == 0:
        axes = [axes]

    def _draw_bar(ax, bgt, label):
        left = 0.0
        handles = []
        for stage, ms in bgt.items():
            color = _C.get(stage, _C["other"])
            ax.barh(0, ms, left=left, height=0.5, color=color,
                    edgecolor="white", linewidth=0.5)
            if ms > max(bgt.max() * 0.02, 0.5):
                ax.text(left + ms / 2, 0, f"{ms:.1f}ms",
                        va="center", ha="center", fontsize=8,
                        color="white", fontweight="bold")
            handles.append(mpatches.Patch(color=color, label=f"{stage}  {ms:.1f}ms"))
            left += ms
        ax.set_xlim(0, left * 1.05)
        ax.set_yticks([])
        ax.set_xlabel("Latency (ms)")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=7.5,
                  framealpha=0.9, bbox_to_anchor=(1.0, 1.02))
        ax.grid(axis="x", alpha=0.3)

    _draw_bar(axes[0], budget,
              f"Mean End-to-End Latency Budget — warmup excluded (n_warmup={n_warmup})"
              if n_warmup > 0 else "Mean End-to-End Latency Budget")

    if n_warmup > 0:
        budget_raw = _build_budget(data_raw)
        if budget_raw is not None:
            budget_raw = budget_raw[budget_raw > 0]
            _draw_bar(axes[1], budget_raw, "All data (including warmup — for reference)")

    plt.tight_layout()
    _savefig(fig, out_dir / "fig1_budget.png", "budget")


def plot_time_series(data: dict, n_warmup: int, out_dir: Path):
    """4-panel time-series of key latencies over chunk/step index."""
    chunk = data["chunk"]
    sent  = data["sent"]
    infer = data["infer"]

    panels = []
    if chunk is not None:
        panels.append(("round_trip_ms",   chunk,  "Round-trip (ms)",         _C["net_c2s"]))
        panels.append(("server_infer_ms", chunk,  "Server inference (ms)",   _C["infer"]))
        panels.append(("deser_ms",        chunk,  "Client deserialize (ms)", _C["deser"]))
    if infer is not None:
        panels.append(("queue_wait_ms",   infer,  "Server queue wait (ms)",  _C["queue_wait"]))
    if sent is not None:
        panels.append(("grpc_send_ms",    sent,   "gRPC send (ms)",          _C["net_c2s"]))

    n = len(panels)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(13, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (col, df, ylabel, color) in zip(axes, panels):
        x = np.arange(len(df))
        y = df[col].values.astype(float)
        valid = ~np.isnan(y)
        ax.plot(x[valid], y[valid], color=color, linewidth=0.9, alpha=0.8)
        if valid.sum() > 0:
            ax.axhline(np.nanmedian(y), color=color, linewidth=1.5,
                       linestyle="--", alpha=0.6,
                       label=f"median={np.nanmedian(y):.1f}ms")
            ax.axhline(np.nanpercentile(y, 95), color="red", linewidth=1,
                       linestyle=":", alpha=0.5,
                       label=f"p95={np.nanpercentile(y, 95):.1f}ms")
        # Shade warmup region
        if n_warmup > 0 and len(df) > n_warmup:
            ax.axvspan(0, n_warmup - 0.5, alpha=0.12, color="red",
                       label=f"warmup (n={n_warmup})")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, len(df) - 1)

    axes[-1].set_xlabel("Chunk / Step index")
    fig.suptitle("Latency Time Series", fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig2_time_series.png", "time_series")


def plot_pipeline(data: dict, out_dir: Path):
    """Box plots for server pipeline stages + client prep stages."""
    rows_data, row_labels, row_colors = [], [], []

    sent  = data["sent"]
    infer = data["infer"]

    client_cols = [
        ("obs_capture_ms",  "capture",      sent,  _C["obs_prep"]),
        ("jpeg_encode_ms",  "jpeg_encode",  sent,  _C["net_c2s"]),    # Issue 1
        ("serialize_ms",    "serialize",    sent,  _C["serialize"]),
        ("grpc_send_ms",    "grpc_send",    sent,  _C["net_c2s"]),
    ]
    server_cols = [
        ("prepare_ms",        "prepare",        infer, _C["prepare"]),
        ("preprocess_ms",     "preprocess",     infer, _C["preprocess"]),
        ("infer_ms",          "infer",          infer, _C["infer"]),
        ("postprocess_ms",    "postprocess",    infer, _C["postprocess"]),
        ("serialize_ms",      "srv_serialize",  infer, _C["serialize"]),
        ("throttle_sleep_ms", "throttle_sleep", infer, _C["other"]),   # Issue 2
    ]
    recv_cols = [
        ("deser_ms",        "deser",        data["chunk"], _C["deser"]),
        ("queue_wait_ms",   "queue_wait",   infer,         _C["queue_wait"]),
    ]

    for col, label, df, color in client_cols + server_cols + recv_cols:
        if df is not None and col in df.columns:
            vals = df[col].dropna().values
            if len(vals) > 0:
                rows_data.append(vals)
                row_labels.append(label)
                row_colors.append(color)

    if not rows_data:
        return

    fig, ax = plt.subplots(figsize=(13, 4))
    bp = ax.boxplot(rows_data, vert=True, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})

    for patch, color in zip(bp["boxes"], row_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(row_labels) + 1))
    ax.set_xticklabels(row_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage Latency Distribution (Box Plot)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()
    _savefig(fig, out_dir / "fig3_pipeline_breakdown.png", "pipeline")


def plot_tail_latency(data: dict, out_dir: Path):
    """Grouped bar chart: p50 / p95 / p99 for key metrics."""
    fields_dfs = [
        ("round_trip",   data["chunk"], "round_trip_ms"),
        ("srv_infer",    data["chunk"], "server_infer_ms"),
        ("grpc_send",    data["sent"],  "grpc_send_ms"),
        ("deser",        data["chunk"], "deser_ms"),
        ("queue_wait",   data["infer"], "queue_wait_ms"),
        ("pipeline",     data["infer"], "total_pipeline_ms"),
        ("model_infer",  data["infer"], "infer_ms"),
        ("preprocess",   data["infer"], "preprocess_ms"),
    ]

    labels, p50s, p95s, p99s = [], [], [], []
    for label, df, col in fields_dfs:
        if df is None or col not in df.columns:
            continue
        v = df[col].dropna()
        if len(v) == 0:
            continue
        labels.append(label)
        p50s.append(float(np.percentile(v, 50)))
        p95s.append(float(np.percentile(v, 95)))
        p99s.append(float(np.percentile(v, 99)))

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 4.5))

    bars50 = ax.bar(x - w, p50s, w, label="p50", color="#4C9BE8", alpha=0.9)
    bars95 = ax.bar(x,     p95s, w, label="p95", color="#F5A623", alpha=0.9)
    bars99 = ax.bar(x + w, p99s, w, label="p99", color="#E84C4C", alpha=0.9)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    for bars in (bars50, bars95, bars99):
        _label_bars(bars)

    for i, (p, q) in enumerate(zip(p50s, p99s)):
        if p > 0 and q / p > 2:
            ax.annotate(f"×{q/p:.1f}", xy=(x[i] + w, q),
                        xytext=(x[i] + w + 0.1, q * 1.05),
                        fontsize=7.5, color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Tail Latency: p50 / p95 / p99", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig4_tail_latency.png", "tail")


def plot_health(data: dict, out_dir: Path):
    """2×2 control-loop health dashboard."""
    sent  = data["sent"]
    recv  = data["recv"]
    infer = data["infer"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Control-Loop Health Dashboard", fontweight="bold")

    # ── (0,0) must_go + episode breakdown ─────────────────────────────────────
    ax = axes[0, 0]
    if sent is not None and "episode" in sent.columns:
        mg_counts = sent.groupby("episode")["must_go"].mean()
        eps = mg_counts.index.tolist()
        ax.bar(eps, mg_counts.values * 100, color=_C["queue_wait"], alpha=0.8)
        ax.axhline(sent["must_go"].mean() * 100, linestyle="--", color="red",
                   label=f"overall {sent['must_go'].mean():.0%}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("must_go rate (%)")
        ax.set_title("must_go rate per episode\n(high → queue empty → stall)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, "No sent/episode data", transform=ax.transAxes, ha="center")

    # ── (0,1) infer_delay distribution ────────────────────────────────────────
    ax = axes[0, 1]
    if sent is not None and "infer_delay" in sent.columns:
        delay_counts = sent["infer_delay"].value_counts().sort_index()
        ax.bar(delay_counts.index.astype(str), delay_counts.values,
               color=_C["preprocess"], alpha=0.85)
        ax.set_xlabel("infer_delay (steps)")
        ax.set_ylabel("Count")
        ax.set_title("RTC infer_delay distribution\n(0 = no RTC, >0 = latency-aware)")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No sent data", transform=ax.transAxes, ha="center")

    # ── (1,0) obs filter pie ───────────────────────────────────────────────────
    ax = axes[1, 0]
    if recv is not None and "enqueued" in recv.columns:
        enq = int(recv["enqueued"].sum())
        fil = int((~recv["enqueued"]).sum())
        # Enqueued but not inferred (queue-replaced)
        n_infer = len(data["infer"]) if data["infer"] is not None else enq
        replaced = max(0, enq - n_infer)
        inferred = n_infer

        if replaced > 0:
            sizes  = [inferred, replaced, fil]
            labels = [f"Inferred\n{inferred} ({inferred/len(recv):.0%})",
                      f"Enqueued→replaced\n{replaced} ({replaced/len(recv):.0%})",
                      f"Filtered\n{fil} ({fil/len(recv):.0%})"]
            colors = [_C["obs_prep"], _C["preprocess"], _C["other"]]
        else:
            sizes  = [enq, fil]
            labels = [f"Enqueued\n{enq} ({enq/len(recv):.0%})",
                      f"Filtered\n{fil} ({fil/len(recv):.0%})"]
            colors = [_C["obs_prep"], _C["other"]]

        ax.pie(sizes, labels=labels, colors=colors,
               autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 9})
        ax.set_title("Server obs funnel\n(filtered = similarity check)")
    else:
        ax.text(0.5, 0.5, "No server recv data", transform=ax.transAxes, ha="center")

    # ── (1,1) server pipeline breakdown pie ───────────────────────────────────
    ax = axes[1, 1]
    if infer is not None:
        stage_cols   = ["queue_wait_ms", "preprocess_ms", "infer_ms", "postprocess_ms",
                        "serialize_ms", "throttle_sleep_ms"]
        stage_labels = ["queue_wait", "preprocess", "infer", "postprocess",
                        "srv_serialize", "throttle_sleep"]
        stage_colors = [_C["queue_wait"], _C["preprocess"], _C["infer"], _C["postprocess"],
                        _C["serialize"], _C["other"]]
        means = [infer[c].mean() for c in stage_cols if c in infer.columns]
        lbls  = [l for c, l in zip(stage_cols, stage_labels) if c in infer.columns]
        colrs = [co for c, co in zip(stage_cols, stage_colors) if c in infer.columns]
        total = sum(means)
        if total > 0:
            ax.pie(means,
                   labels=[f"{l}\n{m:.1f}ms" for l, m in zip(lbls, means)],
                   colors=colrs, autopct="%1.0f%%", startangle=90,
                   textprops={"fontsize": 9})
        ax.set_title("Server pipeline breakdown\n(mean per inference, all data)")
    else:
        ax.text(0.5, 0.5, "No server infer data", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    _savefig(fig, out_dir / "fig5_health.png", "health")


def plot_episode_stats(data: dict, results: dict | None, out_dir: Path):
    """Per-episode breakdown: round-trip, server infer, queue-wait, obs count."""
    chunk = data.get("chunk")
    infer = data.get("infer")
    sent  = data.get("sent")

    if chunk is None or "episode" not in chunk.columns:
        return

    episodes = sorted(chunk["episode"].unique().astype(int))
    if len(episodes) < 2:
        return

    # Collect per-episode metrics from chunk records
    rt_p50   = [np.percentile(chunk[chunk["episode"] == ep]["round_trip_ms"].dropna(), 50)
                for ep in episodes]
    si_p50   = [np.percentile(chunk[chunk["episode"] == ep]["server_infer_ms"].dropna(), 50)
                for ep in episodes]
    n_chunks = [int((chunk["episode"] == ep).sum()) for ep in episodes]

    # Server infer per episode (via wall_time join)
    qw_p50 = [np.nan] * len(episodes)
    if infer is not None and "episode" in infer.columns:
        for j, ep in enumerate(episodes):
            ep_inf = infer[infer["episode"] == ep]["queue_wait_ms"].dropna()
            if len(ep_inf) > 0:
                qw_p50[j] = np.percentile(ep_inf, 50)

    # Success/fail coloring from results
    success_map = {}
    if results and "episodes" in results:
        for r in results["episodes"]:
            success_map[r.get("episode_id", -1)] = r.get("success", None)

    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    fig.suptitle("Per-Episode Stats", fontweight="bold")
    x = np.array(episodes)

    def _ep_colors():
        if not success_map:
            return ["#4C9BE8"] * len(episodes)
        return ["#22C55E" if success_map.get(ep, None) else
                ("#EF4444" if success_map.get(ep, None) is False else "#9CA3AF")
                for ep in episodes]

    ep_colors = _ep_colors()

    # (0,0) Round-trip p50 per episode
    ax = axes[0, 0]
    bars = ax.bar(x, rt_p50, color=ep_colors, alpha=0.85, edgecolor="white")
    ax.axhline(np.nanmedian(rt_p50), linestyle="--", color="gray", linewidth=1,
               label=f"median={np.nanmedian(rt_p50):.1f}ms")
    ax.set_title("Round-trip p50 (ms) per episode", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (0,1) Server inference p50 per episode
    ax = axes[0, 1]
    ax.bar(x, si_p50, color=ep_colors, alpha=0.85, edgecolor="white")
    ax.axhline(np.nanmedian(si_p50), linestyle="--", color="gray", linewidth=1,
               label=f"median={np.nanmedian(si_p50):.1f}ms")
    ax.set_title("Server inference p50 (ms) per episode", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (1,0) Number of chunks per episode
    ax = axes[1, 0]
    ax.bar(x, n_chunks, color=ep_colors, alpha=0.85, edgecolor="white")
    ax.axhline(np.nanmedian(n_chunks), linestyle="--", color="gray", linewidth=1,
               label=f"median={np.nanmedian(n_chunks):.0f}")
    ax.set_title("Chunks received per episode\n(proxy for episode length)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (1,1) Queue-wait p50 per episode (if server data available)
    ax = axes[1, 1]
    valid_qw = [(ep, qw) for ep, qw in zip(episodes, qw_p50) if not np.isnan(qw)]
    if valid_qw:
        qw_eps, qw_vals = zip(*valid_qw)
        col = [ep_colors[episodes.index(ep)] for ep in qw_eps]
        ax.bar(np.array(qw_eps), qw_vals, color=col, alpha=0.85, edgecolor="white")
        ax.axhline(np.nanmedian(qw_vals), linestyle="--", color="gray", linewidth=1,
                   label=f"median={np.nanmedian(qw_vals):.1f}ms")
        ax.set_title("Server queue_wait p50 (ms) per episode", fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Server queue_wait p50 per episode", fontweight="bold")
        ax.text(0.5, 0.5, "No server infer data with episode info",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ms")
    ax.grid(axis="y", alpha=0.3)

    if success_map:
        legend_elems = [
            mpatches.Patch(color="#22C55E", label="Success"),
            mpatches.Patch(color="#EF4444", label="Failed"),
            mpatches.Patch(color="#9CA3AF", label="Unknown"),
        ]
        fig.legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    _savefig(fig, out_dir / "fig6_episode_stats.png", "episode_stats")


def plot_timeline(data: dict, n_warmup: int, out_dir: Path):
    """Wall-clock event timeline: obs sent, chunks received, server inferences."""
    sent  = data.get("sent")
    chunk = data.get("chunk")
    infer = data.get("infer")

    if sent is None and chunk is None:
        return

    # Gather all wall_times to compute t0
    all_wt = []
    for df in [sent, chunk, infer]:
        if df is not None and "wall_time" in df.columns:
            all_wt.extend(df["wall_time"].values.tolist())
    if not all_wt:
        return
    t0 = min(all_wt)

    fig, axes = plt.subplots(3 if infer is not None else 2, 1,
                             figsize=(14, 7 if infer is not None else 5),
                             sharex=True)
    if (infer is None):
        axes = list(axes)
    axes = list(axes)

    # ── Panel 0: Obs sent ──────────────────────────────────────────────────────
    ax = axes[0]
    if sent is not None and "wall_time" in sent.columns:
        t_sent = sent["wall_time"].values - t0
        colors = ["#E84C4C" if m else "#4C9BE8"
                  for m in sent.get("must_go", pd.Series([False] * len(sent)))]
        ax.scatter(t_sent, np.ones(len(t_sent)), c=colors, s=8, alpha=0.6, linewidths=0)
        # Episode boundaries: color change events
        if "episode" in sent.columns:
            ep_starts = sent.groupby("episode")["wall_time"].min() - t0
            for ep, ts in ep_starts.items():
                ax.axvline(ts, color="gray", linewidth=0.5, alpha=0.5)
                ax.text(ts, 1.05, f"ep{int(ep)}", fontsize=6, rotation=90,
                        va="bottom", ha="center", color="gray")
    ax.set_yticks([])
    ax.set_ylabel("Obs\nsent", fontsize=9, rotation=0, labelpad=40)
    ax.set_ylim(0.8, 1.2)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Wall-Clock Event Timeline", fontweight="bold")

    # ── Panel 1: Chunks received ───────────────────────────────────────────────
    ax = axes[1]
    if chunk is not None and "wall_time" in chunk.columns:
        t_chunk = chunk["wall_time"].values - t0
        # Color by episode if available
        ep_col = plt.cm.tab20(np.linspace(0, 1, 20))
        if "episode" in chunk.columns:
            eps = chunk["episode"].values.astype(int)
            colors_c = [ep_col[ep % 20] for ep in eps]
        else:
            colors_c = [_C["infer"]] * len(t_chunk)
        ax.scatter(t_chunk, np.ones(len(t_chunk)), c=colors_c, s=12, alpha=0.8, linewidths=0)
        # Round-trip as horizontal bars (rough — drawn from t_chunk - round_trip to t_chunk)
        if "round_trip_ms" in chunk.columns:
            for t, rt, col in zip(t_chunk, chunk["round_trip_ms"].values, colors_c):
                ax.plot([t - rt/1000, t], [1, 1], color=col, linewidth=1.5, alpha=0.3)
    ax.set_yticks([])
    ax.set_ylabel("Chunks\nrecvd", fontsize=9, rotation=0, labelpad=40)
    ax.set_ylim(0.8, 1.2)
    ax.grid(axis="x", alpha=0.2)

    # ── Panel 2: Server inference duration ────────────────────────────────────
    if infer is not None and len(axes) > 2:
        ax = axes[2]
        t_infer = infer["wall_time"].values - t0
        heights  = infer["infer_ms"].values / 1000.0  # convert to seconds
        bar_colors = ["#FFB3B3" if i < n_warmup else _C["infer"]
                      for i in range(len(t_infer))]
        ax.bar(t_infer, heights, width=0.3, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_ylabel("Infer\nduration (s)", fontsize=9, rotation=0, labelpad=40)
        ax.grid(axis="x", alpha=0.2)
        if n_warmup > 0:
            ax.annotate(f"warmup\n({n_warmup} records)",
                        xy=(t_infer[0], heights[0]),
                        xytext=(t_infer[0] + 2, heights[0] * 0.9),
                        fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1))

    axes[-1].set_xlabel("Wall-clock time (seconds from start)")
    plt.tight_layout()
    _savefig(fig, out_dir / "fig7_timeline.png", "timeline")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1–3: infer_delay calibration, queue starvation, obs filter rate
# ══════════════════════════════════════════════════════════════════════════════

def _infer_fps(data: dict, fallback: float = 30.0) -> float:
    """Estimate control-loop fps from obs send intervals, or return fallback."""
    sent = data.get("sent")
    if sent is not None and "wall_time" in sent.columns and len(sent) > 10:
        diffs = np.diff(np.sort(sent["wall_time"].values))
        diffs = diffs[(diffs > 0.005) & (diffs < 0.5)]  # 2–200 fps plausible range
        if len(diffs) > 5:
            return float(1.0 / np.median(diffs))
    return fallback


def _join_sent_chunk(data: dict) -> pd.DataFrame | None:
    """Inner-join sent obs records with chunk records on (episode, timestep==first_timestep).

    Returns a DataFrame with columns from both tables, or None if join fails.
    """
    sent  = data.get("sent")
    chunk = data.get("chunk")
    if sent is None or chunk is None:
        return None
    if "infer_delay" not in sent.columns or "first_timestep" not in chunk.columns:
        return None

    # Client send overhead fields (jpeg_encode_ms + serialize_ms + grpc_send_ms) live in
    # the sent record and are needed to reconstruct complete_s (Option B).
    # Older recordings that pre-date these columns fall back gracefully.
    _send_overhead_cols = [c for c in ("grpc_send_ms", "serialize_ms", "jpeg_encode_ms")
                           if c in sent.columns]
    # Tier 2 split-estimator diagnostic fields (present only in newer recordings).
    _split_cols = [c for c in ("split_active", "infer_delay_raw") if c in sent.columns]
    sent_cols  = ["timestep", "infer_delay", "wall_time"] + _send_overhead_cols + _split_cols
    chunk_cols = ["first_timestep", "round_trip_ms", "wall_time"]
    # Include deser_ms when present (recorded since Option B).
    if "deser_ms" in chunk.columns:
        chunk_cols = ["first_timestep", "round_trip_ms", "deser_ms", "wall_time"]
    # server_infer_ms is needed to reconstruct the split components (Tier 2).
    if "server_infer_ms" in chunk.columns:
        chunk_cols = chunk_cols + ["server_infer_ms"]

    use_ep = "episode" in sent.columns and "episode" in chunk.columns
    if use_ep:
        sent_cols  = ["episode"] + sent_cols
        chunk_cols = ["episode"] + chunk_cols
        joined = pd.merge(
            sent[sent_cols].rename(columns={"wall_time": "sent_wall_time"}),
            chunk[chunk_cols].rename(columns={"wall_time": "chunk_wall_time"}),
            left_on=["episode", "timestep"],
            right_on=["episode", "first_timestep"],
            how="inner",
        )
    else:
        joined = pd.merge(
            sent[sent_cols].rename(columns={"wall_time": "sent_wall_time"}),
            chunk[chunk_cols].rename(columns={"wall_time": "chunk_wall_time"}),
            left_on="timestep",
            right_on="first_timestep",
            how="inner",
        )
    return joined if not joined.empty else None


def print_infer_delay_calibration(data: dict, fps: float):
    """Analysis 1: Compare sent infer_delay vs actual round-trip delay in steps.

    Shows whether the RTC latency hint is well-calibrated.  A large ratio
    (sent >> actual) means the LatencyTracker was returning an inflated max
    (e.g. all-time max from warmup) and RTC predicts actions far outside the
    training distribution.
    """
    _divider("INFER_DELAY CALIBRATION  (Analysis 1)", 80)

    joined = _join_sent_chunk(data)
    if joined is None:
        print("  (need both sent+chunk records with matching timestep/first_timestep)")
        return

    dt_ms = 1000.0 / fps
    # Option B: latency_tracker measures obs.timestamp → receive_after_deser, i.e.
    #   complete_ms = jpeg_encode_ms + serialize_ms + grpc_send_ms   (client send overhead)
    #               + round_trip_ms                                   (server + net_s2c)
    #               + deser_ms                                        (client deser)
    # Reconstruct the same quantity from the joined record so "actual_steps" matches
    # what the LatencyTracker saw.  Fields may be absent in older recordings — fall back
    # to whatever subset is available.
    _complete_ms = joined["round_trip_ms"].copy()
    _parts = ["round_trip_ms"]
    for col in ("deser_ms", "grpc_send_ms", "serialize_ms", "jpeg_encode_ms"):
        if col in joined.columns:
            _complete_ms = _complete_ms + joined[col]
            _parts.append(col)
    joined["actual_steps"] = _complete_ms / dt_ms
    actual_label = "+".join(_parts)
    joined["delay_error"]   = joined["infer_delay"] - joined["actual_steps"]

    sent_id = joined["infer_delay"]
    actual  = joined["actual_steps"]
    error   = joined["delay_error"]
    n       = len(joined)
    zero_frac = (sent_id == 0).mean()

    print(f"  fps used           : {fps:.1f}  (dt = {dt_ms:.1f} ms/step)")
    print(f"  actual_steps basis : {actual_label}")
    print(f"  matched records    : {n}")
    print(f"  infer_delay == 0   : {zero_frac:.1%}  (RTC not active or LatencyTracker empty)")
    print()
    header = f"  {'metric':<8}  {'actual_steps':>12}  {'sent_infer_delay':>17}  {'error (sent−actual)':>20}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for label, fn in [("mean", lambda s: float(s.mean())),
                      ("p50",  lambda s: float(np.percentile(s.dropna(), 50))),
                      ("p95",  lambda s: float(np.percentile(s.dropna(), 95))),
                      ("max",  lambda s: float(s.max()))]:
        print(f"  {label:<8}  {fn(actual):>12.1f}  {fn(sent_id):>17.1f}  {fn(error):>20.1f}")

    nonzero = joined[sent_id > 0]
    if len(nonzero) > 0:
        overcorr = (nonzero["infer_delay"] / nonzero["actual_steps"].clip(lower=0.01)).mean()
        print(f"\n  Mean overcorrection (when infer_delay>0) : {overcorr:.1f}×")
        if overcorr > 5:
            print(f"  ⚠  infer_delay is ~{overcorr:.0f}× too large.")
            print(f"     Root cause: LatencyTracker uses a sliding window (maxlen=100, ~20 s at 5 inf/s),")
            print(f"     but genuine network jitter spikes (complete_s up to ~1.4 s) pass the 2.5 s filter")
            print(f"     and dominate the window max.  Because p99 events occur every ~100 inferences")
            print(f"     (~20 s), at least one spike is almost always present in the window, so")
            print(f"     max() consistently returns the spike value → ceil(1400 ms / 50 ms) ≈ 28 steps.")
            print(f"     Fix: switch LatencyTracker.max() → .percentile(0.99) in base_client.py to")
            print(f"     reduce infer_delay while keeping a safety margin above typical round-trip p50.")
        elif overcorr > 1.5:
            print(f"  △  Moderate overcorrection — check LatencyTracker window and warmup handling.")
        else:
            print(f"  ✓  infer_delay well-calibrated (overcorrection = {overcorr:.2f}×).")


def print_starvation_analysis(data: dict, fps: float):
    """Analysis 2: Detect action queue starvation events from chunk timing.

    A starvation event occurs when a chunk's actions are exhausted before the
    next chunk arrives.  During starvation the robot repeats or holds its last
    action, which appears as jitter or freezing.
    """
    _divider("ACTION QUEUE STARVATION  (Analysis 2)", 80)
    chunk = data.get("chunk")
    sent  = data.get("sent")

    if chunk is None or "chunk_size" not in chunk.columns or "wall_time" not in chunk.columns:
        print("  (need chunk records with chunk_size and wall_time)")
        return

    dt = 1.0 / fps
    chunk_s = chunk.sort_values("wall_time").copy()
    chunk_s["exhaust_time"] = chunk_s["wall_time"] + chunk_s["chunk_size"] * dt
    chunk_s["next_arrival"] = chunk_s["wall_time"].shift(-1)

    if "episode" in chunk_s.columns:
        same_ep = chunk_s["episode"] == chunk_s["episode"].shift(-1)
        chunk_s.loc[~same_ep, "next_arrival"] = np.nan  # no starvation at episode boundary

    chunk_s["gap_s"] = (chunk_s["next_arrival"] - chunk_s["exhaust_time"]).clip(lower=0)
    starved   = chunk_s[chunk_s["gap_s"] > 0]
    n_intervals = int((~chunk_s["next_arrival"].isna()).sum())
    n_starve    = len(starved)

    print(f"  fps used             : {fps:.1f}")
    print(f"  chunk intervals      : {n_intervals}")
    rate_str = f"({n_starve/n_intervals:.1%} of intervals)" if n_intervals > 0 else ""
    print(f"  starvation events    : {n_starve}  {rate_str}")
    if n_starve > 0:
        total_gap  = float(starved["gap_s"].sum())
        max_gap    = float(starved["gap_s"].max())
        print(f"  total starvation     : {total_gap:.2f} s")
        print(f"  max single gap       : {max_gap:.3f} s  ({max_gap * fps:.0f} control steps @ {fps:.0f} fps)")

    if sent is not None and "must_go" in sent.columns:
        mg_rate = float(sent["must_go"].mean())
        print(f"  must_go rate         : {mg_rate:.1%}  (client-side queue-empty marker)")

    if n_starve > 0:
        print(f"\n  Top-5 starvation events (by gap duration):")
        top5 = starved.nlargest(5, "gap_s")
        for _, row in top5.iterrows():
            gap_steps = row["gap_s"] * fps
            csz = int(row["chunk_size"]) if not np.isnan(row["chunk_size"]) else "?"
            print(f"    gap={row['gap_s']:.3f}s ({gap_steps:.0f} steps), chunk_size={csz}")

        if n_intervals > 0 and n_starve / n_intervals > 0.05:
            print(f"\n  ⚠  >5% starvation rate → robot likely executing repeated/stale actions.")
            print(f"     Suggestions: increase actions_per_chunk, reduce fps, or reduce inference latency.")
        elif n_intervals > 0 and n_starve / n_intervals > 0.01:
            print(f"\n  △  Occasional starvation — monitor; may worsen with slower inference.")
        else:
            print(f"\n  ✓  Starvation rate low (<1%).")


def plot_infer_delay_calibration(data: dict, fps: float, out_dir: Path):
    """Fig 8: infer_delay sent vs actual round-trip steps — calibration scatter + time series.

    Left panel also shows ``leftover_steps`` (buffer remaining when obs was sent) so the
    reader can see three quantities on the same axis:
      - sent infer_delay  : LatencyTracker p99 estimate (control signal)
      - leftover_steps    : actual buffer depth at obs-send time (ideal ≈ infer_delay)
      - actual steps      : reconstructed complete_s / dt (ground truth)

    Ideal calibration: sent ≈ leftover_steps ≈ actual.
    If sent >> actual: LatencyTracker overcorrects (p99 dominated by spike outliers).
    If leftover < actual: buffer ran out before chunk arrived → starvation risk.
    """
    joined = _join_sent_chunk(data)
    if joined is None:
        return

    dt_ms = 1000.0 / fps
    _complete_ms = joined["round_trip_ms"].copy()
    _parts = ["rt"]
    for col, short in (("deser_ms", "deser"), ("grpc_send_ms", "grpc"), ("serialize_ms", "ser"), ("jpeg_encode_ms", "jpeg")):
        if col in joined.columns:
            _complete_ms = _complete_ms + joined[col]
            _parts.append(short)
    joined["actual_steps"] = _complete_ms / dt_ms
    actual_basis = f"({'+'.join(_parts)}) / {dt_ms:.1f}ms"
    t0 = joined["chunk_wall_time"].min()
    joined["t"] = joined["chunk_wall_time"] - t0

    # Merge leftover_steps from sent records if available.
    sent = data.get("sent")
    has_leftover = (
        sent is not None
        and "leftover_steps" in sent.columns
        and "timestep" in sent.columns
        and "timestep" in joined.columns
    )
    if has_leftover:
        lv = sent[["timestep", "leftover_steps"]].rename(columns={"leftover_steps": "_lv"})
        joined = joined.merge(lv, on="timestep", how="left")
        has_leftover = "_lv" in joined.columns and joined["_lv"].notna().any()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # ── Left: time series of sent / leftover / actual ─────────────────────────
    ax = axes[0]
    ax.plot(joined["t"], joined["infer_delay"], color=_C["infer"], linewidth=0.9,
            alpha=0.85, label="sent infer_delay (steps)")
    if has_leftover:
        ax.scatter(joined["t"], joined["_lv"], s=10, color=_C["net_c2s"], alpha=0.7,
                   linewidths=0, zorder=3,
                   label="pre-leftover_steps (buffer depth at obs-send)")
    ax.plot(joined["t"], joined["actual_steps"], color=_C["obs_prep"], linewidth=1.2,
            alpha=0.9, label=f"actual steps ({actual_basis})")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_title("RTC infer_delay: sent vs actual\n"
                 "(+ pre-leftover = buffer depth when obs sent)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Right: scatter sent vs actual ─────────────────────────────────────────
    ax = axes[1]
    ax.scatter(joined["actual_steps"], joined["infer_delay"],
               alpha=0.4, s=14, color=_C["infer"], linewidths=0, label="sent vs actual")
    if has_leftover:
        ax.scatter(joined["actual_steps"], joined["_lv"],
                   alpha=0.3, s=10, color=_C["net_c2s"], linewidths=0,
                   marker="^", label="leftover vs actual")
    lim = max(float(joined["infer_delay"].max()), float(joined["actual_steps"].max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, alpha=0.5, label="ideal (sent = actual)")
    ax.plot([0, lim / 2], [0, lim], color="orange", linewidth=0.8,
            linestyle=":", alpha=0.5, label="2×")
    ax.set_xlabel(f"Actual steps ({actual_basis})")
    ax.set_ylabel("Sent infer_delay / leftover_steps")
    ax.set_title("Calibration scatter\n(ideal = on diagonal)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f"Analysis 1: RTC infer_delay Calibration  (fps={fps:.0f})",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig8_infer_delay_calibration.png", "infer_delay_calibration")


def print_split_diagnosis(data: dict, infer_q: float, overhead_q: float) -> None:
    """Console summary for the Tier 2 split-component infer_delay estimator.

    Reconstructs the two components (server_infer / overhead) from existing fields
    and, when the newer split_active / infer_delay_raw fields are present, also reports
    bootstrap-fallback fraction and hysteresis-suppression rate.
    """
    joined = _join_sent_chunk(data)
    if joined is None or "server_infer_ms" not in joined.columns:
        return
    _divider("SPLIT-COMPONENT infer_delay  (Tier 2)", 80)

    # Reconstruct complete_s and the two components.
    complete_ms = joined["round_trip_ms"].astype(float).copy()
    for col in ("deser_ms", "grpc_send_ms", "serialize_ms", "jpeg_encode_ms"):
        if col in joined.columns:
            complete_ms = complete_ms + joined[col].fillna(0.0)
    infer_ms = joined["server_infer_ms"].astype(float)
    overhead_ms = (complete_ms - infer_ms).clip(lower=0.0)

    def _p(s, q):
        return float(np.percentile(s.dropna(), q * 100)) if len(s.dropna()) else float("nan")

    print(f"  quantiles          : infer p{int(infer_q*100)}  +  overhead p{int(overhead_q*100)}")
    print(f"  server_infer (ms)  : p50={_p(infer_ms,0.5):.0f}  "
          f"p{int(infer_q*100)}={_p(infer_ms,infer_q):.0f}  max={infer_ms.max():.0f}  "
          f"(σ/μ={infer_ms.std()/max(infer_ms.mean(),1e-6):.2f} → stable component)")
    print(f"  overhead     (ms)  : p50={_p(overhead_ms,0.5):.0f}  "
          f"p{int(overhead_q*100)}={_p(overhead_ms,overhead_q):.0f}  "
          f"p99={_p(overhead_ms,0.99):.0f}  max={overhead_ms.max():.0f}  "
          f"(σ/μ={overhead_ms.std()/max(overhead_ms.mean(),1e-6):.2f} → heavy-tailed)")

    # Control-flow stats from the new fields (if present).
    if "split_active" in joined.columns:
        frac = float(joined["split_active"].mean())
        print(f"  split_active       : {frac:.1%}  (fraction using Tier 2 formula; "
              f"rest = bootstrap fallback — should →100% after warmup)")
    if "infer_delay_raw" in joined.columns:
        raw = joined["infer_delay_raw"]
        valid = raw >= 0
        if valid.any():
            held = ((joined.loc[valid, "infer_delay"] != raw[valid])).mean()
            print(f"  hysteresis hold    : {held:.1%}  (obs where sent infer_delay != raw "
                  f"pre-hysteresis value → ±1-step jitter suppressed)")
    print()


def plot_split_component_calibration(data: dict, fps: float, out_dir: Path,
                                     infer_q: float = 0.90, overhead_q: float = 0.75) -> None:
    """Fig 14: Tier 2 split-component infer_delay decomposition & calibration.

    Reconstructs (zero extra logging required) from existing fields:
      server_infer_ms              (chunk record)
      overhead_ms = complete_ms − server_infer_ms
    and overlays the reconstructed estimate ceil((rolling pQ_infer + rolling pQ_overhead)/dt)
    against the actually-sent infer_delay and the ground-truth actual_steps.

    Panel 0: the two latency components over time + their rolling quantile lines.
    Panel 1: steps calibration — reconstructed estimate vs sent infer_delay vs actual_steps.
    Panel 2: control-flow diagnostics (split_active fraction + hysteresis holds) when the
             newer fields are present; otherwise annotated as unavailable.
    """
    joined = _join_sent_chunk(data)
    if joined is None or "server_infer_ms" not in joined.columns:
        return

    dt_ms = 1000.0 / fps
    joined = joined.sort_values("chunk_wall_time").reset_index(drop=True)
    t0 = joined["chunk_wall_time"].min()
    joined["t"] = joined["chunk_wall_time"] - t0

    complete_ms = joined["round_trip_ms"].astype(float).copy()
    for col in ("deser_ms", "grpc_send_ms", "serialize_ms", "jpeg_encode_ms"):
        if col in joined.columns:
            complete_ms = complete_ms + joined[col].fillna(0.0)
    joined["infer_ms"]    = joined["server_infer_ms"].astype(float)
    joined["overhead_ms"] = (complete_ms - joined["infer_ms"]).clip(lower=0.0)
    joined["actual_steps"] = complete_ms / dt_ms

    # Rolling quantiles mirroring the live trackers (infer maxlen=50, overhead maxlen=20).
    roll_infer    = joined["infer_ms"].rolling(50, min_periods=3).quantile(infer_q)
    roll_overhead = joined["overhead_ms"].rolling(20, min_periods=3).quantile(overhead_q)
    joined["recon_delay"] = np.ceil((roll_infer + roll_overhead) / dt_ms)

    has_ctrl = "split_active" in joined.columns or "infer_delay_raw" in joined.columns
    n_panels = 3 if has_ctrl else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.4 * n_panels), sharex=True)
    axes = list(axes)

    # ── Panel 0: components over time ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(joined["t"], joined["infer_ms"], color=_C["infer"], lw=0.7, alpha=0.5,
            label="server_infer (per-chunk)")
    ax.plot(joined["t"], roll_infer, color=_C["infer"], lw=1.8,
            label=f"server_infer rolling p{int(infer_q*100)}")
    ax.plot(joined["t"], joined["overhead_ms"], color=_C["net_c2s"], lw=0.7, alpha=0.4,
            label="overhead (per-chunk)")
    ax.plot(joined["t"], roll_overhead, color=_C["net_c2s"], lw=1.8,
            label=f"overhead rolling p{int(overhead_q*100)}")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Analysis 14: infer_delay split components — stable server_infer (high-q) "
                 "vs heavy-tailed overhead (mod-q)", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── Panel 1: steps calibration ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(joined["t"], joined["actual_steps"], color=_C["obs_prep"], lw=1.0, alpha=0.7,
            label="actual_steps (complete_s / dt, ground truth)")
    ax.plot(joined["t"], joined["recon_delay"], color="green", lw=1.6,
            label=f"reconstructed ceil((p{int(infer_q*100)}+p{int(overhead_q*100)})/dt)")
    ax.plot(joined["t"], joined["infer_delay"], color=_C["infer"], lw=1.2, alpha=0.9,
            label="sent infer_delay (actual control signal)")
    ax.set_ylabel("Steps")
    ax.set_title("Estimate vs sent vs ground truth  (sent should track reconstructed; "
                 "both ≥ actual most of the time)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2: control-flow diagnostics ─────────────────────────────────────
    if has_ctrl:
        ax = axes[2]
        if "split_active" in joined.columns:
            frac = joined["split_active"].astype(float).rolling(30, min_periods=1).mean()
            ax.fill_between(joined["t"], 0, frac, color=_C["queue_wait"], alpha=0.25,
                            label="split_active (rolling frac; 1=Tier2, 0=fallback)")
            ax.set_ylim(-0.05, 1.15)
        if "infer_delay_raw" in joined.columns:
            raw = joined["infer_delay_raw"]
            held = joined[(raw >= 0) & (joined["infer_delay"] != raw)]
            if len(held) > 0:
                # Plot at normalized height 1.05 as event ticks.
                ax.scatter(held["t"], [1.05] * len(held), marker="|", s=60,
                           color="red", alpha=0.7,
                           label=f"hysteresis hold ({len(held)} obs, sent≠raw)")
        ax.set_ylabel("frac / events")
        ax.set_title("Control-flow: Tier 2 vs bootstrap fallback + hysteresis suppression",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Wall-clock time (s from start)")
    plt.suptitle(f"Analysis 14: Split-Component infer_delay Calibration  (fps={fps:.0f})",
                 fontweight="bold", y=1.005)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig14_split_calibration.png", "split_calibration")


def plot_starvation(data: dict, fps: float, out_dir: Path):
    """Fig 9: Chunk coverage timeline with starvation gaps highlighted."""
    chunk = data.get("chunk")
    sent  = data.get("sent")

    if chunk is None or "chunk_size" not in chunk.columns or "wall_time" not in chunk.columns:
        return

    dt = 1.0 / fps
    chunk_s = chunk.sort_values("wall_time").copy()
    chunk_s["exhaust_time"] = chunk_s["wall_time"] + chunk_s["chunk_size"] * dt
    chunk_s["next_arrival"] = chunk_s["wall_time"].shift(-1)
    if "episode" in chunk_s.columns:
        same_ep = chunk_s["episode"] == chunk_s["episode"].shift(-1)
        chunk_s.loc[~same_ep, "next_arrival"] = np.nan
    chunk_s["gap_s"] = (chunk_s["next_arrival"] - chunk_s["exhaust_time"]).clip(lower=0)

    t0 = float(chunk_s["wall_time"].min())
    chunk_s["t_start"]   = chunk_s["wall_time"]   - t0
    chunk_s["t_exhaust"] = chunk_s["exhaust_time"] - t0
    chunk_s["t_next"]    = chunk_s["next_arrival"] - t0

    has_must_go = sent is not None and "must_go" in sent.columns and "wall_time" in sent.columns
    n_panels = 3 if has_must_go else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    axes = list(axes)

    # Panel 0: chunk coverage bars
    ax = axes[0]
    for _, row in chunk_s.iterrows():
        covered_color = _C["obs_prep"] if row["gap_s"] == 0 else _C["queue_wait"]
        ax.barh(0, row["t_exhaust"] - row["t_start"], left=row["t_start"],
                height=0.6, color=covered_color, alpha=0.75, edgecolor="none")
        if row["gap_s"] > 0 and not np.isnan(row["t_next"]):
            ax.barh(0, row["t_next"] - row["t_exhaust"], left=row["t_exhaust"],
                    height=0.6, color="#E84C4C", alpha=0.85, edgecolor="none")
    ax.set_yticks([])
    ax.set_ylabel("Coverage", fontsize=9, rotation=0, labelpad=40)
    ax.set_title("Analysis 2: Action Queue Coverage (blue=covered, orange=warned, red=starved)",
                 fontweight="bold")
    legend_elems = [
        mpatches.Patch(color=_C["obs_prep"],    label="Queue covered"),
        mpatches.Patch(color=_C["queue_wait"],  label="Prev chunk starved"),
        mpatches.Patch(color="#E84C4C",         label="Starvation gap"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right")

    # Panel 1: starvation gap duration per event
    ax = axes[1]
    starved = chunk_s[chunk_s["gap_s"] > 0]
    if not starved.empty:
        ax.bar(starved["t_start"], starved["gap_s"] * fps,
               width=0.3, color="#E84C4C", alpha=0.85, label="gap (steps)")
        ax.axhline(float(starved["gap_s"].mean() * fps), linestyle="--",
                   color="gray", linewidth=1,
                   label=f"mean={starved['gap_s'].mean() * fps:.1f} steps")
        ax.legend(fontsize=8)
    ax.set_ylabel("Gap (steps)", fontsize=9)
    ax.set_title(f"Starvation gap per event (in control steps at {fps:.0f} fps)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2 (optional): must_go events
    if has_must_go:
        ax = axes[2]
        t_sent = sent["wall_time"].values - t0
        mg = sent["must_go"].values.astype(bool)
        ax.scatter(t_sent[mg], np.ones(int(mg.sum())), c="#E84C4C", s=18,
                   alpha=0.7, linewidths=0, label=f"must_go=True  ({mg.mean():.1%})")
        ax.scatter(t_sent[~mg], np.zeros(int((~mg).sum())), c="#4C9BE8", s=4,
                   alpha=0.25, linewidths=0, label="must_go=False")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["normal", "must_go"], fontsize=8)
        ax.set_ylabel("must_go", fontsize=9, rotation=0, labelpad=40)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("Wall-clock time (s from start)")
    plt.tight_layout()
    _savefig(fig, out_dir / "fig9_starvation.png", "starvation")


def plot_filter_rate(data: dict, out_dir: Path, window: int = 50):
    """Fig 10: Rolling obs enqueue rate over time (sliding window of server recv records).

    Low rate during warmup (server blocked) and for stationary robots (obs_similar filter).
    """
    recv = data.get("recv")
    if recv is None or "enqueued" not in recv.columns or "wall_time" not in recv.columns:
        return

    recv_s = recv.sort_values("wall_time").copy()
    recv_s["enq"] = recv_s["enqueued"].astype(int)
    recv_s["rolling_rate"] = recv_s["enq"].rolling(window, min_periods=1).mean()
    t0 = float(recv_s["wall_time"].min())
    recv_s["t"] = recv_s["wall_time"] - t0

    overall_rate = float(recv_s["enq"].mean())

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

    # Panel 0: rolling enqueue rate time series
    ax = axes[0]
    ax.fill_between(recv_s["t"], recv_s["rolling_rate"] * 100,
                    alpha=0.25, color=_C["obs_prep"])
    ax.plot(recv_s["t"], recv_s["rolling_rate"] * 100,
            color=_C["obs_prep"], linewidth=1.2,
            label=f"rolling enqueue rate (window={window} obs)")
    ax.axhline(overall_rate * 100, linestyle="--", color="gray", linewidth=1,
               label=f"overall {overall_rate:.1%}")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Enqueue rate (%)")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_title(
        "Analysis 3: Server Obs Enqueue Rate Over Time\n"
        "(near 0% during warmup deadlock; drops when robot is stationary)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 1: binned enqueue rate histogram across time
    ax = axes[1]
    t_max = float(recv_s["t"].max())
    n_bins = min(60, max(10, len(recv_s) // 10))
    bin_edges = np.linspace(0, t_max, n_bins + 1)
    recv_s["bucket"] = np.digitize(recv_s["t"].values, bin_edges).clip(1, n_bins) - 1
    bucket_rate  = recv_s.groupby("bucket")["enq"].mean()
    bucket_count = recv_s.groupby("bucket")["enq"].count()
    valid = bucket_count[bucket_count >= 3].index
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if len(valid) > 0:
        ax.bar(centers[valid], bucket_rate.loc[valid].values * 100,
               width=(t_max / n_bins) * 0.85, color=_C["obs_prep"], alpha=0.75)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Enqueue rate (%)")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_title(f"Binned enqueue rate  (n_bins={n_bins})", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, out_dir / "fig10_filter_rate.png", "filter_rate")


# ══════════════════════════════════════════════════════════════════════════════
# Gripper state machine event analysis
# ══════════════════════════════════════════════════════════════════════════════

def load_sm_events(client_dir: str | Path | None) -> pd.DataFrame | None:
    """Load gripper SM event records from <client_dir>/gripper_sm_events_records.jsonl.

    Returns None (silently) when the file does not exist — plain RobotClient runs
    never produce this file, so absence is the normal case.
    """
    if client_dir is None:
        return None
    df = _load(Path(client_dir) / "gripper_sm_events_records.jsonl")
    if df is not None:
        print(f"  Loaded sm_events : {len(df):5d} records")
    else:
        print(f"  Loaded sm_events : (not found — SmartRobotClient not used or timing not enabled)")
    return df


def print_sm_event_summary(sm: pd.DataFrame | None) -> None:
    """Console table: event counts, per-type sensor stats, per-episode breakdown."""
    _divider("GRIPPER STATE MACHINE EVENTS", 80)
    if sm is None or len(sm) == 0:
        print("  (no SM event records)")
        return

    total  = len(sm)
    counts = sm["event_type"].value_counts()

    # ── Event count table ──────────────────────────────────────────────────────
    rows = []
    for et in _SM_ORDER:
        n = int(counts.get(et, 0))
        if n == 0:
            continue
        rows.append({"event_type": et, "count": n,
                     "% of total": f"{n / total:.0%}" if total > 0 else "—"})
    print(pd.DataFrame(rows).set_index("event_type").to_string())
    print(f"\n  Total events : {total}")

    n_success  = int(counts.get("grasp_success", 0))
    n_fail     = sum(int(counts.get(et, 0)) for et in ("empty_grasp", "slip", "stop"))
    n_attempts = n_success + n_fail
    if n_attempts > 0:
        print(f"  Grasp attempts : {n_attempts}  →  success rate {n_success / n_attempts:.1%}")

    # ── Recovery / lift intervention counts ────────────────────────────────────
    n_recovery = int(counts.get("recovery_home_ready", 0))
    n_lift     = int(counts.get("lift_position_ready", 0))
    if n_recovery + n_lift > 0:
        print(f"\n  Interventions:")
        if n_recovery > 0:
            print(f"    recovery_home_ready  : {n_recovery}  (recovery trajectories completed)")
        if n_lift > 0:
            print(f"    lift_position_ready  : {n_lift}  (LIFT_RETRY trajectories completed)")
        # settle_ms stats from ready events
        if "settle_ms" in sm.columns:
            ready_mask = sm["event_type"].isin(("recovery_home_ready", "lift_position_ready"))
            ready = sm[ready_mask & (sm["settle_ms"] > 0)]
            if len(ready) > 0:
                s = ready["settle_ms"]
                print(f"    settle_ms (all ready): mean={s.mean():.0f}  "
                      f"p50={np.percentile(s, 50):.0f}  max={s.max():.0f}  ms")

    # ── Per-event-type sensor statistics ──────────────────────────────────────
    float_fields = [c for c in ("gripper_load", "gripper_pos", "peak_load") if c in sm.columns]
    if float_fields:
        print()
        hdr = f"  {'event_type':<16}  {'n':>4}"
        for f in float_fields:
            hdr += f"    {f[:12]:<12}(mean / p50 / p95)"
        print(hdr)
        print("  " + "─" * max(60, len(hdr) - 2))
        for et in _SM_ORDER:
            sub = sm[sm["event_type"] == et]
            if len(sub) == 0:
                continue
            row = f"  {et:<16}  {len(sub):>4}"
            for col in float_fields:
                v = sub[col].dropna()
                if len(v) > 0:
                    row += (f"    {float(v.mean()):7.1f} / "
                            f"{float(np.percentile(v, 50)):7.1f} / "
                            f"{float(np.percentile(v, 95)):7.1f}")
                else:
                    row += f"    {'—':>7}   {'—':>7}   {'—':>7}"
            print(row)

    # ── Per-episode event breakdown ────────────────────────────────────────────
    if "episode" in sm.columns and sm["episode"].nunique() > 1:
        ep_ct = sm.groupby(["episode", "event_type"]).size().unstack(fill_value=0)
        for et in _SM_ORDER:
            if et not in ep_ct.columns:
                ep_ct[et] = 0
        print(f"\n  Per-episode event counts:")
        print(ep_ct[[et for et in _SM_ORDER if et in ep_ct.columns]].to_string())


def print_sm_diagnosis(sm: pd.DataFrame | None) -> None:
    """Rule-based diagnostic for gripper SM events."""
    if sm is None or len(sm) == 0:
        return

    _divider("GRIPPER SM DIAGNOSIS", 80)
    counts    = sm["event_type"].value_counts()
    n_success = int(counts.get("grasp_success", 0))
    n_empty   = int(counts.get("empty_grasp", 0))
    n_slip    = int(counts.get("slip", 0))
    n_stop    = int(counts.get("stop", 0))
    issues: list[tuple[str, str]] = []

    if n_stop > 0:
        stop_eps = (sm[sm["event_type"] == "stop"]["episode"].tolist()
                    if "episode" in sm.columns else [])
        ep_note = f"  Episode(s): {stop_eps}." if stop_eps else ""
        issues.append(("STOP TRIGGERED",
                       f"{n_stop} run(s) exhausted max retries — robot halted. {ep_note}\n"
                       "      → Check object placement / approach, or raise max_empty_grasp_retries."))

    if n_empty > 0:
        sub       = sm[sm["event_type"] == "empty_grasp"]
        load_mean = float(sub["gripper_load"].mean()) if "gripper_load" in sub.columns else float("nan")
        pos_mean  = float(sub["gripper_pos"].mean())  if "gripper_pos"  in sub.columns else float("nan")
        hint = ""
        if not np.isnan(pos_mean) and pos_mean < 3.0:
            hint = "\n      Pos≈0 → fully closed on air; adjust approach or lower gripper_pos_empty_threshold."
        elif not np.isnan(load_mean):
            hint = "\n      Load near threshold → lower gripper_load_grasp_threshold slightly."
        issues.append(("EMPTY GRASPS",
                       f"{n_empty} event(s).  mean_load={load_mean:.1f}  mean_pos={pos_mean:.1f}.{hint}"))

    if n_slip > 0:
        sub       = sm[sm["event_type"] == "slip"]
        peak_mean = float(sub["peak_load"].mean()) if "peak_load" in sub.columns else float("nan")
        hint = ""
        if not np.isnan(peak_mean):
            if peak_mean < 100:
                hint = "\n      Low peak_load → object barely held; check approach force / surface."
            else:
                hint = "\n      High peak_load → try raising gripper_slip_drop_ratio (40% drop triggers now)."
        issues.append(("SLIP EVENTS",
                       f"{n_slip} event(s).  mean_peak_load={peak_mean:.1f}.{hint}"))

    if not issues:
        if n_success > 0:
            print(f"  ✓  All {n_success} grasp(s) succeeded, no failures.")
        else:
            print("  (no grasp attempts recorded)")
        return

    print(f"  {len(issues)} issue(s) found:\n")
    for i, (title, detail) in enumerate(issues, 1):
        print(f"  [{i}] {title}")
        print(f"      {detail}\n")


def print_sm_gap_analysis(sm: pd.DataFrame | None) -> None:
    """Analysis: recovery/lift gap durations and settle time impact.

    Uses wall_time of ready events vs. preceding failure events to compute how
    long obs-send was suspended during recovery/lift trajectories + settle sleeps.
    """
    if sm is None or len(sm) == 0:
        return
    counts = sm["event_type"].value_counts()
    if not (counts.get("recovery_home_ready", 0) + counts.get("lift_position_ready", 0)):
        return

    _divider("RECOVERY / LIFT GAP ANALYSIS", 80)

    sm_s = sm.sort_values("wall_time").reset_index(drop=True)
    failure_types = {"empty_grasp", "slip"}
    ready_types   = {"recovery_home_ready", "lift_position_ready"}

    gaps = []
    for i, row in sm_s.iterrows():
        if row["event_type"] not in ready_types:
            continue
        # Find the most recent failure event in the same episode that precedes this ready event
        ep = row.get("episode", None)
        prior = sm_s.iloc[:i]
        if ep is not None and "episode" in sm_s.columns:
            prior = prior[prior["episode"] == ep]
        failures = prior[prior["event_type"].isin(failure_types)]
        if failures.empty:
            continue
        trigger = failures.iloc[-1]
        gap_s = float(row["wall_time"] - trigger["wall_time"])
        gaps.append({
            "episode":     ep,
            "ready_type":  row["event_type"],
            "gap_s":       gap_s,
            "settle_ms":   float(row.get("settle_ms", 0.0)),
        })

    if not gaps:
        print("  (no failure→ready pairs found — check that episodes match)")
        return

    gap_df = pd.DataFrame(gaps)
    print(f"  Failure→ready pairs found : {len(gap_df)}")
    g = gap_df["gap_s"]
    print(f"  Total gap duration (s)     : sum={g.sum():.1f}  mean={g.mean():.2f}  "
          f"p50={np.percentile(g, 50):.2f}  max={g.max():.2f}")
    print(f"  (gap = wall_time of ready event − wall_time of preceding failure event)")
    print(f"  Includes: recovery/lift trajectory steps + settle sleep + any queued processing")

    # Split by ready type
    for rt in ("recovery_home_ready", "lift_position_ready"):
        sub = gap_df[gap_df["ready_type"] == rt]
        if len(sub) == 0:
            continue
        g2 = sub["gap_s"]
        print(f"\n  {rt} (n={len(sub)}):")
        print(f"    gap_s : mean={g2.mean():.2f}  p50={np.percentile(g2, 50):.2f}  max={g2.max():.2f}")
        if "settle_ms" in sub.columns and sub["settle_ms"].max() > 0:
            sm_vals = sub["settle_ms"]
            print(f"    settle_ms : mean={sm_vals.mean():.0f}  "
                  f"p50={np.percentile(sm_vals, 50):.0f}  max={sm_vals.max():.0f}")
            settle_frac = sub["settle_ms"].mean() / (g2.mean() * 1000) if g2.mean() > 0 else 0
            print(f"    settle fraction of gap : {settle_frac:.0%}")


def plot_sm_overview(sm: pd.DataFrame | None, out_dir: Path) -> None:
    """Fig 11: SM event overview — timeline, load distribution, pos distribution.

    Panel 0  Wall-clock scatter: each event as a dot, Y=gripper_load, colour=event_type.
             Episode boundaries shown as vertical grey lines.
    Panel 1  Box plots of gripper_load grouped by event_type.
    Panel 2  Box plots of gripper_pos  grouped by event_type.
    """
    if sm is None or len(sm) == 0:
        return

    present  = [et for et in _SM_ORDER if et in sm["event_type"].values]
    has_wt   = "wall_time"    in sm.columns
    has_load = "gripper_load" in sm.columns
    has_pos  = "gripper_pos"  in sm.columns

    if not present:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Gripper State Machine Event Overview", fontweight="bold")

    # ── Panel 0: wall-clock timeline scatter ──────────────────────────────────
    ax = axes[0]
    if has_wt:
        t0 = float(sm["wall_time"].min())
        for et in present:
            sub   = sm[sm["event_type"] == et]
            t_rel = sub["wall_time"].values - t0
            y_val = sub["gripper_load"].values if has_load else np.ones(len(sub))
            ax.scatter(t_rel, y_val, c=_SM_COLORS[et], s=60, label=et,
                       alpha=0.85, linewidths=0.5, edgecolors="white", zorder=3)
        ax.set_xlabel("Wall-clock time (s from start)")
        ax.set_ylabel("gripper_load at event" if has_load else "event")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        if "episode" in sm.columns:
            for ep, ts in (sm.groupby("episode")["wall_time"].min() - t0).items():
                ax.axvline(ts, color="gray", linewidth=0.7, alpha=0.4)
                ax.text(ts, 0.97, f"ep{int(ep)}", fontsize=7, ha="center",
                        color="gray", rotation=90,
                        transform=ax.get_xaxis_transform(), va="top")
    else:
        ax.text(0.5, 0.5, "No wall_time data", transform=ax.transAxes, ha="center")
    ax.set_title("Event timeline  (Y = gripper_load,  colour = event type)", fontweight="bold")

    # ── Panel 1: gripper_load box plots ───────────────────────────────────────
    ax = axes[1]
    if has_load:
        _bx_vals, _bx_lbls = [], []
        for et in present:
            v = sm[sm["event_type"] == et]["gripper_load"].dropna().values
            if len(v):
                _bx_vals.append(v)
                _bx_lbls.append(et)
        if _bx_vals:
            bp = ax.boxplot(_bx_vals, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2},
                            flierprops={"marker": ".", "markersize": 4, "alpha": 0.5})
            for patch, et in zip(bp["boxes"], _bx_lbls):
                patch.set_facecolor(_SM_COLORS[et])
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(_bx_lbls) + 1))
            ax.set_xticklabels(_bx_lbls, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No gripper_load data", transform=ax.transAxes, ha="center")
    ax.set_ylabel("gripper_load")
    ax.set_title("gripper_load distribution by event type", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: gripper_pos box plots ─────────────────────────────────────────
    ax = axes[2]
    if has_pos:
        _bx_vals, _bx_lbls = [], []
        for et in present:
            v = sm[sm["event_type"] == et]["gripper_pos"].dropna().values
            if len(v):
                _bx_vals.append(v)
                _bx_lbls.append(et)
        if _bx_vals:
            bp = ax.boxplot(_bx_vals, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2},
                            flierprops={"marker": ".", "markersize": 4, "alpha": 0.5})
            for patch, et in zip(bp["boxes"], _bx_lbls):
                patch.set_facecolor(_SM_COLORS[et])
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(_bx_lbls) + 1))
            ax.set_xticklabels(_bx_lbls, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No gripper_pos data", transform=ax.transAxes, ha="center")
    ax.set_ylabel("gripper_pos")
    ax.set_title("gripper_pos distribution by event type", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, out_dir / "fig11_sm_overview.png", "sm_overview")


def plot_sm_trajectory(sm: pd.DataFrame | None, out_dir: Path) -> None:
    """Fig 12: When in the action sequence do SM events occur?

    Panel 0  Timestep scatter, Y = event_type (categorical).
    Panel 1  Timestep histogram: failures vs grasp_success.
    Panel 2  Per-episode event bar chart (only when ≥2 episodes present).
    """
    if sm is None or len(sm) == 0:
        return

    has_ts  = "timestep" in sm.columns
    has_ep  = "episode" in sm.columns and sm["episode"].nunique() > 1
    present = [et for et in _SM_ORDER if et in sm["event_type"].values]
    if not present:
        return

    n_panels = 3 if has_ep else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
    fig.suptitle("Gripper SM: Trajectory Failure Pattern", fontweight="bold")

    # ── Panel 0: timestep vs event type scatter ────────────────────────────────
    ax = axes[0]
    if has_ts:
        y_pos = {et: i for i, et in enumerate(_SM_ORDER)}
        for et in present:
            sub = sm[sm["event_type"] == et]
            ax.scatter(sub["timestep"].values, [y_pos[et]] * len(sub),
                       c=_SM_COLORS[et], s=60, label=et,
                       alpha=0.8, linewidths=0.5, edgecolors="white")
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(y_pos.keys()), fontsize=9)
        ax.set_xlabel("Timestep (executed action index)")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No timestep data", transform=ax.transAxes, ha="center")
    ax.set_title("Event type vs timestep  (left = early in trajectory, right = late)",
                 fontweight="bold")

    # ── Panel 1: timestep histogram ────────────────────────────────────────────
    ax = axes[1]
    if has_ts:
        fail_ts    = sm[sm["event_type"].isin(("empty_grasp", "slip", "stop"))]["timestep"].dropna()
        success_ts = sm[sm["event_type"] == "grasp_success"]["timestep"].dropna()
        ts_all     = sm["timestep"].dropna()
        if len(ts_all) > 0:
            ts_max    = float(ts_all.max())
            n_bins    = min(40, max(10, int(ts_max / 5)))
            bin_range = (0, ts_max * 1.05)
            if len(fail_ts) > 0:
                ax.hist(fail_ts.values, bins=n_bins, range=bin_range,
                        color=_SM_COLORS["empty_grasp"], alpha=0.7,
                        label=f"failures (n={len(fail_ts)})")
            if len(success_ts) > 0:
                ax.hist(success_ts.values, bins=n_bins, range=bin_range,
                        color=_SM_COLORS["grasp_success"], alpha=0.6,
                        label=f"grasp_success (n={len(success_ts)})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Event count")
        ax.set_title("Timestep distribution: failures vs successes", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # ── Panel 2 (optional): per-episode event bar chart ───────────────────────
    if has_ep:
        ax = axes[2]
        ep_ct = sm.groupby(["episode", "event_type"]).size().unstack(fill_value=0)
        for et in _SM_ORDER:
            if et not in ep_ct.columns:
                ep_ct[et] = 0
        episodes = ep_ct.index.tolist()
        x = np.arange(len(episodes))
        w = 0.18
        for j, et in enumerate(_SM_ORDER):
            offset = (j - 1.5) * w
            ax.bar(x + offset, ep_ct[et].values, w,
                   label=et, color=_SM_COLORS[et], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"ep{int(e)}" for e in episodes], fontsize=9)
        ax.set_ylabel("Event count")
        ax.set_title("SM events per episode", fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, out_dir / "fig12_sm_trajectory.png", "sm_trajectory")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    """Duplicate stdout to a report file.

    Instantiation immediately replaces sys.stdout.
    Call close() — or use as a context manager — to restore it.

    Usage (manual)::
        tee = _Tee(out_dir / "report.txt")
        print("goes to terminal AND file")
        tee.close()

    Usage (context manager)::
        with _Tee(out_dir / "report.txt"):
            print("goes to terminal AND file")
    """

    def __init__(self, path: Path) -> None:
        self._path   = path
        self._file   = path.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        """Restore sys.stdout and close the backing file."""
        if sys.stdout is self:
            sys.stdout = self._stdout
        self._file.close()

    def __enter__(self) -> "_Tee":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def main():
    ap = argparse.ArgumentParser(
        description="Analyze async-inference timing records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "eval_dir", nargs="?", type=str, default=None,
        help="Evaluation output directory (auto-discovers client_timing/ and server_timing/ subdirs)",
    )
    ap.add_argument("--client_dir",  type=str, default=None,
                    help="Directory with client timing JSONL files (overrides eval_dir)")
    ap.add_argument("--server_dir",  type=str, default=None,
                    help="Directory with server timing JSONL files (overrides eval_dir)")
    ap.add_argument("--results_dir", type=str, default=None,
                    help="Directory with aggregate.json / episodes.json (overrides eval_dir)")
    ap.add_argument("--out_dir",     type=str, default=None,
                    help="Output directory for PNG figures (default: <eval_dir>/timing_analysis or ./timing_analysis)")
    ap.add_argument("--warmup_n",    type=str, default="auto",
                    help="Number of warmup inferences to exclude ('auto' or integer, default: auto)")
    ap.add_argument("--fps",         type=float, default=None,
                    help="Control-loop fps used for starvation / calibration analyses "
                         "(default: auto-detected from obs send intervals)")
    ap.add_argument("--no_plots",    action="store_true",
                    help="Skip figure generation (tables only)")
    ap.add_argument("--infer_q",     type=float, default=0.90,
                    help="Quantile of the stable server_infer component for the Tier 2 "
                         "split-calibration analysis (match client infer_latency_quantile)")
    ap.add_argument("--overhead_q",  type=float, default=0.75,
                    help="Quantile of the heavy-tailed overhead component for the Tier 2 "
                         "split-calibration analysis (match client overhead_latency_quantile)")
    args = ap.parse_args()

    # ── Resolve directories ────────────────────────────────────────────────────
    client_dir  = args.client_dir
    server_dir  = args.server_dir
    results_dir = args.results_dir

    if args.eval_dir and not (client_dir or server_dir):
        cd, sd, rd = _auto_discover(args.eval_dir)
        client_dir  = client_dir  or (str(cd) if cd else None)
        server_dir  = server_dir  or (str(sd) if sd else None)
        results_dir = results_dir or (str(rd) if rd else None)
        if not client_dir and not server_dir:
            ap.error(f"Could not find client_timing/ or server_timing/ under '{args.eval_dir}'")

    if client_dir is None and server_dir is None:
        ap.error("Provide eval_dir (positional) or at least one of --client_dir / --server_dir")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.eval_dir:
        out_dir = Path(args.eval_dir) / "timing_analysis"
    else:
        out_dir = Path("./timing_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    _report_path = out_dir / "analysis_report.txt"
    _tee = _Tee(_report_path)

    print("\n" + "═" * 80)
    print("  Async-Inference Timing Analysis")
    print("═" * 80)
    if args.eval_dir:
        print(f"  eval_dir   : {args.eval_dir}")
    print(f"  client_dir : {client_dir or '(none)'}")
    print(f"  server_dir : {server_dir or '(none)'}")
    print(f"  results_dir: {results_dir or '(none)'}")
    print(f"  out_dir    : {out_dir}")

    print("\nLoading records:")
    data_raw  = load_data(client_dir, server_dir)
    results   = load_results(results_dir)
    sm_events = load_sm_events(client_dir)

    if all(v is None for v in data_raw.values()):
        print("ERROR: No timing files found in the specified directories.")
        _tee.close()
        print(f"  Console output → {_report_path}")
        sys.exit(1)

    # ── Warmup detection ───────────────────────────────────────────────────────
    if args.warmup_n == "auto":
        n_warmup = _detect_warmup_n(data_raw["infer"])
    else:
        try:
            n_warmup = int(args.warmup_n)
        except ValueError:
            ap.error(f"--warmup_n must be 'auto' or an integer, got '{args.warmup_n}'")

    data_filtered = _filter_warmup(data_raw, n_warmup)
    # Assign episode to server records via wall_time join (use warmup-filtered chunk data)
    data_raw_ep      = _assign_episodes(data_raw)
    data_filtered_ep = _assign_episodes(data_filtered)

    # ── fps: auto-detect or use provided value ─────────────────────────────────
    fps = args.fps if args.fps is not None else _infer_fps(data_raw_ep)
    print(f"  fps (for analyses 1-3) : {fps:.1f}"
          + ("  (auto-detected)" if args.fps is None else "  (from --fps)"))

    # ── Console output ─────────────────────────────────────────────────────────
    print_warmup_info(data_raw, n_warmup)
    print_obs_funnel(data_raw_ep)
    print_stats_table(data_raw_ep, data_filtered_ep, n_warmup)
    print_budget_table(data_raw_ep, data_filtered_ep, n_warmup)
    print_tail_table(data_filtered_ep)
    print_per_episode_table(data_filtered_ep, results)
    print_diagnosis(data_filtered_ep, n_warmup)

    if sm_events is not None:
        print_sm_event_summary(sm_events)
        print_sm_diagnosis(sm_events)
        print_sm_gap_analysis(sm_events)

    # ── Analyses 1–3 ──────────────────────────────────────────────────────────
    print_infer_delay_calibration(data_filtered_ep, fps)
    print_split_diagnosis(data_filtered_ep, args.infer_q, args.overhead_q)
    print_starvation_analysis(data_filtered_ep, fps)

    if args.no_plots:
        _tee.close()
        print(f"  Console output → {_report_path}")
        return

    # ── Figures ────────────────────────────────────────────────────────────────
    _divider("GENERATING FIGURES", 80)
    plot_budget(data_raw_ep, data_filtered_ep, n_warmup, out_dir)
    plot_time_series(data_raw_ep, n_warmup, out_dir)
    plot_pipeline(data_filtered_ep, out_dir)
    plot_tail_latency(data_filtered_ep, out_dir)
    plot_health(data_raw_ep, out_dir)
    plot_episode_stats(data_filtered_ep, results, out_dir)
    plot_timeline(data_raw_ep, n_warmup, out_dir)

    # ── Figures 8–10 (analyses 1–3) ───────────────────────────────────────────
    plot_infer_delay_calibration(data_filtered_ep, fps, out_dir)
    plot_starvation(data_filtered_ep, fps, out_dir)
    plot_filter_rate(data_raw_ep, out_dir)

    if sm_events is not None:
        plot_sm_overview(sm_events, out_dir)
        plot_sm_trajectory(sm_events, out_dir)

    plot_frequency_stats(data_raw_ep, out_dir)
    plot_split_component_calibration(data_filtered_ep, fps, out_dir,
                                     infer_q=args.infer_q, overhead_q=args.overhead_q)

    print(f"\n  All figures saved to: {out_dir}/\n")
    _tee.close()
    print(f"  Console output → {_report_path}")


if __name__ == "__main__":
    main()
