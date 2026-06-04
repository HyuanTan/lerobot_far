#!/usr/bin/env bash
# analyze_tools_script/analyze_all_eval_outputs.sh
#
# Run per-combo timing / RTC analysis on all eval output dirs, then run
# cross-combo sweep analysis (solve rate vs. delay / horizon).
#
# Per-combo analysis (analyze_timing + analyze_rtc):
#   Finds every client_timing/ directory under ROOT, treats its parent as an
#   eval_dir, and runs:
#     1. analyze_timing  → <eval_dir>/timing_analysis/
#     2. analyze_rtc     → <eval_dir>/rtc_analysis/    (skipped if RTC data absent)
#
# Cross-combo sweep analysis (analyze_sweep):
#   Reads all results/aggregate.json files under ROOT and generates:
#     3. analyze_sweep   → <ROOT>/analysis/
#
# Usage:
#   bash analyze_tools_script/analyze_all_eval_outputs.sh <ROOT>
#   bash analyze_tools_script/analyze_all_eval_outputs.sh outputs/eval_thesis/libero
#
#   # Dry-run (print commands without executing):
#   DRY_RUN=1 bash analyze_tools_script/analyze_all_eval_outputs.sh <ROOT>
#
#   # Skip already-existing analysis output dirs:
#   SKIP_EXISTING=1 bash analyze_tools_script/analyze_all_eval_outputs.sh <ROOT>
#
#   # Skip sweep analysis (only run per-combo):
#   NO_SWEEP=1 bash analyze_tools_script/analyze_all_eval_outputs.sh <ROOT>

set -uo pipefail

ROOT="${1:-outputs/eval/so101_V2.0_0521}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_SWEEP="${NO_SWEEP:-0}"

if [ ! -d "$ROOT" ]; then
    echo "[ERROR] ROOT not found: $ROOT"
    exit 1
fi

echo "[INFO] ROOT          = $ROOT"
echo "[INFO] DRY_RUN       = $DRY_RUN"
echo "[INFO] SKIP_EXISTING = $SKIP_EXISTING"
echo "[INFO] NO_SWEEP      = $NO_SWEEP"
echo "[INFO] Searching experiments with client_timing/ ..."

run_cmd() {
    echo ""
    echo "[CMD] $*"
    if [ "$DRY_RUN" = "0" ]; then
        "$@"
    fi
}

count=0
ok=0
fail=0

# ── Per-combo: analyze_timing + analyze_rtc ───────────────────────────────────
while IFS= read -r client_timing_dir; do
    exp_dir="$(dirname "$client_timing_dir")"

    echo ""
    echo "============================================================"
    echo "[INFO] Experiment: $exp_dir"
    echo "============================================================"

    count=$(( count + 1 ))

    timing_out="$exp_dir/timing_analysis"
    rtc_out="$exp_dir/rtc_analysis"

    # ── 1. analyze_timing ─────────────────────────────────────────────────
    # auto-discovers:
    #   client_timing/  — always present
    #   server_timing   — symlink (created by eval script) or real dir
    #   results/        — aggregate.json (also accepts sim_test_results/)
    if [ "$SKIP_EXISTING" = "1" ] && [ -d "$timing_out" ]; then
        echo "[SKIP] timing_analysis exists: $timing_out"
    else
        if run_cmd uv run python -m lerobot.async_inference.analyze_timing \
            "$exp_dir" \
            --out_dir "$timing_out"; then
            echo "[OK]   analyze_timing → $timing_out"
        else
            echo "[WARN] analyze_timing failed: $exp_dir"
            fail=$(( fail + 1 ))
            continue
        fi
    fi

    # ── 2. analyze_rtc ────────────────────────────────────────────────────
    # Only meaningful when RTC data is present (client_aggregate_records.jsonl
    # contains leftover_steps).  Silently skip if the key file is absent.
    has_rtc_data=0
    if [ -f "${client_timing_dir}/client_aggregate_records.jsonl" ]; then
        # Quick check: does the file contain leftover_steps field?
        if grep -q "leftover_steps" "${client_timing_dir}/client_aggregate_records.jsonl" 2>/dev/null; then
            has_rtc_data=1
        fi
    fi

    if [ "$has_rtc_data" = "0" ]; then
        echo "[SKIP] analyze_rtc — no leftover_steps data in ${client_timing_dir}"
    elif [ "$SKIP_EXISTING" = "1" ] && [ -d "$rtc_out" ]; then
        echo "[SKIP] rtc_analysis exists: $rtc_out"
    else
        if run_cmd uv run python -m lerobot.async_inference.analyze_rtc \
            --client_dir "$client_timing_dir" \
            --out_dir "$rtc_out"; then
            echo "[OK]   analyze_rtc → $rtc_out"
        else
            echo "[WARN] analyze_rtc failed: $exp_dir"
            fail=$(( fail + 1 ))
            continue
        fi
    fi

    ok=$(( ok + 1 ))

done < <(find "$ROOT" -type d -name "client_timing" | sort)

echo ""
echo "============================================================"
echo "[DONE] Per-combo analysis"
echo "  Found experiments : $count"
echo "  Succeeded         : $ok"
echo "  Failed            : $fail"
echo "============================================================"

# ── Cross-combo: analyze_sweep ────────────────────────────────────────────────
# Reads all results/aggregate.json under ROOT and produces solve rate plots.
if [ "$NO_SWEEP" = "1" ]; then
    echo ""
    echo "[SKIP] analyze_sweep (NO_SWEEP=1)"
else
    n_agg=$(find "$ROOT" -path "*/results/aggregate.json" | wc -l)
    if [ "$n_agg" -eq 0 ]; then
        echo ""
        echo "[SKIP] analyze_sweep — no results/aggregate.json found under $ROOT"
    else
        echo ""
        echo "============================================================"
        echo "[INFO] analyze_sweep — ${n_agg} aggregate.json found"
        echo "============================================================"
        if run_cmd uv run python -m lerobot.async_inference.analyze_sweep \
            "$ROOT" \
            --out_dir "$ROOT/analysis"; then
            echo "[OK]   analyze_sweep → $ROOT/analysis"
        else
            echo "[WARN] analyze_sweep failed"
        fi
    fi
fi
