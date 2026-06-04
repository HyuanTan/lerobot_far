#!/usr/bin/env bash
# analyze_tools_script/analyze_so101_all.sh
#
# Run per-method timing + RTC analysis on every (method/policy/param) leaf
# under the merged SO-101 eval tree, then run the cross-method comparison.
#
# Steps per leaf directory:
#   1. analyze_timing  → <leaf>/timing_analysis/
#   2. analyze_rtc     → <leaf>/rtc_analysis/   (skipped if no RTC data)
#
# Final step (unless NO_COMPARE=1):
#   3. analyze_so101_comparison → <ROOT>/comparison/
#
# Usage:
#   bash analyze_tools_script/analyze_so101_all.sh [ROOT]
#   bash analyze_tools_script/analyze_so101_all.sh outputs/eval_thesis/so101
#
# Environment overrides:
#   DRY_RUN=1        — print commands without executing
#   SKIP_EXISTING=1  — skip leaf if timing_analysis/ already exists  (default: 1)
#   NO_COMPARE=1     — skip the final analyze_so101_comparison step
#   FPS=10           — control-loop fps passed to both analyzers
#   POLICY=          — filter to a single policy (e.g. POLICY=pi05)
#   PARAM=           — filter to a single param  (e.g. PARAM=H15)

set -uo pipefail

ROOT="${1:-outputs/eval_thesis/so101}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
NO_COMPARE="${NO_COMPARE:-0}"
FPS="${FPS:-20}"
FILTER_POLICY="${POLICY:-}"
FILTER_PARAM="${PARAM:-}"

# ── Validate root ─────────────────────────────────────────────────────────────
if [ ! -d "$ROOT" ]; then
    echo "[ERROR] ROOT not found: $ROOT"
    exit 1
fi

echo "============================================================"
echo "  analyze_so101_all — per-method timing & RTC analysis"
echo "============================================================"
echo "  ROOT          = $ROOT"
echo "  DRY_RUN       = $DRY_RUN"
echo "  SKIP_EXISTING = $SKIP_EXISTING"
echo "  NO_COMPARE    = $NO_COMPARE"
echo "  FPS           = $FPS"
[ -n "$FILTER_POLICY" ] && echo "  POLICY filter = $FILTER_POLICY"
[ -n "$FILTER_PARAM"  ] && echo "  PARAM  filter = $FILTER_PARAM"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
run_cmd() {
    echo "[CMD] $*"
    if [ "$DRY_RUN" = "0" ]; then
        "$@"
    fi
}

count=0
ok=0
fail=0

# ── Per-leaf: analyze_timing + analyze_rtc ────────────────────────────────────
# Walk method → policy → param; skip the special `comparison` output dir.
while IFS= read -r client_timing_dir; do
    leaf="$(dirname "$client_timing_dir")"

    # Skip the comparison output directory
    case "$leaf" in
        */comparison*) continue ;;
    esac

    # Extract method / policy / param from path relative to ROOT
    rel="${leaf#"$ROOT"/}"           # e.g. async_rtc_sm/pi05/H15
    method="$(echo "$rel" | cut -d/ -f1)"
    policy="$(echo "$rel" | cut -d/ -f2)"
    param="$(echo "$rel"  | cut -d/ -f3)"

    # Apply optional filters
    if [ -n "$FILTER_POLICY" ] && [ "$policy" != "$FILTER_POLICY" ]; then
        continue
    fi
    if [ -n "$FILTER_PARAM" ] && [ "$param" != "$FILTER_PARAM" ]; then
        continue
    fi

    echo ""
    echo "============================================================"
    echo "[LEAF] $method / $policy / $param"
    echo "       $leaf"
    echo "============================================================"

    count=$(( count + 1 ))
    timing_out="$leaf/timing_analysis"
    rtc_out="$leaf/rtc_analysis"

    # ── 1. analyze_timing ─────────────────────────────────────────────────────
    if [ "$SKIP_EXISTING" = "1" ] && [ -d "$timing_out" ]; then
        echo "[SKIP] timing_analysis already exists: $timing_out"
    else
        if run_cmd uv run python -m lerobot.async_inference.analyze_timing \
                "$leaf" \
                --out_dir "$timing_out" \
                --fps "$FPS"; then
            echo "[OK]   analyze_timing → $timing_out"
        else
            echo "[WARN] analyze_timing failed for $leaf"
            fail=$(( fail + 1 ))
            continue
        fi
    fi

    # ── 2. analyze_rtc ────────────────────────────────────────────────────────
    # leftover_steps lives in client_chunk_action_records.jsonl, not aggregate.
    action_records="${client_timing_dir}/client_chunk_action_records.jsonl"
    has_rtc=0
    if [ -f "$action_records" ] && grep -q "leftover_steps" "$action_records" 2>/dev/null; then
        has_rtc=1
    fi

    if [ "$has_rtc" = "0" ]; then
        echo "[SKIP] analyze_rtc — no leftover_steps data in $client_timing_dir"
    elif [ "$SKIP_EXISTING" = "1" ] && [ -d "$rtc_out" ]; then
        echo "[SKIP] rtc_analysis already exists: $rtc_out"
    else
        if run_cmd uv run python -m lerobot.async_inference.analyze_rtc \
                --client_dir "$client_timing_dir" \
                --out_dir "$rtc_out" \
                --fps "$FPS"; then
            echo "[OK]   analyze_rtc → $rtc_out"
        else
            echo "[WARN] analyze_rtc failed for $leaf"
            fail=$(( fail + 1 ))
            continue
        fi
    fi

    ok=$(( ok + 1 ))

done < <(find "$ROOT" -type d -name "client_timing" | sort)

echo ""
echo "============================================================"
echo "[DONE] Per-method analysis"
echo "  Leaves processed  : $count"
echo "  Succeeded         : $ok"
echo "  Failed            : $fail"
echo "============================================================"

# ── Cross-method comparison ───────────────────────────────────────────────────
if [ "$NO_COMPARE" = "1" ]; then
    echo ""
    echo "[SKIP] analyze_so101_comparison (NO_COMPARE=1)"
else
    echo ""
    echo "============================================================"
    echo "[INFO] analyze_so101_comparison — cross-method summary"
    echo "============================================================"

    compare_args=("$ROOT" --out_dir "$ROOT/comparison")
    [ -n "$FILTER_POLICY" ] && compare_args+=(--policy "$FILTER_POLICY")
    [ -n "$FILTER_PARAM"  ] && compare_args+=(--param  "$FILTER_PARAM")
    compare_args+=(--fps "$FPS")

    if run_cmd uv run python -m lerobot.async_inference.analyze_so101_comparison \
            "${compare_args[@]}"; then
        echo "[OK]   analyze_so101_comparison → $ROOT/comparison/"
    else
        echo "[WARN] analyze_so101_comparison failed"
    fi
fi

echo ""
echo "Done."
