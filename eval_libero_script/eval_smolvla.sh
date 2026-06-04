#!/usr/bin/env bash
# =============================================================================
# eval_smolvla.sh  —  SmolVLA  LIBERO sweep
#
# Reproduces two plots (matching the paper figure):
#   LEFT:  Solve Rate vs. Inference Delay   d ∈ DELAYS   (fixed s = DELAY_FIXED_S)
#   RIGHT: Solve Rate vs. Execution Horizon s ∈ HORIZONS (fixed d = HORIZON_FIXED_D)
#
# ──────────────────────────────────────────────────────────────────────────────
# Parameter semantics (consistent with bt-libero/it_rtc/eval_libero_rtc.py):
#
#   async_delay  (d):  env steps the policy receives a STALE observation.
#                      Policy sees obs from d steps ago.
#                      → --eval.async_delay
#
#   execution_horizon (s):  actions executed per planning cycle before re-plan.
#                           Must satisfy 1 ≤ s ≤ chunk_size.
#                           RTC:      effective_horizon = max(d, s)  [d ≤ s required]
#                           Baseline: effective_horizon = s          [if 0 < s < chunk_size]
#                                   = chunk_size                    [if s=0 or s≥chunk_size]
#                           → --eval.execution_horizon
#
# ──────────────────────────────────────────────────────────────────────────────
# Difference from original bt-libero/it_rtc/eval_itrtc_badresults.sh:
#
#   Original: ASYNC_DELAYS=(0 1 2 4 8 15 20), EXECUTION_HORIZON=10 (fixed for RTC).
#             Baseline IGNORES execution_horizon → always uses full chunk (50 steps).
#             WARNING: original had d=15,20 > s=10 violating RTC d≤s constraint.
#             No horizon sweep.
#
#   This script: Both d and s are swept. Baseline uses execution_horizon for
#             replanning frequency (controlled comparison where s matches).
#             Set DELAY_FIXED_S=0 to restore original full-chunk baseline.
#
# 4 method combinations:
#   baseline    — chunked execution + delayed obs,  no RTC,  no SM
#   rtc         — chunked execution + delayed obs,  RTC on,  no SM
#   baseline_sm — baseline + gripper SM (empty-grasp detection + set_state rewind)
#   rtc_sm      — RTC     + gripper SM
#
# Usage:
#   bash eval_smolvla.sh [--dry-run] [--delay-only] [--horizon-only] [--no-sm]
#
# Environment variables (set before running):
#   CKPT       path to pretrained checkpoint                (required)
#   GPU        comma-separated GPU ids, e.g. "0,1,2,3"     (default: "0")
#   SUITES     space-separated suite list                   (default: all 4)
#   N_EPISODES episodes per task                            (default: 10)
#   BATCH_SIZE parallel envs per batch for non-SM runs      (default: 10, must ≤ N_EPISODES)
#   OUT_ROOT   output root dir                              (default: outputs/eval/pi05)
#   SEED       random seed                                  (default: 42)
# =============================================================================
set -euo pipefail

# ── Working directory: must be the LeRobot project root ──────────────────────
# eval_smolvla.sh may be called from any cwd; cd to project root so that
# `uv run` picks up .venv at lerobot_v0.5.2/ and module paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── MuJoCo headless rendering (server without display) ───────────────────────
# Without these, libero → robosuite → OpenCV triggers "libGL.so.1: not found".
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT="${CKPT:-HollyTan/libero_smolvla_500MSmolVLM2_multitask}"
# GPU: comma-separated GPU ids for round-robin job assignment.
#   Single GPU: GPU=3           → all jobs use GPU 3
#   Multi  GPU: GPU=0,1,2,3    → jobs cycle GPU 0→1→2→3→0→...
# GPU="${GPU:-1,2,3}"

# ── GPU selection ─────────────────────────────────────────────────────────────
# Priority:
#   1. Use GPU env if explicitly set, e.g. GPU=0,1 bash eval_smolvla.sh
#   2. Use Slurm-assigned CUDA_VISIBLE_DEVICES if available
#   3. Fall back to all visible GPUs from nvidia-smi
#
# In Slurm jobs with --gres=gpu:1, CUDA_VISIBLE_DEVICES is usually already
# restricted to the allocated GPU(s). Inside the job, using GPU=0 is safest.

if [[ -n "${GPU:-}" ]]; then
  GPU="${GPU}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Slurm may set CUDA_VISIBLE_DEVICES to physical IDs or remapped IDs.
  # For CUDA inside the job, visible GPUs are usually re-indexed from 0.
  IFS=',' read -ra _VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  _N_VISIBLE=${#_VISIBLE_GPU_LIST[@]}

  if (( _N_VISIBLE == 1 )); then
    GPU="0"
  else
    GPU="$(seq -s, 0 $((_N_VISIBLE - 1)))"
  fi
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    _N_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    if (( _N_GPU > 0 )); then
      GPU="$(seq -s, 0 $((_N_GPU - 1)))"
    else
      echo "[ERROR] No GPU found by nvidia-smi."
      exit 1
    fi
  else
    echo "[ERROR] GPU is not set, CUDA_VISIBLE_DEVICES is empty, and nvidia-smi is unavailable."
    echo "        Please set GPU manually, e.g. GPU=0 bash eval_smolvla.sh"
    exit 1
  fi
fi

echo "[INFO] Using GPU=${GPU}"

SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
# SUITES="${SUITES:-libero_spatial libero_object}"
N_EPISODES="${N_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-10}"   # EvalConfig requires BATCH_SIZE <= N_EPISODES
OUT_ROOT="${OUT_ROOT:-outputs/eval_thesis_sim/libero/smolvla}"
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
SEED="${SEED:-42}"

# ── Sweep ranges ──────────────────────────────────────────────────────────────
# NOTE: HORIZONS must start at 2 (not 1) so that HORIZON_FIXED_D=1 < min(HORIZONS).
#       This keeps every horizon-sweep point in the valid RTC regime d < s.
# Dense (matches figure exactly): uncomment both lines below
# DELAYS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
# HORIZONS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)

# # Sparse (step-2, same range, fewer GPU-hours):
# DELAYS=(0 2 4 6 8 10 12 14)     # async_delay  d ∈ [0, DELAY_FIXED_S)
# HORIZONS=(2 4 6 8 10 12 14)     # execution_horizon s ∈ [2, chunk_size); min=2 > HORIZON_FIXED_D=1

DELAYS=(0 2)     # async_delay  d ∈ [0, DELAY_FIXED_S)
HORIZONS=(2 4)   # execution_horizon s ∈ [2, chunk_size)

# ── Fixed values for cross-sweeps ─────────────────────────────────────────────
# execution_horizon s used during the DELAY sweep (Sweep 1).
#
# CRITICAL for RTC:  max(DELAYS) < DELAY_FIXED_S < CHUNK_SIZE(=50)
#   Lower bound (> max DELAYS): so RTC constraint d < s holds (effective_horizon=s).
#   Upper bound (< chunk_size): so a non-empty leftover prefix exists each cycle.
#     If s >= chunk_size the whole chunk is consumed before replanning, the
#     leftover is empty, and RTC guidance (err = (0 - x1_t)*weights) pulls every
#     action toward ZERO → SR collapses to ~10-20% (verified: d0_s50_rtc = 13-21%
#     vs baseline 90%). The reference impl eval_real_robot_rtc_analysis.py uses
#     execution_horizon=10 << 50 for exactly this reason.
#   → 16 satisfies 14 < 16 < 50.  Set to 0 or >= 50 only to test the baseline path.
DELAY_FIXED_S=16         # s fixed during the delay sweep; max(DELAYS)=14 < 16 < 50 ✓

# async_delay d used during the HORIZON sweep (Sweep 2).
# Must be strictly LESS THAN min(HORIZONS) to avoid the d=s RTC-degenerate case:
#   When d = s: effective_horizon = max(d,s) = d = s
#               new_actions = new_chunk[d:s] = new_chunk[d:d] = EMPTY
#               → RTC executes 0 new actions; all actions come from the old chunk.
#               RTC guidance is computed but its actions are never executed.
# Rule: HORIZON_FIXED_D < min(HORIZONS) ensures new_chunk[d:s] is always non-empty.
#
# Plan A (matches the paper figure's right panel, "d=1"):
#   d=1 gives a non-empty hard-anchor zone [0,1) so RTC actually anchors to the
#   previous chunk (unlike d=0 where RTC ≈ baseline). Requires min(HORIZONS) > 1,
#   hence HORIZONS starts at 2 above.
HORIZON_FIXED_D=1        # d fixed during the horizon sweep; must be < min(HORIZONS)=2

# Policy chunk size (pi05 and smolvla both default to 50).
# Passed only as a comment reference; actual chunk_size comes from the checkpoint config.
CHUNK_SIZE=50

# ── RTC config ────────────────────────────────────────────────────────────────
MAX_GUIDANCE_WEIGHT=5.0
PREFIX_ATTENTION_SCHEDULE=EXP  # ZEROS | ONES | LINEAR | EXP

# ── SM config ─────────────────────────────────────────────────────────────────
SM_REWIND_BUFFER=60
SM_REWIND_WARMUP=10
SM_CONFIRM_STEPS=3

# ── Method selection ──────────────────────────────────────────────────────────
# METHODS: space-separated subset of {baseline, rtc, baseline_sm, rtc_sm}
# Default (empty) = all four.  CLI flags override this.
# Each entry maps to: method_type and enable_sm in eval_libero_rtc:
#   baseline    → --eval.method_type=baseline  --eval.enable_sm=false
#   rtc         → --eval.method_type=rtc       --eval.enable_sm=false
#   baseline_sm → --eval.method_type=baseline  --eval.enable_sm=true
#   rtc_sm      → --eval.method_type=rtc       --eval.enable_sm=true
METHODS="${METHODS:-}"     # empty = all; set via env or --method=<name>

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=0
RUN_DELAY=1
RUN_HORIZON=1
# FORCE=1: re-run combinations that already have eval_results.json.
# Default 0 = skip existing results (safe resume after interruption).
# Override via env: FORCE=1 bash eval_smolvla.sh, or pass --force flag.
FORCE="${FORCE:-0}"

for arg in "$@"; do
  case "$arg" in
    --dry-run)        DRY_RUN=1 ;;
    --delay-only)     RUN_HORIZON=0 ;;
    --horizon-only)   RUN_DELAY=0 ;;
    --force)          FORCE=1 ;;
    --no-force)       FORCE=0 ;;
    # Method selection
    --method=*)       METHODS="${arg#--method=}" ;;
    --baseline-only)  METHODS="baseline" ;;
    --rtc-only)       METHODS="rtc" ;;
    --no-rtc)         METHODS="baseline baseline_sm" ;;
    --no-baseline)    METHODS="rtc rtc_sm" ;;
    --no-sm)          METHODS="baseline rtc" ;;
    --sm-only)        RUN_DELAY=0; RUN_HORIZON=0; METHODS="baseline_sm rtc_sm" ;;
  esac
done

# Resolve METHODS to an array (default: all four)
if [[ -z "$METHODS" ]]; then
  _METHODS=(baseline rtc baseline_sm rtc_sm)
  # _METHODS=(baseline baseline_sm)
  # _METHODS=(rtc rtc_sm)
else
  read -ra _METHODS <<< "$METHODS"
fi

# Validate each method name
for _m in "${_METHODS[@]}"; do
  case "$_m" in
    baseline|rtc|baseline_sm|rtc_sm) ;;
    *) echo "[ERROR] Unknown method '${_m}'. Valid: baseline rtc baseline_sm rtc_sm"; exit 1 ;;
  esac
done

# Helper: call run_eval for all selected methods at given (d, s, suite)
run_methods() {
  local suite="$1" d="$2" s="$3"
  for _m in "${_METHODS[@]}"; do
    case "$_m" in
      baseline)    run_eval "$suite" "baseline" "false" "$d" "$s" "${suite}/baseline" ;;
      rtc)         run_eval "$suite" "rtc"      "false" "$d" "$s" "${suite}/rtc" ;;
      baseline_sm) run_eval "$suite" "baseline" "true"  "$d" "$s" "${suite}/baseline+SM" ;;
      rtc_sm)      run_eval "$suite" "rtc"      "true"  "$d" "$s" "${suite}/rtc+SM" ;;
    esac
  done
}

# ── Signal handling: Ctrl+C kills only our background jobs ────────────────────
_ALL_PIDS=()   # every background PID launched by this script

# kill_job_tree <pid>
# Kills a process AND its entire descendant tree (so Python children of
# `uv run` don't become GPU-hogging orphans after we kill the parent).
# Guards against PID reuse by verifying the process still belongs to us
# (command line must contain "eval_libero_rtc").
kill_job_tree() {
  local pid="$1"

  # 1. Is the process still alive?
  kill -0 "$pid" 2>/dev/null || return 0

  # 2. Verify the process is owned by the current user (not another user's PID).
  local proc_uid
  proc_uid=$(ps -p "$pid" -o uid= 2>/dev/null | tr -d ' ' || true)
  if [[ -z "$proc_uid" || "$proc_uid" != "$(id -u)" ]]; then
    echo "[INTERRUPT] PID ${pid} not owned by $(id -un), skipping."
    return 0
  fi

  # 3. Verify it's our job: command line must contain our module name.
  #    Guards against accidental kill of a reused PID owned by the same user.
  local cmdline
  cmdline=$(ps -p "$pid" -o args= 2>/dev/null || true)
  if [[ "$cmdline" != *"eval_libero_rtc"* && "$cmdline" != *"uv"* ]]; then
    echo "[INTERRUPT] PID ${pid} not an eval job (cmd: ${cmdline:0:60}), skipping."
    return 0
  fi

  # 3. Kill the whole descendant tree (children, grandchildren, …) first,
  #    then the process itself.  `pkill -P` sends SIGTERM to direct children;
  #    we recurse for deeper descendants.
  local children
  children=$(pgrep -P "$pid" 2>/dev/null || true)
  for child in $children; do
    kill_job_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

cleanup() {
  echo ""
  echo "[INTERRUPT] Stopping all eval jobs (${#_ALL_PIDS[@]} launched)..."
  local killed=0
  for pid in "${_ALL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill_job_tree "$pid"
      killed=$((killed + 1))
    fi
  done
  # Give processes 1 s to exit gracefully, then SIGKILL survivors
  sleep 1
  for pid in "${_ALL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[INTERRUPT] Done (sent SIGTERM to ${killed} job tree(s))."
  exit 130   # 128 + SIGINT(2)
}

trap cleanup INT TERM

# ── GPU helpers ───────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST <<< "$GPU"
N_GPU=${#GPU_LIST[@]}
JOB_IDX=0

# Round-robin GPU assignment.
# MUST be called without $() so JOB_IDX increments in the parent shell.
# Result is stored in _NEXT_GPU (not echoed), avoiding the subshell trap.
_NEXT_GPU=""
assign_gpu() {
  _NEXT_GPU="${GPU_LIST[$((JOB_IDX % N_GPU))]}"
  JOB_IDX=$((JOB_IDX + 1))
}

PIDS=()

# ── Sanity checks ─────────────────────────────────────────────────────────────
# Compute true max/min regardless of array ordering.
max_delay=$(printf '%s\n' "${DELAYS[@]}"   | sort -n | tail -1)
min_horizon=$(printf '%s\n' "${HORIZONS[@]}" | sort -n | head -1)

# Check 1: Delay-sweep horizon must be strictly greater than max(DELAYS) for RTC validity
if (( DELAY_FIXED_S > 0 && max_delay >= DELAY_FIXED_S )); then
  echo "[ERROR] max(DELAYS)=${max_delay} >= DELAY_FIXED_S=${DELAY_FIXED_S}"
  echo "        RTC requires d < s → effective_horizon = max(d,s) = s for all d."
  echo "        Fix: increase DELAY_FIXED_S or reduce DELAYS."
  exit 1
fi

# Check 2: Horizon-sweep fixed delay must be strictly less than min(HORIZONS).
# d >= s is degenerate: new_chunk[d:s] is EMPTY → 0 new actions per cycle.
# Changed from WARN to ERROR — degenerate combinations are now skipped per-job
# by run_eval(), so the global config should be self-consistent.
if (( HORIZON_FIXED_D >= min_horizon )); then
  echo "[ERROR] HORIZON_FIXED_D=${HORIZON_FIXED_D} >= min(HORIZONS)=${min_horizon}"
  echo "        At d=${HORIZON_FIXED_D}, s=${min_horizon}: new_chunk[d:s] = EMPTY (RTC degenerate)."
  echo "        Fix: HORIZON_FIXED_D < ${min_horizon}, e.g. HORIZON_FIXED_D=0"
  echo "        Or remove ${min_horizon} from HORIZONS so min(HORIZONS) > HORIZON_FIXED_D."
  exit 1
fi

# Check 3: BATCH_SIZE <= N_EPISODES (EvalConfig hard constraint)
if (( BATCH_SIZE > N_EPISODES )); then
  echo "[ERROR] BATCH_SIZE=${BATCH_SIZE} > N_EPISODES=${N_EPISODES}"
  echo "        EvalConfig.__post_init__ will raise ValueError."
  exit 1
fi

echo "══ Configuration ═══════════════════════════════════════════"
echo "  DELAYS           = (${DELAYS[*]})   → async_delay"
echo "  HORIZONS         = (${HORIZONS[*]})   → execution_horizon"
echo "  DELAY_FIXED_S    = ${DELAY_FIXED_S}   (s fixed during delay sweep)"
echo "  HORIZON_FIXED_D  = ${HORIZON_FIXED_D}   (d fixed during horizon sweep)"
echo "  METHODS          = (${_METHODS[*]})"
echo "  N_EPISODES/BATCH = ${N_EPISODES} / ${BATCH_SIZE}"
echo "  SUITES           = ${SUITES}"
echo "  SEED             = ${SEED}"
echo "  FORCE            = ${FORCE}   (1=re-run existing, 0=skip existing)"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Run helper ────────────────────────────────────────────────────────────────
# run_eval <suite> <method_type> <enable_sm> <async_delay_d> <execution_horizon_s> <tag>
#
# What happens inside eval_libero_rtc.py (rollout_chunked):
#   RTC:      effective_horizon = max(d, s)
#             → mixes old[:d] + new[d:s] actions per cycle
#   Baseline: effective_horizon = s  (if 0 < s < chunk_size=50)
#             → executes s actions, replans, repeats
#             (set s=0 or s≥50 for original full-chunk behavior)
run_eval() {
  local suite="$1"  method="$2"  sm="$3"  d="$4"  s="$5"  tag="$6"

  # ── Per-combination (d, s) constraint filter ──────────────────────────────
  # RTC requires d < s:  d == s → new_chunk[d:s] = EMPTY (0 new actions);
  #                      d >  s → effective_horizon = d, still 0 new actions.
  # Baseline: d > s means the obs is older than the entire execution horizon,
  #           producing a (still runnable but) scientifically meaningless point.
  # In both cases we skip rather than waste GPU time on degenerate data.
  case "$method" in
    rtc|rtc_sm)
      if (( d >= s )); then
        echo "[SKIP] ${tag}  d=${d} s=${s}: d>=s violates RTC d<s — new_chunk[d:s]=EMPTY, skip"
        return 0
      fi
      ;;
    baseline|baseline_sm)
      if (( d > s )); then
        echo "[SKIP] ${tag}  d=${d} s=${s}: d>s — obs older than execution horizon, skip"
        return 0
      fi
      ;;
  esac

  # Optional 7th arg: explicit GPU id (overrides round-robin).
  # When omitted, assign_gpu() is called to get the next GPU in rotation.
  local gpu_id
  if [[ -n "${7:-}" ]]; then
    gpu_id="$7"
  else
    assign_gpu          # sets _NEXT_GPU in parent shell (no subshell)
    gpu_id="$_NEXT_GPU"
  fi
  local sm_tag=""; [[ "$sm" == "true" ]] && sm_tag="_sm"
  local out_dir="${OUT_ROOT}/${POLICY_TYPE}/${suite}/d${d}_s${s}_${method}${sm_tag}"

  if [[ -f "${out_dir}/eval_results.json" && "$FORCE" -eq 0 ]]; then
    echo "[SKIP] ${tag}  d=${d} s=${s}  (exists; use --force to re-run)"
    return 0
  fi

  local cmd=(
    env CUDA_VISIBLE_DEVICES="$gpu_id"
    uv run python -m lerobot.async_libero_inference.it_rtc.eval_libero_rtc
      --policy.path="${CKPT}"
      --env.type=libero
      --env.task="${suite}"
      --env.obs_type=pixels_agent_pos
      --eval.method_type="${method}"
      --eval.async_delay="${d}"
      --eval.execution_horizon="${s}"
      --eval.n_episodes="${N_EPISODES}"
      --eval.batch_size="${BATCH_SIZE}"
      --eval.enable_sm="${sm}"
      --eval.max_guidance_weight="${MAX_GUIDANCE_WEIGHT}"
      --eval.prefix_attention_schedule="${PREFIX_ATTENTION_SCHEDULE}"
      --eval.sm_rewind_buffer_steps="${SM_REWIND_BUFFER}"
      --eval.sm_rewind_warmup_steps="${SM_REWIND_WARMUP}"
      --eval.sm_gripper_confirm_steps="${SM_CONFIRM_STEPS}"
      --seed="${SEED}"
      --output_dir="${out_dir}"
  )

  echo "[RUN] GPU=${gpu_id}  ${tag}  d=${d}  s=${s}  → ${out_dir}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "      CMD: ${cmd[*]}"
    return 0
  fi

  mkdir -p "${out_dir}"
  "${cmd[@]}" &>> "${out_dir}.log" &
  local pid=$!
  PIDS+=("$pid")
  _ALL_PIDS+=("$pid")   # tracked for Ctrl+C cleanup
}

wait_all() {
  local failed=0
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    for pid in "${PIDS[@]}"; do
      wait "$pid" || { echo "[WARN] job ${pid} exited non-zero"; failed=$((failed+1)); }
    done
  fi
  PIDS=()
  [[ $failed -gt 0 ]] && echo "[WARN] ${failed} jobs failed" || true
}

# ── Sweep 1: SR vs. Inference Delay ──────────────────────────────────────────
# Sweeps: async_delay d ∈ DELAYS
# Fixed:  execution_horizon s = DELAY_FIXED_S
#
# RTC:      effective_horizon = max(d, s) = s   (since all d < s)
# Baseline: effective_horizon = s               (since 0 < s < chunk_size=50)
#
# Effect on baseline vs. original:
#   Original bt-libero baseline: s=50 (full chunk, ignores execution_horizon)
#   This script with s=15:       replan every 15 steps (more adaptive than original)
#   → Set DELAY_FIXED_S=0 or =50 to match original full-chunk baseline.
if [[ "$RUN_DELAY" -eq 1 ]]; then
  echo "── Sweep 1: SR vs. Delay ────────────────────────────────"
  echo "   d ∈ (${DELAYS[*]}),  s = ${DELAY_FIXED_S} (fixed)"
  echo ""

  for suite in $SUITES; do
    for d in "${DELAYS[@]}"; do
      run_methods "$suite" "$d" "${DELAY_FIXED_S}"
    done
    echo "[WAIT] delay-sweep suite=${suite}"
    wait_all
  done
fi

# ── Sweep 2: SR vs. Execution Horizon ────────────────────────────────────────
# Sweeps: execution_horizon s ∈ HORIZONS
# Fixed:  async_delay d = HORIZON_FIXED_D
#
# RTC:      effective_horizon = max(d, s) = s  (since d=1 ≤ all s in HORIZONS)
# Baseline: effective_horizon = s              (since 0 < s < chunk_size=50)
#
# Both methods see obs from d=1 step ago; only replanning frequency varies.
if [[ "$RUN_HORIZON" -eq 1 ]]; then
  echo ""
  echo "── Sweep 2: SR vs. Horizon ──────────────────────────────"
  echo "   s ∈ (${HORIZONS[*]}),  d = ${HORIZON_FIXED_D} (fixed)"
  echo ""

  for suite in $SUITES; do
    for s in "${HORIZONS[@]}"; do
      run_methods "$suite" "${HORIZON_FIXED_D}" "$s"
    done
    echo "[WAIT] horizon-sweep suite=${suite}"
    wait_all
  done
fi

wait_all
echo ""
echo "All sweeps complete."
echo ""


# # ── Plot results ──────────────────────────────────────────────────────────────
# PLOT_DIR="${OUT_ROOT}/plots"
# mkdir -p "${PLOT_DIR}"

# echo "Generating plots → ${PLOT_DIR}"
# uv run python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \
#   "${OUT_ROOT}" \
#   --all-plots \
#   --delay-sweep-horizon "${DELAY_FIXED_S}" \
#   --horizon-sweep-delay "${HORIZON_FIXED_D}" \
#   --ci \
#   --output-dir "${PLOT_DIR}"

# echo "Done. Plots saved to: ${PLOT_DIR}"
