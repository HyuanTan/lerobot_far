#!/usr/bin/env bash
# =============================================================================
# async_libero_smolvla_eval.sh  —  async LIBERO SmolVLA method comparison
#
# Single-point evaluation (NO sweep). Each method runs once per suite with
# FIXED parameters. Compares the 6 async method variants head-to-head.
#
# ── Delay model ───────────────────────────────────────────────────────────────
#   server --inference_latency=0  → NO artificial latency injection.
#   The client measures the REAL inference delay (server compute + gRPC) on the
#   fly and uploads it per-observation as `inference_delay` for RTC guidance.
#   So inference_delay reflects the genuine ~10-step model latency, not a
#   synthetic value — and RTC_HORIZON=15 > that, satisfying RTC's s > d rule.
#
# ── Fixed parameters ──────────────────────────────────────────────────────────
#   actions_per_chunk      = 50
#   chunk_size_threshold   = 0.5   (re-infer when queue drops to 50% = 25 steps)
#   rtc_execution_horizon  = 15    (RTC methods only; non-RTC = 0)
#
# ── Supported methods ─────────────────────────────────────────────────────────
#   nortc          async, no RTC, no SM   (port 8085)
#   rtc            async, RTC, no SM      (port 8083)
#   nortc_sm       async, no RTC, SM      (port 8086)
#   rtc_sm         async, RTC, SM         (port 8084)
#   nortc_multicand async, no RTC, MC     (port 8087)
#   rtc_multicand  async, RTC, MC         (port 8098)
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/libero_smolvla/async_libero_smolvla_eval.sh   # skips done methods
#   METHODS="rtc rtc_sm" bash ...
#   bash ... --dry-run
#   bash ... --force                  # re-run even if results already exist
#
# ── Environment variables ─────────────────────────────────────────────────────
#   CKPT          checkpoint path/HF id       (default: HollyTan/...)
#   GPU           CUDA_VISIBLE_DEVICES         (default: 3)
#   SUITES        space-separated suite list   (default: libero_object)
#   METHODS       space-separated method list  (default: all 6)
#   EPISODES_PER_TASK                          (default: 10)
#   OUT_ROOT      output root dir             (default: outputs/eval_thesis/libero)
#   SKIP_EXISTING true/false                   (default: true; a done method is skipped)
#   SAVE_VIDEO    true/false                   (default: false)
# =============================================================================
set -uo pipefail

# ── Working directory ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU:-3}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export USER="${USER:-$(id -un)}"

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT="${CKPT:-HollyTan/libero_smolvla_500MSmolVLM2_multitask}"
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
FPS="${FPS:-30}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_thesis/libero}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"   # default: skip methods already done; --force to re-run
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SUITES="${SUITES:-libero_object}"

# ── Fixed parameters (no sweep) ───────────────────────────────────────────────
INFERENCE_LATENCY=0                   # server injects NO artificial delay
ACTIONS_PER_CHUNK=50                  # K
CHUNK_SIZE_THRESHOLD=0.5              # re-infer when queue ≤ 50% (25/50 steps)
RTC_HORIZON=15                        # rtc_execution_horizon for RTC methods

# ── Server settings ───────────────────────────────────────────────────────────
SERVER_HOST="localhost"
SERVER_STARTUP_TIMEOUT=180
SERVER_SHUTDOWN_TIMEOUT=30

# ── Method selection ──────────────────────────────────────────────────────────
# Valid: nortc  rtc  nortc_sm  rtc_sm  nortc_multicand  rtc_multicand
METHODS="${METHODS:-}"   # empty = all six

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --method=*)      METHODS="${arg#--method=}" ;;
    --no-mc)         METHODS="${METHODS:-nortc rtc nortc_sm rtc_sm}" ;;
    --no-sm)         METHODS="${METHODS:-nortc rtc}" ;;
    --rtc-only)      METHODS="rtc rtc_sm" ;;
    --nortc-only)    METHODS="nortc nortc_sm" ;;
    --skip-existing) SKIP_EXISTING=true ;;
    --force)         SKIP_EXISTING=false ;;   # re-run even if results exist
    --save-video)    SAVE_VIDEO=true ;;
  esac
done

# Resolve METHODS array
if [[ -z "$METHODS" ]]; then
  _METHODS=(nortc rtc nortc_sm rtc_sm nortc_multicand rtc_multicand)
else
  read -ra _METHODS <<< "$METHODS"
fi

# Validate
for _m in "${_METHODS[@]}"; do
  case "$_m" in
    nortc|rtc|nortc_sm|rtc_sm|nortc_multicand|rtc_multicand) ;;
    *) echo "[ERROR] Unknown method '${_m}'."; exit 1 ;;
  esac
done

# ── Port / config / module mapping ───────────────────────────────────────────
method_port()         { case "$1" in nortc) echo 8085;; rtc) echo 8083;; nortc_sm) echo 8086;; rtc_sm) echo 8084;; nortc_multicand) echo 8087;; rtc_multicand) echo 8098;; esac; }
method_server_cfg()   { case "$1" in nortc|rtc) echo async_server.yaml;; nortc_sm|rtc_sm) echo async_server_sm.yaml;; nortc_multicand|rtc_multicand) echo async_server_sm_multican.yaml;; esac; }
method_client_cfg()   { case "$1" in nortc|rtc) echo async_client.yaml;; nortc_sm|rtc_sm) echo async_client_sm.yaml;; nortc_multicand|rtc_multicand) echo async_client_sm_multican.yaml;; esac; }
method_client_mod()   { case "$1" in nortc|rtc) echo run_libero_test;; nortc_sm|rtc_sm) echo run_libero_smart_test;; nortc_multicand|rtc_multicand) echo run_libero_multicand_test;; esac; }
method_server_mod()   { case "$1" in nortc_multicand|rtc_multicand) echo lerobot.async_inference.multi_candidate_server;; *) echo lerobot.async_inference.policy_server;; esac; }
method_is_rtc()       { [[ "$1" == rtc* ]] && echo true || echo false; }
method_is_multicand() { [[ "$1" == *multicand ]] && echo true || echo false; }

# ── Port helpers ──────────────────────────────────────────────────────────────
wait_for_port() {
  local port=$1 timeout=${2:-$SERVER_STARTUP_TIMEOUT}
  for i in $(seq 1 "$timeout"); do
    nc -z 127.0.0.1 "$port" 2>/dev/null && return 0
    sleep 1
  done
  echo "[port] ERROR: :${port} not open after ${timeout}s" >&2
  return 1
}

wait_port_free() {
  local port=$1 timeout=${2:-$SERVER_SHUTDOWN_TIMEOUT}
  for i in $(seq 1 "$timeout"); do
    lsof -ti:"$port" >/dev/null 2>&1 || return 0
    sleep 1
  done
  while IFS= read -r stale_pid; do
    local owner; owner=$(ps -o user= -p "$stale_pid" 2>/dev/null | tr -d ' ')
    [[ "$owner" == "$USER" ]] && kill -9 "$stale_pid" 2>/dev/null || true
  done < <(lsof -ti:"$port" 2>/dev/null)
  sleep 1
}

stop_server() {
  local pid=$1 port=$2
  if kill -0 "$pid" 2>/dev/null; then
    local owner; owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ "$owner" == "$USER" ]]; then
      kill "$pid" 2>/dev/null || true
      for i in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
  wait_port_free "$port"
}

echo "══ Configuration (single-point, no sweep) ══════════════════════════"
echo "  CKPT               = ${CKPT}"
echo "  SUITES             = ${SUITES}"
echo "  METHODS            = (${_METHODS[*]})"
echo "  FPS                = ${FPS}"
echo "  inference_latency  = ${INFERENCE_LATENCY}  (server injects no delay; client uploads real delay)"
echo "  actions_per_chunk  = ${ACTIONS_PER_CHUNK}"
echo "  chunk_size_thresh  = ${CHUNK_SIZE_THRESHOLD}"
echo "  rtc_horizon (RTC)  = ${RTC_HORIZON}  (non-RTC = 0)"
echo "  EPISODES_PER_TASK  = ${EPISODES_PER_TASK}"
echo "  OUT_ROOT           = ${OUT_ROOT}"
echo "  SKIP_EXISTING      = ${SKIP_EXISTING}   (skip methods with results; --force to re-run)"
echo "  SAVE_VIDEO         = ${SAVE_VIDEO}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ── run_combo: one (method, suite) — fixed params ─────────────────────────────
run_combo() {
  local method="$1" suite="$2"

  local rtc_eh
  if [[ "$(method_is_rtc "$method")" == "true" ]]; then
    rtc_eh=$RTC_HORIZON
  else
    rtc_eh=0
  fi

  local port server_cfg client_cfg client_mod server_mod
  port=$(method_port "$method")
  server_cfg=$(method_server_cfg "$method")
  client_cfg=$(method_client_cfg "$method")
  client_mod=$(method_client_mod "$method")
  server_mod=$(method_server_mod "$method")

  local log_root="${OUT_ROOT}/${suite}/${method}"
  local combo_label="${suite}  ${method}  rtc_eh=${rtc_eh}  thresh=${CHUNK_SIZE_THRESHOLD}  K=${ACTIONS_PER_CHUNK}"

  if [[ "${SKIP_EXISTING}" == "true" && -f "${log_root}/results/aggregate.json" ]]; then
    echo "[SKIP] ${suite}/${method} already done — ${log_root}/results/aggregate.json exists (use --force to re-run)"
    return 2   # distinct from done(0)/failed(1)
  fi

  echo ""
  echo "── [${combo_label}] ──"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    server: uv run python -m ${server_mod} --port=${port} --fps=${FPS} --inference_latency=${INFERENCE_LATENCY}"
    echo "    client: uv run python -m lerobot.async_inference.sim_test.${client_mod}"
    echo "            --rtc_execution_horizon=${rtc_eh} --chunk_size_threshold=${CHUNK_SIZE_THRESHOLD} --actions_per_chunk=${ACTIONS_PER_CHUNK}"
    echo "            → ${log_root}"
    return 0
  fi

  mkdir -p "${log_root}/server_timing" "${log_root}/results" "${log_root}/client_timing"
  [[ "${SAVE_VIDEO}" == "true" ]] && mkdir -p "${log_root}/videos"
  if [[ "$(method_is_multicand "$method")" == "true" ]]; then
    mkdir -p "${log_root}/mc_data" "${log_root}/mc_trajectories"
  fi

  # ── Start server (no artificial latency) ────────────────────────────────────
  stdbuf -oL -eL uv run python -m "${server_mod}" \
    --config_path "src/lerobot/async_inference/config/libero/${server_cfg}" \
    --host="${SERVER_HOST}" --port="${port}" \
    --timing_output_dir="${log_root}/server_timing" \
    --fps="${FPS}" --inference_latency="${INFERENCE_LATENCY}" \
    --log_level=WARNING \
    > >(tee "${log_root}/server_$(date +%Y%m%d_%H%M%S).log") 2>&1 &
  local server_pid=$!
  echo "[server] PID=${server_pid} port=${port}"

  if ! wait_for_port "$port" "$SERVER_STARTUP_TIMEOUT"; then
    echo "[FAIL] Server did not start for ${method} — skipping" >&2
    stop_server "$server_pid" "$port"
    return 1
  fi

  # ── Build client command ────────────────────────────────────────────────────
  local client_args=(
    stdbuf -oL -eL uv run python
    -m "lerobot.async_inference.sim_test.${client_mod}"
    --config_path "src/lerobot/async_inference/config/libero/${client_cfg}"
    --env_task="${suite}"
    --policy_type="${POLICY_TYPE}"
    --pretrained_name_or_path="${CKPT}"
    --server_address="${SERVER_HOST}:${port}"
    --results_dir="${log_root}/results"
    --timing_output_dir="${log_root}/client_timing"
    --save_video="${SAVE_VIDEO}"
    --video_camera=image
    --video_dir="${log_root}/videos"
    --queue_size_monitor_path="${log_root}/queue.png"
    --fps="${FPS}"
    --actions_per_chunk="${ACTIONS_PER_CHUNK}"
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}"
    --rtc_execution_horizon="${rtc_eh}"
    --aggregate_fn_name=latest_only
    --episodes_per_task="${EPISODES_PER_TASK}"
    --log_level=WARNING
  )
  # Multicand extras
  if [[ "$(method_is_multicand "$method")" == "true" ]]; then
    client_args+=(
      --data_collect_dir="${log_root}/mc_data"
      --trajectory_dir="${log_root}/mc_trajectories"
      --record_trajectory=false
    )
  fi

  # ── Run client ──────────────────────────────────────────────────────────────
  "${client_args[@]}" 2>&1 | tee "${log_root}/client_$(date +%Y%m%d_%H%M%S).log"
  local client_exit=${PIPESTATUS[0]}

  # ── Stop server ─────────────────────────────────────────────────────────────
  stop_server "$server_pid" "$port"

  if (( client_exit == 0 )); then
    echo "[OK]   ${combo_label}"
  else
    echo "[FAIL] ${combo_label}  exit=${client_exit}" >&2
    return 1
  fi
}

# ── Main: each method once per suite ──────────────────────────────────────────
done_combos=0
skipped_combos=0
failed_combos=0
for suite in $SUITES; do
  echo "── suite=${suite} ──────────────────────────────────────────────────"
  for m in "${_METHODS[@]}"; do
    run_combo "$m" "$suite"; rc=$?
    case "$rc" in
      0) done_combos=$((done_combos+1)) ;;
      2) skipped_combos=$((skipped_combos+1)) ;;
      *) failed_combos=$((failed_combos+1)) ;;
    esac
  done
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Complete."
echo "  done    : ${done_combos}"
echo "  skipped : ${skipped_combos}  (already had results; --force to re-run)"
echo "  failed  : ${failed_combos}"
echo "════════════════════════════════════════════════════════════════════"
