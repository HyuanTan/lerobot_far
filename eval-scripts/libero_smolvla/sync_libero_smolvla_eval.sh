#!/usr/bin/env bash
# =============================================================================
# sync_libero_smolvla_eval.sh  —  SYNC LIBERO SmolVLA method comparison
#
# Single-point evaluation (NO sweep). Each method runs once per suite with
# FIXED parameters. Compares sync_nortc vs sync_nortc_sm head-to-head.
#
# ── Delay model ───────────────────────────────────────────────────────────────
#   server --inference_latency=0  → NO artificial latency injection.
#   The client measures the REAL inference delay and uploads it per-observation.
#   (In sync mode RTC is off, but the upload keeps the timing pipeline consistent.)
#
# ── Synchronous semantics ─────────────────────────────────────────────────────
#   chunk_size_threshold = 0  → client waits for the full inference response
#   before continuing; rtc_execution_horizon = 0 → RTC disabled.
#
# ── Fixed parameters ──────────────────────────────────────────────────────────
#   actions_per_chunk     = 25
#   chunk_size_threshold  = 0   (synchronous)
#   rtc_execution_horizon = 0   (no RTC)
#
# ── Supported methods ─────────────────────────────────────────────────────────
#   nortc       synchronous, no RTC, no SM   (port 8081)
#   nortc_sm    synchronous, no RTC, SM      (port 8082)
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/libero_smolvla/sync_libero_smolvla_eval.sh
#   METHODS="nortc_sm" bash ...
#   bash ... --dry-run
#   bash ... --force                  # re-run even if results already exist
#
# ── Environment variables ─────────────────────────────────────────────────────
#   CKPT          checkpoint path/HF id       (default: HollyTan/...)
#   GPU           CUDA_VISIBLE_DEVICES         (default: 3)
#   SUITES        space-separated suite list   (default: libero_object)
#   METHODS       space-separated method list  (default: nortc nortc_sm)
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
ACTIONS_PER_CHUNK=25                  # K
CHUNK_SIZE_THRESHOLD=0                # synchronous (block for full response)
RTC_HORIZON=0                         # RTC disabled in sync mode

# ── Server settings ───────────────────────────────────────────────────────────
SERVER_HOST="localhost"
SERVER_STARTUP_TIMEOUT=180
SERVER_SHUTDOWN_TIMEOUT=30

# ── Method selection ──────────────────────────────────────────────────────────
# Valid: nortc  nortc_sm
METHODS="${METHODS:-}"   # empty = both

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --method=*)      METHODS="${arg#--method=}" ;;
    --no-sm)         METHODS="nortc" ;;
    --sm-only)       METHODS="nortc_sm" ;;
    --skip-existing) SKIP_EXISTING=true ;;
    --force)         SKIP_EXISTING=false ;;   # re-run even if results exist
    --save-video)    SAVE_VIDEO=true ;;
  esac
done

# Resolve METHODS array
if [[ -z "$METHODS" ]]; then
  _METHODS=(nortc nortc_sm)
else
  read -ra _METHODS <<< "$METHODS"
fi

# Validate
for _m in "${_METHODS[@]}"; do
  case "$_m" in
    nortc|nortc_sm) ;;
    *) echo "[ERROR] Unknown method '${_m}'. Valid: nortc nortc_sm"; exit 1 ;;
  esac
done

# ── Port / config / module mapping ───────────────────────────────────────────
# Sync uses async_server.yaml on the server side; the SM variant only changes
# the CLIENT (sync_client vs async_client_sm + smart-test module).
method_port()       { case "$1" in nortc) echo 8081;; nortc_sm) echo 8082;; esac; }
method_client_cfg() { case "$1" in nortc) echo sync_client.yaml;; nortc_sm) echo async_client_sm.yaml;; esac; }
method_client_mod() { case "$1" in nortc) echo run_libero_test;; nortc_sm) echo run_libero_smart_test;; esac; }

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
echo "  chunk_size_thresh  = ${CHUNK_SIZE_THRESHOLD}  (0 = synchronous)"
echo "  EPISODES_PER_TASK  = ${EPISODES_PER_TASK}"
echo "  OUT_ROOT           = ${OUT_ROOT}"
echo "  SKIP_EXISTING      = ${SKIP_EXISTING}   (skip methods with results; --force to re-run)"
echo "  SAVE_VIDEO         = ${SAVE_VIDEO}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ── run_combo: one (method, suite) — fixed params ─────────────────────────────
run_combo() {
  local method="$1" suite="$2"

  local port client_cfg client_mod
  port=$(method_port "$method")
  client_cfg=$(method_client_cfg "$method")
  client_mod=$(method_client_mod "$method")

  local method_tag="sync_${method}"
  local log_root="${OUT_ROOT}/${suite}/${method_tag}"
  local combo_label="${suite}  ${method_tag}  K=${ACTIONS_PER_CHUNK}  thresh=${CHUNK_SIZE_THRESHOLD}"

  if [[ "${SKIP_EXISTING}" == "true" && -f "${log_root}/results/aggregate.json" ]]; then
    echo "[SKIP] ${suite}/${method_tag} already done — ${log_root}/results/aggregate.json exists (use --force to re-run)"
    return 2   # distinct from done(0)/failed(1)
  fi

  echo ""
  echo "── [${combo_label}] ──"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    server: uv run python -m lerobot.async_inference.policy_server --port=${port} --inference_latency=${INFERENCE_LATENCY}"
    echo "    client: uv run python -m lerobot.async_inference.sim_test.${client_mod}"
    echo "            --actions_per_chunk=${ACTIONS_PER_CHUNK} --chunk_size_threshold=${CHUNK_SIZE_THRESHOLD} --rtc_execution_horizon=${RTC_HORIZON}"
    echo "            → ${log_root}"
    return 0
  fi

  mkdir -p "${log_root}/server_timing" "${log_root}/results" "${log_root}/client_timing"
  [[ "${SAVE_VIDEO}" == "true" ]] && mkdir -p "${log_root}/videos"

  # ── Start server (always async_server.yaml; sync behaviour is client-side) ──
  stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
    --config_path "src/lerobot/async_inference/config/libero/async_server.yaml" \
    --host="${SERVER_HOST}" --port="${port}" \
    --timing_output_dir="${log_root}/server_timing" \
    --fps="${FPS}" --inference_latency="${INFERENCE_LATENCY}" \
    --log_level=WARNING \
    > >(tee "${log_root}/server_$(date +%Y%m%d_%H%M%S).log") 2>&1 &
  local server_pid=$!
  echo "[server] PID=${server_pid} port=${port}"

  if ! wait_for_port "$port" "$SERVER_STARTUP_TIMEOUT"; then
    echo "[FAIL] Server did not start for ${method_tag} — skipping" >&2
    stop_server "$server_pid" "$port"
    return 1
  fi

  # ── Run client (synchronous: chunk_size_threshold=0) ────────────────────────
  stdbuf -oL -eL uv run python \
    -m "lerobot.async_inference.sim_test.${client_mod}" \
    --config_path "src/lerobot/async_inference/config/libero/${client_cfg}" \
    --env_task="${suite}" \
    --policy_type="${POLICY_TYPE}" \
    --pretrained_name_or_path="${CKPT}" \
    --server_address="${SERVER_HOST}:${port}" \
    --results_dir="${log_root}/results" \
    --timing_output_dir="${log_root}/client_timing" \
    --save_video="${SAVE_VIDEO}" \
    --video_camera=image \
    --video_dir="${log_root}/videos" \
    --queue_size_monitor_path="${log_root}/queue.png" \
    --fps="${FPS}" \
    --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
    --rtc_execution_horizon="${RTC_HORIZON}" \
    --aggregate_fn_name=latest_only \
    --episodes_per_task="${EPISODES_PER_TASK}" \
    --log_level=WARNING \
    2>&1 | tee "${log_root}/client_$(date +%Y%m%d_%H%M%S).log"
  local client_exit=${PIPESTATUS[0]}

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
