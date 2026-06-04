#!/usr/bin/env bash
# eval-scripts/async_libero_smolvla_eval-rtc.sh
#
# Sweep: env_task × inference_latency × rtc_execution_horizon
# Mode : asynchronous inference with RTC (Real-Time Control conditioning)
#
# ── Key design: H couples chunk_size_threshold and rtc_execution_horizon ──────
#
#   rtc_execution_horizon = H  : model is conditioned on "execute H steps"
#   chunk_size_threshold  = H/K: re-inference triggers when queue ≤ H steps remain
#
#   These are the SAME value expressed differently.  Setting both consistently
#   ensures the model's conditioning matches when re-inference actually fires:
#     - model assumes: "I have H steps to act"
#     - client fires re-inference at: remaining == H steps
#
#   Single sweep axis H replaces separate T and rtc_execution_horizon lists.
#   chunk_size_threshold (ratio) = H / ACTIONS_PER_CHUNK  (derived, not swept)
#
# ── Per-combo independent design ──────────────────────────────────────────────
#   Server restarts for every (suite, d, H) combo.
#   All data live in the same eval_dir — no shared server directory, no symlinks.
#
#   eval_dir = outputs/eval_thesis/libero/<suite>/async_rtc/smolvla/latency_s<d>/H<H>/
#     server_timing/    ← server timing records (this combo only)
#     server_<ts>.log
#     client_timing/
#     results/
#     queue.png
#     client_<ts>.log
#
# ── Skip condition ────────────────────────────────────────────────────────────
#   H ≤ d: queue exhausted before inference returns → robot stalls.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/async_libero_smolvla_eval-rtc.sh
#   SKIP_EXISTING=true bash eval-scripts/async_libero_smolvla_eval-rtc.sh
#   SAVE_VIDEO=true    bash eval-scripts/async_libero_smolvla_eval-rtc.sh

set -uo pipefail   # -e omitted: client failure must not abort the entire sweep

# ══ GPU / environment ═════════════════════════════════════════════════════════
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export USER=huoyuan

# ══ Model ═════════════════════════════════════════════════════════════════════
pretrained_name_or_path=HollyTan/libero_smolvla_500MSmolVLM2_multitask
model_type=smolvla
benchmark_robot_type=libero

# ══ Method tag ════════════════════════════════════════════════════════════════
METHOD=async_rtc

# ══ Server settings ═══════════════════════════════════════════════════════════
SERVER_PORT=8090
SERVER_HOST=localhost
SERVER_STARTUP_TIMEOUT=180
SERVER_SHUTDOWN_TIMEOUT=30

# ══ Eval settings ═════════════════════════════════════════════════════════════
FPS=30
EPISODES_PER_TASK=10
ACTIONS_PER_CHUNK=50     # fixed: H/K gives chunk_size_threshold ratio

# ══ Sweep axes ════════════════════════════════════════════════════════════════
# SUITES=(libero_object libero_spatial libero_goal libero_10)
INFERENCE_LATENCIES_STEPS=(0 8 12 16 20 30 40)

SUITES=(libero_object libero_spatial)
# INFERENCE_LATENCIES_STEPS=(0 8)

# RTC execution horizon H (steps).
# H simultaneously controls:
#   --rtc_execution_horizon = H          (model conditioning)
#   --chunk_size_threshold  = H / K      (re-inference trigger ratio)
# Skip combos where H ≤ d (robot stalls before inference returns).
# H=45 covers d=40 with a 5-step safety margin.
RTC_HORIZON_LIST=(10 20 30 40 45)
# RTC_HORIZON_LIST=(10 20)

# ══ Optional flags ════════════════════════════════════════════════════════════
SKIP_EXISTING=${SKIP_EXISTING:-false}
SAVE_VIDEO=${SAVE_VIDEO:-false}

# ══ Port helpers ══════════════════════════════════════════════════════════════
wait_for_port() {
    local port=$1 timeout=${2:-$SERVER_STARTUP_TIMEOUT}
    echo "[port] Waiting for :${port} to open (max ${timeout}s)..."
    for i in $(seq 1 "$timeout"); do
        if nc -z 127.0.0.1 "$port" 2>/dev/null; then
            echo "[port] :${port} open after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[port] ERROR: :${port} not open after ${timeout}s" >&2
    return 1
}

wait_port_free() {
    local port=$1 timeout=${2:-$SERVER_SHUTDOWN_TIMEOUT}
    echo "[port] Waiting for :${port} to be free (max ${timeout}s)..."
    for i in $(seq 1 "$timeout"); do
        if ! lsof -ti:"$port" >/dev/null 2>&1; then
            echo "[port] :${port} free after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[port] WARNING: :${port} still in use after ${timeout}s — force-killing (USER=${USER})" >&2
    while IFS= read -r stale_pid; do
        local stale_owner
        stale_owner=$(ps -o user= -p "$stale_pid" 2>/dev/null | tr -d ' ')
        if [ "$stale_owner" = "$USER" ]; then
            echo "[port] Force-killing PID=${stale_pid} (owner=${stale_owner})"
            kill -9 "$stale_pid" 2>/dev/null || true
        else
            echo "[port] Skipping PID=${stale_pid} (owner=${stale_owner} ≠ ${USER})"
        fi
    done < <(lsof -ti:"$port" 2>/dev/null)
    sleep 2
}

stop_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        local owner
        owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$owner" != "$USER" ]; then
            echo "[server] WARNING: PID=${pid} owned by '${owner}', not '${USER}' — refusing to kill" >&2
        else
            # Kill by PID only (not PGID) to avoid hitting other server instances
            # on different ports that share the same session process group.
            echo "[server] Stopping PID=${pid} (owner=${owner})..."
            kill "$pid" 2>/dev/null || true
            for i in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    fi
    wait_port_free "$SERVER_PORT"
}

# ══ Summary counters ══════════════════════════════════════════════════════════
total_combos=$(( ${#SUITES[@]} * ${#INFERENCE_LATENCIES_STEPS[@]} * ${#RTC_HORIZON_LIST[@]} ))
done_combos=0
skipped_combos=0
failed_combos=0

echo "════════════════════════════════════════════════════════════════════"
echo "  eval-scripts/async_libero_smolvla_eval-rtc.sh"
echo "  method              : ${METHOD}"
echo "  model               : ${pretrained_name_or_path}"
echo "  fps                 : ${FPS}"
echo "  actions_per_chunk   : ${ACTIONS_PER_CHUNK}  (fixed)"
echo "  latency steps (d)   : ${INFERENCE_LATENCIES_STEPS[*]}"
echo "  latency (s)         : $(for d in "${INFERENCE_LATENCIES_STEPS[@]}"; do awk "BEGIN{printf \"%.4f \",${d}/${FPS}}"; done)"
echo "  rtc_horizon H       : ${RTC_HORIZON_LIST[*]}  (= chunk_size_threshold steps)"
echo "  threshold ratio H/K : $(for h in "${RTC_HORIZON_LIST[@]}"; do awk "BEGIN{printf \"%.2f \",${h}/${ACTIONS_PER_CHUNK}}"; done)"
echo "  suites              : ${SUITES[*]}"
echo "  total combos        : ${total_combos}  (before H≤d skips)"
echo "  skip_existing       : ${SKIP_EXISTING}"
echo "  save_video          : ${SAVE_VIDEO}"
echo "  design              : per-combo independent (server restarts each combo)"
echo "════════════════════════════════════════════════════════════════════"

# ══ Main sweep ════════════════════════════════════════════════════════════════
# Loop: d → H → suite
# Server starts and stops once per (d, H, suite).
# All outputs go into eval_dir — self-contained, no symlinks.

for delay_steps in "${INFERENCE_LATENCIES_STEPS[@]}"; do
    latency=$(awk "BEGIN { printf \"%.6f\", ${delay_steps} / ${FPS} }")
    latency_tag="s${delay_steps}"

    for rtc_horizon in "${RTC_HORIZON_LIST[@]}"; do

        # Skip degenerate combos: H ≤ d → queue exhausted before inference returns.
        if (( rtc_horizon <= delay_steps )); then
            echo "[skip] H=${rtc_horizon} ≤ d=${delay_steps} steps — degenerate combo (robot stalls), skipping"
            (( skipped_combos += ${#SUITES[@]} )) || true
            continue
        fi

        # Derive chunk_size_threshold ratio from H and K.
        # chunk_size_threshold (ratio) = H / K ensures re-inference fires
        # exactly when H steps remain — matching the model's RTC conditioning.
        threshold_ratio=$(awk "BEGIN { printf \"%.6f\", ${rtc_horizon} / ${ACTIONS_PER_CHUNK} }")

        for env_task in "${SUITES[@]}"; do

            log_root="./outputs/eval_thesis/${benchmark_robot_type}/${env_task}/${METHOD}/${model_type}/latency_${latency_tag}/H${rtc_horizon}"
            combo_label="suite=${env_task}  d=${delay_steps}steps(${latency}s)  H=${rtc_horizon}(ratio=${threshold_ratio})  K=${ACTIONS_PER_CHUNK}"
            combo_num=$(( done_combos + skipped_combos + failed_combos + 1 ))

            if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${log_root}/results/aggregate.json" ]; then
                echo "[skip] (${combo_num}/${total_combos})  ${combo_label}"
                (( skipped_combos++ )) || true
                continue
            fi

            mkdir -p "${log_root}/server_timing" "${log_root}/results" "${log_root}/client_timing"
            [ "${SAVE_VIDEO}" = "true" ] && mkdir -p "${log_root}/videos"

            echo ""
            echo "── [${combo_num}/${total_combos}]  ${combo_label} ──"

            # ── Start server ──────────────────────────────────────────────
            stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
                --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_inference_server.yaml" \
                --host="${SERVER_HOST}" \
                --port="${SERVER_PORT}" \
                --timing_output_dir="${log_root}/server_timing" \
                --fps="${FPS}" \
                --inference_latency="${latency}" \
                --log_level=WARNING \
                > >(tee "${log_root}/server_$(date +%Y%m%d_%H%M%S).log") 2>&1 &
            SERVER_PID=$!
            echo "[server] Launched PID=${SERVER_PID}"

            if ! wait_for_port "$SERVER_PORT" "$SERVER_STARTUP_TIMEOUT"; then
                echo "[server] Failed to start — skipping combo" >&2
                stop_server "$SERVER_PID"
                (( failed_combos++ )) || true
                continue
            fi

            # ── Run client ────────────────────────────────────────────────
            stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_test \
                --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_smolvla_client.yaml" \
                --env_task="${env_task}" \
                --policy_type="${model_type}" \
                --pretrained_name_or_path="${pretrained_name_or_path}" \
                --server_address="${SERVER_HOST}:${SERVER_PORT}" \
                --results_dir="${log_root}/results" \
                --timing_output_dir="${log_root}/client_timing" \
                --save_video="${SAVE_VIDEO}" \
                --video_camera=image \
                --video_dir="${log_root}/videos" \
                --queue_size_monitor_path="${log_root}/queue.png" \
                --fps="${FPS}" \
                --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
                --chunk_size_threshold="${threshold_ratio}" \
                --rtc_execution_horizon="${rtc_horizon}" \
                --aggregate_fn_name=latest_only \
                --episodes_per_task="${EPISODES_PER_TASK}" \
                --log_level=WARNING \
                2>&1 | tee "${log_root}/client_$(date +%Y%m%d_%H%M%S).log"
            client_exit=${PIPESTATUS[0]}

            # ── Stop server ───────────────────────────────────────────────
            stop_server "$SERVER_PID"

            if [ "${client_exit}" -eq 0 ]; then
                (( done_combos++ )) || true
                echo "[ok]   ${combo_label}"
            else
                (( failed_combos++ )) || true
                echo "[FAIL] ${combo_label}  exit=${client_exit}" >&2
            fi

        done  # env_task
    done  # rtc_horizon
done  # delay_steps

# ══ Final summary ═════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Sweep complete."
echo "  done    : ${done_combos}"
echo "  skipped : ${skipped_combos}"
echo "  failed  : ${failed_combos}"
echo "  total   : ${total_combos}"
echo ""
echo "  Next steps:"
echo "  1. Run per-combo timing + RTC analysis:"
echo "     bash analyze_tools_script/analyze_all_eval_outputs.sh \\"
echo "         outputs/eval_thesis/${benchmark_robot_type}"
echo ""
echo "  2. Run cross-combo sweep analysis:"
echo "     uv run python -m lerobot.async_inference.analyze_sweep \\"
echo "         outputs/eval_thesis/${benchmark_robot_type} \\"
echo "         --method ${METHOD} --model ${model_type}"
echo "════════════════════════════════════════════════════════════════════"
