#!/usr/bin/env bash
# eval-scripts/async_libero_smolvla_eval-nortc.sh
#
# Sweep: env_task × inference_latency × chunk_size_threshold
# Mode : asynchronous inference (chunk_size_threshold > 0), no RTC
#
# ── Per-combo independent design ──────────────────────────────────────────────
#   Server restarts for every (suite, d, T) combo.
#   All data live in the same eval_dir — no shared server directory, no symlinks.
#
#   eval_dir = outputs/eval_thesis/libero/<suite>/async_nortc/smolvla/latency_s<d>/T<T>/
#     server_timing/    ← server timing records (this combo only)
#     server_<ts>.log
#     client_timing/
#     results/
#     queue.png
#     client_<ts>.log
#
# ── Parameter relationships ────────────────────────────────────────────────────
#   ACTIONS_PER_CHUNK = K = 50  (fixed)
#   chunk_size_threshold (ratio) = T / K  (derived; T is the sweep axis in steps)
#   Re-inference triggers when queue ≤ T steps remain.
#
# ── Skip condition ────────────────────────────────────────────────────────────
#   T ≤ d: queue exhausted before inference returns → robot stalls.
#
# ── Comparison with sync_nortc ────────────────────────────────────────────────
#   Both use the same d axis and K=50, so solve_rate_vs_delay curves are
#   directly comparable at fixed T (async) vs K=50 (sync).
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/async_libero_smolvla_eval-nortc.sh
#   SKIP_EXISTING=true bash eval-scripts/async_libero_smolvla_eval-nortc.sh
#   SAVE_VIDEO=true    bash eval-scripts/async_libero_smolvla_eval-nortc.sh

set -uo pipefail

# ══ GPU / environment ═════════════════════════════════════════════════════════
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export USER=huoyuan

# ══ Model ═════════════════════════════════════════════════════════════════════
pretrained_name_or_path=HollyTan/libero_smolvla_500MSmolVLM2_multitask
model_type=smolvla
benchmark_robot_type=libero

# ══ Method tag ════════════════════════════════════════════════════════════════
METHOD=async_nortc_sm_multicand

# ══ Server settings ═══════════════════════════════════════════════════════════
SERVER_PORT=8080
SERVER_HOST=localhost
SERVER_STARTUP_TIMEOUT=180
SERVER_SHUTDOWN_TIMEOUT=30

# ══ Eval settings ═════════════════════════════════════════════════════════════
FPS=30
EPISODES_PER_TASK=10
ACTIONS_PER_CHUNK=50     # fixed: T/K gives chunk_size_threshold ratio

# ══ Sweep axes ════════════════════════════════════════════════════════════════
# SUITES=(libero_object libero_spatial libero_goal libero_10)
INFERENCE_LATENCIES_STEPS=(0 8 12 16 20 30 40)

SUITES=(libero_object libero_spatial)
# INFERENCE_LATENCIES_STEPS=(0 8)

# T = chunk_size_threshold in steps (ratio = T/K passed to --chunk_size_threshold).
# Skip T ≤ d. T=40 covers d=30 (margin=10); add T=45 if d=40 is needed.
CHUNK_SIZE_THRESHOLD_LIST=(10 20 30 40)
# CHUNK_SIZE_THRESHOLD_LIST=(10 20)

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
total_combos=$(( ${#SUITES[@]} * ${#INFERENCE_LATENCIES_STEPS[@]} * ${#CHUNK_SIZE_THRESHOLD_LIST[@]} ))
done_combos=0
skipped_combos=0
failed_combos=0

echo "════════════════════════════════════════════════════════════════════"
echo "  eval-scripts/async_libero_smolvla_eval-nortc.sh"
echo "  method              : ${METHOD}"
echo "  model               : ${pretrained_name_or_path}"
echo "  fps                 : ${FPS}"
echo "  actions_per_chunk   : ${ACTIONS_PER_CHUNK}  (fixed)"
echo "  latency steps (d)   : ${INFERENCE_LATENCIES_STEPS[*]}"
echo "  latency (s)         : $(for d in "${INFERENCE_LATENCIES_STEPS[@]}"; do awk "BEGIN{printf \"%.4f \",${d}/${FPS}}"; done)"
echo "  threshold steps (T) : ${CHUNK_SIZE_THRESHOLD_LIST[*]}"
echo "  threshold ratio T/K : $(for t in "${CHUNK_SIZE_THRESHOLD_LIST[@]}"; do awk "BEGIN{printf \"%.2f \",${t}/${ACTIONS_PER_CHUNK}}"; done)"
echo "  suites              : ${SUITES[*]}"
echo "  total combos        : ${total_combos}  (before T≤d skips)"
echo "  skip_existing       : ${SKIP_EXISTING}"
echo "  save_video          : ${SAVE_VIDEO}"
echo "  design              : per-combo independent (server restarts each combo)"
echo "════════════════════════════════════════════════════════════════════"

# ══ Main sweep ════════════════════════════════════════════════════════════════
# Loop: d → T → suite
# Server starts and stops once per (d, T, suite).
# All outputs go into eval_dir — self-contained, no symlinks.

for delay_steps in "${INFERENCE_LATENCIES_STEPS[@]}"; do
    latency=$(awk "BEGIN { printf \"%.6f\", ${delay_steps} / ${FPS} }")
    latency_tag="s${delay_steps}"

    for chunk_size_threshold in "${CHUNK_SIZE_THRESHOLD_LIST[@]}"; do

        # Skip T ≤ d: queue exhausted before inference returns → robot stalls.
        if (( chunk_size_threshold <= delay_steps )); then
            echo "[skip] T=${chunk_size_threshold} ≤ d=${delay_steps} steps — degenerate combo, skipping"
            (( skipped_combos += ${#SUITES[@]} )) || true
            continue
        fi

        # Derive ratio once per (d, T) block.
        threshold_ratio=$(awk "BEGIN { printf \"%.6f\", ${chunk_size_threshold} / ${ACTIONS_PER_CHUNK} }")

        for env_task in "${SUITES[@]}"; do

            log_root="./outputs/eval_thesis/${benchmark_robot_type}/${env_task}/${METHOD}/${model_type}/latency_${latency_tag}/T${chunk_size_threshold}"
            combo_label="suite=${env_task}  d=${delay_steps}steps(${latency}s)  T=${chunk_size_threshold}(ratio=${threshold_ratio})  K=${ACTIONS_PER_CHUNK}"
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
            stdbuf -oL -eL uv run python -m lerobot.async_inference.multi_candidate_server \
                --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_smolvla_server_sm_multican.yaml" \
                --host="${SERVER_HOST}" \
                --port="${SERVER_PORT}" \
                --timing_output_dir="${log_root}/server_timing" \
                --data_collect_dir="${log_root}/mc_data" \
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
            stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_multicand_test \
                --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_smolvla_client_sm_multican.yaml" \
                --env_task="${env_task}" \
                --policy_type="${model_type}" \
                --pretrained_name_or_path="${pretrained_name_or_path}" \
                --server_address="${SERVER_HOST}:${SERVER_PORT}" \
                --results_dir="${log_root}/results" \
                --timing_output_dir="${log_root}/client_timing" \
                --record_trajectory=false \
                --trajectory_dir="${log_root}/mc_trajectories" \
                --data_collect_dir="${log_root}/mc_data" \
                --save_video="${SAVE_VIDEO}" \
                --video_camera=image \
                --video_dir="${log_root}/videos" \
                --queue_size_monitor_path="${log_root}/queue.png" \
                --fps="${FPS}" \
                --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
                --chunk_size_threshold="${threshold_ratio}" \
                --rtc_execution_horizon=0 \
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
    done  # chunk_size_threshold
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
echo "  1. Run per-combo timing analysis:"
echo "     bash analyze_tools_script/analyze_all_eval_outputs.sh \\"
echo "         outputs/eval_thesis/${benchmark_robot_type}"
echo ""
echo "  2. Run cross-combo sweep analysis:"
echo "     uv run python -m lerobot.async_inference.analyze_sweep \\"
echo "         outputs/eval_thesis/${benchmark_robot_type} \\"
echo "         --method ${METHOD} --model ${model_type}"
echo "════════════════════════════════════════════════════════════════════"
