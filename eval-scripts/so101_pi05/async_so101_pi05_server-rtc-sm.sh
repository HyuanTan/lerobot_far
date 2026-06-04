#!/usr/bin/env bash
# eval-scripts/so101_pi05/async_so101_pi05_server-rtc.sh
#
# Policy server for SO-101 pi05 — runs on the GPU machine.
# The model is loaded when the client connects (sent via gRPC RemotePolicyConfig).
#
# ── Workflow ─────────────────────────────────────────────────────────────────
#   1. Run this script on the GPU machine (start first).
#   2. Run async_so101_pi05_client-rtc.sh on the robot machine.
#   3. Stop this server manually with Ctrl-C when done.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/so101_pi05/async_so101_pi05_server-rtc.sh
#   CUDA_VISIBLE_DEVICES=0 bash eval-scripts/so101_pi05/async_so101_pi05_server-rtc.sh

set -uo pipefail

# ══ GPU / environment ═════════════════════════════════════════════════════════
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1

# ══ Task identity (for log naming only) ═══════════════════════════════════════
benchmark_robot_type=so101
model_type=pi05

# ══ Method tag ════════════════════════════════════════════════════════════════
METHOD=async_nortc_sm # async_rtc_sm, async_rtc_sm_inter

FPS=20                   # robot hardware control rate (Hz)

# ══ RTC horizon (for path naming — must match client's RTC_HORIZON) ════════════
RTC_HORIZON=0 #16

# ══ Output ════════════════════════════════════════════════════════════════════
#   log_root = outputs/eval_thesis/<robot>/<method>/<model>/H<H>/
#     server_timing/    ← server timing records
#     server_<ts>.log
log_root="./outputs/eval_thesis/${benchmark_robot_type}/${METHOD}/${model_type}/H${RTC_HORIZON}"
mkdir -p "${log_root}/server_timing"

echo "════════════════════════════════════════════════════════════════════"
echo "  SO-101 pi05 — RTC policy server"
echo "  method      : ${METHOD}"
echo "  CUDA device : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  config      : src/lerobot/async_inference/config/${benchmark_robot_type}/async_server_sm.yaml"
echo "  fps         : ${FPS}"
echo "  log_root    : ${log_root}"
echo "  Stop        : Ctrl-C"
echo "════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
    --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_server_sm.yaml" \
    --timing_output_dir="${log_root}/server_timing" \
    --fps="${FPS}" \
    2>&1 | tee "${log_root}/server_$(date +%Y%m%d_%H%M%S).log"
