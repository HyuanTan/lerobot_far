#!/usr/bin/env bash
# eval-scripts/so101_smolvla/async_so101_smolvla_server-rtc-sm-multican.sh
#
# Policy server for SO-101 smolvla (multi-candidate SM) — runs on the GPU machine.
# The model is loaded when the client connects (sent via gRPC RemotePolicyConfig).
#
# ── Workflow ─────────────────────────────────────────────────────────────────
#   1. Run this script on the GPU machine (start first).
#   2. Run async_so101_smolvla_client-rtc-sm-multican.sh on the robot machine.
#   3. Stop this server manually with Ctrl-C when done.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/so101_smolvla/async_so101_smolvla_server-rtc-sm-multican.sh
#   CUDA_VISIBLE_DEVICES=0 bash eval-scripts/so101_smolvla/async_so101_smolvla_server-rtc-sm-multican.sh

set -uo pipefail

# ══ GPU / environment ═════════════════════════════════════════════════════════
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1

# ══ Task identity (for log naming only) ═══════════════════════════════════════
benchmark_robot_type=so101
model_type=smolvla

# ══ Method tag ════════════════════════════════════════════════════════════════
NCANDIDATE=4
TOPK=2
METHOD=async_rtc_sm_multicand_${NCANDIDATE}n_${TOPK}k

FPS=20                   # robot hardware control rate (Hz)

# ══ RTC horizon (for path naming — must match client's RTC_HORIZON) ════════════
RTC_HORIZON=20

# ══ Output ════════════════════════════════════════════════════════════════════
#   log_root = outputs/eval_thesis/<robot>/<method>/<model>/H<H>/
#     server_timing/    ← server timing records
#     server_<ts>.log
log_root="./outputs/eval_thesis/${benchmark_robot_type}/${METHOD}/${model_type}/H${RTC_HORIZON}"
mkdir -p "${log_root}/server_timing"

echo "════════════════════════════════════════════════════════════════════"
echo "  SO-101 smolvla — multi-candidate SM policy server"
echo "  method      : ${METHOD}"
echo "  CUDA device : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  config      : src/lerobot/async_inference/config/${benchmark_robot_type}/async_server_sm_multican.yaml"
echo "  fps         : ${FPS}"
echo "  log_root    : ${log_root}"
echo "  Stop        : Ctrl-C"
echo "════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL uv run python -m lerobot.async_inference.multi_candidate_server \
    --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_server_sm_multican.yaml" \
    --timing_output_dir="${log_root}/server_timing" \
    --n_candidates=${NCANDIDATE} \
    --top_k=${TOPK} \
    --fps="${FPS}" \
    2>&1 | tee "${log_root}/server_$(date +%Y%m%d_%H%M%S).log"
