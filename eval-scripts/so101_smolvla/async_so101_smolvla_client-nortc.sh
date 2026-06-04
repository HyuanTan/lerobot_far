#!/usr/bin/env bash
# eval-scripts/so101_smolvla/async_so101_smolvla_client-rtc.sh
#
# Robot client for SO-101 smolvla RTC eval — runs on the robot machine.
# Transmits the model identity to the server on connect; server loads the model.
#
# ── Workflow ─────────────────────────────────────────────────────────────────
#   1. Start the matching async_so101_smolvla_server-rtc*.sh on the GPU machine first.
#   2. Set SERVER_IP to the GPU machine's IP if running on separate machines.
#   3. Run this script on the robot machine.
#
# ── RTC parameter coupling ─────────────────────────────────────────────────────
#   chunk_size_threshold = RTC_HORIZON / ACTIONS_PER_CHUNK  (derived below)
#   Re-inference fires when queue ≤ RTC_HORIZON steps remain.
#   Both are derived from RTC_HORIZON here to keep them consistent.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash eval-scripts/so101_smolvla/async_so101_smolvla_client-rtc.sh
#   SERVER_IP=192.168.1.100 bash eval-scripts/so101_smolvla/async_so101_smolvla_client-rtc.sh

set -uo pipefail

export PYTHONUNBUFFERED=1

# ══ Model ═════════════════════════════════════════════════════════════════════
pretrained_name_or_path="jadenovalight/smolvla_pick-place_v2.4"
# pretrained_name_or_path="jadenovalight/smolvla_pick-place_v2.2"
model_type=smolvla
benchmark_robot_type=so101

# ══ Method tag ════════════════════════════════════════════════════════════════
METHOD=async_nortc

# ══ Server ════════════════════════════════════════════════════════════════════
# Set SERVER_IP to the GPU machine's network IP when running on separate machines.
# If both run on the same machine (tunnel or localhost): SERVER_IP=127.0.0.1
SERVER_IP=${SERVER_IP:-127.0.0.1}
SERVER_PORT=8080

FPS=20                   # robot hardware control rate (Hz)

# ══ RTC settings ══════════════════════════════════════════════════════════════
RTC_HORIZON=0 # 20 # 15           # H (steps): rtc_execution_horizon
ACTIONS_PER_CHUNK=50     # K (fixed, must match server's chunk size)

chunk_size_threshold=0.6        # hardcoded; adjust when RTC_HORIZON changes

# ══ Task ══════════════════════════════════════════════════════════════════════
TASK="Pick up the yellow cube and put it into the box."

# ══ Robot hardware ════════════════════════════════════════════════════════════
ROBOT_PORT=/dev/ttyACM_so101follower
ROBOT_ID=cse_so101follower

# ══ Output ════════════════════════════════════════════════════════════════════
#   log_root = outputs/eval_thesis/<robot>/<method>/<model>/H<H>/
#     client_timing/    ← client timing records
#     trajectories/     ← recorded trajectories
#     queue.png         ← queue size monitor
#     client_<ts>.log
log_root="./outputs/eval_thesis/${benchmark_robot_type}/${METHOD}/${model_type}/H${RTC_HORIZON}"
mkdir -p "${log_root}/client_timing" "${log_root}/trajectories"

echo "════════════════════════════════════════════════════════════════════"
echo "  SO-101 smolvla — RTC robot client"
echo "  method  : ${METHOD}"
echo "  server  : ${SERVER_IP}:${SERVER_PORT}"
echo "  model   : ${pretrained_name_or_path}"
echo "  H       : ${RTC_HORIZON} steps  threshold=${chunk_size_threshold}  K=${ACTIONS_PER_CHUNK}"
echo "  fps     : ${FPS}"
echo "  task    : ${TASK}"
echo "  log_root: ${log_root}"
echo "════════════════════════════════════════════════════════════════════"

stdbuf -oL -eL python -m lerobot.async_inference.robot_client \
    --config_path "src/lerobot/async_inference/config/${benchmark_robot_type}/async_client.yaml" \
    --task="${TASK}" \
    --policy_type=${model_type} \
    --pretrained_name_or_path="${pretrained_name_or_path}" \
    --server_address="${SERVER_IP}:${SERVER_PORT}" \
    --robot.type=so100_follower \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --rtc_execution_horizon="${RTC_HORIZON}" \
    --obs_image_use_model_resize=true \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}" \
    --obs_image_jpeg_quality=85 \
    --chunk_size_threshold="${chunk_size_threshold}" \
    --timing_output_dir="${log_root}/client_timing" \
    --record_trajectory=false \
    --trajectory_output_dir="${log_root}/trajectories" \
    --queue_size_monitor_interval=0 \
    --queue_size_monitor_path="${log_root}/queue.png" \
    2>&1 | tee "${log_root}/client_$(date +%Y%m%d_%H%M%S).log"
