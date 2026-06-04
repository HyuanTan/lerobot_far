**vlash_robot_client.py**
VLASHRobotClient extends RobotClient with two overrides:

_aggregate_action_queues() — called in receive_actions() thread after each chunk arrives. Captures incoming[-1].get_action() (last action of the chunk, by timestep) as _vlash_future_state (thread-safe with a lock).

_capture_raw_obs() — calls super() for real robot obs, then replaces each motor joint scalar with the corresponding value from _vlash_future_state. When raw_observation_to_observation() runs on the server, it reads these overridden values to build observation.state, so the policy sees the robot's predicted future pose instead of its current pose
---
**vlash_policy_server.py**
VLASHPolicyServer extends PolicyServer with two overrides:

SendPolicyInstructions() — loads VLASH's PI05Policy (or PI0Policy) via _load_vlash_policy(), sets preprocessor = postprocessor = None, disables RTC. VLASH's from_pretrained is used directly. The VLASH package path is resolved from env var VLASH_PACKAGE_PATH (default: ~/VLA/vla_asyn_arena_ori_repo/vlash/vlash_main).

_predict_action_chunk() — builds the batch dict via raw_observation_to_observation() (same robot→lerobot format conversion as the base server), moves tensors to device, calls policy.predict_action_chunk(batch) directly. VLASH handles normalization + VLM + ODE + unnormalization internally. Returns ActionChunk(original_actions=None) to disable RTC leftover tracking
```bash
# Server
VLASH_PACKAGE_PATH=/data/users/huoyuan/VLA/vla_asyn_arena_ori_repo/vlash/vlash_main \
stdbuf -oL -eL uv run python -m lerobot.async_inference.vlash_policy_server \
    --host=127.0.0.1 --port=8080 --fps=10 \
    --inference_latency=0.00 \
    --obs_queue_timeout=1 \
    2>&1 | tee ${log_path}/server_$(date +%Y%m%d_%H%M%S).log

# Client
export pretrained_name_or_path=HollyTan/pi05_vlash_so101_2.4-8b_async
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=vlash_pi05
export env_task=so101_no_rtc_abs_10hz
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.vlash_robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --task="Pick up the yellow cube and put it into the orange box." \
    --server_address=127.0.0.1:8080 \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda \
    --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=0 \
    --interpolation_multiplier=1 \
    --fps=10 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}"
    --obs_image_jpeg_quality=85



```

-----
