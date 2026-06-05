# Client–Server Experiments

← [Back to README](../README.md)

All commands assume `cd ~/VLA/LeRobot/lerobot_v0.5.2` and a GPU workstation as server.  
Client commands run inside the Jetson container (`docker exec -it lerobot_so101_v0.5.2 /bin/bash`).

---

## 1. Policy Server

The server is hardware-agnostic. Launch once and reuse across all client experiments.

```bash
# Server — common environment
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export benchmark_robot_type=so101          # or: libero
export model_type=pi05                     # or: smolvla
export env_task=so101_rtc_20hz
export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=20 \
    --inference_latency=0.00 \
    --obs_queue_timeout=1 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/server_timing \
    2>&1 | tee ${log_path}/server_$(date +%Y%m%d_%H%M%S).log

# Verify port is open
nc -vz 127.0.0.1 8080
```

**SSH tunnel** (run on Jetson, forwards Jetson local port 8080 to the server):

```bash
ssh -J <gateway> <server-host> -L 8080:127.0.0.1:8080
```

> `server --fps` and `client --fps` must match. Effective control rate = `fps × interpolation_multiplier`.

---

## 2. SO-101 Real Robot — robot_client

### Environment variables (shared)

```bash
export benchmark_robot_type=so101
export robot_type=so100_follower
export robot_port=/dev/ttyACM_so101follower
export CAMERAS="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}"
export TASK="Pick up the yellow cube and put it into the orange box."
```

### No RTC — Pi0.5

```bash
export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_no_rtc_abs_image_crop_20hz
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.robot_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="${CAMERAS}" \
    --robot.id=cse_so101follower \
    --task="${TASK}" \
    --server_address=127.0.0.1:8080 \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=0 \
    --interpolation_multiplier=1 \
    --fps=10 \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```

### With RTC — Pi0.5

```bash
export env_task=so101_rtc_abs_image_crop_10hz
# Same as above with:
    --rtc_execution_horizon=20
```

### No RTC — SmolVLA

```bash
export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_no_rtc
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

python -m lerobot.async_inference.robot_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="${CAMERAS}" \
    --robot.id=cse_so101follower \
    --task="${TASK}" \
    --server_address=127.0.0.1:8080 \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=0 \
    --interpolation_multiplier=2 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}" \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```

---

## 3. SO-101 Real Robot — smart_robot_client (Gripper SM)

`smart_robot_client` adds the gripper-feedback state machine. Set `--enable_gripper_sm=false` to fall back to plain `robot_client` behavior.

### SM parameters reference

| Parameter | Default | Tuning basis |
|-----------|---------|-------------|
| `gripper_load_grasp_threshold` | 80.0 | Successful grasp load ≈ 400–500, empty ≈ 0–50; set 80 for safety margin |
| `gripper_pos_empty_threshold` | 8.0 | Empty close pos ≈ 2–5, held object ≈ 10–15; set 8 to distinguish |
| `gripper_pos_open_threshold` | 20.0 | Fully open pos ≈ 28–30; set 20 to detect "open" state |
| `gripper_slip_drop_ratio` | 0.4 | Load drops to 40% of peak → slip; start loose (0.3) to reduce false triggers |
| `gripper_confirm_steps` | 3 | 3 steps at 30 fps = 100 ms, eliminates transient noise |
| `max_empty_grasp_retries` | 3 | Stop after 3 consecutive empty grasps to prevent infinite loop |

> Tune thresholds by running teleoperate with `--robot.record_motor_state='["gripper"]'` and observing `gripper/load` in Rerun. See [jetson-so101_hardware.md](jetson-so101_hardware.md#motor-feedback-monitoring-during-teleoperation).

### SM + LIFT_RETRY — SmolVLA

```bash
export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_sm_no_rtc_lift_retry
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

python -m lerobot.async_inference.smart_robot_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="${CAMERAS}" \
    --robot.id=cse_so101follower \
    --task="${TASK}" \
    --server_address=127.0.0.1:8080 \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=0 \
    --interpolation_multiplier=2 \
    --fps=20 \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --empty_grasp_lift_retry_enabled=true \
    --empty_grasp_lift_delta_xyz_m="[0.0, 0.0, 0.12]" \
    --empty_grasp_lift_warmup_steps=1 \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}" \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --log_level=DEBUG \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```

### SM + RTC + REWIND_RETRY — Pi0.5

```bash
export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_sm_20rtc_imag-crop_rewind_retry_20hz
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.smart_robot_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="${CAMERAS}" \
    --robot.id=cse_so101follower \
    --task="${TASK}" \
    --server_address=127.0.0.1:8080 \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=1 \
    --fps=20 \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_grasp_confirm_steps=0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=false \
    --recovery_smooth_steps=140 \
    --enable_recapture_home_positions=false \
    --task_done_home_check_mode=ee \
    --task_done_home_ee_tolerance_m=0.06 \
    --task_done_home_confirm_steps=1 \
    --empty_grasp_lift_retry_enabled=false \
    --empty_grasp_rewind_enabled=true \
    --empty_grasp_rewind_buffer_steps=160 \
    --empty_grasp_rewind_steps=100 \
    --empty_grasp_rewind_min_displacement_deg=40.0 \
    --empty_grasp_rewind_warmup_steps=1 \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --log_level=DEBUG \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```

**Home-check modes:**
```bash
# EE mode (recommended)
--task_done_home_check_mode=ee \
--task_done_home_ee_tolerance_m=0.06 \
--task_done_home_confirm_steps=1

# Joint mode (default)
--task_done_home_check_mode=joint \
--task_done_home_tolerance=15.0 \
--task_done_home_confirm_steps=1
```

---

## 4. LIBERO Simulation — sim_client

### No RTC

```bash
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_object          # libero_goal / libero_object / libero_spatial / libero_10
export model_type=pi05
export benchmark_robot_type=libero_no-rtc
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}
mkdir -p ${log_path}

# Server: policy_server --fps=30 (same as §1 but with fps=30)

stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_test \
    --env_task=${env_task} \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --server_address=localhost:8080 \
    --transmit_images_as_uint8=true \
    --actions_per_chunk=50 \
    --episodes_per_task=2 \
    --fps=30 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=0 \
    --results_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/sim_test_results \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/client_timing \
    --save_video=true \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/videos \
    --video_camera=image \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```

### With RTC

Same as above with `--rtc_execution_horizon=20` and `export benchmark_robot_type=libero_rtc`.

### With SM + action_replay rewind

```bash
export benchmark_robot_type=libero_rtc_sm_action_replay_home_reset

stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_smart_test \
    --env_task=${env_task} \
    --obs_type=pixels_agent_pos \
    --enable_gripper_sm=true \
    --enable_home_reset=true \
    --home_reset_warmup_steps=5 \
    --rewind_mode=action_replay \
    --rewind_buffer_steps=25 \
    --rewind_warmup_steps=5 \
    --gripper_pos_sum_empty_threshold=0.04 \
    --max_empty_grasp_retries=3 \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --server_address=localhost:8080 \
    --transmit_images_as_uint8=true \
    --actions_per_chunk=50 \
    --episodes_per_task=2 \
    --fps=30 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=20 \
    --results_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/sim_test_results \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/client_timing \
    --save_video=true \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/videos \
    --video_camera=image \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

# --rewind_mode=set_state       MuJoCo exact state restore
# --rewind_mode=action_replay   replay action history in reverse
```

---

## 5. Multi-Candidate Server

Runs N parallel inference calls per observation; selects the best action chunk by a scoring function (jerk, velocity peak, consistency).

### LIBERO — Pi0.5

```bash
export CUDA_VISIBLE_DEVICES=3
export benchmark_robot_type=libero
export model_type=pi05
export env_task=libero_object
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export pretrained_short_name="${pretrained_name_or_path##*/}"
export log_path=./logs/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

# Terminal 1 — MultiCandidatePolicyServer
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --fps=10 \
    --n_candidates=16 \
    --top_k=4 \
    --delay_delta=1 \
    --w_jerk=1.0 --w_vel_peak=0.5 --w_consistency=0.3 \
    --record_all_candidates=true \
    --data_collect_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_data

# Terminal 2 — sim client with SM
uv run python -m lerobot.async_inference.sim_test.run_libero_multicand_test \
    --env_task=${env_task} \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --server_address=127.0.0.1:8080 \
    --actions_per_chunk=50 --fps=10 \
    --obs_type=pixels_agent_pos \
    --enable_gripper_sm=true \
    --enable_home_reset=true \
    --home_reset_warmup_steps=5 \
    --rewind_mode=action_replay \
    --rewind_buffer_steps=25 \
    --rewind_warmup_steps=5 \
    --gripper_pos_sum_empty_threshold=0.04 \
    --max_empty_grasp_retries=3 \
    --client_smooth_alpha=0.4 \
    --episodes_per_task=2 \
    --rtc_execution_horizon=20 \
    --record_trajectory=true \
    --trajectory_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_trajectories \
    --data_collect_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_data \
    --results_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_results_pi05 \
    --save_video=true \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/videos
```

`n_candidates` / `top_k` tradeoffs:
```bash
--n_candidates=8  --top_k=2   # low diversity
--n_candidates=16 --top_k=4   # client can override
--n_candidates=16 --top_k=1   # server picks best, no client override
--delay_delta=0               # noise diversity only (no RTC delay variant)
--delay_delta=1               # RTC delay variant (recommended)
```

### SO-101 — via YAML config (recommended)

```bash
export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_mc_sm_10rtc_imag-crop_rewind_retry_20hz
export benchmark_robot_type=so101
export robot_type=so100_follower
export robot_port=/dev/ttyACM_so101follower
export log_root=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_root}

# Terminal 1 — server via YAML
uv run python -m lerobot.async_inference.multi_candidate_server \
    --config_path src/lerobot/async_inference/config/${benchmark_robot_type}/multicand_server.yaml \
    --save_root_path=${log_root} \
    2>&1 | tee ${log_root}/server_$(date +%Y%m%d_%H%M%S).log

# Terminal 2 — client via YAML (override any value from CLI)
python -m lerobot.async_inference.run_so101_multicand_client \
    --config_path src/lerobot/async_inference/config/${benchmark_robot_type}/multicand_client.yaml \
    --task="Pick up the yellow cube and put it into the box." \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="${CAMERAS}" \
    --robot.id=cse_so101follower \
    --save_root_path=${log_root} \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --obs_image_use_model_resize=true \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --rtc_execution_horizon=10 \
    --queue_size_monitor_path=${log_root}/queue.png \
    2>&1 | tee ${log_root}/client_$(date +%Y%m%d_%H%M%S).log
```

### SO-101 — full inline (SM + REWIND_RETRY)

```bash
# Terminal 1
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --inference_latency=0.00 --obs_queue_timeout=1 \
    --fps=20 \
    --n_candidates=2 --top_k=2 --delay_delta=0 \
    --w_consistency=0.0 \
    --record_all_candidates=true \
    --log_level=DEBUG \
    --data_collect_dir=${log_root}/mc_data \
    2>&1 | tee ${log_root}/server_$(date +%Y%m%d_%H%M%S).log

# Terminal 2
python -m lerobot.async_inference.run_so101_multicand_client \
    --robot.type=${robot_type} --robot.port=${robot_port} \
    --robot.cameras="${CAMERAS}" --robot.id=cse_so101follower \
    --task="Pick up the yellow cube and put it into the box." \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda --client_device=cuda \
    --actions_per_chunk=50 --server_address=127.0.0.1:8080 \
    --fps=20 --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=10 \
    --interpolation_multiplier=1 \
    --client_smooth_alpha=0.4 \
    --server_score_normalize=softmax \
    --spread_uncertainty_threshold=0.15 \
    --spread_slow_alpha_scale=1.5 --spread_slow_mode_window=5 \
    --action_limit_min=-10.0 --action_limit_max=310.0 \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_grasp_confirm_steps=0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --recovery_smooth_steps=155 \
    --empty_grasp_lift_retry_enabled=false \
    --empty_grasp_rewind_enabled=true \
    --empty_grasp_rewind_buffer_steps=150 \
    --empty_grasp_rewind_steps=120 \
    --empty_grasp_rewind_min_displacement_deg=50.0 \
    --empty_grasp_rewind_warmup_steps=1 \
    --empty_grasp_rewind_settle_time=1.0 \
    --recovery_warmup_steps=2 \
    --recovery_home_settle_time=1.0 \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --enable_recapture_home_positions=false \
    --task_done_home_check_mode=joint \
    --task_done_home_tolerance=8.0 \
    --task_done_home_confirm_steps=5 \
    --retry_anti_repeat_steps=30 \
    --retry_anti_min_dist=15.0 --retry_anti_penalty=0.35 \
    --record_trajectory=true \
    --trajectory_output_dir=${log_root}/trajectories \
    --trajectory_dir=${log_root}/mc_trajectories \
    --data_collect_dir=${log_root}/mc_data \
    --results_dir=${log_root}/mc_results \
    --timing_output_dir=${log_root}/timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=${log_root}/queue.png \
    --log_level=INFO \
    2>&1 | tee ${log_root}/client_$(date +%Y%m%d_%H%M%S).log
```

**LIBERO vs. SO-101 parameter mapping:**

| Parameter | LIBERO | SO-101 | Note |
|-----------|--------|--------|------|
| `action_limit_min/max` | `-1.5 / 1.5` | `-10.0 / 310.0` | LIBERO normalised; SO-101 in degrees |
| `so101_gripper_open_deg` | N/A | `20.0` | Must equal `gripper_pos_open_threshold` |
| `so101_gripper_empty_deg` | N/A | `8.0` | Must equal `gripper_pos_empty_threshold` |
| `spread_uncertainty_threshold` | `0.08` | `0.15` | Real robot has more noise |
| `gripper_load_grasp_threshold` | N/A (qpos) | `150` | Real grasp load ≈ 300–500; empty ≈ 80–120 |
| `interpolation_multiplier` | 1 | `3` | Effective rate = fps × 3 |

> **Warning:** if `so101_gripper_open_deg ≠ gripper_pos_open_threshold` (diff >1°) the client prints a WARNING. Keep them consistent.

---

## 6. Attention / Feature Visualization

### SmolVLA — LIBERO cross-attention (standalone, no server)

```bash
export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=HuggingFaceVLA/smolvla_libero
export model_type=smolvla
export env_task=libero_10
export benchmark_robot_type=libero
export pretrained_short_name="${pretrained_name_or_path##*/}"
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_attn_test \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --env_task=${env_task} \
    --episodes_per_task=1 \
    --attn_save_every_n=5 \
    --attn_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/smolvla_attention/${env_task}/client_attention \
    2>&1 | tee ${log_path}/client_attention_$(date +%Y%m%d_%H%M%S).log
```

### Pi0.5 — LIBERO features (standalone)

```bash
export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export model_type=pi05
export env_task=libero_10

stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_pi05_feat_test \
    --env_task=${env_task} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda \
    --episodes_per_task=1 \
    --feat_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/pi05_features/${env_task} \
    --feat_save_every_n=5 \
    2>&1 | tee ${log_path}/client_attention_$(date +%Y%m%d_%H%M%S).log
```

### Integrated visualization — video + attention overlay

```bash
# SmolVLA — cross-attention heatmap
uv run python -m lerobot.async_inference.sim_test.run_libero_vis_test \
    --policy_type=smolvla \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --env_task=${env_task} \
    --enable_attn_vis=true \
    --attn_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/smolvla_attention/${env_task}/attention \
    --attn_save_every_n=3 \
    --save_video=true \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/smolvla_attention/${env_task}/videos \
    --video_camera=agentview_image \
    --video_fps=30

# Pi0.5 — lang→image + action→image heatmap + episode drift plot
uv run python -m lerobot.async_inference.sim_test.run_libero_vis_test \
    --policy_type=pi05 \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --env_task=${env_task} \
    --enable_attn_vis=true \
    --attn_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/pi05_attention/${env_task}/attention \
    --attn_save_every_n=5 \
    --save_episode_plots=true \
    --save_video=true \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/pi05_attention/${env_task}/videos \
    --video_camera=agentview_image \
    --video_fps=30

# Eval only, zero visualization overhead
python -m lerobot.async_inference.sim_test.run_libero_vis_test \
    --policy_type=smolvla --enable_attn_vis=false
```

> `--attn_save_every_n=N`: N=1 saves every call; N=3–10 is a good tradeoff for long episodes.  
> `--video_camera`: `agentview_image` (default) or `robot0_eye_in_hand_image`.

### Client-server attention (attn_policy_server)

Replace `policy_server` with `attn_policy_server` to enable attention recording from any client without changing the client command:

```bash
python -m lerobot.async_inference.attn_policy_server \
    --host=127.0.0.1 --port=8080 --fps=10 \
    --attn_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/attention/${env_task} \
    --attn_save_every_n=3 \
    2>&1 | tee ${log_path}/server_attention_$(date +%Y%m%d_%H%M%S).log

# Client unchanged
python -m lerobot.async_inference.sim_test.run_libero_test \
    --policy_type=pi05 \
    --pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044 \
    --server_address=localhost:8080 \
    --env_task=libero_spatial
```

---

## 7. Analysis

### Timing analysis

```bash
uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name} \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/timing_analysis

# Text only (no plots)
uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name} \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/timing_analysis \
    --no_plots
```

### RTC analysis

```bash
uv run python -m lerobot.async_inference.analyze_rtc \
    --client_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --out_dir    ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/rtc_analysis
```

### Trajectory analysis

```bash
# SO-101 trajectory (EE-space)
python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \
    --traj_dir=${log_root}/mc_trajectories \
    --out_dir=${log_root}/mc_viz \
    --action_dim_names=shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper \
    --robot_type=so101 \
    --viz_mode=ee

# LIBERO trajectory
uv run python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \
    --traj_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_trajectories \
    --out_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_viz \
    --action_dim_names=j0,j1,j2,j3,j4,j5,grip

# Simple trajectory viewer (requires python3-tk)
# sudo apt install python3-tk
python src/lerobot/async_inference/analyze_trajectory.py ${log_root}/trajectories
```

### Outcome summary

```bash
cat ${log_root}/mc_results/summary.txt
cat ${log_root}/mc_results/aggregate.json

cat ${log_root}/mc_data/client_outcomes.jsonl | python3 -c "
import sys, json
recs = [json.loads(l) for l in sys.stdin]
success = sum(r['success'] for r in recs)
print(f'Episodes: {len(recs)}  Success: {success}  SR: {success/len(recs):.1%}')
for r in recs:
    print(f'  ep={r[\"episode_id\"]} success={r[\"success\"]} steps={r[\"steps\"]} retries={r[\"sm_retries\"]}')
"
```

### Copy outputs from Jetson to server

```bash
sudo chown -R $USER:$USER outputs/eval/so101
sudo apt install -y rsync
rsync -avh --progress outputs/eval/so101 \
    huoyuan@minerva.cse.chalmers.se:/data/users/huoyuan/VLA/LeRobot/lerobot_v0.5.2/outputs/eval_client/
```

### Batch analysis scripts

```bash
./analyze_tools_script/copy_client_eval_outputs.sh
./analyze_tools_script/analyze_all_eval_outputs.sh
```

---

## 8. Image Transmission Parameters

```bash
# Per-camera resize before sending over gRPC (values must be ≥ policy input shape)
--obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}"
--obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}"
--obs_image_resize_hw="[224, 224]"        # apply same size to all cameras

# Use model's own input shape for resize (avoids specifying manually)
--obs_image_use_model_resize=true

# JPEG quality (85 reduces bandwidth ~40% with minimal quality loss)
--obs_image_jpeg_quality=85

# Default (no flags): send raw uint8 at original camera resolution
--transmit_images_as_uint8=true
```

---

## 9. Queue Monitor

```bash
--queue_size_monitor_interval=5 \
--queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png
```

Watch it update:
```bash
watch -n 5 ls -lh ./outputs/eval/.../queue.png
feh --reload 5 ./outputs/eval/.../queue.png &
```

Queue size interpretation:

| Pattern | Meaning |
|---------|---------|
| Sustained 0 | Robot waiting for server — action starved, discontinuous control |
| Sustained high and stable | Server fast, buffer full — actions may be stale |
| Periodic 0 then spike | Normal chunk arrival pattern |
| Gradually decreasing | Server can't keep up with robot consumption rate |

---

## 10. LIBERO Simulation Sweep (eval_libero_script/)

These scripts run **standalone simulation sweeps** (no client–server; model is loaded directly). They sweep `async_delay d` and `execution_horizon s` to reproduce the delay/horizon plots.

### Module

```
lerobot.async_libero_inference.it_rtc.eval_libero_rtc
```

4 methods per (d, s) cell:

| Method | method_type | enable_sm |
|--------|-------------|-----------|
| `baseline` | baseline | false |
| `rtc` | rtc | false |
| `baseline_sm` | baseline | true |
| `rtc_sm` | rtc | true |

RTC semantics: `effective_horizon = max(d, s)` (requires `d < s`). Baseline: `effective_horizon = s` (requires `d ≤ s`).

### Pi0.5 — eval_libero_script/eval_pi05.sh

```bash
# Run all suites, all methods, both sweeps (multi-GPU)
GPU=1,2,3 \
CKPT=lerobot/pi05_libero_finetuned_v044 \
N_EPISODES=10 BATCH_SIZE=10 \
bash eval_libero_script/eval_pi05.sh

# Flags
bash eval_libero_script/eval_pi05.sh --dry-run       # print commands only
bash eval_libero_script/eval_pi05.sh --delay-only    # sweep d only
bash eval_libero_script/eval_pi05.sh --horizon-only  # sweep s only
bash eval_libero_script/eval_pi05.sh --force         # re-run existing results
bash eval_libero_script/eval_pi05.sh --no-sm         # skip SM methods
bash eval_libero_script/eval_pi05.sh --rtc-only      # rtc + rtc_sm only

# Scope to specific suites or methods
SUITES="libero_spatial libero_object" \
METHODS="baseline rtc" \
bash eval_libero_script/eval_pi05.sh
```

**Key defaults:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CKPT` | `lerobot/pi05_libero_finetuned_v044` | Checkpoint |
| `GPU` | `1,2,3` | Round-robin GPU assignment |
| `SUITES` | all 4 LIBERO | `libero_spatial libero_object libero_goal libero_10` |
| `N_EPISODES` | 10 | Episodes per task |
| `BATCH_SIZE` | 10 | Parallel envs (must ≤ N_EPISODES) |
| `DELAYS` | `0 2 4 6 8 10 12 14` | async_delay sweep |
| `HORIZONS` | `2 4 6 8 10 12 14` | execution_horizon sweep |
| `DELAY_FIXED_S` | 16 | s fixed during delay sweep |
| `HORIZON_FIXED_D` | 1 | d fixed during horizon sweep |
| `OUT_ROOT` | `outputs/eval_thesis_sim/libero/pi05` | Output root |
| `FORCE` | 0 | Skip existing results |

Output layout: `OUT_ROOT/pi05/<suite>/d<d>_s<s>_<method>[_sm]/eval_results.json`

### SmolVLA — eval_libero_script/eval_smolvla.sh

Same interface as `eval_pi05.sh`, different defaults:

| Variable | Default |
|----------|---------|
| `CKPT` | `HollyTan/libero_smolvla_500MSmolVLM2_multitask` |
| `GPU` | auto-detect (CUDA_VISIBLE_DEVICES / nvidia-smi) |
| `DELAYS` | `0 2` (quick test; expand for full sweep) |
| `HORIZONS` | `2 4` |
| `OUT_ROOT` | `outputs/eval_thesis_sim/libero/smolvla` |

```bash
GPU=3 bash eval_libero_script/eval_smolvla.sh

# Full sweep
GPU=0,1,2,3 \
DELAYS="0 2 4 6 8 10 12 14" \
HORIZONS="2 4 6 8 10 12 14" \
bash eval_libero_script/eval_smolvla.sh
```

### Plot results

```bash
uv run python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \
  outputs/eval_thesis_sim/libero/pi05 \
  --all-plots \
  --delay-sweep-horizon 16 \
  --horizon-sweep-delay 1 \
  --ci \
  --output-dir outputs/eval_thesis_sim/libero/pi05/plots
```

---

## 11. LIBERO Single-Point Async/Sync Eval (eval-scripts/libero_*)

These scripts use the **real async client–server pipeline** for single-point evaluation (no sweep). Server is auto-started and stopped per method. YAML configs in `src/lerobot/async_inference/config/libero/`.

### Methods and ports

| Method | Mode | RTC | SM | MC | Port |
|--------|------|-----|----|----|------|
| `nortc` | async | ✗ | ✗ | ✗ | 8085 |
| `rtc` | async | ✓ | ✗ | ✗ | 8083 |
| `nortc_sm` | async | ✗ | ✓ | ✗ | 8086 |
| `rtc_sm` | async | ✓ | ✓ | ✗ | 8084 |
| `nortc_multicand` | async | ✗ | ✓ | ✓ | 8087 |
| `rtc_multicand` | async | ✓ | ✓ | ✓ | 8098 |
| `nortc` (sync) | sync | ✗ | ✗ | ✗ | 8081 |
| `nortc_sm` (sync) | sync | ✗ | ✓ | ✗ | 8082 |

Fixed async parameters: `actions_per_chunk=50`, `chunk_size_threshold=0.5`, `rtc_execution_horizon=15` (RTC methods).  
Fixed sync parameters: `actions_per_chunk=25`, `chunk_size_threshold=0` (synchronous blocking).

### Pi0.5 — async

```bash
# All 6 methods, all 4 suites (skips existing results)
GPU=3 \
CKPT=lerobot/pi05_libero_finetuned_v044 \
EPISODES_PER_TASK=10 \
bash eval-scripts/libero_pi05/async_libero_pi05_eval.sh

# Flags
bash ... --dry-run
bash ... --force                          # re-run even if aggregate.json exists
bash ... --no-sm                          # nortc + rtc only
bash ... --no-mc                          # skip multicand
bash ... --rtc-only                       # rtc + rtc_sm
METHODS="rtc rtc_sm" bash ...            # specific methods
SUITES="libero_object" bash ...          # single suite
SAVE_VIDEO=true bash ...
```

Output: `outputs/eval_thesis/libero/<suite>/<method>/results/aggregate.json`

### SmolVLA — async

```bash
GPU=3 \
CKPT=HollyTan/libero_smolvla_500MSmolVLM2_multitask \
SUITES="libero_spatial libero_object libero_goal libero_10" \
bash eval-scripts/libero_smolvla/async_libero_smolvla_eval.sh
```

Default `SUITES` in the SmolVLA script is `libero_object` (single suite); override to run all.

### Pi0.5 / SmolVLA — sync (baseline)

```bash
# Pi0.5 sync (nortc + nortc_sm, chunk_size_threshold=0)
GPU=3 bash eval-scripts/libero_pi05/sync_libero_pi05_eval.sh

# SmolVLA sync
GPU=3 bash eval-scripts/libero_smolvla/sync_libero_smolvla_eval.sh

# Flags
bash ... --no-sm          # nortc only
bash ... --sm-only        # nortc_sm only
bash ... --force
```

---

## 12. SO-101 Real Robot Eval Scripts (eval-scripts/so101_*)

These are individual server/client scripts for the SO-101 real robot. Unlike the LIBERO eval scripts, the server and client must be started **manually in separate terminals**.

Workflow:
1. Start the **server** script on the GPU machine
2. Set `SERVER_IP` if running on separate machines (default: `127.0.0.1`)
3. Start the **client** script on the Jetson (inside container)
4. Stop the server with `Ctrl-C` when done

Output layout: `outputs/eval_thesis/<robot>/<METHOD>/<model>/H<H>/`

### YAML config files

All scripts load defaults from:
```
src/lerobot/async_inference/config/so101/
  async_server.yaml
  async_client.yaml
  async_server_sm.yaml
  async_client_sm.yaml
  async_server_sm_multican.yaml
  async_client_sm_multican.yaml
```
CLI arguments override any YAML value.

### Pi0.5 — server

```bash
# Start server (GPU machine) — adjust CUDA_VISIBLE_DEVICES as needed
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_pi05/async_so101_pi05_server-rtc.sh

# SM variant
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_pi05/async_so101_pi05_server-rtc-sm.sh

# Multi-candidate SM variant
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_pi05/async_so101_pi05_server-rtc-sm-multican.sh
```

Key settings in `async_so101_pi05_server-rtc.sh`:

| Variable | Default | Edit in script |
|----------|---------|----------------|
| `CUDA_VISIBLE_DEVICES` | 1 | or override via env |
| `FPS` | 20 | must match client |
| `METHOD` | `async_nortc` | tag for output path |
| `RTC_HORIZON` | 0 | 0 = no RTC; set to 16 for RTC |

### Pi0.5 — clients (Jetson)

```bash
# No RTC
SERVER_IP=127.0.0.1 bash eval-scripts/so101_pi05/async_so101_pi05_client-nortc.sh

# No RTC + SM
SERVER_IP=127.0.0.1 bash eval-scripts/so101_pi05/async_so101_pi05_client-nortc-sm.sh

# RTC (H=16)
SERVER_IP=127.0.0.1 bash eval-scripts/so101_pi05/async_so101_pi05_client-rtc.sh

# RTC + SM (H=16)
SERVER_IP=127.0.0.1 bash eval-scripts/so101_pi05/async_so101_pi05_client-rtc-sm.sh

# RTC + SM + Multi-Candidate (N=4, top_k=2)
SERVER_IP=127.0.0.1 bash eval-scripts/so101_pi05/async_so101_pi05_client-rtc-sm-multican.sh
```

Key settings per client:

| Script | `RTC_HORIZON` | `chunk_size_threshold` | module |
|--------|--------------|----------------------|--------|
| `client-nortc.sh` | 0 | 0.4 | `robot_client` |
| `client-nortc-sm.sh` | 0 | 0.4 | `smart_robot_client` |
| `client-rtc.sh` | 16 | 0.4 | `robot_client` |
| `client-rtc-sm.sh` | 16 | 0.4 | `smart_robot_client` |
| `client-rtc-sm-multican.sh` | 16 | 0.4 | `run_so101_multicand_client` |

> `chunk_size_threshold = RTC_HORIZON / ACTIONS_PER_CHUNK = 16/50 ≈ 0.32` for strict coupling; the scripts use `0.4` as a slightly looser value. Adjust `RTC_HORIZON` and `chunk_size_threshold` together.

### SmolVLA — server

```bash
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_smolvla/async_so101_smolvla_server-rtc.sh
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_smolvla/async_so101_smolvla_server-rtc-sm.sh
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_smolvla/async_so101_smolvla_server-rtc-sm-multican.sh
```

### SmolVLA — clients (Jetson)

```bash
SERVER_IP=127.0.0.1 bash eval-scripts/so101_smolvla/async_so101_smolvla_client-nortc.sh
SERVER_IP=127.0.0.1 bash eval-scripts/so101_smolvla/async_so101_smolvla_client-rtc.sh
SERVER_IP=127.0.0.1 bash eval-scripts/so101_smolvla/async_so101_smolvla_client-rtc-sm.sh
SERVER_IP=127.0.0.1 bash eval-scripts/so101_smolvla/async_so101_smolvla_client-rtc-sm-multican.sh
```

Key difference from Pi0.5: SmolVLA uses `obs_image_resize_hw={'top': [480, 640], ...}` (full resolution), Pi0.5 uses `[224, 224]` (model input size with `--obs_image_use_model_resize=true`).

### SmolVLA localhost (server + client on same machine)

```bash
# Server (port 8081)
CUDA_VISIBLE_DEVICES=1 bash eval-scripts/so101_smolvla_localhost/async_so101_smolvla_server-rtc.sh

# Client (points to 127.0.0.1:8081)
bash eval-scripts/so101_smolvla_localhost/async_so101_smolvla_client-rtc.sh
bash eval-scripts/so101_smolvla_localhost/async_so101_smolvla_client-rtc-sm.sh
```

Localhost scripts use `SERVER_PORT=8081` to avoid conflict with the real robot pipeline on `8080`.

### Editing method variants in the scripts

Each server/client script has a `METHOD=` tag at the top that controls the output path. Edit directly or override the variables before running:

```bash
# Override model and method tag for Pi0.5 client
pretrained_name_or_path="HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b" \
METHOD=async_rtc_sm_inter \
RTC_HORIZON=16 \
bash eval-scripts/so101_pi05/async_so101_pi05_client-rtc-sm.sh
```
