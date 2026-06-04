# sim_client test

--chunk_size_threshold=0.5 \
--aggregate_fn_name=latest_only \
--rtc_execution_horizon=0 \  

--rtc_execution_horizon=0 --- no rtc
## server
```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_object
export model_type=pi05
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}
mkdir -p ./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}

stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/server_timing \
    2>&1 | tee ${log_path}/server_$(date +%Y%m%d_%H%M%S).log


nc -vz 127.0.0.1 8080
```
## client
```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_object
export model_type=pi05
export benchmark_robot_type=libero_no-rtc
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}
mkdir -p ./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}

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
    --save_video=True \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/videos \
    --video_camera=image \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log



# 不同 fps / chunk size 对比实验
for fps in 10 20 30; do
  uv run python -m lerobot.async_inference.sim_test.run_sim_test \
      --fps=$fps \
      --num_episodes=5 \
      --max_steps_per_episode=50 \
      --server_port=809$fps \
      --timing_output_dir=./timing_fps${fps} \
      --save_results=false
done

```

Using RTC
```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_object
export model_type=pi05
export benchmark_robot_type=libero_rtc
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}
mkdir -p ./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}

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
    --rtc_execution_horizon=20 \
    --results_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/sim_test_results \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/client_timing \
    --save_video=True \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/videos \
    --video_camera=image \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```
--transmit_images_as_uint8=true: robot_client 走 obs_pre_mapped=False 路径，服务端调 raw_observation_to_observation() → prepare_image()，内部已做 uint8→float32 转换。真机相机本身输出 uint8，链路上本来就没有 float32 开销，robot_client 无需改动。

--- _policy_broken = True	5次连续失败阈值到达
显存碎片化：多次推理后 GPU 显存碎片化，PyTorch 分配器需要在新地址分配张量，而 CUDA Graph 捕获的是原始地址偏移，导致 "offset increment outside graph"。

动态分支：Pi05 的 flow matching denoising loop 在某些 obs 状态下触发了不同迭代步数，或条件分支（如 clipping 逻辑），CUDA Graph 无法处理动态控制流。

obs_fps=12.3Hz 是压力信号：预期 30Hz 降至 12.3Hz，说明 gRPC 线程与 inference GPU 内存之间存在竞争，加速了显存碎片化。

----

libero_goal
libero_object
libero_spatial
libero_10
```bash

cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_10
export model_type=pi05
export benchmark_robot_type=libero_rtc_sm_action_replay_home_reset
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}
mkdir -p ./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}

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
    --save_video=True \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/videos \
    --video_camera=image \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

--rewind_mode=set_state       # (默认) MuJoCo 精确回退
--rewind_mode=action_replay   # 动作历史回放

```

```bash

cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_object
export model_type=pi05
export benchmark_robot_type=libero_rtc_sm_set_state
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}
mkdir -p ./logs/${benchmark_robot_type}/${model_type}/sim/${env_task}

stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_smart_test \
    --env_task=${env_task} \
    --obs_type=pixels_agent_pos \
    --enable_gripper_sm=true \
    --rewind_mode=set_state \
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
    --save_video=True \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/videos \
    --video_camera=image \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

--rewind_mode=set_state       # (默认) MuJoCo 精确回退
--rewind_mode=action_replay   # 动作历史回放

```

## Anylize
```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=libero_object
export model_type=pi05
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/sim

uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task} \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/timing_analysis --no_plots 2>&1

uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task} \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/timing_analysis 2>&1


uv run python -m lerobot.async_inference.analyze_rtc \
    --client_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/client_timing \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/rtc_analysis
```

# SO101 test
## server
```BASH
cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=1
export env_task=so101
export model_type=pi05
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}
mkdir -p ./logs/${benchmark_robot_type}/${model_type}

stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=20 \
    --inference_latency=0.00 \
    --obs_queue_timeout=1 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/server_timing \
    2>&1 | tee ${log_path}/server_$(date +%Y%m%d_%H%M%S).log
    
```
--use_relative_actions=True \
## client

HollyTan/pi05_so101_pick_place-v2.1
HollyTan/pi05_so101_pick_place-v2.1_ori-img
HollyTan/pi05_so101_pick_place-v2.1_ori-img_abs
HollyTan/pi05_so101_pick_place-v2.2_delta

HollyTan/pi05_so101_pick_place-v2.3_delta
HollyTan/pi05_so101_pick_place-v2.3_abs

HollyTan/pi05_so101_pick_place-v2.2-100eps_abs_nofreeze
HollyTan/pi05_so101_pick_place-v2.4_abs_nofreeze_8b
HollyTan/pi05_so101_pick_place-v2.4_delta_nofreeze-8b
HollyTan/so101_pick-place-v2.4_abs_nofreeze-16b-7.5k
HollyTan/so101_pick-place-v2.4_delta_nofreeze-16b-7.5k

HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
HollyTan/pi05_so101_pick_place-v2.2-100eps_delta_nofreeze-8b

HollyTan/pi05_so101_pick_place-merge-v2.2-v2.3-v2.4_abs_nofreeze_8b, 抖动

HollyTan/pi05_so101_pick_place-v2.4-base-mixtrain_abs_nofreeze_8b, 抖动

--policy.n_action_steps=10

- 都不设置时：原始图片大小，uint8 上传
- obs_image_resize_hw 设的值必须 ≥ policy 的 image_features.shape，保证 server 只做等比缩小或 NO-OP，不做上采样

--obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}" \
--obs_image_resize_hw="{'top': [512, 512], 'wrist': [512, 512], 'front': [512, 512]}"
--obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
--obs_image_resize_hw="[224, 224]"

--obs_image_jpeg_quality=85

NO RTC:
```bash
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_no_rtc_abs_image_crop_20hz
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.robot_client \
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
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
    
    
    --obs_image_jpeg_quality=85


```

Use RTC:
```bash
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_rtc_abs_image_crop_10hz
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=1 \
    --fps=10 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log
```

--------
### Test Smolvla
jadenovalight/smolvla_pick-place_v2.1_3cam
jadenovalight/smolvla_pick-place_v2.2

jadenovalight/smolvla_pick-place_v2.2_100eps
jadenovalight/smolvla_pick-place_v2.3_2cam
jadenovalight/smolvla_pick-place_v2.3

jadenovalight/smolvla_pick-place_v2.4
jadenovalight/smolvla_pick-place_v2.4_top-front
jadenovalight/smolvla_pick-place_v2.4_top-wrist

--obs_image_resize_hw="{'top': (480, 640), 'wrist': (480, 640), 'front': (480, 640)}" \
--obs_image_jpeg_quality=85

NO RTC:
```bash
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_no_rtc
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

python -m lerobot.async_inference.robot_client \
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
    --interpolation_multiplier=2 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}"
    
    --obs_image_resize_hw="{'top': [512, 512], 'wrist': [512, 512], 'front': [512, 512]}"


python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
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
    --interpolation_multiplier=2 \
    --fps=30 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png
    
```


Use RTC:
```bash
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_rtc-img_crop_100q
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

python -m lerobot.async_inference.robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=2 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}"
```

--------
### Test PI0

HollyTan/pi0_so101_pick_place-v2.4_abs_nofreeze

USE RTC:
```bash
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=HollyTan/pi05_vlash_so101_2.4-8b_async
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_rtc_abs-img_crop_100q
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

python -m lerobot.async_inference.robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=2 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}"


```

## Anylize
```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2

./analyze_tools_script/copy_client_eval_outputs.sh

./analyze_tools_script/analyze_all_eval_outputs.sh

uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name} \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/timing_analysis --no_plots 2>&1

uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name} \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/timing_analysis 2>&1


uv run python -m lerobot.async_inference.analyze_rtc \
    --client_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/rtc_analysis


watch -n 5 ls -lh ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png          # 确认文件在更新
# 或用支持自动刷新的图片查看器
feh --reload 5 ./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png &
```


```
sudo chown -R $USER:$USER outputs/eval/so101

sudo apt install -y rsync
rsync -avh --progress outputs/eval/so101 huoyuan@minerva.cse.chalmers.se:/data/users/huoyuan/VLA/LeRobot/lerobot_v0.5.2/outputs/eval_client/
```


----------------
robot_client          obs_pre_mapped=False          sim_client
(物理机器人)                                          (仿真环境)
     │                                                    │
     │  raw_obs: {top: uint8 array,                      │  已经是 lerobot 格式:
     │             wrist: ...,                           │  {observation.images.top: Tensor,
     │             state: [j1,j2,...],                   │   observation.state: Tensor, ...}
     │             task: "pick up"}                      │
     │                                                    │
     └────────────────────┬───────────────────────────────┘
                          │ gRPC → pickle → TimedObservation
                          ▼
                    policy_server
                          │
             obs_pre_mapped == False?
              ├─ YES → 直接使用，跳过 raw_observation_to_observation()
              │        (sim_client path)
              └─ NO  → 调用 raw_observation_to_observation()
                       把 {top: array, state: [...]} 转换成
                       {observation.images.top: Tensor(C,H,W),
                        observation.state: Tensor(1,D), ...}
                       (robot_client path)
-------------------

--queue_size_monitor_interval=10 \      # 每 10 秒刷新一次 PNG
--queue_size_monitor_path=queue.png     # 默认 queue_size.png


队列大小	含义
持续 0	robot 在等待 server，action starved，控制不连续
持续高且稳定	server 推理很快，buffer 充足，但 action 可能偏旧
周期性 0 后突然升高	正常的 chunk 到达模式（每隔 N 步一个 chunk 补充）
整体下降趋势	server 跟不上 robot 消耗速度

## Test
GraspPhase(Enum)              APPROACHING / CLOSING / HOLDING / OPENING
GripperDecision(Enum)         CONTINUE / REINFER / RECOVERY / STOP
GripperStateMonitor           状态机核心，scan + classify + debounce
SmartRobotClientConfig        继承 RobotClientConfig，新增 12 个字段（全有默认值）
SmartRobotClient              继承 RobotClient，覆写 control_loop()
smart_async_client()          draccus 入口，与 robot_client.py 结构相同


```bash
# 启用状态机（默认）：
python -m lerobot.async_inference.smart_robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --task="pick up the cube" \
    --server_address=127.0.0.1:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=user/model \
    --actions_per_chunk=50 \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3

# 关闭状态机（完全回退）：
python -m lerobot.async_inference.smart_robot_client \
    ... \
    --enable_gripper_sm=false

# --enable_gripper_sm=false 时 control_loop() 第一行执行 return super().control_loop()，走完全相同的父类代码路径。

```

参数	默认值	调参依据
gripper_load_grasp_threshold	80.0	图2抓取成功 load ~400-500，空抓 ~0-50，设 80 留安全裕量
gripper_pos_empty_threshold	8.0	图3空抓闭合 pos ~2-5，持物 ~10-15，设 8 区分两者
gripper_pos_open_threshold	20.0	图3全开 pos ~28-30，设 20 判断"开"状态
gripper_slip_drop_ratio	0.4	load 降到峰值 40% 以下判滑落，可先设宽松（0.3）减少误触发
gripper_confirm_steps	3	30fps 下 3 步 = 100ms，消除瞬态噪声
max_empty_grasp_retries	3	超过 3 次连续空抓停止，防止无限循环

修改 recovey tohome 的速度:
 - recovery_home_steps
 - 如果太快，设置 max_relative_target 限制，per-step 软件限速保护。如果配置了 max_relative_target=None（默认），send_action() 每步会读一次总线位置做 clamp：
present_pos = self.bus.sync_read("Present_Position")  # 额外一次总线读
这在 recovery 中额外增加了每步 ~2ms 的延迟。如果 max_relative_target 设置过小（如 2.0），而 recovery 步长 D/n > 2.0，电机会被 clamp，轨迹在 n 步内走不完全程，会停在中间某处而不是精确到 home。但不会崩溃，只是"基本到家"。

```BASH
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_sm_no_rtc_lift_retry
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

python -m lerobot.async_inference.smart_robot_client \
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
    --interpolation_multiplier=2 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --log_level=DEBUG \
    --empty_grasp_lift_retry_enabled=true \
    --empty_grasp_lift_delta_xyz_m="[-0.035, 0.0, 0.08]" \
    --empty_grasp_lift_warmup_steps=10 \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}"

```


HollyTan/pi05_so101_pick_place-v2.2-100eps_abs_nofreeze
HollyTan/pi05_so101_pick_place-v2.4_abs_nofreeze_8b
HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b

```BASH
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_sm_20rtc_imag-crop_lift_retry_10hz
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.smart_robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=1 \
    --fps=10 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --log_level=DEBUG \
    --empty_grasp_lift_retry_enabled=true \
    --empty_grasp_lift_delta_xyz_m="[0.0, 0.0, 0.12]" \
    --empty_grasp_lift_warmup_steps=1 \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    
    --empty_grasp_lift_delta_xyz_m="[-0.035, 0.0, 0.08]" \

```

HollyTan/pi05_so101_pick_place-v2.4basev2.2_abs_nofreeze_8b
compile_model=true, compile_mode="reduce-overhead"-- not work
compile_model=false
```BASH
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.4basev2.2_abs_nofreeze_8b

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_sm_20rtc_imag-crop_rewind_retry_20hz
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.smart_robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=1 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_grasp_confirm_steps=0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=false \
    --recovery_smooth_steps=140 \
    --log_level=DEBUG \
    --enable_recapture_home_positions=false \
    --task_done_home_check_mode=joint \
    --task_done_home_tolerance=8.0 \
    --task_done_home_ee_tolerance_m=0.01 \
    --task_done_home_ee_tolerance_xyz_m=[0.01,0.01,0.01] \
    --task_done_home_check_gripper=true \
    --task_done_home_gripper_tolerance_deg=1.0 \
    --task_done_home_confirm_steps=5 \
    --empty_grasp_lift_retry_enabled=false \
    --empty_grasp_lift_delta_xyz_m="[0.0, 0.0, 0.12]" \
    --empty_grasp_lift_warmup_steps=10 \
    --empty_grasp_rewind_enabled=true \
    --empty_grasp_rewind_buffer_steps=160 \
    --empty_grasp_rewind_steps=100 \
    --empty_grasp_rewind_min_displacement_deg=40.0 \
    --empty_grasp_rewind_warmup_steps=1 \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --task_done_home_check_mode=ee \
    --task_done_home_ee_tolerance_m=0.06 \
    --task_done_home_confirm_steps=1 \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    
    --empty_grasp_lift_delta_xyz_m="[-0.035, 0.0, 0.08]" \


EE 模式（推荐）：

--task_done_home_check_mode=ee \
--task_done_home_ee_tolerance_m=0.06 \
--task_done_home_confirm_steps=1 \

Joint 模式（原始，默认）：

--task_done_home_check_mode=joint \
--task_done_home_tolerance=15.0 \
--task_done_home_confirm_steps=1 \

```
----

```BASH
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_sm_20rtc_imag-crop_gohome_retry_10hz
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.smart_robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=1 \
    --fps=10 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --log_level=DEBUG \
    --empty_grasp_lift_retry_enabled=false \
    --empty_grasp_rewind_enabled=false \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    
    --empty_grasp_lift_delta_xyz_m="[-0.035, 0.0, 0.08]" \

```



-----


```BASH
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_20rtc-imgcrop_sm_lift_retry
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.smart_robot_client \
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
    --rtc_execution_horizon=20 \
    --interpolation_multiplier=1 \
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_use_model_resize=true \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}" \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --log_level=DEBUG \
    --empty_grasp_lift_retry_enabled=true \
    --empty_grasp_lift_delta_xyz_m="[0.0, 0.0, 0.12]" \
    --empty_grasp_lift_warmup_steps=1 \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    
    --empty_grasp_lift_delta_xyz_m="[-0.035, 0.0, 0.08]" \

```



```BASH
docker exec -it lerobot_so101_v0.5.2 /bin/bash

export pretrained_name_or_path=jadenovalight/smolvla_pick-place_v2.4
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=so101_no_rtc-imgcrop_sm_rewind_retry
export benchmark_robot_type=so101
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL python -m lerobot.async_inference.smart_robot_client \
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
    --fps=20 \
    --timing_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/client_timing \
    --queue_size_monitor_interval=5 \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    --obs_image_use_model_resize=true \
    --obs_image_resize_hw="{'top': [480, 640], 'wrist': [480, 640], 'front': [480, 640]}" \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --log_level=DEBUG \
    --empty_grasp_lift_retry_enabled=false \
    --empty_grasp_lift_delta_xyz_m="[0.0, 0.0, 0.12]" \
    --empty_grasp_lift_warmup_steps=5 \
    --empty_grasp_rewind_enabled=true \
    --empty_grasp_rewind_buffer_steps=100 \
    --empty_grasp_rewind_steps=100 \
    --empty_grasp_rewind_min_displacement_deg=40.0 \
    --empty_grasp_rewind_warmup_steps=4 \
    --record_trajectory=true \
    --trajectory_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/trajectories \
    2>&1 | tee ${log_path}/client_$(date +%Y%m%d_%H%M%S).log

    
    --empty_grasp_lift_delta_xyz_m="[-0.035, 0.0, 0.08]" \

```
