# SO-101

Start SO-101 server:

```bash
export CUDA_VISIBLE_DEVICES=3

export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=so101_mc_sm_10rtc_imag-crop_rewind_retry_20hz
export benchmark_robot_type=so101
export robot_type=so100_follower
export robot_port=/dev/ttyACM_so101follower
export PYTHONUNBUFFERED=1
export log_root=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/
mkdir -p ${log_root}


uv run python -m lerobot.async_inference.multi_candidate_server \
    --config_path src/lerobot/async_inference/config/${benchmark_robot_type}/multicand_server.yaml \
    --save_root_path=${log_root} \
    2>&1 | tee ${log_root}/server_$(date +%Y%m%d_%H%M%S).log

```
Start SO-101 client:

```bash
python -m lerobot.async_inference.run_so101_multicand_client \
    --config_path src/lerobot/async_inference/config/${benchmark_robot_type}/multicand_client.yaml \
    --task="Pick up the yellow cube and put it into the box." \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --save_root_path=${log_root} \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --obs_image_use_model_resize=true \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --rtc_execution_horizon=10 \
    --aggregate_fn_name=latest_only \
    --enable_recapture_home_positions=false \
    --task_done_home_check_mode=joint \
    --task_done_home_tolerance=8.0 \
    --task_done_home_ee_tolerance_m=0.01 \
    --task_done_home_ee_tolerance_xyz_m=[0.01,0.01,0.01] \
    --task_done_home_confirm_steps=5 \
    --trajectory_output_dir=trajectories \
    --trajectory_dir=mc_trajectories \
    --data_collect_dir=mc_data \
    --results_dir=mc_results \
    --timing_output_dir=timing \
    --queue_size_monitor_path=${log_root}/queue.png \
    2>&1 | tee ${log_root}/client_$(date +%Y%m%d_%H%M%S).log
    

```

Override any YAML value from CLI:
```bash
# Use 30 Hz instead of the YAML's 20 Hz, keep everything else
python -m lerobot.async_inference.run_so101_multicand_client \
    --config_path src/lerobot/async_inference/config/so101/multicand_client.yaml \
    --save_root_path ./outputs/eval/so101/run1 \
    --fps=30 --client_smooth_alpha=0.3 \
    --task="pick the block" ...

```
# LIBERO simulation:
```bash
# Server
python -m lerobot.async_inference.multi_candidate_server \
    --config_path src/lerobot/async_inference/config/libero/multicand_server.yaml \
    --save_root_path ./outputs/eval/so101/run1 \
    --pretrained_name_or_path /path/to/pi05_model

# Client
python -m lerobot.async_inference.sim_test.run_libero_multicand_test \
    --config_path src/lerobot/async_inference/config/libero/multicand_client.yaml \
    --save_root_path ./outputs/eval/so101/run1 \
    --pretrained_name_or_path /path/to/pi05_model \
    --env_task=libero_10

```
