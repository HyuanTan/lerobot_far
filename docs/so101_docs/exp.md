# Sync
## Libero
### smolvla
HollyTan/libero_smolvla_500MSmolVLM2_multitask

libero_goal
libero_object
libero_spatial
libero_10
#### no RTC
```bash
export CUDA_VISIBLE_DEVICES=3

export pretrained_name_or_path=HollyTan/libero_smolvla_500MSmolVLM2_multitask
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=libero_object
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1

export log_root=./outputs/eval_thesis/${benchmark_robot_type}/${env_task}/${model_type}/sync/
mkdir -p ${log_root}

stdbuf -oL -eL uv run python -m lerobot.async_inference.policy_server \
    --config_path src/lerobot/async_inference/config/${benchmark_robot_type}/async_inference_server.yaml \
    --timing_output_dir=${log_root}/server_timing \
    --inference_latency=0.0 \
    2>&1 | tee ${log_root}/server_$(date +%Y%m%d_%H%M%S).log


stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_test \
    --config_path src/lerobot/async_inference/config/${benchmark_robot_type}/sync_smolvla_client.yaml \
    --task="Pick up the yellow cube and put it into the box." \
    --env_task=${env_task} \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --results_dir=${log_root}/results \
    --timing_output_dir=${log_root}/client_timing \
    --save_video=True \
    --video_camera=image \
    --video_dir=${log_root}/videos \
    --queue_size_monitor_path=${log_root}/queue.png \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0 \
    --episodes_per_task=10 \
    2>&1 | tee ${log_root}/client_$(date +%Y%m%d_%H%M%S).log
```
####  RTC


## SO101
20hz, too fast
export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b

20hz, smooth:
export pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.4basev2.2_abs_nofreeze_8b


# Async
## Libero

## SO101