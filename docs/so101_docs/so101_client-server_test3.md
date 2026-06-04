
# 注意力可视化
## Visulize SmolVLA
```bash
# Replace policy_server with attn_policy_server:
stdbuf -oL -eL uv run python -m lerobot.async_inference.attn_policy_server \
    --host=localhost --port=8080 --fps=30 \
    --attn_save_every_n=3 \
    --attn_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/smolvla_attention/${env_task}/server_attention \
    2>&1 | tee ${log_path}/server_attention_$(date +%Y%m%d_%H%M%S).log

run_libero_test.py ...
```


```bash
libero_goal
libero_object
libero_spatial
libero_10



cd ~/VLA/LeRobot/lerobot_v0.5.2

export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=HuggingFaceVLA/smolvla_libero
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=libero_10
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
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

## Visulize PI05
```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2

export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=libero_10
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

stdbuf -oL -eL uv run python -m lerobot.async_inference.sim_test.run_libero_pi05_feat_test \
    --env_task=${env_task} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda \
    --episodes_per_task=1 \
    --feat_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/pi05_features/${env_task} \
    --feat_save_every_n=5 \
    2>&1 | tee ${log_path}/client_attention_$(date +%Y%m%d_%H%M%S).log

```

## Integerate
```bash
export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=HuggingFaceVLA/smolvla_libero
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=libero_object
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

# SmolVLA — 跨注意力热力图
uv run python -m lerobot.async_inference.sim_test.run_libero_vis_test \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --env_task=${env_task} \
    --enable_attn_vis=true \
    --attn_output_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/smolvla_attention/${env_task}/attention \
    --attn_save_every_n=3 \
    --save_video=true \
    --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/smolvla_attention/${env_task}/videos \
    --video_camera=agentview_image \
    --video_fps=30


export CUDA_VISIBLE_DEVICES=0
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=libero_object
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}

# PI05 — lang→image + action→image 热力图 + episode 漂移图
uv run python -m lerobot.async_inference.sim_test.run_libero_vis_test \
    --policy_type=${model_type} \
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

# 任意模型 — 纯评测，零可视化开销
python -m lerobot.async_inference.sim_test.run_libero_vis_test \
    --policy_type=smolvla \
    --enable_attn_vis=false

```
--video_camera：指定录制哪个摄像头（agentview_image 或 robot0_eye_in_hand_image），默认 agentview_image
帧收集在 _run_episode_inprocess 内每个 env step 完成，episode 结束后立即写入 mp4
文件名格式：ep0000_success.mp4 / ep0001_failed.mp4
使用 imageio.v3 + libx264，与 sim_client.py 完全一致
视频录制与注意力可视化完全独立，可以任意组合开关

# Client-Server visulize
新增方法：_submit_smolvla() 和 _submit_pi05() — 各自处理 token layout、图像提取和可视化保存，均在后台线程异步执行，不阻塞 gRPC 响应路径

```bash
# 1. 启动注意力服务器（支持 smolvla 或 pi05，自动根据客户端 policy_type 分派）
python -m lerobot.async_inference.attn_policy_server \
    --host=127.0.0.1 --port=8080 --fps=10 \
    --attn_output_dir=/tmp/pi05_attn \
    --attn_save_every_n=3

# 2. 客户端完全不变
python -m lerobot.async_inference.sim_test.run_libero_test \
    --policy_type=pi05 \
    --pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044 \
    --server_address=localhost:8080 \
    --env_task=libero_spatial

```
--attn_save_every_n=3 控制每隔多少次 inference 调用保存一次可视化图片。

具体来说：

=1：每次 inference 都保存（默认值）
=3：每 3 次 inference 保存一次（第 0、3、6、9… 次）
=5：每 5 次 inference 保存一次
为什么需要这个参数？

PI05/SmolVLA 每次推理产生一个 action chunk（通常 16 步动作），但新的 inference 请求频率取决于 actions_per_chunk 和 fps。例如 fps=10、chunk_size=16 时，机器人每 ~1.6 秒请求一次新 chunk。如果每次都保存可视化（=1），磁盘写入和 matplotlib 渲染会产生额外开销：

SmolVLA：保存约 3 张 PNG/inference，每张约 200-500KB
PI05：保存约 3-6 张 PNG/inference
设置 =3 表示每 3 次 inference 保存一次，减少 IO 和 CPU 开销约 3x，同时仍能观察到注意力随时间的演变趋势。对于长时间评测（几百步的 episode），=3 到 =10 是合理的权衡值。