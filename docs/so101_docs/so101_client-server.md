```bash
ssh -L 8080:sensai:8080 cid@minerva.cse.chalmers.se

ssh -J cid@minerva.cse.chalmers.se -L 8080:127.0.0.1:8080 cid@sensai
```

## 启动 Policy Server
```bash
cd ~/VLA/LeRobot/lerobot

uv run python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=20 \
    --inference_latency=0.020 \
    --obs_queue_timeout=1
 
```

参数	默认值	说明
--host	127.0.0.1	gRPC 服务器监听地址；0.0.0.0 允许外部连接
--port	8080	gRPC 服务器端口
--fps	30	控制循环目标帧率
--inference_latency	0.033	推理延迟目标（秒）；服务器会sleep以达到此目标
--obs_queue_timeout	1	等待观察的超时时间（秒）

## 启动 SO101 Client
### jestson
```
jetson-containers run -it \
  --name lerobot_so101_v0.5.1 \
  -v /data/code/lerobot_server:/opt/lerobot \
  -v /data/hf:/data/hf \
  -e HF_HOME=/data/hf \
  -w /opt/lerobot \
  $(autotag lerobot)


docker exec -it lerobot_so101_v0.5.1 /bin/bash

export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
env | grep -i pip
export HF_HOME=/data/hf
export HF_USER=HollyTan

/opt/venv/bin/python3 -m pip install -U pip setuptools wheel
/opt/venv/bin/python3 -m pip install -e .  --no-build-isolation
```
### 启动 SO101 Client


HollyTan/so101_smolvla_pick_place
```BASH
python -m src/lerobot/async_inference/robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --task="Grab the red cube and put it on the yellow sticker." \
    --server_address=127.0.0.1:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=HollyTan/so101_smolvla_pick_place \
    --policy_device=cuda \
    --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
--aggregate_fn_name=weighted_average \

HollyTan/so101_pi05_pick-place-v2.0 
```BASH

```

## Using newest lerobot version-5.2
## Server
```
uv venv --python 3.12
uv pip install -U pip setuptools wheel
uv pip install -e .

uv pip install -e ".[libero]"
uv pip install -e ".[pi]"
uv pip install -e ".[smolvla]"
uv pip install -e ".[async]"

```

```bash
 
# 在client登录
-f → 后台
-N → 不执行命令

ssh -J huoyuan@minerva.cse.chalmers.se huoyuan@sensai -L 8080:127.0.0.1:8080


ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'

cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=3
uv run python -m lerobot.async_inference.policy_server \
--host=127.0.0.1 \
--fps=30 \
--port=8080

# 测试
nc -vz 127.0.0.1 8080


```
 
### jestson
```bash
jetson-containers run -it \
  --name lerobot_so101_v0.5.2 \
  -v /data/code/lerobot_v0.5.2:/opt/lerobot \
  -v /data/hf:/data/hf \
  -e HF_HOME=/data/hf \
  -w /opt/lerobot \
  $(autotag lerobot)


docker exec -it lerobot_so101_v0.5.2 /bin/bash

export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
env | grep -i pip
export HF_HOME=/data/hf
export HF_USER=HollyTan

/opt/venv/bin/python3 -m pip install -U pip setuptools wheel
/opt/venv/bin/python3 -m pip install -e .  --no-build-isolation

/opt/venv/bin/python3 -m pip install "deepdiff>=7.0.1,<9.0.0"
/opt/venv/bin/python3 -m pip install "placo>=0.9.6,<0.9.17"

/opt/venv/bin/python -m pip uninstall -y placo cmeel-urdfdom cmeel-console-bridge cmeel-tinyxml2
/opt/venv/bin/python -m pip install \
  "placo==0.9.16" \
  "cmeel-urdfdom==4.0.1" \
  "cmeel-console-bridge==1.0.2.3" \
  "cmeel-tinyxml2==10.0.0.0"

```
### client
HollyTan/so101_smolvla_pick_place
```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --task="Grab the red cube and put it on the yellow sticker." \
    --server_address=127.0.0.1:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=HollyTan/so101_smolvla_pick_place \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --debug_visualize_queue_size=True
```
--aggregate_fn_name=weighted_average \

------

```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2
CUDA_LAUNCH_BLOCKING=3 \
uv run python -m lerobot.async_inference.policy_server \
--host=127.0.0.1 \
--fps=10 \
--port=8080
```


jadenovalight/master-thesis
```BASH
python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --task="Pick up the red cube and put it into the orange box." \
    --server_address=127.0.0.1:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=jadenovalight/master-thesis \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --fps=10 \
    --interpolation_multiplier=2 \
    --rtc_execution_horizon=20 \
    --debug_visualize_queue_size=True
```

------------------
    
HollyTan/pi05_so101_pick_place-v2.0_subset_50eps_20k

-- 找到下载模型缓存的地方，修改`config.json`文件中的参数为 "compile_model": false, 原因： CUDA graph 初始化后被修改----推理速度下降约 20-40%?

-- 在 configuration_pi05.py 中改 compile_mode
compile_mode: str = "reduce-overhead"   # 只用 CUDA graph 减少 Python overhead，不做 autotuning
-- 或 compile_mode: str = "default"           # 最保守，较少使用 CUDA graph

compile_model=false（当前，稳定）
    → 验证 libero_object/libero_10 基础 SR 后
    → 尝试 compile_mode="reduce-overhead"，观察是否复现 CUDA Graph 错误
    → 若稳定，保持；若仍崩溃，退回 false



```
服务器参数（不变）:
  --fps                  推理帧率（决定 environment_dt）
  --inference_latency    模拟最低推理延迟（0.0 = 不限速）
  --obs_queue_timeout    等待观测超时（秒）

客户端新增参数:
  --interpolation_multiplier=N   控制频率倍率（1=关闭, 2/3=推荐）
  --rtc_execution_horizon=N      RTC 重规划窗口（0=关闭, 20=推荐）
                                 只对 smolvla/pi0/pi05 生效

```

```bash
# 时钟同步
# timedatectl status

cd ~/VLA/LeRobot/lerobot_v0.5.2
export CUDA_VISIBLE_DEVICES=3
uv run python -m lerobot.async_inference.policy_server \
--host=127.0.0.1 \
--fps=10 \
--port=8080
```

```BASH
python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --task="Pick up the red cube and put it into the orange box." \
    --server_address=127.0.0.1:8080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=HollyTan/pi05_so101_pick_place-v2.0_subset_50eps_20k \
    --policy_device=cuda \
    --client_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --fps=10 \
    --interpolation_multiplier=3 \
    --rtc_execution_horizon=20 \
    --debug_visualize_queue_size=True
```
--aggregate_fn_name=weighted_average \
--actions_per_chunk=50 \
--chunk_size_threshold=0.5 \

服务器和客户端的`--fps=10` 需要保持一致,最终控制频率=fps*interpolation_multiplier
### Local Test

```bash
rm -rf /data/hf/lerobot/HollyTan/eval_so101_pick-place

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM_so101follower \
  --robot.id=cse_so101follower \
  --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}}" \
  --dataset.repo_id=HollyTan/eval_so101_pick-place \
  --dataset.single_task="Grab the red cube and drop to the yellow sticker." \
  --dataset.push_to_hub=false \
  --policy.path=HollyTan/so101_smolvla_pick_place \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10000 \
  --dataset.reset_time_s=6 \
  --play_sounds=false \
  --display_async=true \
  --display_image_interval_s=0.2 \
  --display_worker_poll_interval_s=0.1 \
  --display_data=true
  
```

------------
jadenovalight/master-thesis
```bash
rm -rf /data/hf/lerobot/HollyTan/eval_so101_pick-place

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM_so101follower \
  --robot.id=cse_so101follower \
  --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
  --dataset.repo_id=HollyTan/eval_so101_pick-place \
  --dataset.single_task="Pick up the red cube and put it into the orange box." \
  --dataset.push_to_hub=false \
  --policy.path=jadenovalight/master-thesis \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10000 \
  --dataset.reset_time_s=6 \
  --play_sounds=false \
  --display_async=true \
  --display_image_interval_s=0.2 \
  --display_worker_poll_interval_s=0.1 \
  --display_data=true
```


### RTC TEST
jadenovalight/master-thesis
```bash
    # Run RTC with Real robot with RTC
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=jadenovalight/master-thesis \
        --policy.device=cuda \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyACM_so101follower \
        --robot.id=cse_so101follower \
        --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
        --task="Pick up the red cube and put it into the orange box." \
        --rtc.enabled=true \
        --duration=120

    # Run RTC with Real robot without RTC
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=jadenovalight/master-thesis \
        --policy.device=cuda \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyACM_so101follower \
        --robot.id=cse_so101follower \
        --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
        --task="Pick up the red cube and put it into the orange box." \
        --duration=120 \
        --rtc.enabled=false
```

HollyTan/pi05_so101_pick_place-v2.0_subset_50eps_10k
```bash
    # Run RTC with Real robot with pi0.5 policy
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=HollyTan/pi05_so101_pick_place-v2.0_subset_50eps_10k \
        --policy.device=cuda \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyACM_so101follower \
        --robot.id=cse_so101follower \
        --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
        --task="Pick up the red cube and put it into the orange box." \
        --duration=120 \
        --rtc.enabled=false
```
