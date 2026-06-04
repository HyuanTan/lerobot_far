```
lsb_release -a
cat /etc/nv_tegra_release
```

```bash
sudo nvpmodel -m 0
sudo jetson_clocks

tegrastats

# 如果占用
pkill -f lerobot-teleoperate

hf login
```

电机id无法识别：两个motorbus断电超过3s, 使用softlink

```
jetson-containers run -it \
  --name lerobot_so101_v0.4.4 \
  -v /data/code/lerobot_v0.4.4:/opt/lerobot \
  -v /data/hf:/data/hf \
  -e HF_HOME=/data/hf \
  -w /opt/lerobot \
  $(autotag lerobot)

docker exec -it lerobot_so101_v0.4.4 /bin/bash

export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
env | grep -i pip
export HF_HOME=/data/hf
export HF_USER=HollyTan

/opt/venv/bin/python3 -m pip install -U pip setuptools wheel
/opt/venv/bin/python3 -m pip install -e .  --no-build-isolation

/opt/venv/bin/python3 -m pip install "pynput>=1.7.7,<1.9.0"
/opt/venv/bin/python3 -m pip install "num2words>=0.5.14,<0.6.0", "accelerate>=1.7.0,<2.0.0", "safetensors>=0.4.3,<1.0.0"
/opt/venv/bin/python3 -m pip install \
  "transformers>=4.57.1,<5.0.0" \
  "accelerate>=1.7.0,<2.0.0" \
  "safetensors>=0.4.3,<1.0.0" \
  "num2words>=0.5.14,<0.6.0"

smolvla = ["lerobot[transformers-dep]", "num2words>=0.5.14,<0.6.0", "accelerate>=1.7.0,<2.0.0", "safetensors>=0.4.3,<1.0.0"]



python3 -m pip show torch torchvision torchaudio transformers
python -c "import sys, torch, torchvision, transformers, cv2; print(sys.version); print(torch.__version__); print(torchvision.__version__); print(transformers.__version__); print(cv2.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```


## Wundows Rerun
```
pip install rerun-sdk
rerun
```

修改 rerun 连接的ip:
`code/lerobot_v0.4.4/src/lerobot/utils/visualization_utils.py`, `init_rerun`
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=cse_so101follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=cse_so101_leader \
    --display_data=true \
    --display_compressed_images=true \
    --display_ip=192.168.1.225 \
    --display_port=9876
    

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader \
    --display_data=true \
    --display_compressed_images=true \
    --display_image_interval_s=0.5 \
    --robot.record_motor_state=["gripper"]
    
    
    
    --display_ip=192.168.1.225 \
    --display_port=9876
```

front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}, 

# Record a dataset
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
```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

# hf auth login --token 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

HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

```bash
rm -rf /data/hf/lerobot/so101_pick-place-v2.4

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --robot.cameras="{
        top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
        wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
        front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}
        }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader \
    --dataset.root=/data/hf/lerobot/${HF_USER}/so101_pick-place-v2.4 \
    --dataset.repo_id=${HF_USER}/so101_pick-place-v2.4 \
    --dataset.push_to_hub=False \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --dataset.fps=20 \
    --display_data=false \
    --play_sounds=false \
    --display_async=true \
    --display_image_interval_s=0.2 \
    --display_worker_poll_interval_s=0.2 \
    --dataset.reset_time_s=8 \
    --dataset.num_episodes=20 \
    --resume=true \
    --dataset.single_task="Pick up the yellow cube and place it in the box."

red
yellow
cube in natural wood
green
orange

cuboid

--resume=true
num_episodes 写“新增数量”

```
默认60s
--dataset.episode_time_s
--dataset.reset_time_s

--display_compressed_images=false \
--dataset.encoder_queue_maxsize=15 \
--dataset.vcodec=auto \
    
    --robot.calibration_dir=/data/models/huggingface/lerobot/calibration/robots/so_follower/cse_so101follower.json \
    --teleop.calibration_dir=/data/models/huggingface/lerobot/calibration/teleoperators/so_leader/cse_so101_leader.json \
    --display_compressed_images=true \
    --display_ip=192.168.1.225 \
    --display_port=9876

--robot.cameras="{
images.front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200},
images.top: {type: opencv, index_or_path: '/dev/videotop', width: 640, height: 480, fps: 30, backend: 200},
images.wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 1280, height: 1024, fps: 30, backend: 200}
}" \

backend: 200, 对应的就是 cv2.CAP_V4L2

--dataset.vcodec=auto
--dataset.vcodec=h264
libsvtav1（默认）
    - 优点：压缩率最高（最省空间），质量最好（同等码率下）
    - 缺点：CPU 非常重（Jetson杀手），延迟高
    - 会导致：teleop卡顿，encoder queue 堆积
h264:非常稳定，硬件支持好（Jetson友好），编码快，实时性强
缺点：压缩率一般，同质量文件更大

auto: 自动选最优（通常走硬件 encoder）,在 Jetson 上通常更优, 在 Jetson 上通常比 h264 还好,最推荐用于数据采集

默认30
--dataset.encoder_queue_maxsize


push your local dataset to the Hub manually, running:
```bash
hf upload HollyTan/so101_pick-place-v2.4 /data/hf/lerobot/HollyTan/so101_pick-place-v2.4 --repo-type dataset
```
## Replay an episode
```
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --dataset.repo_id=${HF_USER}/so101_pick-place \
    --dataset.root=/data/hf/lerobot/${HF_USER}/so101_pick-place \
    --dataset.fps=10 \
    --play_sounds=false \
    --dataset.episode=0

```
--dataset.episode=0 # choose the episode you want to replay

```BASH
lerobot-dataset-viz \
  --repo-id=${HF_USER}/so101_pick-place \
  --root=/data/hf/lerobot/${HF_USER}/so101_pick-place \
  --episode-index=0

```
# Update notes
`code/lerobot_v0.4.4/src/lerobot/robots/so_follower/so_follower.py`:
`cam.read_latest()`

`src/lerobot/scripts/lerobot_teleoperate.py`: teleop_loop，降低rerun 显示频率，控制 log_rerun_data(...) 的调用频率，而不是每个 control loop 都发一次：图像按时间降频，不降 action/state， 添加参数`display_image_interval_s`

`src/lerobot/utils/visualization_utils.py`，log_rerun_data

`src/lerobot/scripts/lerobot_record.py`:
主循环只负责采集、控制、写盘
Rerun 显示拆到后台线程
图像走 latest-only
action/state 不降频
图像按时间降频
主循环提交显示数据时永不阻塞

添加参数：
display_async=True
display_image_interval_s=0.2

------
record_motor_state=[]（默认）：1 次总线读取，Rerun 6 个标量/帧，与原来完全一致。
record_motor_state=["gripper"]：

1 次 sync_read("Present_Position")（全部 6 个电机）
1 次 sync_read_motor_state(["gripper"])（仅 gripper：块读 + current，共 2 次总线）
Rerun 发送 6（pos）+ 5（gripper 扩展字段）= 11 个标量/帧，远少于之前的 36 个
CLI 用法：
--robot.record_motor_state="[gripper]"
或多个电机:
--robot.record_motor_state="[gripper, wrist_roll]"

record/teleoperate 添加以下电机反馈
    "Present_Velocity":    "vel",
    "Present_Load":        "load",
    "Present_Voltage":     "voltage",
    "Present_Temperature": "temp",

addr 56-57: Present_Position  (2B)  ┐
addr 58-59: Present_Velocity  (2B)  │  8字节连续块 ✓
addr 60-61: Present_Load      (2B)  │
addr 62:    Present_Voltage   (1B)  │
addr 63:    Present_Temperature(1B) ┘
addr 64:    ← 未定义，停在这里

---- Update
record_motor_state=[]（默认）:
  └─ sync_read("Present_Position")  ← 全部6个电机，1次总线
  └─ Rerun: 6个标量/帧 (6×pos)

record_motor_state=["gripper"]:
  ├─ sync_read("Present_Position")   ← 全部6个电机，1次总线 ✓（保留）
  ├─ sync_read("Present_Load",    ["gripper"])  ← 仅gripper，1次总线
  └─ sync_read("Present_Current", ["gripper"])  ← 仅gripper，1次总线
  └─ Rerun: 6(pos) + 2(gripper.load + gripper.current) = 8个标量/帧

---- Update
默认从 2 次 feedback 寄存器读取（Load + Current）变为 1 次（仅 Load），加上原有的 Position 读取，总共 2 次 sync_read（之前是 3 次），节省约 1-2ms/帧。

若将来需要 Current（如电流过载检测），可通过命令行参数临时恢复：
--gripper_sm_feedback_registers="{'Present_Load': 'load', 'Present_Current': 'current'}"


## Local test
```bash
rm -rf /data/hf/lerobot/HollyTan/eval_so101_pick-place

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM_so101follower \
  --robot.id=cse_so101follower \
  --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}}" \
  --display_data=true \
  --dataset.repo_id=HollyTan/eval_so101_pick-place \
  --dataset.single_task="Grab the red cube and drop to the yellow sticker." \
  --dataset.push_to_hub=false \
  --dataset.fps=10 \
  --policy.path=HollyTan/so101_smolvla_pick_place \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10000 \
  --dataset.reset_time_s=6 \
  --play_sounds=false \
  --display_async=true \
  --display_image_interval_s=0.2 \
  --display_worker_poll_interval_s=0.1
```



```bash
rm -rf /data/hf/lerobot/HollyTan/eval_smolvla_pick-place_v2.4

HollyTan/so101_smolvla_pick_place-v2.0-40k
HollyTan/so101_smolvla_pick_place-v2.0-30k
HollyTan/so101_smolvla_pick_place-v2.0-25k

jadenovalight/smolvla_pick-place_v2.4

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM_so101follower \
  --robot.id=cse_so101follower \
  --robot.cameras="{
    top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
    wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
    front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}
    }" \
  --display_data=true \
  --dataset.repo_id=HollyTan/eval_smolvla_pick-place_v2.4 \
  --dataset.single_task="Pick up the yellow cube and place it in the box." \
  --dataset.push_to_hub=false \
  --dataset.fps=20 \
  --policy.path=jadenovalight/smolvla_pick-place_v2.4 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10000 \
  --dataset.reset_time_s=6 \
  --play_sounds=false \
  --display_async=true \
  --display_image_interval_s=0.2 \
  --display_worker_poll_interval_s=0.1
```
