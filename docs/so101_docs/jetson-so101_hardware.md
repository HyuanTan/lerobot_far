# Install jetson-containers
https://github.com/dusty-nv/jetson-containers/tree/master

lerobot:
https://github.com/dusty-nv/jetson-containers/tree/master/packages/physicalAI/lerobot


# Usage with Real-World Robot (so101)
https://huggingface.co/docs/lerobot/so101
## Before starting the container : Set udev rule
### Set udev rule for motor

On Jetson host side, we set an udev rule so that arms always get assigned the same device name as following.

- `/dev/ttyACM_so101leader`   : Leader arm
- `/dev/ttyACM_so101follower` : Follower arm

First only connect the leader arm to Jetson and record the serial ID by running the following:

```bash
ll /dev/serial/by-id/

udevadm info -a -n /dev/ttyACM0
udevadm info -a -n /dev/ttyACM1 | grep serial

udevadm info -a -n /dev/ttyACM1
udevadm info -a -n /dev/ttyACM1 | grep serial

sudo touch /etc/udev/rules.d/99-so101.rules


SUBSYSTEM=="tty", ATTRS{serial}=="5AB9067356", SYMLINK+="ttyACM_so101leader"
SUBSYSTEM=="tty", ATTRS{serial}=="5AB9067974", SYMLINK+="ttyACM_so101follower"


sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Set udev rule for camera
web camera 没有唯一的 serial,idVendor, idProduct, product 和 KERNELS
不同 USB 口进行绑定
```bash
udevadm info -q property -n /dev/video0 | grep -E 'ID_PATH|DEVPATH|ID_SERIAL'
udevadm info -q property -n /dev/video3 | grep -E 'ID_PATH|DEVPATH|ID_SERIAL'

udevadm info -a -n /dev/video4 | grep -E 'idVendor|idProduct|serial'
```

sudo touch /etc/udev/rules.d/99-webcam.rules
```
# Front camera (USB port 2.1)
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.1:1.0", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videotop", MODE="0666"

# Top camera (USB port 2.4)
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.4:1.0", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videofront", MODE="0666"

# Wrist camera
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:1.3:1.0", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videowrist", MODE="0666"
```

SUBSYSTEM=="video4linux", ENV{ID_VENDOR_ID}=="0bda", ENV{ID_MODEL_ID}=="5883", ENV{ID_SERIAL_SHORT}=="YHTek", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videowrist", MODE="0666"


```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

ll /dev/videotop
ll /dev/videofront

sudo apt install v4l-utils
sudo apt install ffmpeg
v4l2-ctl --version
v4l2-ctl --list-devices

# 查看分辨率 / 帧率
v4l2-ctl --device=/dev/video2 --list-formats-ext

# ffplay -f v4l2 -video_size 1280x1024 -framerate 30 /dev/videotop
ffplay /dev/videotop
ffplay /dev/videofront
ffplay /dev/videowrist

```
```bash
ffplay -f v4l2 \
  -input_format mjpeg \
  -video_size 1920x1080 \
  -framerate 30 \
  /dev/videofront
  
ffplay -f v4l2 \
  -input_format mjpeg \
  -video_size 1920x1080 \
  -framerate 30 \
  /dev/videotop


ffplay -f v4l2 \
  -input_format mjpeg \
  -video_size 1280x1024 \
  -framerate 30 \
  /dev/videowrist

ffplay -f v4l2 \
  -input_format mjpeg \
  -video_size 640x480 \
  -framerate 30 \
  /dev/video4
```


## Use container for Lerobot


```bash
jetson-containers run -it \
  --name lerobot_so101 \
  -v /data/code/lerobot:/opt/lerobot \
  -w /opt/lerobot \
  dustynv/lerobot:r36.4.0-cu128-24.04 

# sudo chown -R $USER:$USER /data/code/lerobot
# curl -LsSf https://astral.sh/uv/install.sh | sh
# Install dependencies
# uv venv --system-site-packages
# uv sync
# uv pip install "numpy>=2.0.0,<2.3.0" "opencv-python-headless>=4.10,<4.12"
# uv pip install -e . --no-deps --no-build-isolation
# source .venv/bin/activate

# 安装到container的系统：/usr/local/bin/python3
# uv pip install --system --break-system-packages -e . --no-deps
# uv pip install --system --break-system-packages -e . --no-build-isolation
# /opt/venv/bin/python3 -m pip install -e .

cd /opt/lerobot

export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
env | grep -i pip

# container 里面很多依赖安装在 /opt/venv/lib/python3.12/site-packages
/opt/venv/bin/python3 -m pip install -U pip setuptools wheel
/opt/venv/bin/python3 -m pip install -e .  --no-build-isolation


docker exec -it lerobot_so101 /bin/bash
```

验证：
```bash
command -v lerobot-find-port
python3 -c "import lerobot; print(lerobot.__file__)"
python3 -c "import sysconfig; print(sysconfig.get_path('scripts'))"
python3 -c "import cv2; print(cv2.__file__)"
python3 -c "import transformers ; print(transformers.__version__)"
python3 -c "import huggingface_hub ; print(huggingface_hub.__version__)"


/opt/venv/bin/python3 -m pip show lerobot
/opt/venv/bin/python3 -m pip show transformers
```

huoyuan@sensai:~/VLA/LeRobot/lerobot$ uv run python3 -c "import huggingface_hub ; print(huggingface_hub.__version__)"
1.7.1
huoyuan@sensai:~/VLA/LeRobot/lerobot$ uv run python3 -c "import transformers ; print(transformers.__version__)"
5.3.0
huoyuan@sensai:~/VLA/LeRobot/lerobot$

Then, start the docker container to run the visualization script.

```bash
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

新版本lerobot-calibrate import 链条长，会导入policy transformer 包，导致版本不兼容

### Calibration mortor 

https://huggingface.co/docs/lerobot/il_robots

/data/hf/lerobot/calibration/robots/so_follower/cse_so101follower.json

```
sudo chmod 666 /dev/ttyACM_so101follower
sudo chmod 666 /dev/ttyACM_so101leader

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader
    
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=cse_so101follower

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=cse_so101_leader
    
```

### Check camera 
```
lerobot-find-cameras opencv
```

# Recored
## Teleoperate

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=cse_so101follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=cse_so101_leader 
```
## Teleoperate with cameras
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --robot.cameras="{ images.front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, images.top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader \
    --display_data=true
    

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=cse_so101follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=cse_so101_leader \
    --display_data=true
```

 ## Example: Visualize datasets
 On Docker host (Jetson native), first launch rerun.io (check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#visualize-datasets))

```bash
jetson-containers run -it \
--name lerobot_so101 \
dustynv/lerobot:r36.4.0-cu128-24.04 
 
pip install rerun-sdk -i https://pypi.org/simple


rerun
```
## Motor
### Acceleration / Goal_Velocity 对采集与 Inference 的影响
**当前初始化写入的寄存器:**
onfigure() → configure_motors() 在每次 connect 时写入：

寄存器	addr	写入值	说明
Return_Delay_Time	9	0	响应延迟最小 2μs
Maximum_Acceleration	85	254	最大加速度上限（仅 Protocol 0）
Acceleration	41	254	加速度曲线，最大值
Operating_Mode	33	0	位置伺服模式
P/I/D_Coefficient	—	16/0/32	PID 参数
Goal_Velocity (addr 46) 和 Goal_Time (addr 44) 未被写入 — 保留电机 EEPROM 中的值（STS3215 出厂默认 Goal_Velocity=0，含义是不限速）。

**STS3215 位置控制的速度决定逻辑:**
Goal_Time=0, Goal_Velocity=0   → 全速运动，由 Acceleration 控制加速曲线
Goal_Time=0, Goal_Velocity=V   → 速度上限为 V，由 Acceleration 控制加速曲线
Goal_Time=T,  Goal_Velocity=*  → 时间控制：在 T ms 内到达目标（自动计算速度）


**Acceleration 改变的影响:**
Acceleration=254（当前最大值）→ 电机从静止到最大速度的时间极短，响应最激进。

降低 Acceleration 后：

加减速曲线变平滑，电机实际位置与 Goal_Position 的跟踪延迟增加
采集时：人类遥操作的运动轨迹在数据集里变得更平滑，但实际关节位置滞后于指令
Inference 时：Policy 下发的 Goal_Position 执行更慢，同样时间内移动距离减少
最关键的问题：采集和 Inference 的 Acceleration 必须一致。 否则 policy 学到的 action → outcome 映射在 inference 时完全失效（distribution shift）。

**调整 Goal_Velocity 的影响**

Goal_Velocity	行为
0（默认）	不限速，电机尽可能快地以 Acceleration 曲线运动
设置为 V	速度硬上限为 V（即使 Acceleration 允许更快也不超过 V）

具体效果：
运动一致性提升：设置 Goal_Velocity 后，所有关节的最大速度被统一约束，大幅度运动和小幅度运动的速度曲线更可预测，减少 policy 训练时的 action-noise

过冲/振荡减少：高 P 增益 + 不限速容易造成目标附近的来回振荡；限速后阻尼效果更好

对电流/负载反馈的影响：限速 → 加速度峰值降低 → Present_Current 和 Present_Load 的峰值显著减小 → 如果用这些值作为 observation，其分布会改变

对 Gripper 影响最大：Gripper 是纯弹性夹持，不限速时电流冲击最大，设置 Goal_Velocity 可以保护电机并让夹持力更均匀

采集和 Inference 必须保证 Acceleration + Goal_Velocity 完全一致，否则相当于在不同动力学特性的机器人上采集数据后跑 inference。建议：

如需修改，在 configure() 中显式写入 Goal_Velocity（而不是依赖 EEPROM 默认值），确保每次 connect 后状态可重复
记录在实验文档里（或加进 SOFollowerConfig），避免不同场次参数漂移