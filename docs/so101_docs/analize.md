

```bash
apt update
apt install python3-tk

# 单文件
python src/lerobot/async_inference/analyze_trajectory.py \
    trajectories/run1/episode_0000_20260511_120000.json

python src/lerobot/async_inference/analyze_trajectory.py --no-show \
    ./outputs/eval/so101_V2.0_0511/pi05/so101_sm_10rtc_imag-crop_lift_retry_10hz/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b/trajectories/episode_0001_20260511_213428.json

# 整个目录（多 episode 叠加到同一 3D 图）
python src/lerobot/async_inference/analyze_trajectory.py outputs/eval/so101/pi05/so101_sm_20rtc_imag-crop_lift_retry_10hz/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b/trajectories/

# 保存图片
python src/lerobot/async_inference/analyze_trajectory.py trajectories/run1/ --out traj.png

```


configs.py — 新增 2 个字段到 RobotClientConfig（SmartRobotClientConfig 自动继承）：

record_trajectory: bool = False
trajectory_output_dir: str = "trajectories"
base_client.py — 最小化侵入：

新增 _on_chunk_received() 空虚方法
在 receive_actions() 的 chunk 到达后调用它（1 行）
robot_client.py — 主要逻辑：

TrajectoryRecorder 类（线程安全，带 threading.Lock）
_reset_loop_state() override → 每个 task 自动新建文件
_on_chunk_received() override → 记录收到的 chunk
control_loop_action() → 记录真正执行的 action，并区分 raw/插值
stop() → flush 最后一个 episode 到磁盘


----
1. TASK_DONE episode 自动切分 ✅
smart_robot_client.py — 在两处 TASK_DONE 确认点（SM 路径 + backup 路径）均加入：


self._on_task_done()   # flush trajectory + open new episode file
robot_client.py — _on_task_done() 实现调用 TrajectoryRecorder.next_episode()，内部维护 _current_ep 自增，确保每次 TASK_DONE 都能写出独立文件，命名如 episode_0001_20260511_120015.json。

2. 记录 feedback state（实际 joint 位姿）✅
base_client.py — 新增 _on_obs_captured(raw_obs) 虚方法，在 control_loop_observation() 的 _capture_raw_obs() 调用之后立即触发（不影响原有流程）。

robot_client.py — _on_obs_captured() override 提取 raw_obs 中所有 .pos 键（跳过 camera 数组和 task string），缓存到 self._last_feedback_state。每次 control_loop_action() 执行时随 executed 记录一并写入 JSON
----

--record_trajectory true
--trajectory_output_dir trajectories


蓝色系渐变线 = 各 chunk 的完整规划轨迹（越新越深）；红色实线 = 真正执行的轨迹；橙色虚线 = 插值子步骤（interpolation_multiplier > 1 时）；绿点=起点，红 X=终点。

绿色实线：从每个 executed 记录的 feedback_state 提取关节位姿，经 FK 转换为 EE (x,y,z)，用绿色绘制（表示机器人真实位置，区别于命令位置的红色）
自动保存：始终保存图片，无需 --out：
单文件输入 → {same_dir}/{filename}_trajectory.png
目录或多文件 → {first_input_dir}/trajectory_plot.png
可用 --out 覆盖路径；--no-show 跳过交互弹窗（SSH 环境）