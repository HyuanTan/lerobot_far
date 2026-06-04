# Multi-Candidate Action-Chunk Selection — SO-101 Real Robot

SO-101 迁移版本：继承 `SmartRobotClient`（Feetech load+pos SM），复用
`multi_candidate_server.py`（硬件无关，action tensor 操作）。

对应脚本：`lerobot.async_inference.run_so101_multicand_client`

---

## 环境变量（所有命令共用）

```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2

export CUDA_VISIBLE_DEVICES=0
export pretrained_name_or_path=/path/to/your/model   # 本地路径或 HF Hub ID
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla                              # smolvla / pi05 / act ...
export robot_type=so101_follower
export robot_port=/dev/ttyUSB0
export task="pick the red block and place it in the bin"
export PYTHONUNBUFFERED=1

export log_root=./outputs/eval/so101/${model_type}/mc_sm
mkdir -p ${log_root}
```

---

## 场景 1：服务器最优 top_k=1（向后兼容，无客户端二次选择）

```bash
# 终端 1：MultiCandidatePolicyServer（服务器选最优）
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --fps=10 \
    --n_candidates=8 \
    --top_k=1 \
    --delay_delta=1 \
    --data_collect_dir=${log_root}/mc_data \
    2>&1 | tee ${log_root}/server.log

# 终端 2：SO-101 Multi-Cand 客户端（Phase 1，server 已选最优）
uv run python -m lerobot.async_inference.run_so101_multicand_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
    --task="${task}" \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --server_address=127.0.0.1:8080 \
    --fps=10 \
    --actions_per_chunk=16 \
    --client_smooth_alpha=0.3 \
    --enable_gripper_sm=true \
    --gripper_load_grasp_threshold=150 \
    --gripper_pos_gap_threshold=7.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_pos_open_threshold=20.0 \
    --so101_gripper_open_deg=20.0 \
    --so101_gripper_empty_deg=8.0 \
    --max_reinfer_retries=2 \
    --max_empty_grasp_retries=6 \
    --record_trajectory=true \
    --trajectory_dir=${log_root}/mc_trajectories \
    --data_collect_dir=${log_root}/mc_data \
    --results_dir=${log_root}/mc_results \
    2>&1 | tee ${log_root}/client.log
```

---

## 场景 2：服务器 Top-K + 客户端二次选优（推荐）

```bash
# 终端 1：返回 top_k=2 候选，启用全候选记录
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --fps=10 \
    --n_candidates=8 \
    --top_k=2 \
    --delay_delta=1 \
    --w_jerk=1.0 --w_vel_peak=0.5 --w_consistency=0.3 \
    --record_all_candidates=true \
    --data_collect_dir=${log_root}/mc_data \
    2>&1 | tee ${log_root}/server.log

# 终端 2：Phase 2 — 客户端 continuity 二次排序 + SM
uv run python -m lerobot.async_inference.run_so101_multicand_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
    --task="${task}" \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --server_address=127.0.0.1:8080 \
    --fps=10 \
    --actions_per_chunk=16 \
    --interpolation_multiplier=3 \
    --client_smooth_alpha=0.4 \
    --server_score_normalize=softmax \
    --action_limit_min=-10.0 \
    --action_limit_max=310.0 \
    --enable_gripper_sm=true \
    --gripper_load_grasp_threshold=150 \
    --gripper_pos_gap_threshold=7.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_pos_open_threshold=20.0 \
    --so101_gripper_open_deg=20.0 \
    --so101_gripper_empty_deg=8.0 \
    --max_reinfer_retries=2 \
    --max_empty_grasp_retries=6 \
    --recovery_return_to_home=true \
    --recovery_home_steps=90 \
    --recovery_warmup_steps=25 \
    --recovery_smooth_steps=60 \
    --record_trajectory=true \
    --trajectory_dir=${log_root}/mc_trajectories \
    --data_collect_dir=${log_root}/mc_data \
    --results_dir=${log_root}/mc_results \
    --timing_output_dir=${log_root}/timing \
    2>&1 | tee ${log_root}/client.log
```

---

## 场景 3：完整配置（SM + LIFT_RETRY + REWIND_RETRY + 不确定性慢模式）

```bash

cd ~/VLA/LeRobot/lerobot_v0.5.2

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

# 终端 1：n_candidates=16，top_k=4
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --inference_latency=0.00 \
    --obs_queue_timeout=1 \
    --obs_similarity_atol=1.0  \
    --fps=20 \
    --n_candidates=2 \
    --top_k=2 \
    --delay_delta=0 \
    --w_consistency=0.0 \
    --log_level=DEBUG \
    --record_all_candidates=true \
    --data_collect_dir=${log_root}/mc_data \
    2>&1 | tee ${log_root}/server_$(date +%Y%m%d_%H%M%S).log


too much chunk, more delay time
--n_candidates=16 \
--top_k=1 \
--delay_delta=0 \

qukickly half go home
--n_candidates=8 \
--top_k=1 \
--delay_delta=1 \

some qukickly half go home
--n_candidates=4 \
--top_k=1 \
--delay_delta=0 \
    
纯噪声多样性（关闭 RTC delay 变体）
--delay_delta=0 \

RTC delay 变体
--delay_delta=1 \

--rtc_execution_horizon=20 ---> --rtc_execution_horizon=10

--empty_grasp_rewind_buffer_steps=160 \
--empty_grasp_rewind_steps=100 \
--empty_grasp_rewind_min_displacement_deg=40.0 \

--task_done_home_confirm_steps=1 \
--empty_grasp_rewind_warmup_steps=1 \

--recovery_warmup_steps=0 \
--empty_grasp_rewind_warmup_steps=0 \

--task_done_home_ee_tolerance_m=0.06 \

--recovery_smooth_steps=140 \

# 终端 2：全功能客户端
python -m lerobot.async_inference.run_so101_multicand_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --robot.id=cse_so101follower \
    --task="Pick up the yellow cube and put it into the box." \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --policy_device=cuda \
    --client_device=cuda \
    --actions_per_chunk=50 \
    --server_address=127.0.0.1:8080 \
    --fps=20 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only \
    --rtc_execution_horizon=10 \
    --interpolation_multiplier=1 \
    --client_smooth_alpha=0.4 \
    --server_score_normalize=softmax \
    --spread_uncertainty_threshold=0.15 \
    --spread_slow_alpha_scale=1.5 \
    --spread_slow_mode_window=5 \
    --action_limit_min=-10.0 \
    --action_limit_max=310.0 \
    --gripper_load_grasp_threshold=80.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_grasp_confirm_steps=0 \
    --max_empty_grasp_retries=3 \
    --enable_gripper_sm=true \
    --recovery_smooth_steps=155 \
    --empty_grasp_lift_retry_enabled=false \
    --empty_grasp_lift_delta_xyz_m="[0.0, 0.0, 0.12]" \
    --empty_grasp_lift_warmup_steps=10 \
    --empty_grasp_rewind_enabled=true \
    --empty_grasp_rewind_buffer_steps=150 \
    --empty_grasp_rewind_steps=120 \
    --empty_grasp_rewind_min_displacement_deg=50.0 \
    --empty_grasp_rewind_warmup_steps=1 \
    --obs_image_resize_hw="{top: [224, 224], wrist: [224, 224], front: [224, 224]}" \
    --obs_image_use_model_resize=true \
    --record_trajectory=true \
    --enable_recapture_home_positions=false \
    --task_done_home_check_mode=joint \
    --task_done_home_tolerance=8.0 \
    --task_done_home_ee_tolerance_m=0.01 \
    --task_done_home_ee_tolerance_xyz_m=[0.01,0.01,0.01] \
    --task_done_home_check_gripper=true \
    --task_done_home_gripper_tolerance_deg=1.0 \
    --task_done_home_confirm_steps=5 \
    --trajectory_output_dir=${log_root}/trajectories \
    --trajectory_dir=${log_root}/mc_trajectories \
    --data_collect_dir=${log_root}/mc_data \
    --results_dir=${log_root}/mc_results \
    --timing_output_dir=${log_root}/timing \
    --bg_obs_sender_send_image=False \
    --recovery_warmup_steps=2 \
    --empty_grasp_rewind_warmup_steps=2 \
    --empty_grasp_rewind_settle_time=1.0 \
    --recovery_home_settle_time=1.0 \
    --empty_grasp_lift_settle_time=0.5 \
    --queue_size_monitor_interval=5 \
    --retry_anti_repeat_steps=30 \
    --retry_anti_min_dist=15.0 \
    --retry_anti_penalty=0.35 \
    --log_level=INFO \
    --queue_size_monitor_path=./outputs/eval/${benchmark_robot_type}/${model_type}/${env_task}/${pretrained_short_name}/queue.png \
    2>&1 | tee ${log_root}/client_$(date +%Y%m%d_%H%M%S).log



--gripper_load_grasp_threshold=80.0 \
    --gripper_pos_gap_threshold=7.0 \
    --gripper_pos_empty_threshold=8.0 \
    --gripper_pos_open_threshold=20.0 \
    --so101_gripper_open_deg=20.0 \
    --so101_gripper_empty_deg=8.0 \
    --gripper_slip_drop_ratio=0.4 \
    --gripper_confirm_steps=3 \
    --gripper_lookahead_steps=15 \
    --max_reinfer_retries=2 \
    --max_empty_grasp_retries=3 \
    --recovery_return_to_home=true \
    --recovery_home_steps=90 \
    --recovery_warmup_steps=25 \
    --recovery_smooth_steps=60 \
    --recovery_smooth_max_delta=3.0 \

# 双重检验：L2（宽松）+ 各轴独立严格
--task_done_home_check_mode=ee \
--task_done_home_ee_tolerance_m=0.06 \        # L2 宽松：不同轴组合 ≤6cm
--task_done_home_ee_tolerance_xyz_m=[0.04,0.06,0.04]  # 各轴：X≤4cm, Y≤6cm, Z≤4cm

--bg_obs_sender_send_image=False   # 轻量模式（trajectory 无需图像）
--bg_obs_sender_send_image=True    # 默认，保留图像（向后兼容）

--task_done_home_check_mode=ee \
    --task_done_home_ee_tolerance_m=0.008 \
    --task_done_home_ee_tolerance_xyz_m=[0.01,0.01,0.01] \
    --task_done_home_check_gripper=true \
    --task_done_home_gripper_tolerance_deg=3.0 \

--task_done_home_check_mode=joint \
--task_done_home_tolerance=15.0 \

retry_anti_repeat_steps=30 — active步数窗口
retry_anti_min_dist=15.0° — 相似度阈值
retry_anti_penalty=0.35 — 惩罚分值

--retry_anti_repeat_steps=30 \
--retry_anti_min_dist=15.0° \
--retry_anti_penalty=0.35 \
```

---

## 场景 4：仅噪声多样性（关闭 RTC delay 变体）

```bash
# delay_delta=0：仅高斯噪声多样性
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --fps=10 \
    --n_candidates=8 \
    --top_k=2 \
    --delay_delta=0 \
    --data_collect_dir=${log_root}/mc_data_noise_only

uv run python -m lerobot.async_inference.run_so101_multicand_client \
    --robot.type=${robot_type} \
    --robot.port=${robot_port} \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
    --task="${task}" \
    --policy_type=${model_type} \
    --pretrained_name_or_path=${pretrained_name_or_path} \
    --server_address=127.0.0.1:8080 \
    --fps=10 --actions_per_chunk=16 \
    --enable_gripper_sm=true \
    --record_trajectory=true \
    --trajectory_dir=${log_root}/mc_trajectories_noise_only \
    --results_dir=${log_root}/mc_results_noise_only
```

---

## 调试模式（逐步验证）

```bash
# Step 1：禁用所有 O1-O3 优化，验证基本 candidate selection 流程
uv run python -m lerobot.async_inference.run_so101_multicand_client \
    ... \
    --enable_gripper_sm=false \
    --server_score_normalize=none \
    --spread_uncertainty_threshold=0.0 \
    --client_smooth_alpha=0.3 \
    --log_level=DEBUG \
    --record_trajectory=true

# Step 2：加入 SM，验证 phase 映射和 gripper 阈值
#   关注日志中的：
#   [mc_so101] phase_read: SO101=CLOSING → scoring=CLOSING
#   [mc_so101] t=XX rank=1 O2_EE_cont: ref_gripper=0.5° prev_gripper=28.0°
#   [mc_so101] t=XX rank=0 CLOSING_PENALTY: gripper_cmds max=25.0° > open_th=20.0° → -0.5
#   [mc_so101] t=XX rank=2 HOLDING_SLIP_GATE: gripper_cmds max=12.0° > empty_th=8.0° → discarded
uv run python -m lerobot.async_inference.run_so101_multicand_client \
    ... \
    --enable_gripper_sm=true \
    --server_score_normalize=none \
    --spread_uncertainty_threshold=0.0 \
    --log_level=DEBUG

# Step 3：开启 softmax 归一化 + O1 不确定性慢模式
uv run python -m lerobot.async_inference.run_so101_multicand_client \
    ... \
    --enable_gripper_sm=true \
    --server_score_normalize=softmax \
    --spread_uncertainty_threshold=0.15 \
    --log_level=INFO
```

---

## 离线分析

```bash
# 轨迹可视化（复用 LIBERO 分析工具，schema 兼容）
/opt/venv/bin/python -m pip uninstall -y placo cmeel-urdfdom cmeel-console-bridge cmeel-tinyxml2
/opt/venv/bin/python -m pip install \
  "placo==0.9.16" \
  "cmeel-urdfdom==4.0.1" \
  "cmeel-console-bridge==1.0.2.3" \
  "cmeel-tinyxml2==10.0.0.0"
  
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH

python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \
    --traj_dir=${log_root}/mc_trajectories \
    --out_dir=${log_root}/mc_viz \
    --action_dim_names=shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper \
    --robot_type=so101 \
    --viz_mode=ee 


--robot_type {libero,so101}
-viz_mode {ee,joint} (default ee)


apt update
apt install python3-tk
python src/lerobot/async_inference/analyze_trajectory.py ${log_root}/trajectories

# 查看汇总
cat ${log_root}/mc_results/summary.txt
cat ${log_root}/mc_results/aggregate.json

# 查看每个 pick-place 周期的结果
cat ${log_root}/mc_data/client_outcomes.jsonl | python3 -c "
import sys, json
recs = [json.loads(l) for l in sys.stdin]
success = sum(r['success'] for r in recs)
print(f'Episodes: {len(recs)}  Success: {success}  SR: {success/len(recs):.1%}')
for r in recs:
    print(f'  ep={r[\"episode_id\"]} success={r[\"success\"]} steps={r[\"steps\"]} retries={r[\"sm_retries\"]}')
"
```

---

## 关键参数对照（LIBERO → SO-101）

| 参数 | LIBERO 典型值 | SO-101 典型值 | 说明 |
|---|---|---|---|
| `action_limit_min` | `-1.5` | `-10.0` | 动作空间单位：LIBERO 归一化 / SO-101 度数 |
| `action_limit_max` | `1.5` | `310.0` | 同上 |
| `so101_gripper_open_deg` | N/A | `20.0` | 夹爪"开"阈值（度数）→ 需与 `gripper_pos_open_threshold` 保持一致 |
| `so101_gripper_empty_deg` | N/A | `8.0` | 夹爪"空闭合"阈值 → 需与 `gripper_pos_empty_threshold` 保持一致 |
| `spread_uncertainty_threshold` | `0.08` | `0.15` | 真机噪声更大，阈值适当调高 |
| `gripper_load_grasp_threshold` | N/A（qpos） | `150` | 真实抓取 load ≈ 300-500，空闭合 ≈ 80-120 |
| `interpolation_multiplier` | 1（仿真） | `3` | 真机建议 3×，控制频率 = fps × 3 |

### 阈值不一致警告
启动时若 `so101_gripper_open_deg ≠ gripper_pos_open_threshold`（差值 >1°），客户端
会打印 WARNING。**请确保两者一致**，否则 Layer-4/5 评分判断与 SM 相位判断基准不同：

```
[mc_so101] RISK: Gripper threshold mismatch between MC scorer and SM:
  so101_gripper_open_deg=25.0  vs gripper_pos_open_threshold=20.0
  → Set so101_gripper_open_deg = gripper_pos_open_threshold
```

---

## 输出文件结构

```
${log_root}/
├── mc_data/
│   ├── client_outcomes.jsonl     # 每个 pick-place 周期：success/steps/retries
│   └── client_steps.jsonl        # 每次 chunk 选择：scores/phase/alpha/uncertainty
├── mc_trajectories/
│   ├── ep0000.json               # 第 0 个周期完整轨迹（chunks + executed steps）
│   ├── ep0001.json
│   └── ...
├── mc_results/
│   ├── summary.txt               # 人类可读汇总（MC telemetry + SM retry stats）
│   └── aggregate.json            # 程序可读 JSON
├── timing/                       # --timing_output_dir 输出（latency breakdown）
│   ├── sm_summary.txt
│   └── ...
├── server.log
└── client.log
```
