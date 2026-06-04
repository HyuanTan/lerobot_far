# select action-chunk--Libero

## 服务器选最优，客户端无感知（向后兼容）
```bash
# 终端 1：启动 MultiCandidatePolicyServer
python -m lerobot.async_inference.multi_candidate_server \
    --policy_type=smolvla \
    --pretrained_name_or_path=/path/to/smolvla \
    --host=127.0.0.1 --port=8080 \
    --fps=30 \
    --n_candidates=4 \
    --top_k=1 \
    --delay_delta=1 \
    --data_collect_dir=./mc_data

# 终端 2：使用标准客户端（无需改动）
python -m lerobot.async_inference.sim_test.run_libero_test \
    --env_task=libero_10 \
    --policy_type=smolvla \
    --pretrained_name_or_path=/path/to/smolvla \
    --server_address=127.0.0.1:8080 \
    --actions_per_chunk=16 --fps=30 \
    --episodes_per_task=10 \
    --results_dir=./mc_results_phase1

```

## 服务器返回 Top-K，客户端二次选优
```bash
# 终端 1：top_k=2，服务器返回最优 2 个候选
python -m lerobot.async_inference.multi_candidate_server \
    --policy_type=smolvla \
    --pretrained_name_or_path=/path/to/smolvla \
    --host=127.0.0.1 --port=8080 \
    --fps=30 \
    --n_candidates=4 \
    --top_k=2 \
    --delay_delta=1 \
    --w_jerk=1.0 --w_vel_peak=0.5 --w_consistency=0.3 \
    --data_collect_dir=./mc_data

# 终端 2：Phase 2 专用客户端，执行连续性再排序
python -m lerobot.async_inference.sim_test.run_libero_multicand_test \
    --env_task=libero_10 \
    --policy_type=smolvla \
    --pretrained_name_or_path=/path/to/smolvla \
    --server_address=127.0.0.1:8080 \
    --actions_per_chunk=50 --fps=30 \
    --rtc_execution_horizon=8 \
    --client_smooth_alpha=0.3 \
    --action_limit_min=-1.5 --action_limit_max=1.5 \
    --episodes_per_task=2 \
    --record_trajectory=true \
    --trajectory_dir=./mc_trajectories \
    --data_collect_dir=./mc_data \
    --results_dir=./mc_results_phase2 \
    --save_video=true --video_dir=./mc_videos

```
###  PI05 

libero_goal
libero_object
libero_spatial
libero_10

```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2

export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=lerobot/pi05_libero_finetuned_v044
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=pi05
export env_task=libero_object
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}


# 终端 1
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --fps=10 \
    --n_candidates=16 \
    --top_k=4 \
    --delay_delta=1 \
    --record_all_candidates=true \
    --w_jerk=1.0 --w_vel_peak=0.5 --w_consistency=0.3 \
    --data_collect_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_data

差异性小:
    --n_candidates=8 \
    --top_k=2 \

client 会override server 的选择
    --n_candidates=16 \
    --top_k=4 \

libero_spatial 比  --top_k=4 放偏更多；libero_object成功率高
    --n_candidates=16 \
    --top_k=1 \


--spread_uncertainty_threshold=0.08   # O1: 触发阈值
--spread_slow_alpha_scale=1.5         # O1: alpha 放大倍数
--spread_slow_mode_window=5           # O1: 滚动窗口
--server_score_normalize=softmax      # O3: none|softmax|minmax

# 终端 2
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
    --save_video=true --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/videos

```

###  smolvla 

libero_goal
libero_object
libero_spatial
libero_10

```bash
cd ~/VLA/LeRobot/lerobot_v0.5.2

export CUDA_VISIBLE_DEVICES=3
export pretrained_name_or_path=HuggingFaceVLA/smolvla_libero
export pretrained_short_name="${pretrained_name_or_path##*/}"
export model_type=smolvla
export env_task=libero_object
export benchmark_robot_type=libero
export PYTHONUNBUFFERED=1
export log_path=./logs/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/${pretrained_short_name}
mkdir -p ${log_path}


# 终端 1
uv run python -m lerobot.async_inference.multi_candidate_server \
    --host=127.0.0.1 --port=8080 \
    --fps=10 \
    --n_candidates=16 \
    --top_k=4 \
    --delay_delta=1 \
    --record_all_candidates=true \
    --data_collect_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/mc_data

--spread_uncertainty_threshold=0.08   # O1: 触发阈值
--spread_slow_alpha_scale=1.5         # O1: alpha 放大倍数
--spread_slow_mode_window=5           # O1: 滚动窗口
--server_score_normalize=softmax      # O3: none|softmax|minmax

# 终端 2
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
    --rewind_mode=set_state \
    --rewind_buffer_steps=25 \
    --rewind_warmup_steps=5 \
    --gripper_pos_sum_empty_threshold=0.04 \
    --max_empty_grasp_retries=3 \
    --client_smooth_alpha=0.4 \
    --episodes_per_task=2 \
    --rtc_execution_horizon=20 \
    --record_trajectory=true \
    --trajectory_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/mc_trajectories \
    --data_collect_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/mc_data \
    --results_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/mc_results_pi05 \
    --save_video=true --video_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/videos

retry 后还是较多不成功：
--rewind_mode=action_replay
--rewind_mode=set_state

```


### 纯噪声多样性（关闭 RTC delay 变体）
```bash
# delay_delta=0：仅高斯噪声多样性，4 个 candidate 使用同一 delay
uv run python -m lerobot.async_inference.multi_candidate_server \
    --policy_type=smolvla \
    --pretrained_name_or_path=/path/to/smolvla \
    --host=127.0.0.1 --port=8080 \
    --fps=30 \
    --n_candidates=4 \
    --top_k=2 \
    --delay_delta=0 \
    --data_collect_dir=./mc_data_noise_only

```
## 离线数据合并（采集后）
merge_phase3.py

```bash
# Then visualize:
uv run python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \
    --traj_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_trajectories \
    --out_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_viz \
    --action_dim_names=j0,j1,j2,j3,j4,j5,grip

uv run python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \
    --traj_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/mc_trajectories \
    --out_dir=./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_smolvla/${env_task}/mc_viz \
    --action_dim_names=j0,j1,j2,j3,j4,j5,grip

uv run python -m lerobot.async_inference.analyze_trajectory --no-show ./outputs/eval/${benchmark_robot_type}/${model_type}/mc_sm_pi05/${env_task}/mc_trajectories/

```
robot_state[:3] 直接对应 eef_pos，
ep JSON 里的 robot_state 格式现在是 [x, y, z, qx, qy, qz, qw, g0, g1]（9维）。

- ep{N}_ee_traj_3d.png	实际执行的 EE 轨迹（3D），颜色按 episode phase 渐变，标出 start/end
- ep{N}_cand_3d.png	每个 chunk 的所有候选 action 轨迹（dims 0,1,2）vs 被选中的，每 chunk 一个子图；绿色=server 选中，红色=client override 选中，灰色=未选中
cand_3d 用的是 action 的前 3 维（joint-space 下是 j0,j1,j2），图标题会标注维度名，所以不会误导。

颜色方案

类型	旧颜色	新颜色
已选中 (server)	#2ecc71 绿	#2ecc71 绿（不变）
已选中 (override)	#e74c3c 红	#e74c3c 红（不变）
未选中候选	#aaaaaa 浅灰	tab10 调色板（蓝/紫/橙/青…每个候选独立颜色，alpha=0.75）
无候选 (Phase-1)	不显示	#888888 中灰，标注 no candidate
布局变化

现在包含所有 chunk（不只是多候选 chunk）
Phase-1 / top_k=1 的 chunk 用灰色显示 selected_actions，subplot 标题标注 no cand
图标题增加 mc=X/Y chunks 说明多候选占
