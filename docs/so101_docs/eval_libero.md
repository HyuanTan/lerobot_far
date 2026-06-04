(1) 整合可行性分析
能否整合？可行，但有若干适配点需要处理：

benchmarks/ (Naive Asynchronous)
原始依赖	LeRobot替换	状态
vlash.policies.factory.make_policy	lerobot.policies.factory.make_policy	✅ 已替换
from benchmarks.benchmark_config import	from .benchmark_config import	✅ 已修复
vlash.configs	已移除（不需要）	✅
预处理pipeline	新增 make_pre_post_processors	✅
主要变化：预处理流程。LeRobot的 predict_action_chunk 接受的是已经过 preprocessor 处理（tokenized + normalized）的观测。原VLASH版本把normalize内置在policy内部，而LeRobot将其外置为 PolicyProcessorPipeline。benchmark 文件中对每个 dataset batch 调用 preprocessor 后再计时。

it_rtc/ (Inference Time RTC)
原始依赖	LeRobot替换	状态
vlash.policies.factory.make_policy	lerobot.policies.factory.make_policy	✅
vlash.libero_gym.make_env_from_suite_task	lerobot.envs.libero.LiberoEnv	✅
libero.SubprocVectorEnv	gymnasium.vector.SyncVectorEnv	✅
patch_policy_with_rtc (monkey-patch)	LeRobot原生RTC（config.rtc_config）	✅ 关键优化
vlash.policies.smolvla.utils.make_att_2d_masks	lerobot.policies.smolvla.modeling_smolvla	已不需要
policy.normalize_inputs/normalize_targets/unnormalize_outputs	make_pre_post_processors pipeline	✅
关键发现：LeRobot 的 SmolVLAPolicy 已经原生支持 RTC（policy.config.rtc_config + predict_action_chunk(inference_delay=..., prev_chunk_left_over=..., execution_horizon=...) kwargs），无需 monkey-patch。VLASH 版需要 monkey-patch 是因为 VLASH 的 SmolVLA 不含原生 RTC。

关键接口差异：

predict_action_chunk 返回的是模型空间（normalized）动作，需要 postprocessor 才能转换为环境可用的动作
prev_chunk_left_over 应该是模型空间的动作（直接使用 predict_action_chunk 的返回值，无需额外 normalize）
(2) 状态机整合
设计：通过 EvalConfig.enable_sm: bool 开关控制：

模式	环境	推理方式	SM
enable_sm=False	SyncVectorEnv(n_envs=batch_size)	向量化batch推理	无
enable_sm=True	SyncVectorEnv(n_envs=1) 顺序运行	单环境顺序推理	OfflineGripperSM
SM的核心逻辑从 SimSmartClient._update_gripper_sm 适配为 OfflineGripperSM：

检测：通过 gripper qpos (abs(q0) + abs(q1) < threshold) 判断 empty_grasp
回卷：调用 env.envs[0]._env.sim.set_state() + sim.forward()（需 SyncVectorEnv）
warmup：回卷后执行 N 步 hold action 填充策略 context 窗口
SM触发后的观测获取（rollout_chunked_with_sm 中的 env.call("_get_obs") 行）是一个已知的注意点，实际运行时可能需要根据 LIBERO env 的实际接口调整，或在 rewind 后直接让 policy 用最后一帧 obs 重新规划即可。

(3) 绘图设计
parse_eval_results_rtc.py 生成两类图：

SR vs. async_delay（已有，扩展）：

每个 execution_horizon 值一张图
2×2 子图按 LIBERO suite
4条曲线：baseline, rtc, baseline/SM, rtc/SM（颜色/marker自动映射）
输出：sr_vs_delay_horizon_{H}.png
SR vs. execution_horizon（新增）：

每个 async_delay 值一张图
2×2 子图按 LIBERO suite
同样4条曲线
输出：sr_vs_horizon_delay_{D}.png
使用方式：


# 只画 SR vs delay（默认）
python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc outputs/eval/...

# 同时画两种图
python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc outputs/eval/... --plot-horizon
文件结构

async_libero_inference/
├── __init__.py
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_config.py          # TrainPipelineConfig 子类
│   └── benchmark_inference_latency.py  # 推理延迟基准
└── it_rtc/
    ├── __init__.py
    ├── gripper_sm.py                # 离线 Gripper 状态机（从 SimSmartClient 适配）
    ├── eval_libero_rtc.py           # 主评估脚本（含 SM 开关）
    └── parse_eval_results_rtc.py   # 解析+绘图（SR-vs-delay + SR-vs-horizon）
典型 Sweep 命令

```bash
# Baseline sweep (no SM)
for DELAY in 0 1 2 3 4; do
for HORIZON in 5 10 20; do
python -m lerobot.async_libero_inference.it_rtc.eval_libero_rtc \
    --policy.path=<ckpt> --env.type=libero --env.task=libero_spatial \
    --eval.method_type=baseline --eval.async_delay=$DELAY \
    --eval.execution_horizon=$HORIZON --eval.n_episodes=20 \
    --output_dir=outputs/eval/baseline/delay${DELAY}_horizon${HORIZON}
done; done

# RTC sweep (no SM) — same but --eval.method_type=rtc
# Baseline + SM — add --eval.enable_sm=true
# RTC + SM — --eval.method_type=rtc --eval.enable_sm=true

python -m lerobot.async_libero_inference.it_rtc.eval_libero_rtc \
    --eval.method_type=baseline \
    --eval.enable_sm=true \
    --eval.async_delay=4 --eval.execution_horizon=10 ...

# smolvla
--policy.path=/path/to/smolvla_checkpoint

# pi05 (完全相同的命令行，factory 根据 config 自动调度)
--policy.path=/path/to/pi05_checkpoint


# Then plot all results together:
python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \
    outputs/eval --plot-horizon --output-dir outputs/plots
```

--------

eval_pi05.sh — 完整 sweep 脚本
参数	图中范围	脚本配置
delay sweep	d = 0..15	DELAYS=(0 1 2 ... 15), 固定 s=15
horizon sweep	s = 1..15 @ d=1	HORIZONS=(1 2 ... 15), 固定 d=1
方法	4条曲线	baseline / rtc / baseline_sm / rtc_sm
suite	4个	libero_spatial/object/goal/libero_10
总runs		496次 (124配置 × 4 suite)
使用示例：


# 单GPU全量运行
CKPT=/path/to/pi05_ckpt bash eval_pi05.sh

# 多GPU并行（4块GPU）
CKPT=/path/to/pi05_ckpt GPU=0,1,2,3 N_EPISODES=20 bash eval_pi05.sh

# 只跑 delay sweep（不含SM）
CKPT=/path/to/pi05_ckpt bash eval_pi05.sh --delay-only --no-sm

# 断点续跑（已有 eval_results.json 的任务自动跳过）
CKPT=/path/to/pi05_ckpt bash eval_pi05.sh

# 预览命令（不实际运行）
CKPT=/path/to/pi05_ckpt bash eval_pi05.sh --dry-run
运行结束后自动调用 parse_eval_results_rtc.py 生成两类图：

sr_vs_delay_horizon_15.png — 对应图中左图（4条曲线 baseline/rtc/+SM）
sr_vs_horizon_delay_1.png — 对应图中右图（d=1 处的 horizon sweep）


-------

新增的两类图
Plot 3: generate_per_combination_plots — 每个方法组合一张图
每种方法各一个文件：per_combo_baseline.png、per_combo_rtc.png、per_combo_baseline_sm.png、per_combo_rtc_sm.png

布局：2行 × 4列（行 = sweep类型，列 = LIBERO suite）


                 Spatial    Object     Goal    Libero-10
Row 0 (delay)  [subplot]  [subplot]  [subplot]  [subplot]
Row 1 (horiz)  [subplot]  [subplot]  [subplot]  [subplot]
每个子图：该 method 下不同 checkpoint 变体作为多条曲线
包含 Wilson 95% CI 误差棒
baseline 图中叠加灰色虚线（VLASH paper 参考）
Plot 4: generate_combined_overview — 单张总图
文件：overview_combined.png

布局：4行 × 2列（行 = LIBERO suite，列 = delay sweep / horizon sweep）


              SR vs Delay (s=15)    SR vs Horizon (d=1)
Spatial      [                  ]  [                  ]
Object       [                  ]  [                  ]
Goal         [                  ]  [                  ]
Libero-10    [                  ]  [                  ]
每个子图：4条曲线（Baseline=蓝, RTC=红, Baseline+SM=绿, RTC+SM=紫）
颜色固定不随 checkpoint 变化
CI 半透明填充区域 + 误差棒
底部全局图例
使用方式

# 四种图全生成
python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \
    outputs/eval/pi05 --all-plots --output-dir outputs/plots

# 只生成组合图 + 总图（最常用）
python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \
    outputs/eval/pi05 --no-plot-delay --plot-combo --plot-overview \
    --delay-sweep-horizon 15 --horizon-sweep-delay 1

# 关闭 CI（数据少时误差棒太大）
python -m lerobot.async_libero_inference.it_rtc.parse_eval_results_rtc \
    outputs/eval/pi05 --all-plots --no-ci --autoscale
eval_pi05.sh 末尾的绘图命令也同步更新（脚本已被用户修改）：默认只开 --all-plots，运行完 sweep 后自动生成全部 4 种图。


```
CKPT=lerobot/pi05_libero_finetuned_v044 \
GPU=3 \
SUITES="libero_spatial libero_object" \
N_EPISODES=10 \
BATCH_SIZE=10 \
OUT_ROOT=outputs/eval_thesis_sim/libero/pi05 \
SEED=42 \
bash eval_libero_script/eval_pi05.sh

```

------

分析：DELAY_FIXED_S=50 和 HORIZONS 的作用是一致的吗？是否跟/data/users/huoyuan/VLA/vla_asyn_arena_ori_repo/async-vla-inference/bt-libero/it_rtc/eval_libero_rtc.py 的实验逻辑一样？

1. DELAY_FIXED_S 和 HORIZONS 不是同一回事
两者都映射到 --eval.execution_horizon（参数 s），但角色完全不同：

变量	角色	含义
DELAY_FIXED_S=50	Sweep 1 的控制常量	延迟扫描时固定的 s，值=50=chunk_size
HORIZONS=(1 3 5 7 ...)	Sweep 2 的自变量	视野扫描时依次取的 s 值范围

Sweep 1: 固定 s=50，扫描 d ∈ {0,2,4,6,8,10,12,14}  → "SR vs. Inference Delay"
Sweep 2: 固定 d=0，  扫描 s ∈ {1,3,5,7,9,11,13,15} → "SR vs. Execution Horizon"
DELAY_FIXED_S=50 的物理含义： s=50=chunk_size → 每 50 步才 replan 一次（全 chunk 执行），让 Sweep 1 只研究延迟 d 的影响，不混入重规划频率 s 的干扰。

2. 实验逻辑与 eval_libero_rtc.py 完全一致
eval_pi05.sh 第 346 行直接调用：


uv run python -m lerobot.async_libero_inference.it_rtc.eval_libero_rtc \
    --eval.async_delay="${d}" \
    --eval.execution_horizon="${s}" ...
即是同一个同步仿真脚本，不使用 async client-server 架构。async_delay=d 由 obs_buffer[-(d+1)] 实现：


# eval_libero_rtc.py:459-462
if async_delay > 0 and len(obs_buffer) > async_delay:
    policy_obs = obs_buffer[-(async_delay + 1)]  # d 步前的 obs
else:
    policy_obs = obs_buffer[-1]                   # 当前 obs
3. 完整执行逻辑对照（eval_libero_rtc.py）

每个 planning cycle（当 action_queue 为空时触发）：

          d=async_delay        s=execution_horizon
          ↓                    ↓
old chunk: [0 ─── d-1]│[d ─── s-1]│[s ──── chunk_size-1]
                  ↑执行旧    ↑执行新      ↑丢弃
                  (d步)     (s-d步)

RTC:
  effective_horizon = max(d, s)  ← 保证 s≥d
  policy 输入：obs[-(d+1)]（d步前）
  prev_chunk_left_over：上次 chunk 左移 effective_horizon 后的剩余
  执行：old[:d] + new[d:s]  共 s 步

Baseline:
  effective_horizon = s（0<s<50）或 50（s=0 or s≥50）
  policy 输入：obs[-(d+1)]
  执行：new_chunk[:s]  共 s 步（无混合）
4. 与原始 bt-libero 的关键差异
原始 bt-libero	eval_pi05.sh
d 范围	(0,1,2,4,8,15,20)	(0,2,4,6,8,10,12,14)
s 固定值	s=10（RTC）/ s=50（Baseline，忽略参数）	s=DELAY_FIXED_S=50 统一
d>s 违约	d=15,20 > s=10 ← 违反 RTC 约束	max(d)=14 < s=50 ✓
Horizon sweep	无	有（Sweep 2）
Baseline 控制	不受 s 控制（永远全 chunk）	s 生效，与 RTC 对齐（公平对比）
DELAY_FIXED_S=50 的设计意图： 让 Sweep 1 中 Baseline 和 RTC 都使用全 chunk（s=50），复现原始 bt-libero 的 Baseline 行为，同时满足 RTC 的 d<s 约束（14<50 ✓）。如果 DELAY_FIXED_S=15，则 Baseline 也会每 15 步 replan，与原始实验不等价。

分析：Sweep 2: 固定 d=0，  扫描 s ∈ {1,3,5,7,9,11,13,15} → "SR vs. Execution Horizon" 中，固定 d=0是否合理？是否应该为1，2，3等非0值？
分析：d=0 是否合理？
核心问题：d=0 时 RTC 的动作执行与 Baseline 完全相同
从 eval_libero_rtc.py:524-529 的执行逻辑：


if rtc_enabled and prev_chunk_left_over is not None and async_delay > 0:
    # RTC: 执行 d 步旧动作 + (s-d) 步新动作
    old_actions = prev_chunk_left_over[:, :async_delay, :]
    new_actions = raw_new_chunk[:, async_delay:effective_horizon, :]
    actions_to_execute = torch.cat([old_actions, new_actions], dim=1)
else:
    # baseline（或 d=0 时的 RTC）：直接执行新 chunk 前 s 步
    actions_to_execute = raw_new_chunk[:, :effective_horizon, :]
当 d=0 时：async_delay > 0 为 False → RTC 和 Baseline 执行完全相同的动作序列 new_chunk[:s]。

唯一区别只有去噪过程中的 RTC guidance（prev_chunk 的软约束），但不存在 d 步"旧动作续接"。

d=0 的实验含义

d=0, s=5:
  Baseline: new_chunk[0:5]         → 执行新 chunk 前 5 步，obs 来自当前时刻 ✓
  RTC:      new_chunk[0:5]         → 执行完全相同的动作，obs 来自当前时刻
            (仅 denoising guidance 不同，无动作混合)

d=0 的 Sweep 2 问题：研究的是 "replanning frequency 对无延迟情况的影响"
这对 RTC 来说不是最有价值的场景，因为 RTC 的核心价值是补偿延迟（latency compensation），而非零延迟下的时序一致性。


------
为什么脚本选择 d=0？——规避退化约束
脚本注释（line 107-110）明确说明：


# d=1: if min(HORIZONS)=1 then d=s=1 is degenerate
#      new_chunk[d:s] = new_chunk[1:1] = EMPTY → RTC 执行 0 步新动作
HORIZON_FIXED_D=0  # must be < min(HORIZONS)
当 d=s 时的退化情形：


d=1, s=1:
  effective_horizon = max(1,1) = 1
  new_actions = new_chunk[1:1] = EMPTY (0步)
  actions_to_execute = old[:1] + [] = 仅 1 步旧动作
  → RTC 永远不执行任何新动作，等同于纯旧 chunk 重放
是否应该用非零 d？
应该用非零 d，但需要调整 HORIZONS 的起点：

HORIZON_FIXED_D	min(HORIZONS) 约束	实验含义
d=0	无约束（当前）	研究无延迟下视野效果，不体现 RTC 核心优势
d=1	HORIZONS 从 ≥2 开始	小延迟下视野效果，RTC 开始体现动作混合
d=2	HORIZONS 从 ≥3 开始	有代表性的延迟，RTC vs Baseline 差异更明显
d=4	HORIZONS 从 ≥5 开始	较大延迟，研究 "最小有效 s"
最有信息量的 Sweep 2 设计（论文建议）：


HORIZON_FIXED_D=2              # 固定中等延迟（对应约 2/FPS 秒）
HORIZONS=(3 5 7 9 11 13 15)   # min(HORIZONS)=3 > d=2 ✓

# 此时 RTC 在每个 s 点执行：2步旧 + (s-2)步新
# Baseline 执行：new_chunk[:s]（基于 d=2 步前的 obs）
# → s 越大，两者差距越明显，揭示 "horizon 对延迟补偿的调节作用"
或者运行多个固定 d 的 Sweep 2，得到 2D 图像（d, s 交互热图）：


for FIXED_D in 0 2 4 8; do
    run_horizon_sweep fixed_d=$FIXED_D s_range=(${FIXED_D+1}...15)
done
# → 揭示：给定延迟 d，需要多大的 s 才能达到最优性能？
结论
d=0 的问题：

RTC 动作执行与 Baseline 相同（无混合），不能体现 RTC 的延迟补偿机制
研究的是"无延迟下频繁规划的影响"，而非 RTC 的设计目标
推荐改进：

Sweep 2 使用 d=2（或 d=4），HORIZONS 从 d+1 起
或者增加 Sweep 2.5：在几个固定 d 值下各做一次 horizon sweep，揭示 d-s 交互


