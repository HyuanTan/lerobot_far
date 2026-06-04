lsof -i :8095
tmux attach -t libero_client

lsof -i -a -u huoyuan


```bash
tmux list-panes -t libero_server3 -F '#{session_name}:#{window_index}.#{pane_index} #{pane_pid}' | while read pane pid; do
  echo "=== $pane pid=$pid ==="
  pstree -ap "$pid"
  echo
done
```

# libero-smolvla
## Sync
INFERENCE_LATENCIES_STEPS=(0 6 8 10 16 20 30)
ACTIONS_PER_CHUNK_LIST=(6 10 20 30 50)

The skipped combos (K < d) are:

d \ K	6	10	20	30	50
0	✓	✓	✓	✓	✓
6	✓	✓	✓	✓	✓
8	✗	✓	✓	✓	✓
10	✗	✓	✓	✓	✓
16	✗	✗	✓	✓	✓
20	✗	✗	✓	✓	✓
30	✗	✗	✗	✓	✓
That saves 9 server starts (out of 35 total) and their associated 4-suite runs — roughly 9 × 45s model-load + 9 × 4 × N eval episodes avoided

## Async no RTC
d ∈ {0, 6, 8, 10, 16, 20, 30}   (同 sync_nortc，跨方法可对比)
T ∈ {5, 10, 15, 20, 30}          (chunk_size_threshold)
跳过：T ≤ d                        (否则 robot stalls，等同 sync)


有效矩阵（✓ = run，✗ = skip）：

d \ T	5	10	15	20	30
0	✓	✓	✓	✓	✓
6	✗	✓	✓	✓	✓
8	✗	✓	✓	✓	✓
10	✗	✗	✓	✓	✓
16	✗	✗	✗	✓	✓
20	✗	✗	✗	✓	✓
30	✗	✗	✗	✗	✗
d=30 全部跳过（T 最大只到 30，T > d 需 T ≥ 31）。可以加一个 T=35 覆盖 d=30



问题	答案
固定 K=50？	✓ 是，K=50 提供足够 buffer，不是 async 的有效对比轴
是否还 sweep d？	✓ 是，d 仍是主轴，且与 sync_nortc 保持对比一致性
sweep T 的意义	T 控制"提前多少步请求新推理"，即 async 的核心设计参数
skip 条件	T ≤ d（robot 会 stall，退化为 sync 行为）
server 重启	仅 per d（而非 per (d,T)），大幅节省开销

CHUNK_SIZE_THRESHOLD_LIST = (5 10 15 20 30 35)   ← 步数，用于 skip 判断和目录名
threshold_ratio = T_steps / K = T_steps / 50     ← 实际传给 --chunk_size_threshold
T_steps	threshold_ratio	触发时机（K=50）
5	0.10	剩余 ≤ 5 步时
10	0.20	剩余 ≤ 10 步时
20	0.40	剩余 ≤ 20 步时
30	0.60	剩余 ≤ 30 步时
35	0.70	剩余 ≤ 35 步时
目录名仍用 T{steps}（如 T10/），combo log 同时打印 steps 和 ratio 方便 debug。skip 条件 T_steps ≤ d 用步数比较，无需转换。

## Async RTC
方面	async_nortc	async_rtc
sweep 轴名	T (chunk步数)	H (rtc_horizon 步数)
目录名	T{T}/	H{H}/
--rtc_execution_horizon	0 (禁用)	${rtc_horizon}
--chunk_size_threshold	T/K	H/K (H和T等价，但语义绑定)
参数列表	CHUNK_SIZE_THRESHOLD_LIST	RTC_HORIZON_LIST
关键语义：在 RTC 脚本里 CHUNK_SIZE_THRESHOLD_LIST 消失了，因为 H 已经完全决定了触发时机（H/K）和模型条件化（H），它们不能独立配置。banner 同时打印 H 和对应的 H/K ratio 方便核查。


Inference delay tracher
更激进（更新鲜的动作，偶发饥饿）
python -m lerobot.async_inference.smart_robot_client \
    --config_path ... --spike_buffer_s=0.05

更保守（减少饥饿风险）
--spike_buffer_s=0.25

## Async no RTC, FSM



# libero-pi05
lerobot/pi05_libero_finetuned_v044

# Visualize
```bash
uv run python -m lerobot.async_inference.analyze_sweep \
    outputs/eval_thesis/libero \
    --method sync_nortc sync_nortc_sm async_nortc async_nortc_sm  async_nortc_sm_multicand async_rtc_sm_multicand async_rtc async_rtc_sm --model smolvla
```

```bash
uv run python -m lerobot.async_inference.analyze_timing \
    ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task} \
    --fps=20 \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/timing_analysis 2>&1


uv run python -m lerobot.async_inference.analyze_rtc \
    --client_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/client_timing \
    --fps=20 \
    --out_dir ./outputs/eval/${benchmark_robot_type}/${model_type}/sim/${env_task}/rtc_analysis

./analyze_tools_script/analyze_so101_all.sh

export FPS=30
analyze_tools_script/analyze_so101_all.sh outputs/eval_thesis/libero/libero_object
```

DRY_RUN=1        # preview commands without running
SKIP_EXISTING=0  # re-run even if output dirs exist  (default: 1 = skip)
NO_COMPARE=1     # skip the final cross-method comparison
POLICY=pi05      # filter to one policy
PARAM=H15        # filter to one param
FPS=20           # control-loop fps (default: 20)



```bash
./analyze_tools_script/copy_so101_client_to_thesis.sh

uv run python -m lerobot.async_inference.analyze_so101_comparison \
    [eval_root]  [--policy pi05] [--param H15] [--fps 10] [--out_dir ...]

uv run python -m lerobot.async_inference.analyze_so101_comparison \
    outputs/eval_thesis/so101  --policy pi05 --param H15 --fps 20 --out_dir outputs/eval_thesis/so101/comparison
```
