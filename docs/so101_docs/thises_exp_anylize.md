```
uv run python -m lerobot.async_inference.analyze_so101_comparison \
    [eval_root]  [--policy pi05] [--param H15] [--fps 10] [--out_dir ...]

```
Key findings from the current data (all 5 methods, pi05/H15):

async_rtc	async_rtc_inter	async_rtc_no_imgcrop	async_rtc_sm	async_rtc_sm_inter
RT p50 (ms)	306	303	350	296	311
RT p95 (ms)	972	802	1015	909	992
diff_l2 p50	5.3	11.8	8.6	3.5	4.2
leftover p50	2697	3447	3379	75	71
overall_sr	—	—	—	100%	100%
sr_lift	—	—	—	+40%	+95.9%
Notable: async_rtc_sm and async_rtc_sm_inter have dramatically smaller leftover_steps (~75 vs ~3000) — this is expected because the SM client uses a different buffering strategy. async_rtc_inter has the lowest p95 round-trip (802 ms) but the highest chunk discontinuity (diff_l2_mean p50 = 11.8 vs 3.5–5.3 for others).

Output figures: outputs/eval_thesis/so101/comparison/fig{1-7}_*.png

------------- SO101-PI05
RTC 设置与运行质量综合分析
一、核心问题：infer_delay 严重过大（所有方法共有）
这是最关键的问题。

方法	actual_steps p50	sent infer_delay p50	过校正倍数	诊断
async_rtc	8.3 steps	29.0 steps	6.1×	⚠️
async_rtc_inter	8.7 steps	29.0 steps	4.7×	△
async_rtc_no_imgcrop	9.4 steps	31.0 steps	5.5×	⚠️
async_rtc_sm	6.9 steps	29.0 steps	7.9×	⚠️
async_rtc_sm_inter	7.1 steps	29.0 steps	7.5×	⚠️
根本原因：LatencyTracker.max() 返回的是历史全局最大值，不是滑动窗口。首次推理（warmup）往往有一次大 latency spike，之后 infer_delay 被永久锁定在峰值（~1400ms ÷ 50ms/step ≈ 28–31 steps），而 steady-state 实际 round_trip p50 只有 300–350ms（≈ 7–10 steps）。

后果：

infer_delay=29 意味着客户端从 chunk 的第 29 步开始执行，跳过了 chunk[0..28]
实际只需要跳过 ~7-10 步，多跳了 20 步，使机器人执行本该是"未来"状态的动作
这会加大 chunk 边界处的动作不连续性（diff_l2_mean）
chunk_size_threshold 与 RTC_HORIZON 不一致：脚本中 RTC_HORIZON=15（期望阈值 = 15/50 = 0.30），但 chunk_size_threshold=0.5（实际触发阈值 = 0.5×50 = 25 steps）。真实 RTC horizon 是 25 steps，不是 15。注释写"adjust when RTC_HORIZON changes"但未实际同步。

二、Action Queue Starvation（饥饿）
方法	starvation rate	max gap	must_go rate
async_rtc	9.4% ⚠️	3.6s (71 steps)	1.6%
async_rtc_inter	4.0% △	3.4s (68 steps)	1.1%
async_rtc_no_imgcrop	6.5% ⚠️	3.8s (76 steps)	1.2%
async_rtc_sm	14.6% ⚠️⚠️	7.2s (144 steps)	6.1%
async_rtc_sm_inter	8.5% ⚠️	7.0s (140 steps)	8.0%
SM 方法饥饿原因（非 bug）：SM recovery 轨迹执行约 6s，期间 policy 队列不补充，而 50 steps × 50ms = 2.5s buffer，必然饥饿约 3.5s。这段时间机器人执行的是硬编码 recovery 轨迹而非 policy 动作，饥饿在功能上无害。

非 SM 方法饥饿原因（可优化）：网络 jitter（gRPC_send p95 = 365ms vs p50 = 67ms，比值 5.9×）导致偶发长延迟。加上 infer_delay=29 使有效 buffer 从 50 steps 减少到约 21 steps（50-29=21），在任何延迟 spike 时都易饥饿。

三、网络 Jitter（持续问题，架构层面）
所有方法一致：

指标	p50	p95	p99	p99/p50
grpc_send_ms	~67ms	~365ms	~390ms	5.8× ⚠️
round_trip_ms	~305ms	~990ms	~1100ms	3.6× ⚠️
gRPC 网络层 p95 比 p50 高 5.5 倍，是 WiFi/UDP 丢包或 TCP 重传导致。这也是 queue_wait_ms 出现双峰分布（p50 ≈ 1ms，p95 ≈ 500-700ms）的原因。

四、SM 方法 one_way_ms 极高（SM 特有）
方法	one_way p50	one_way p95	one_way p99
async_rtc	-24ms (clock skew)	83ms	284ms
async_rtc_inter	-5ms	108ms	304ms
async_rtc_sm	+10ms	1264ms	2311ms
async_rtc_sm_inter	+9ms	1021ms	2259ms
SM server 接收到的 obs 数量远超 non-SM（async_rtc_sm: client sent 3191 → server received 8050 = 252%，非 SM 约 135%）。这暗示 SM server 或 client 有额外的重发/同步逻辑，导致 one_way 出现 >2s 的极端值。

五、Gripper SM 参数可微调
两个 SM 方法都触发了同一诊断：

EMPTY GRASPS: mean_load=78–87, mean_pos=3.9–4.5. Load near threshold → lower gripper_load_grasp_threshold slightly.

empty_grasp 次数较多（async_rtc_sm: 64次 / 30 episodes = 2.1次/episode；async_rtc_sm_inter: 56次 / 49 episodes = 1.1次/episode）。抓取时 gripper_load 在阈值附近抖动，轻微降低 gripper_load_grasp_threshold 可以减少假阴性（empty_grasp 误判）。

六、leftover_steps 指标不可比（实现差异）
non-SM 方法（robot_client.py）报告 leftover_steps p50 = 2697–3447，SM 方法（smart_robot_client.py）报告 70–80。两者语义不同，不可直接比较。

总结：推荐修复优先级
优先级	问题	修复方向
🔴 P0	LatencyTracker.max() 导致 infer_delay 永久过大	改为滑动窗口（N=50 samples）+ 去除 warmup outliers
🔴 P0	chunk_size_threshold=0.5 与 RTC_HORIZON=15 不一致	同步为 chunk_size_threshold = RTC_HORIZON/ACTIONS_PER_CHUNK = 0.30
🟡 P1	非 SM 方法 starvation 9.4%（async_rtc）	infer_delay 修复后自然改善；网络层优化降 jitter
🟡 P1	SM 方法 must_go 6–8%	排查 SM client 是否在 recovery 期间阻塞 obs 发送
🟢 P2	empty_grasp 过多	小幅降低 gripper_load_grasp_threshold
🟢 P2	SM one_way_ms p99 > 2s	排查 SM server 是否有 obs 重传/堆积
最高影响力的单点修复是 LatencyTracker 改为滑动窗口，可将 infer_delay 从 ~29 步降至 ~10 步，直接改善 chunk 连续性和 starvation 率。

-----
chunk_size_threshold 不需要与 RTC_HORIZON 一致；分析：1）infer_delay 严重过大为什么没有使用滑动窗口？代码是否已经有还是没有生效？

滑动窗口已经存在且生效，诊断文本是错的，真正根因是网络抖动尖峰主导窗口最大值。

详细分析
层	代码位置	实际行为
LatencyTracker	latency_tracker.py:32	deque(maxlen=100) — 滑动窗口，一直存在
max()	latency_tracker.py:59	返回窗口内最大值，注释明确说"不是全时间最大值"
过滤器	base_client.py:483	仅当 0 < complete_s ≤ 2.5s 时加入窗口
infer_delay 计算	base_client.py:781-783	ceil(window_max / dt)
真正根因
真机测试数据显示 round_trip p99 ≈ 1100 ms，对应 complete_s ≈ 1450 ms。这个值：

✅ 通过了 2.5s 过滤器（1450 < 2500）
✅ 被正确加入滑动窗口
❌ 但每 ~100 次推理（~20 秒）必然出现一次 p99 事件
窗口 maxlen=100、推理频率 ~5 次/秒 → 窗口覆盖 ~20 秒。p99 事件每 20 秒发生一次，所以窗口内始终有一个尖峰，max() 始终返回 ~1450ms → ceil(1450/50) = 29 步。

analyze_timing.py 诊断文本已修复
旧文本（错误）：LatencyTracker.max() returned an all-time (not windowed) max

新文本（正确）：描述滑动窗口正常工作，但 p99 网络抖动事件频率足以始终驻留在窗口中，并建议将 max() 换为 .percentile(0.99) 来降低过校正。

可选修复：base_client.py:781
将 _max_lat = self.latency_tracker.max() 改为 _max_lat = self.latency_tracker.percentile(0.99) 可以将 infer_delay 从 ~29 步降至 ~8 步（p99 ≈ 365ms → ceil(365/50) = 8），同时仍保留对 99% 以上延迟的覆盖。这会改变实机行为，建议单独确认后再修改。

------
分析：能否通过将窗口 maxlen=100改小解决？现有过滤器是如何工作的？如何确认 _max_lat = self.latency_tracker.max() 改为 _max_lat = self.latency_tracker.percentile(0.99) 的方案是否正确（是否适用所有方法）


数据汇总（5 种方法）
方法	rt p50 (ms)	rt p95 (ms)	rt p99 (ms)	rt max (ms)	grpc_send p50	grpc_send p99
async_rtc	306	972	1102	1425	67	391
async_rtc_sm	296	909	1231	1329	65	390
async_rtc_inter	303	802	999	1289	67	392
async_rtc_sm_inter	311	992	1266	1343	67	390
async_rtc_no_imgcrop	350	1015	1314	1441	67	400
问题 1：缩小 maxlen 能解决问题吗？
不能，反而会更差。

设推理频率为 $r$（次/秒），窗口 maxlen = N：

$$\text{窗口覆盖时长} = \frac{N}{r} \text{ 秒}$$

$$\text{p99 事件间隔} = \frac{100}{r} \text{ 秒}$$

$$\text{窗口中期望尖峰数} = \frac{N/r}{100/r} = \frac{N}{100}$$

当前 N=100 → 期望 1 个尖峰 → 始终在窗口中。

如果 N=50 → 期望 0.5 → 50% 时间有尖峰，另 50% 无尖峰。

但"无尖峰"期间 max() 返回的是 p95 非极端样本，对 async_rtc 约 972ms → ceil(972/50)=20 步，而有尖峰时约 23 步。结果变成在 20 和 23 之间震荡，机器人行为变得不稳定。核心原因是这个等式：

maxlen/100 = 窗口期望尖峰数 ≈ 1 只要 maxlen ≈ 100

缩小 maxlen 只是把"确定性过高"变成"随机振荡"，两者都不是好状态。

问题 2：现有过滤器如何工作？
base_client.py:478-482:


_chunk_lifetime_s = actions_per_chunk * environment_dt  # 50 × 0.05 = 2.5s
complete_s = receive_after_deser - timed_actions[0].get_timestamp()
if 0 < complete_s <= _chunk_lifetime_s:
    self.latency_tracker.add(complete_s)
complete_s 覆盖的完整路径：


obs 构建开始 → gRPC 发送 → 服务器推理 → gRPC 接收 → 客户端反序列化
 ≈ 13ms(prep)  +  67ms(正常)/390ms(尖峰)  +  300ms  +  67/390ms  +  27ms
过滤器只拒绝两类：

≤ 0：无效（计时器错误）
> 2.5s：chunk 到达时所有动作已超时（没有执行意义）
所有真实网络尖峰（max ≈ 1440ms）都在 2.5s 以内，全部通过过滤器。 过滤器被设计用来拒绝"warmup 时的首次慢推理"和"已失效的 chunk"，而不是网络抖动尖峰。

问题 3：max() → percentile(0.99) 是否正确，是否适用所有方法？
改变量计算
方法	max→infer_delay	p99→infer_delay	p95→infer_delay	p50→infer_delay
async_rtc	ceil(1425/50)=29	ceil(1102/50)=23	ceil(972/50)=20	ceil(306/50)=7
async_rtc_sm	ceil(1329/50)=27	ceil(1231/50)=25	ceil(909/50)=19	ceil(296/50)=6
async_rtc_inter	ceil(1289/50)=26	ceil(999/50)=20	ceil(802/50)=17	ceil(303/50)=7
async_rtc_sm_inter	ceil(1343/50)=27	ceil(1266/50)=26	ceil(992/50)=20	ceil(311/50)=7
async_rtc_no_imgcrop	ceil(1441/50)=29	ceil(1314/50)=27	ceil(1015/50)=21	ceil(350/50)=7
改为 p99 是否正确？
是，但改善有限（节约 2–6 步），且一致性跨方法。

grpc_send_ms 在 5 种方法中极度一致（p50=65-67ms，p99=390-400ms），说明这是网络行为，与模型无关。因此 p99 的改善效果对所有 base 方法均匀一致。

为什么改善仍然有限？
根本原因是双峰分布：


grpc_send_ms:  正常模式 ~67ms  |  尖峰模式 ~390ms（5.8×）

round_trip ≈ server(300ms) + send + recv:
  双正常：  300 + 67 + 67   =  434ms  → p50
  单侧尖峰：300 + 390 + 67  =  757ms  → ~p95
  双侧尖峰：300 + 390 + 390 = 1080ms  → p99
双侧同时尖峰概率 ≈ 0.05 × 0.05 = 0.25%，即每 400 次推理发生一次。任何高于 p99.75 的分位数估计都能排除这类事件，但 p99 只覆盖到 1/100，每 100 次还有一次双侧尖峰在窗口中。

对 multi-candidate 方法的适用性
目前数据里没有 async_rtc_sm_multicand_4n_2k。该方法服务端跑 4 次推理后择优，server_infer_ms 可能更高（约 4× 如串行，或与单推理相当如并行）。在修改前需先确认多候选服务端是否并行推理，否则 p99 估计会偏低。

结论与建议
方案	效果	风险
减小 maxlen	❌ 反效果，行为震荡	增加 starvation 概率
max() → p99()	✅ 改善 2–6 步，适用所有 base 方法	1% 情况下 chunk 晚到
max() → p95()	✅ 改善 6–12 步，覆盖更激进	5% 情况下 chunk 晚到
过滤器收紧（拒绝 > 2×median）	✅ 直接隔离双侧尖峰	需要在线 median 估计
最务实的修改：将 base_client.py:781 改为 p99()，它在所有 5 种有数据的 base 方法上均有一致改善，且 1% 的漏保概率对 RTC 的容错机制（leftover_steps / starvation 检测）是可接受的。确认 multi-candidate 服务端为并行推理后再一起应用。
----
能否确认 multi-candidate 服务端是否为并行推理？

for delay in delays: 是顺序循环（multi_candidate_server.py:652），两次 forward pass 串行执行，每次内部是 batch_size=2 的批推理（GPU 内并行）。


N=4 candidates 的执行路径：
  Pass 0: batch_size=2, delay=delay_low  → 2 candidates (GPU 批并行)
  Pass 1: batch_size=2, delay=base_delay → 2 candidates (GPU 批并行)
  两次 pass 串行，total server_infer ≈ 2 × single-pass(batch=2)
对 infer_delay 的影响
现有数据里没有 async_rtc_sm_multicand 的 analysis_report.txt，但可以从 single-inference 基线推算：

pi05 8B 模型，base 方法 server_infer_ms p50 ≈ 300ms（batch=1）。

multi-candidate batch=2 的两次串行 pass：每次 ≈ 350–450ms（batch=2 显存带宽增加约 1.1–1.5×）→ 总计 ≈ 700–900ms。

成分	base 正常	base 尖峰	multicand 正常	multicand 尖峰
grpc_send	67	390	67	390
server_infer	300	300	800	800
grpc_recv	67	390	67	390
prep+deser	40	40	40	40
round_trip	474ms	1120ms	974ms	1620ms
infer_delay(max)	—	ceil(1424/50)=29	—	ceil(1620/50)=33（但 cap=49）
infer_delay(p99)	—	ceil(1102/50)=23	—	≈ceil(1580/50)=32（估算）
关键结论：multi-candidate 的 server_infer_ms 是主导项（~800ms）。此时：

网络尖峰的相对贡献变小（+390ms 在 974ms 基础上 = +40%，而 base 下 +390ms 在 474ms 基础上 = +82%）
p99 和 max() 给出的 infer_delay 差异收窄（32 vs 33，差约 1 步）
p99 对 multi-candidate 几乎没有改善效果
最终适用性结论
方法类别	p99 vs max() 改善	推荐
base（async_rtc, no_imgcrop）	6 步（29→23）	✅ 值得改
SM（async_rtc_sm, sm_inter）	2–3 步（27→25）	✅ 轻微改善
multi-candidate（4n_2k）	≤1 步（~33→~32）	❌ 几乎无效，server_infer 主导
对 multi-candidate，真正降低 infer_delay 的办法不是换分位数，而是减少推理次数或用真正并行的 GPU 多流。目前两次串行 predict_action_chunk 这个设计决定了 server_infer_ms ≈ 2× base，这才是 multi-candidate 下 infer_delay 高的根本。

----

如果  multi-candidate 中 n_candidates >=4,  delay_delta: 0，noise diversity only (no delay split)  是串行 pass吗？

不是 1 pass——delay_delta: 0 的注释具有误导性，实际运行中存在两条路径，关键在于运行时 delay_low 的值。

决策树（rtc_on=True 时）

rtc_on = True  (RTC capable policy + rtc_execution_horizon > 0)
│
├── delay_low > 0  AND  delay_low < base_delay   → delays = [delay_low, base_delay]
│   (稳态，latency tracker 有数据)                   n_per_delay = 2
│                                                  ★ 2 次串行 pass，每次 batch=2
│
├── delta > 0                                     → delays = [base-delta, base+delta]
│   (delay_delta 在 config 里 = 0，永远不走这里)      n_per_delay = 2
│
└── else (delta=0 AND delay_low==0 OR delay_low>=base_delay)
    (warmup 阶段，tracker < 3 samples)              delays = [base_delay]
                                                   n_per_delay = n (=4)
                                                   ★ 1 次 pass，batch=4
实际运行结果
从 base_client.py 可知：

delay_low = ceil(latency_tracker.p50() / dt) ← 稳态约 6 步（p50≈300ms）
base_delay = ceil(latency_tracker.max() / dt) ← 稳态约 29 步
稳态时 delay_low=6 > 0 且 delay_low=6 < base_delay=29 → 走数据驱动分支，即使 delay_delta=0：


delays = [6, 29]  →  2 次串行 pass（各 batch=2）
delay_delta: 0 只关闭了 ±delta 固定回退，不关闭数据驱动的延迟拆分。配置里的注释 "noise diversity only (no delay split)" 只在两种情况成立：

warmup（< 3 个 latency 样本，delay_low 仍为 0）
RTC 关闭时（rtc_on=False）
对 server_infer_ms 的影响
阶段	passes	batch_size	估算 server_infer_ms
Warmup（前 3 次推理）	1	4	~400–550 ms
稳态（绝大多数时间）	2 串行	2+2	~700–900 ms
稳态下与我之前估算一致：server_infer_ms ≈ 700–900ms，p99 改善对 multi-candidate 效果有限（max 与 p99 相差约 1 步）的结论仍然成立。

若要真正实现"noise diversity only, single pass, batch=4"，需要在代码里让 RTC 关闭或强制 delays = [base_delay]，而不是靠 delay_delta: 0。
---------------

分析 /data/users/huoyuan/VLA/LeRobot/lerobot_v0.5.2/outputs/eval_thesis/so101/ 下的rtc_fig1_leftover_health.png 总的Overlap vs new steps


Overlap vs New Steps 分析
数据汇总
方法	n_overlap均值	n_new均值	overlap%	n_new==0% (stale)	n_overlap==0% (starvation)	n_new==50% (fresh)
async_rtc	26.4	16.9	60.9%	0.2%	7.6%	6.7%
async_rtc_sm	24.1	19.9	54.8%	8.6%	19.0%	16.5%
async_rtc_inter	28.3	14.9	65.5%	0.1%	3.8%	3.8%
async_rtc_sm_inter	25.2	18.2	58.1%	8.0%	16.2%	12.6%
async_rtc_no_imgcrop	26.3	15.9	62.3%	0.2%	5.2%	5.0%
现象 1：所有方法 overlap 严重过高（根因：infer_delay 过大）
理想 RTC 下，infer_delay 应匹配实际延迟 p50：


round_trip p50 = 300ms / 50ms = 6 steps
→ 理想 infer_delay ≈ 6，n_new ≈ 44，overlap% ≈ 12%
   当前 infer_delay ≈ 29，n_new ≈ 17，overlap% ≈ 60%
每次推理只有 17 步 = 0.85s 真正新的动作被使用，而不是理想的 44 步 = 2.2s。系统推理频率约为理想的 2.6×，服务端在无效地浪费算力。

现象 2：n_new 的双峰分布（base 方法）
base 方法（rtc、inter、no_imgcrop）中 n_new 呈双峰：

Mode 1 (1–5 步，~45%)：chunk 在网络尖峰下晚到，robot 已消耗大部分执行窗口，只剩 1–5 步新内容。对应 grpc_send_ms 尖峰模式（~390ms）。
Mode 2 (16–25 步，~42%)：正常到达，infer_delay=29 → 新步数 = 50-29 = 21，波动来自实际延迟与预测延迟的微小差异。
这两个峰直接映射到 grpc_send_ms 的双峰（正常 67ms vs 尖峰 390ms），与之前的 latency 分析完全吻合。

现象 3：SM 方法特有的极端值
SM 方法在 base 方法的双峰之上，叠加了两类特殊事件：

n_new == 0（8–9%，stale chunk）：新 chunk 到达时 robot 执行位置已经超过 chunk 范围，整个 chunk 成为"纯 overlap"——没有任何新动作。这发生在 SM 恢复轨迹（~6s = 120 步）执行期间，新 chunk 的时间戳已落后于实际执行位置。

n_overlap == 0（16–19%，starvation）：旧 chunk 耗尽后新 chunk 才到，robot 在此期间重复执行最后一步。与 timing 报告的 starvation rate（14.6% for async_rtc_sm）一致，且 SM 的 top starvation 事件长达 7.2s（144 步），远超 base 方法的 3.6s（71 步）。

n_new == 50（12–17%）：starvation 结束后第一个 chunk 完全新鲜（无可 blend 的旧 buffer），直接替换。

视觉图解读（对照图像）
面板	async_rtc	async_rtc_sm
Analysis 4（leftover_steps per chunk）	单调增长，RTC 健康	不规律，SM 恢复期间出现跳降
Overlap vs new steps	蓝条（overlap）稳定 ~26 步，橙条（new）短且稳定约 17 步，双峰均匀	橙条高度变化剧烈：大批 0（stale）和 50（post-starvation），蓝条宽度波动大
Analysis 5（L2 norm）	稳定波动 ≈ 2.0–2.5	更高峰值，SM 切换时出现 boundary discontinuity
async_rtc_inter 表现为更短的橙条（n_new mean=14.9 最低），overlap% 最高（65.5%）。inter 方法即使延迟略低（infer_delay mean=28.5）却产生更多 overlap，说明 interpolation 插值引入了更多时间步对齐开销。

结论
60% overlap 是 infer_delay 过大的直接后果，每次推理有效利用率不到 40%。修复 infer_delay（max() → p99()）后，预期 n_new 提升到 ~27 步，overlap% 降至 ~45%。
SM 方法的 n_overlap==0 率（16–19%）是 base（4–8%）的 2–5×，源于恢复轨迹造成的 chunk 时间戳错位，而非 infer_delay，这无法通过换分位数解决。
n_new 双峰结构是 grpc_send_ms 双峰的镜像，与 latency 分析完全一致——不是策略问题，是网络抖动问题。

-----------------

fig8_infer_delay_calibration.png 中的infer_delay 是指完整的round trips 时间吗？每隔一段时间的尖峰 actual 来自那里？gRCP 握手是否会影响？

fig8 中 actual_steps 的完整定义
1. actual_steps 包含什么？
代码 (analyze_timing.py:1405-1411)：


_complete_ms = round_trip_ms + deser_ms + grpc_send_ms + serialize_ms + jpeg_encode_ms
actual_steps = _complete_ms / dt_ms   # dt_ms = 50ms
图轴标签 (rt+deser+grpc+ser+jpeg) / 50.0ms 对应完整的端到端延迟链：


obs_build_start
  ↓ serialize_ms     ≈  2ms   (obs 序列化)
  ↓ grpc_send_ms     ≈ 67ms   (gRPC call: net_c2s + server recv_deser + ACK 返回)
    ── send_wall 时间戳打在此处 ──
  ↓ server inference ≈ 300ms  ┐
  ↓ net_s2c          ≈  6ms   ├─ round_trip_ms = (receive_time − send_wall) ≈ 306ms
    ── receive_time 打在此处 ──┘
  ↓ deser_ms         ≈ 27ms   (客户端反序列化)
  ↓ receive_after_deser
actual_steps = LatencyTracker 实际测量的 complete_s 的近似值（差 13ms 的 total_prep，可忽略）。

不是单纯 round_trip_ms，而是包含了 grpc_send（client→server 网络）、server 推理、net_s2c、和 deser 的全程。

p50 数值验证：306 + 27 + 67 + 2 = 402ms / 50ms = 8.0 steps，与散点图中左侧的点簇一致。

2. 周期性尖峰从哪里来？
scatter 图中明显的两簇（base 方法：主要在 ~8 步和 ~22–25 步）对应 grpc_send_ms 的双峰分布：


正常模式 (90%): grpc_send≈67ms  →  rt≈306ms  →  actual = (306+67+27) / 50 ≈  8 步
单侧尖峰 (5%):  grpc_send≈390ms →  rt≈306ms  →  actual = (306+390+27) / 50 ≈ 14 步
双侧尖峰 (1%):  grpc_send≈390ms →  rt≈1102ms →  actual = (1102+390+100) / 50 ≈ 32 步
物理原因是TCP 重传。gRPC 在 HTTP/2 over TCP 上运行：当一个 TCP 段丢失或延迟，接收方等待 RTO（≈300ms 初始值）后触发重传，造成 grpc_send_ms 从 67ms 跳到 ~390ms（=67+300+23 overhead）。这不是 gRPC 层的问题，而是底层 TCP 的行为。

SM 方法的极端尖峰（async_rtc_sm 在 t≈400s 处出现 ~60+ actual_steps = 3000ms）：这些样本的 complete_s > 2.5s，会被 LatencyTracker 的过滤器 拒绝（不加入窗口），所以不影响 infer_delay 的计算。它们仍然显示在 fig8 的时间轴里，但粉色 sent_infer_delay 曲线保持平稳——这说明过滤器对这类极端事件是正确有效的。

3. gRPC 握手是否影响？
不影响稳态。gRPC 使用 HTTP/2 持久连接：

事件	影响	出现时机
TLS + HTTP/2 SETTINGS 初始握手	一次性，≈100–300ms	服务端启动后首次连接
per-call 握手	不存在	—
gRPC PING keepalive	单帧 ≈1ms，极少竞争	每隔 keepalive_time（默认2小时）
HTTP/2 流重置（stream error）	+1–5ms	偶发，非主要原因
TCP 重传（RTO ≈ 300ms）	⚠ 主要根因	网络拥塞时，频率约 5%
SM 方法的极端高延迟还有另一个机制：SM 恢复轨迹（6s）期间服务端积压了大量 obs（250% vs 135%），恢复结束时短时间内一批推理请求并发，导致服务端 obs_queue 积压→ server 开始处理延迟 obs → round_trip_ms 激增至 2000ms+。这也是 one_way_ms p99 在 SM 方法中高达 2310ms（而 base 方法仅 284ms）的原因。

总结
量	内容	p50	p99	对 infer_delay 影响
actual_steps (fig8)	complete_s 全程步数	~8 步	~32 步	= LatencyTracker 见到的值
round_trip_ms 单独	server_infer + net_s2c	306ms = 6步	1102ms = 22步	仅是 actual_steps 的一部分
尖峰来源	TCP 重传（grpc_send+round_trip 同步尖峰）	—	×4–5	支配 window max → infer_delay=29
gRPC 握手	一次性，无 per-call 开销	—	—	无影响


------

lerobot.async_inference.policy_server 和 lerobot.async_inference.multi_candidate_server 的 obs_queue_timeout设置了1.0， SM 恢复轨迹（6s）期间服务端 应该也只保留最新的obs, 应该是没有推理积压的？那可能是网络拥挤？但是在设置 bg_obs_sender_send_image: false 情况下也只会上传obs.state 用于保持steps?


三个机制的精确分析
1. obs_queue_timeout=1.0 + Queue(maxsize=1)：服务端确实无积压
policy_server.py:88,534-536：


self.observation_queue = Queue(maxsize=1)
...
if self.observation_queue.full():
    _ = self.observation_queue.get_nowait()   # 立即驱逐旧 obs
self.observation_queue.put(obs)              # 放入新 obs
新 obs 到达时旧 obs 被立即驱逐，服务端 永远只持有最新的一个 obs。obs_queue_timeout=1.0 是推理线程等待 obs 的超时，不是积压限制。没有推理积压，这个分析是正确的。

2. bg_obs_sender_send_image: false 的实际作用
smart_robot_client.py:1325-1390 的 _BackgroundObsSender：


RECOVERY / LIFT_RETRY / REWIND_RETRY 期间：
  主线程：每步捕获 obs → 存入 bg 队列（非阻塞，~2ms）→ 继续控制循环
  bg 线程：从队列取 obs → gRPC 发送（skip_inference=True）
服务端收到 skip_inference=True 的 obs 后（policy_server.py:501-506）：


if getattr(obs, "skip_inference", False):
    return False   # 直接返回，不推理、不相似性检查、不更新 last_processed_obs
send_image=false 减少的是 网络带宽，不是推理量：

配置	每步 payload	6s 恢复×120步总流量	GPU 触发
send_image=true	~443 KB	~53 MB	0（skip）
send_image=false	<1 KB	<120 KB	0（skip）
图像永远不被推理（skip），关闭 send_image 只避免 53MB 的无效网络流量。

3. one_way_ms p99 > 2000ms 的真实来源
SM 方法的异常大 one_way_ms 并非推理积压造成，而是 drain() 突发冲洗引起的测量噪声：


恢复期间（6s × 20fps = 120步）：
  bg 队列以 20 obs/sec 积累
  bg 线程以 ~15 obs/sec 发送（send_time≈67ms/次）
  恢复结束时：drain() 被调用 → 等待队列清空
    剩余 obs × send_time ≈ (120-90) × 67ms ≈ 2000ms
drain() 会在 main thread 阻塞等待 bg 队列完全清空。有了 send_image=false 后：

每次 bg gRPC 发送 <1KB → grpc_send_ms 降到 ~5–10ms（而非 67ms）
drain() 耗时：30 × 10ms = 300ms（而不是 2000ms）
这是 send_image=false 最关键的实际效果：不是减少 GPU 负载，而是让 drain() 更快完成，从而缩短 SM 恢复 → 下一次推理之间的空窗期。

SM starvation 的真实机制

RECOVERY 开始 (6s)
  └─ bg_obs_sender 发送 120 × skip_inference obs（轻量，快速）
  └─ 客户端 action queue 依赖旧 chunk 直到耗尽 → starvation

RECOVERY 结束
  └─ drain() 等待 bg 队列清空（若 send_image=false：≈300ms；若 true：≈2000ms）
  └─ must_go 推理请求发出
  └─ 等待 round_trip ≈ 300ms
  └─ 第一个新 chunk 到达

SM starvation 最长 gap = 6s(recovery) + drain_wait + round_trip
  send_image=false: ≈ 6 + 0.3 + 0.3 = 6.6s  （top-1 gap 实测 7.2s，吻合）
  send_image=true:  ≈ 6 + 2.0 + 0.3 = 8.3s
base 方法最长 gap = 3.6s = 只有网络尖峰（无 SM 恢复），完全独立。

结论：obs_queue_timeout 和 send_image=false 的组合设计是合理的——无推理积压（queue=1），bg obs 快速穿透服务端（skip_inference），send_image=false 的核心价值在于缩短 drain() 阻塞时间，从而减少 SM 恢复后的 starvation 窗口。

----
有SM 情况下 drain，warmup, 待轨迹回退后sleep 等待稳定等是否也统计进了流程时间？

各阶段是否进入计时统计？
Starvation gap 的计算方式
analyze_timing.py:1355-1362:


chunk_s["exhaust_time"] = chunk_s["wall_time"] + chunk_s["chunk_size"] * dt
chunk_s["next_arrival"] = chunk_s["wall_time"].shift(-1)
gap_s = next_arrival - exhaust_time        # 正值 = starvation
chunk_s.loc[~same_ep, "next_arrival"] = np.nan  # 跨 episode 边界不计
关键判断：same_ep 是按 chunk record 的 episode 字段来区分的，episode 只在用户主动开始新 episode（按按钮）时递增，SM 内部 RECOVERY 不会增加 episode 计数。因此 SM 恢复全程是同一 episode 内的 starvation。

各阶段在哪些统计里出现？
阶段	耗时 (so101 config)	starvation gap_s	round_trip_ms / grpc_send_ms	actual_steps (fig8)
恢复轨迹执行	90步 × 50ms = 4.5s	✅ 全部计入	❌ 无推理	❌ 无推理 obs
time.sleep(_settle)	1.0s	✅ 全部计入	❌	❌
drain() (等 bg 队列清空)	send_image=false：~0.1–0.3s	✅ 计入	❌	❌
暖机 warmup (2步 obs)	2 × 50ms = 0.1s	✅ 计入	❌ skip_inference obs	❌
等待 must_go 推理 chunk 到达	round_trip ~0.3–1.4s	✅ 计入	✅ 这次计入	✅ 这次计入
完整 SM 最大 starvation gap 推算：


恢复轨迹:  90步 × 50ms  = 4.5s
settle:    1.0s
drain:     ~0.2s  (send_image=false)
warmup:    2×50ms = 0.1s
首次推理:  round_trip p50~0.3s ~ max~1.4s
──────────────────────────────────
总计:  6.1s (p50) ~ 7.2s (spike)
与实测 top-1 starvation gap=7.201s 完全吻合。

对"250% obs 量"数据的重新解读
bg_obs_sender 发送的 obs 全部携带 skip_inference=True：

服务端立即返回 False，不进任何推理队列
不会被计入 server_infer_records.jsonl，不影响 round_trip_ms 或 server_infer_ms 统计
但会出现在 client_obs_sent_records.jsonl 和 one_way_ms 中
所以 SM 方法的 one_way_ms p99 > 2000ms 有一部分来自 bg obs drain 期间突发发送的轻量 gRPC 调用——它们的时钟戳对比（client 发送时间 vs server 接收时间）受 drain 时序影响，不代表真实推理路径的网络延迟。推理链的真实网络延迟（base 和 SM 方法）在 grpc_send_ms 上几乎相同（p50≈67ms，p99≈390ms）。

结论
SM 方法中 starvation rate 14.6% 和 top-5 gap 5–7s 完全反映了恢复流程的全部耗时（轨迹 + sleep + drain + warmup + 首次推理），不是纯网络问题。要降低 SM starvation：

减少 recovery_home_settle_time（当前 1.0s，可酌情降低）
减少 recovery_warmup_steps（当前 2，已经很少）
bg_obs_sender_send_image=false 已将 drain 缩到最短
根本上需要减短 recovery_home_steps（当前 90 步 = 4.5s）

---
fig8_infer_delay_calibration.png中，具体从那里知道actual delay 的尖峰来源是TCP 重传？

fig8 能直接观察到的
能确认的：

actual_steps 的尖峰是双峰分布（8步 正常 vs 22–32步 尖峰），不是渐变退化，说明有某个离散事件触发，而不是持续负载增加。

server_infer_ms 非常稳定（p50=300ms，p99=406ms，仅 1.35×），所以尖峰不来自服务端推理。

actual_steps 的公式是 (rt + deser + grpc + ser) / dt。分拆来看：

分量	p50	p99	尖峰 Δ
server_infer_ms	300ms	406ms	+106ms（稳定）
grpc_send_ms (c2s)	67ms	391ms	+324ms
net_s2c（rt − server_infer）	~6ms	~696ms	+690ms
deser_ms	27ms	100ms	+73ms
→ 两个方向的网络时延同时尖峰，且幅度不对称（s2c 尖峰远大于 c2s）。

为什么推断是 TCP 重传（而不是直接观测到的）
间接证据：

grpc_send_ms 的尖峰幅度 ≈ +324ms，接近 TCP 初始 RTO 典型值（200–500ms）
双峰特征（67ms 或 390ms，中间几乎没有）：TCP 重传是一次性事件（要么没丢包，要么触发 RTO 等待），这与随机排队延迟（会有连续分布）不同
grpc_send_ms p99/p50 = 5.9×（报告里标记为 ⚠），这种倍数关系在纯排队延迟场景中极少见
无法从 fig8 排除的其他原因：

Wi-Fi / 交换机突发拥塞（造成缓冲区积压）
ARP/ICMP 超时重新解析
gRPC 层的流控（HTTP/2 WINDOW_UPDATE 阻塞）
网卡中断合并（interrupt coalescing）引起的批量延迟
要真正确认 TCP 重传，需要

# 在机器人端或 GPU 端捕包期间
ss -s                         # 查看 TCP 重传计数器
netstat -s | grep retransmit  # 观察 retransmit 是否在尖峰时增加
# 或
tcpdump -i eth0 -w capture.pcap 'tcp[13] & 4 != 0'  # 抓 RST/重传帧
或者：若 grpc_send_ms 的尖峰值精确聚集在 RTO 的整数倍（如 300ms、600ms），则强支持 TCP 重传。若散落在 200–800ms 之间则更像拥塞。

结论
从 fig8 能确认的是：

两个方向的网络传输同时出现 ~300ms 级别的离散跳变，服务端计算不参与。

"TCP 重传"是这个现象的最简单解释，但不是 fig8 的直接观测结果。在实验报告中更准确的表述应该是："网络层双向抖动尖峰，幅度与 TCP RTO 量级一致，具体机制需结合 tcpdump 或 ss 统计确认"


----
client-server 是否有重连/重新初始化/等原因导致actual delay 的尖峰？确定是s2c 尖峰远大于 c2s吗？c2s 传输3张resize到224*224的图片+6 joints states,而s2c 返回50*6 的chun_size * action dimention


关键定义澄清
send_wall 的精确位置（base_client.py:379-384）：


_ = self.stub.SendObservations(obs_iter)   # 阻塞等待服务端 ACK
grpc_send_t = time.perf_counter() - t_grpc
send_wall = time.time()                    # 在 gRPC 返回后才打时间戳
SendObservations 是 client-streaming + unary response 模式：客户端流式发送所有 obs chunk，服务端在收完后发 ACK 响应，stub 才返回。所以：

grpc_send_ms 包含了完整的 c2s 网络传输时间（442KB 数据到达服务端并收到 ACK）
send_wall 在服务端已收到 obs 之后才被记录
round_trip_ms = receive_time − send_wall 的起点已经是服务端收到 obs 的时刻
问题 1：重连/重初始化是否导致尖峰？
不会在 round_trip_ms 上产生尖峰，只会导致记录缺失。

若连接中断，stub.SendObservations() 抛出 RpcError，函数返回 False，send_wall 不写入 _send_wall_buf
receive_actions 线程同样捕获 RpcError 并 continue
该 obs 的 round_trip_ms 永远不会被记录（NaN 缺失）→ 表现为分析报告中的 (N NaN) 条目
重连透明重试（gRPC 内部）若恰好在 GetActions 阻塞期间发生：会让 GetActions 等到重连完成再继续，增加 round_trip，但这种情况难以与正常 server-side 延迟区分。

问题 2：s2c 尖峰真的远大于 c2s 吗？——之前的分析是错的
Payload 对比：

方向	内容	大小
c2s	3× 224×224 raw RGB + 6 joints + 元数据	~442 KB（与报告一致）
s2c	50 × 6 × float32 + TimedAction 封装	~2 KB
round_trip_ms 的正确分解：

send_wall 已在服务端收到 obs 之后，所以：


round_trip_ms = receive_time − send_wall
             = 服务端队列等待  +  服务端推理  +  s2c 网络传输
其中 s2c 网络传输：2KB / (100 Mbps / 8) = 0.16ms，可以忽略。

因此：

round_trip 模式	来源	估算
正常 ~306ms	server_infer p50(300ms) + queue_wait(0ms) + s2c(0ms)	✓
尖峰 ~1102ms	server_infer max(689ms) + 排在前一次推理后的等待(~406ms)	689+406=1095ms ≈ p99 ✓
服务端 Queue(maxsize=1)，当新 obs 到达时服务端正在做一次慢推理（689ms max），obs 等到那次推理完才开始，加上自己的 p99 推理（406ms）= 1095ms ≈ round_trip p99 = 1102ms。

之前的错误结论 → 正确版本
分量	之前错误说法	正确分析
grpc_send_ms 尖峰 (+324ms)	c2s 尖峰	✅ 正确：442KB obs，TCP 拥塞
round_trip_ms 尖峰 (+796ms)	s2c 尖峰，s2c "远大于 c2s"	❌ 错误：实际是 服务端队列等待 + 推理抖动；s2c (2KB) 传输 <0.2ms，不是原因
用户的直觉是对的：s2c 只有 ~2KB，不可能在 100Mbps LAN 上产生几百毫秒的网络尖峰。round_trip 的尖峰来源是服务端排队和推理抖动，和网络无关。

----------
结合图fig1_budget.png分析

fig1_budget 综合分析
数据汇总
方法	obs_prep	net_c2s	recv_deser_srv	queue_wait	prepare	infer	deser	Total
async_rtc	12.9	72.2	27.3	124.3	14.2	288.2	31.0	~577ms
async_rtc_sm	13.0	85.7	13.3	125.1	17.6	270.1	32.0	~564ms
async_rtc_inter	12.7	71.5	27.4	123.8	12.7	284.8	31.4	~572ms
async_rtc_no_imgcrop	12.8	71.2	27.7	140.8	34.5	292.6	30.9	~617ms
async_rtc_sm_inter†	12.9	81.5	17.6	(absorbed)	—	303.5	32.0	~529ms
†: async_rtc_sm_inter 无服务端 server_recv records，net_s2c=81.9ms 实为 queue_wait + s2c 的残差。

核心发现 1：queue_wait 是第二大延迟项
所有 base 方法均显示 queue_wait ≈ 124ms，超过 net_c2s（72ms），仅次于 infer（288ms）。

这直接回答了 round_trip_ms 为何远大于 server_infer_ms：


round_trip_ms ≈ queue_wait(124ms) + recv_deser_srv(27ms) + prepare(14ms)
              + preprocess(3ms) + infer(288ms) + srv_ser(2ms) + net_s2c(~0ms)
            ≈ 458ms  →  均值 392ms（join 的不同样本集有偏差）
round_trip_ms p99 = 1102ms 的成因：

queue_wait 最坏情况 ≈ 上一次推理时长 max（~689ms）
自身 server_infer_ms p99 ≈ 406ms
两者叠加 ≈ 1095ms ≈ 1102ms，与 s2c 网络无关
核心发现 2：net_c2s（c2s 网络）≠ grpc_send_ms，且已拆分
grpc_send_ms（报告中 p50=67ms）= net_c2s + recv_deser_srv（服务端 obs 反序列化）：

net_c2s ≈ 72ms = 纯 c2s 网络传输 442KB
recv_deser_srv ≈ 27ms = 服务端 pickle.loads 解码
c2s 带宽：442KB / 72ms ≈ 6.1 MB/s（≈ 49 Mbps），与 Gigabit LAN 相比偏低，说明网络不是 Gigabit，或图像未压缩导致带宽受限
核心发现 3：no_imgcrop 的 prepare 显著更高
方法	prepare	含义
async_rtc / inter	12.7–14.2ms	正常 server obs preprocessing（已 resize 到 224×224）
async_rtc_no_imgcrop	34.5ms	额外 +20ms = 服务端需要执行 crop/resize
no_imgcrop 的额外 20ms 花在服务端图像处理上，同时造成 queue_wait 最高（140.8ms）：prepare 时间更长 → 推理占用时间更长 → 服务端更忙 → 新 obs 等待更长。

核心发现 4：async_rtc_sm 的异常
async_rtc_sm 的 recv_deser_srv = 13.3ms（其他方法 27ms）与 infer = 270.1ms（其他方法 285–293ms）都偏低。这很可能是 SM 方法的 server recv 记录来自不同时间段（包含 skip_inference obs 的时间戳），导致 join 偏差，而非真实推理更快。

对前几条分析的更正和强化
之前的假设	正确结论
"s2c 尖峰远大于 c2s"	❌ s2c = ~2KB，传输 <0.2ms；round_trip 尖峰来自 server queue_wait
"round_trip = server_infer"	❌ 实际 = queue_wait(124ms) + server_infer(307ms)
"net_c2s 是 grpc_send_ms"	❌ grpc_send_ms = net_c2s(72ms) + recv_deser_srv(27ms)
async_rtc_sm_inter "net_s2c=82ms 是真实网络"	❌ = queue_wait + 实际 s2c(<1ms) 的残差
实际端到端延迟的优化优先级（从大到小）：

model infer（288ms）— 模型本身，换更小模型可改善
queue_wait（124ms）— 降低推理频率或减少并发 obs 流量可改善
net_c2s（72ms）— 压缩图像（JPEG）或减小分辨率可改善
recv_deser_srv（27ms）— 与 net_c2s 共享带宽/序列化瓶颈
deser（31ms）— 客户端反序列化，改用 protobuf 等可改善


---
模型固定的情况下推理时长不是会比较稳定（为什么会出现 queue_wait 最坏情况 ≈ 上一次推理时长 max~689ms）？理论上queue_wait 是否应该小于等于server_infer？ 分析如何 queue_wait（124ms）— 降低推理频率或减少并发 obs 流量?

queue_wait 机制分析
Q1: 模型固定为何 server_infer max=689ms？
Pi05 是扩散策略（diffusion policy），推理时长本质上是不确定的：

迭代去噪步数可变: DDPM/DDIM 系列在每一步 denoising 中做条件采样，GPU kernel 调度受 CUDA stream 竞争、内存带宽、SM occupancy 影响，单步抖动会累积
GPU 热态: 长时间评测中 GPU 温度→功耗限制（power throttling）会使 FLOP 吞吐量下降 10-30%
首次推理 / JIT 重编译: PyTorch Dynamo 或 XLA 遇到新输入 shape 会触发重编译，单次可 >1s（被 filter complete_s ≤ 2.5s 截断）
Python GIL + gRPC 后处理: pickle.dumps() 在 inference 结束后序列化，GIL 争用加剧尾延迟
所以 689ms 是分布右尾，不是异常值——对扩散策略而言 p50:300ms vs max:689ms 的 2.3× 比值是正常的。

Q2: queue_wait 理论上是否 ≤ server_infer？
是的，严格成立。

服务端结构是单推理线程 + Queue(maxsize=1)：


obs 到达 → get() 等待 → 开始 inference → 写回 chunk
             ↑
        queue_wait = 从入队到被取出的时间
                   = 当前正在运行的 inference 剩余时间
                   ∈ [0, server_infer_current]
因为队列最大深度=1，新 obs 等待的只有"当前那次推理"的剩余部分，所以：


queue_wait_max ≤ server_infer_max = 689ms  ✓
queue_wait_mean ≈ ρ × E[S_remaining]
               = ρ × E[server_infer] / 2   (M/G/1 残差公式近似)
其中 ρ = server utilization = λ × E[server_infer]，λ = obs 发送频率。

Q3: mean queue_wait=124ms — 是真实值还是统计伪影？
很可能是被 rt_mean 的右偏拉高的伪影。

Budget 中 queue_wait 是通过残差估算的：


queue_wait_mean ≈ round_trip_mean − server_infer_mean − net_s2c_mean
                ≈ 392ms − 280ms − 0ms = 112ms
问题：round_trip_mean 包含 p99 事件（~1100ms），算术均值被拉高远超中位数（p50 ≈ 310ms）。如果改用中位数估算：


queue_wait_median ≈ round_trip_p50 − server_infer_p50 − 0
                  ≈ 310ms − 280ms = 30ms
真实稳态 queue_wait ≈ 30-50ms，124ms 是均值膨胀的统计伪影。

如何降低 queue_wait（实际有效的手段）
手段	机制	预期效果
① infer_delay 用 p99 替代 max()	p99 → 更大 n_new（21→27步）→ 推理触发频率降低 ~23% → λ 降低 → ρ 降低 → queue_wait 降低	稳态 queue_wait -20~30%
② c2s JPEG 压缩	减少 grpc_send_ms（72→~30ms）→ 客户端轮询加快 → λ 上升（反向！）但同时 recv_deser_srv 降低	需权衡；主要收益是 n_overlap 减少
③ 降低 obs 发送频率（显式限速）	直接降低 λ → ρ↓ → 排队等待↓	最直接；但会增加 n_overlap（旧 chunk 复用更多步）
④ multi-candidate 改并行	将串行 for delay in delays 改为 CUDA stream 并行	server_infer_max 降低 → queue_wait_max 降低；需模型层修改
最高性价比：手段①（p99 替代 max()），一行代码改动，降低 λ 同时减少 infer_delay 过估计，对所有 5 种方法均有效，且不改变机器人硬件行为。

---
fig1_budget.png 的统计整个周期600ms 左右，20hz 下actual delay 大概10steps, 跟 fig8_infer_delay_calibration.png 里的 actual steps 的尖刺差太多，分析原因

根本原因：两张图测量的是不同统计矩
fig1_budget — 时间均值堆叠
fig1 中每个组件显示的是 均值（mean），总和 ≈ 600ms：


grpc_send_ms   ~72ms  (p50=67ms, 均值被 p99≈390ms 稍微拉高)
server_queue   ~50ms  (真实稳态均值，剔除尖刺后)
server_infer   ~280ms (p50，扩散策略分布较窄)
net_s2c        ~0ms
deser          ~20ms
serialize      ~20ms
───────────────────
合计           ~440–600ms ≈ 12 steps  ← fig1 展示这个
fig8_actual_steps — 每次推理的实时采样
fig8 中每个点是单次推理周期的完整耗时：


actual_steps = (round_trip_ms + deser_ms + grpc_send_ms + serialize_ms) / dt_ms
尖刺事件（p99）时各组件同时触发高值：


grpc_send_ms   390ms  (TCP 慢路径，p99)
server_queue   689ms  (上次推理恰好还在跑，max=server_infer_max)
server_infer   406ms  (GPU jitter p99)
deser          30ms
───────────────────
合计           1515ms / 50ms = 30 steps  ← fig8 spike
关键：grpc_send 与 queue_wait 的正相关性
这是尖刺幅度如此大的原因——两个最大的组件同时出现：


正常路径:  grpc_send=67ms → 到达时上次推理已结束 → queue_wait≈0ms
慢路径:    grpc_send=390ms → 晚了 323ms 到达 → 上次推理还剩更多时间
                             → queue_wait 正比于迟到的时间 → 尖刺叠加
这就是 1% 的网络慢路径事件 可以造成 2.4× 均值 的单次延迟的原因。

为什么 infer_delay 却被设成 29 steps（≫ 12 steps）
LatencyTracker 用 complete_s（obs.timestamp 到 receive_after_deser 的完整路径）喂窗口 max：


complete_s 在尖刺时 = serialize + grpc_send(390) + queue(689) + infer(406) + deser ≈ 1515ms
max() 返回 ≈ 1450ms → ceil(1450/50) = 29 steps
这个尖刺 每 ~100 次推理发生一次（p99），而窗口 maxlen=100 保留最近 100 个样本（约 20s）。所以窗口内几乎始终存在至少一个 p99 样本，max() 被持续"锁定"在 ~1450ms。

两张图的关系总结
fig1_budget	fig8_actual_steps
统计量	各组件均值之和	每次推理的实时值
代表什么	典型（p50）吞吐压力	时序分布（含尖刺）
值	~600ms ≈ 12 steps	稳态 8-12 steps，尖刺 28-30 steps
影响什么	robot 平均等待时间	LatencyTracker.max() → infer_delay
结论：fig1 的 12 steps 才是真实的典型延迟；fig8 的 29-step 尖刺是 p99 事件，但因为 max() 把它"冻结"进 infer_delay，导致所有推理都按 29 steps 偏移执行——这就是 infer_delay 过估计的完整因果链。

----
结合 fig2_time_series.png 分析

Fig2 直接揭示三个分层事实
① server_infer 是稳定的（p95/median ≈ 1.2×）
每个方法的 server_infer 时序面板几乎是水平线：median ≈ 295-323ms，p95 ≈ 337-388ms，比例 ~1.15-1.25×。这直接证明 GPU 推理本身是稳定的，不是尖刺的来源。

② queue_wait 是极度突发的（p95/median ≈ 10-720×）
queue_wait 面板显示：绝大多数时刻接近 0，但周期性跳到 500-720ms。这是 round_trip 尖刺的直接来源：


round_trip = server_queue_wait + server_infer + net_s2c(≈0)
正常: queue_wait(0ms) + infer(300ms) = 300ms
尖刺: queue_wait(680ms) + infer(360ms) = 1040ms ≈ p95=972ms ✓
③ gRPC send 是双峰分布（median=67ms，p95=365ms）
gRPC 面板呈现散点式尖刺，p95 ≈ 365ms（5.4× median）。这是 TCP 慢路径触发的。

用 Fig2 数据重建 actual_steps 分布
actual_steps = (round_trip + grpc_send + deser) / dt


典型时刻 (各组件取 median):
  (306 + 67 + 27) / 50 = 400ms / 50 = 8 steps   ← fig8 中的基线密集带

尖刺时刻 (round_trip p95 + gRPC p95):
  (972 + 365 + 47) / 50 = 1384ms / 50 = 27.7 steps ≈ 28 steps  ← fig8 尖刺 ✓

LatencyTracker 测量的 complete_s ≈ actual_steps × dt:
  peak ≈ 1384ms → max() ≈ 1440ms → ceil(1440/50) = 29 steps → infer_delay=29
与 fig1 600ms 的统一解释
fig1 "600ms 整体周期" 是各组件的算术均值之和，均值被右尾拉高：


round_trip_mean ≈ 390ms  (median=306ms 被 p95=972ms 向右拉)
gRPC_mean       ≈ 100ms  (median=67ms  被 p95=365ms 向右拉)
deser_mean      ≈ 27ms
─────────────────────────
mean total      ≈ 517-600ms → ~12 steps (均值)

但分布并非对称：
  典型 (p50): 400ms → 8 steps
  均值 (mean): ~600ms → 12 steps   ← fig1 看到的
  尖刺 (p95+): 1384ms → 28 steps  ← fig8 spike
  LatencyTracker max: ~1450ms → 29 steps → infer_delay
这三者的关系：8 steps（典型）→ 12 steps（均值/fig1）→ 28-29 steps（max/fig8 尖刺/infer_delay），是同一个重尾分布的不同统计量。

Fig2 揭示的额外规律
SM 方法 p95 异常低（async_rtc_sm: 569ms vs 其他 800-1000ms）

SM 每次恢复后服务端队列被清空、推理重新冷启动，打断了 queue_wait 的级联积累。基础 RTC 方法中，高频推理请求会持续喂压服务端 → queue_wait 均值更高（67ms vs 0.7-6ms）。

async_rtc queue_wait median=67ms 远高于 inter 方法（0.7ms）

interpolation 方法的 n_new 更大 → 每次 chunk 使用更多步 → 触发下次推理的频率更低（λ↓）→ 服务端利用率低 → queue 几乎为空。这是 interpolation 对 latency budget 的隐性贡献。
------

实现 ① + ② 的优化

latency_tracker.py:32 — 窗口从 100 → 30

在 5 inf/s 下：100 samples ≈ 20s → 30 samples ≈ 6s
p99 尖刺在网络恢复后最多 6s 内即从窗口淡出（原来需要 20s）
base_client.py:781,786 — max() → percentile(0.99)

_max_lat 从窗口最大值改为 p99，消除单个极端尖刺对 infer_delay 的永久占据
base_client.py:515,523,794 — 日志标签同步更新为 latency_p99 / Latency tracker p99

两个优化叠加的预期效果：infer_delay 从 29 steps 降至 ~16-18 steps，且网络恢复后 6s 内自动回落，不再持续过估计

----

根本原因链：一次 actual 尖刺 → 进入窗口 → p99 ≈ max → 高位锁定 ~50s

修改建议（从低到高复杂度）
① 降低百分位数（一行，立竿见影）

# base_client.py:781（当前）
_max_lat = self.latency_tracker.percentile(0.99) or 0.0

# 建议改为 p75 或 p80
_max_lat = self.latency_tracker.percentile(0.75) or 0.0
百分位	maxlen=30 对应样本	预估 sent（散点图读数）
p99（当前）	第 30/30 = max	20-44 steps
p90	第 27/30	18-25 steps
p75	第 22/30	~15-18 steps
p50	第 15/30	~10-12 steps
p75 能过滤掉最坏的 25% 尖刺，同时保留对 actual 典型值（8-15 steps）的合理安全裕量。

② 入队前截断离群值（防止窗口被单次尖刺污染）
在 latency_tracker.add() 之前，将极端值 clamp 到 3× 近期中位数：


# base_client.py，latency_tracker.add(complete_s) 之前插入
if len(self.latency_tracker) >= 5:
    _median = self.latency_tracker.percentile(0.5) or complete_s
    complete_s_clamped = min(complete_s, _median * 3.0)
else:
    complete_s_clamped = complete_s
self.latency_tracker.add(complete_s_clamped)
效果：1450ms 尖刺被截断为 ~3×400ms = 1200ms → infer_delay 从 29 降到 24，且更快恢复。

③ 双分量估算（分离"典型延迟" + "安全裕量"）
从散点图可以看到：actual 典型值 8-12 steps，偶发尖刺 25-45 steps。这两部分成因不同，应该分开处理：


# base_client.py Stage 2 infer_delay 计算部分
_p50_lat  = self.latency_tracker.percentile(0.50) or 0.0  # 典型延迟
_p90_lat  = self.latency_tracker.percentile(0.90) or 0.0  # 尖刺保护

# 用 p50 作为基础，加固定步数安全缓冲
_SPIKE_BUFFER_STEPS = 3
_infer_delay = _to_delay_steps(_p50_lat) + _SPIKE_BUFFER_STEPS

# 用 p90 做上限 cap（不被极端事件完全控制）
_infer_delay = min(_infer_delay, _to_delay_steps(_p90_lat))
效果：


p50_lat ≈ 400ms → 8 steps + 3 buffer = 11 steps
p90_lat ≈ 600ms → 12 steps（上限 cap）
→ sent ≈ 11-12 steps，比当前 25-44 steps 大幅改善
效果对比预估
方案	预估 sent infer_delay	散点图位置
当前（p99，maxlen=30）	20-44 steps	对角线 2-4× 上方
① p75	~15-18 steps	对角线 1.5× 上方
① + ② p75 + clamp	~12-15 steps	接近对角线 1×
③ 双分量	~10-13 steps	接近理想对角线
偶发 actual 尖刺（25-45 steps）超过 sent 时：force_must_go（10步=500ms）机制会自动触发补救，无需让 infer_delay 覆盖 p99 极端事件。


根本原因链：一次 actual 尖刺 → 进入窗口 → p99 ≈ max → 高位锁定 ~50s

修改建议（从低到高复杂度）
① 降低百分位数（一行，立竿见影）

# base_client.py:781（当前）
_max_lat = self.latency_tracker.percentile(0.99) or 0.0

# 建议改为 p75 或 p80
_max_lat = self.latency_tracker.percentile(0.75) or 0.0
百分位	maxlen=30 对应样本	预估 sent（散点图读数）
p99（当前）	第 30/30 = max	20-44 steps
p90	第 27/30	18-25 steps
p75	第 22/30	~15-18 steps
p50	第 15/30	~10-12 steps
p75 能过滤掉最坏的 25% 尖刺，同时保留对 actual 典型值（8-15 steps）的合理安全裕量。

② 入队前截断离群值（防止窗口被单次尖刺污染）
在 latency_tracker.add() 之前，将极端值 clamp 到 3× 近期中位数：


# base_client.py，latency_tracker.add(complete_s) 之前插入
if len(self.latency_tracker) >= 5:
    _median = self.latency_tracker.percentile(0.5) or complete_s
    complete_s_clamped = min(complete_s, _median * 3.0)
else:
    complete_s_clamped = complete_s
self.latency_tracker.add(complete_s_clamped)
效果：1450ms 尖刺被截断为 ~3×400ms = 1200ms → infer_delay 从 29 降到 24，且更快恢复。

③ 双分量估算（分离"典型延迟" + "安全裕量"）
从散点图可以看到：actual 典型值 8-12 steps，偶发尖刺 25-45 steps。这两部分成因不同，应该分开处理：


# base_client.py Stage 2 infer_delay 计算部分
_p50_lat  = self.latency_tracker.percentile(0.50) or 0.0  # 典型延迟
_p90_lat  = self.latency_tracker.percentile(0.90) or 0.0  # 尖刺保护

# 用 p50 作为基础，加固定步数安全缓冲
_SPIKE_BUFFER_STEPS = 3
_infer_delay = _to_delay_steps(_p50_lat) + _SPIKE_BUFFER_STEPS

# 用 p90 做上限 cap（不被极端事件完全控制）
_infer_delay = min(_infer_delay, _to_delay_steps(_p90_lat))
效果：


p50_lat ≈ 400ms → 8 steps + 3 buffer = 11 steps
p90_lat ≈ 600ms → 12 steps（上限 cap）
→ sent ≈ 11-12 steps，比当前 25-44 steps 大幅改善
效果对比预估
方案	预估 sent infer_delay	散点图位置
当前（p99，maxlen=30）	20-44 steps	对角线 2-4× 上方
① p75	~15-18 steps	对角线 1.5× 上方
① + ② p75 + clamp	~12-15 steps	接近对角线 1×
③ 双分量	~10-13 steps	接近理想对角线
偶发 actual 尖刺（25-45 steps）超过 sent 时：force_must_go（10步=500ms）机制会自动触发补救，无需让 infer_delay 覆盖 p99 极端事件。

----------

分析：当前实现的通用性问题
_SPIKE_BUFFER_STEPS = 3 是以步数硬编码的，不随 fps 缩放：


20Hz (dt=50ms):  3 步 = 150ms  ✓
10Hz (dt=100ms): 3 步 = 300ms  → 过大，p90 cap 频繁触发
30Hz (dt=33ms):  3 步 =  99ms  → 过小，安全裕量不足
用 10Hz 举例：


p50=400ms → 4步, +3buffer → 7步
p90=600ms → 6步
min(7, 6) = 6  ← p90 cap 直接接管，双分量退化为单 p90
正确做法：buffer 用时间（秒）定义，动态折算为步数：


_SPIKE_BUFFER_S = 0.15          # 150ms，与频率无关
_SPIKE_BUFFER_STEPS = max(1, math.ceil(_SPIKE_BUFFER_S / dt))

fps	dt	_SPIKE_BUFFER_STEPS	p50=400ms→步	+buffer	p90=600ms→步	最终
10Hz	100ms	ceil(0.15/0.1)=2	4步	4+2=6	6步	min(6,6)=6
20Hz	50ms	ceil(0.15/0.05)=3	8步	8+3=11	12步	min(11,12)=11
30Hz	33ms	ceil(0.15/0.033)=5	12步	12+5=17	18步	min(17,18)=17
三种频率下 buffer 在时间上均为 ≈150ms，双分量逻辑行为一致。_SPIKE_BUFFER_S = 0.15 是唯一需要根据策略调整的参数。

_SPIKE_BUFFER_STEPS 改为可配置参数？

三处完全一致。变更总结：

文件	改动
configs.py:199	新增 spike_buffer_s: float = 0.15，含完整 help 文档
configs.py:356	__post_init__ 验证 >= 0
base_client.py:790	_SPIKE_BUFFER_STEPS = max(1, ceil(self.config.spike_buffer_s / dt))
async_client_sm.yaml:42	注释说明参数含义，默认值注释掉（可直接取消注释修改）
使用方式：


# 更激进（更新鲜的动作，偶发饥饿）
python -m lerobot.async_inference.smart_robot_client \
    --config_path ... --spike_buffer_s=0.05

# 更保守（减少饥饿风险）
--spike_buffer_s=0.25

----------------
分析：项目 /data/users/huoyuan/VLA/LeRobot/lerobot_v0.5.2 下 lerobot.async_inference.sim_test.run_libero_smart_test 和 lerobot.async_inference.policy_server 的流程中，如果client 端开启RTC或者没有开启RTC , server 端设置了 --inference_latency>0 的话， server 端会如何使用inference_latency？

libero 仿真 vs 真机的具体场景
libero sim（smolvla 快速策略，实际推理 < 33ms）：


_elapsed = queue_wait(~0) + infer(~20ms) + serialize(~5ms) = 25ms
inference_latency = 33ms → sleep_s = 33 - 25 = 8ms

complete_s ≈ 33ms + net + deser → infer_delay 与真机近似
→ inference_latency 有效，模拟真实延迟条件，保证 sim 中 infer_delay 有意义。

真机 pi05（慢策略，推理 ~300ms）：


_elapsed = queue_wait + infer(300ms) + serialize ≫ 33ms
sleep_s = 0 → 无效（触发一次 warning）
→ 服务端打印警告："inference_latency has no effect for this policy"，建议设置为 0。

eval 脚本中 --inference_latency 的正确使用
场景	建议值	原因
libero sim + smolvla（<33ms）	0.033（33ms）或目标仿真延迟	让 sim 的 round-trip ≈ 真机
libero sim + pi0/pi05（>200ms）	0.0	实际推理已超过任何合理 inference_latency
真机任何策略	0.0	使用真实测量延迟
故意加压测试（压力测试 RTC）	任意正值	人工拉高 infer_delay 验证 RTC 鲁棒性

-----
分析：在libero sim 中，inference_latency 是 queue_wait + infer + serialize 的最小总时长限制, 其中忽略了客户端序列化，网络传输，服务端反序列化，预处理等用时，是否会导致 计算的 inference latency： queue_wait + infer + serialize 比实际的偏小，导致_sleep_s = max(0, inference_latency - _elapsed) 额外增加了延时？


用户的分析是正确的：存在系统性偏差
精确的时间边界对比

obs.timestamp (obs 构建完成前)
    │ client_serialize_ms        (~20ms)   ← _elapsed 不含
    │ grpc_send_ms               (~67ms)   ← _elapsed 不含
    │ recv_deser_srv_ms          (~27ms)   ← _elapsed 不含
    │ server_queue_wait_ms       ← getactions_starts = perf_counter() ↓
    │ server_prepare_ms          (raw→lerobot)          ┐
    │ server_preprocess_ms       (tokenize/normalize)   │ _elapsed
    │ server_infer_ms            (model forward)        │ = queue_wait
    │ server_postprocess_ms                             │ + pipeline
    │ server_serialize_ms        (pickle.dumps chunk)   ┘ + serialize
    │ ← _elapsed 到此结束 ─────────────────────────────────────────
    │ sleep_s                    ← 基于 _elapsed 计算   ← complete_s 含此
    │ net_s2c_ms                 (~0ms, 1.2KB)          ← _elapsed 不含
    │ client_deser_ms            (~15ms)                ← _elapsed 不含
receive_after_deser → complete_s
_elapsed 漏掉的组件（@p50 条件）：

组件	典型值
client_serialize_ms	~20ms
grpc_send_ms	~67ms
recv_deser_srv_ms	~27ms
net_s2c_ms	~0ms
client_deser_ms	~15ms
合计 offset	~129ms
sleep_s 引发的额外延时链
当 _elapsed < inference_latency（fast policy 在 sim 中的典型情况）：


sleep_s = max(0, inference_latency - _elapsed)

complete_s = client_overhead(~129ms) + _elapsed + sleep_s + net_s2c + client_deser
           = 129ms + _elapsed + (inference_latency - _elapsed)
           = 129ms + inference_latency                          ← 恒成立！
结论：complete_s 比 inference_latency 多出固定的 client/network overhead（~129ms），与实际推理速度无关。

对 infer_delay 的系统性高估
目标 complete_s	用户设置 inference_latency	实际 complete_s	实际 infer_delay（@20Hz）
300ms（模拟真机）	300ms	~429ms	ceil(429/50) = 9
300ms（模拟真机）	171ms	~300ms	ceil(300/50) = 6 ← 正确
500ms（压力测试）	500ms	~629ms	ceil(629/50) = 13（预期10）
正确的 libero sim 设置公式：


inference_latency_correct = target_complete_s - client_network_overhead
                          ≈ target_complete_s - 129ms

例：target 300ms → inference_latency = 300 - 129 = 171ms
   target 500ms → inference_latency = 500 - 129 = 371ms
其中 client_network_overhead 由实测数据得：


# 从 analyze_timing 输出估算 overhead
overhead = complete_s_mean - (server_infer_mean + server_queue_wait_mean + server_serialize_mean)
         ≈ sent.grpc_send_ms.mean() + client_serialize_ms.mean() + client_deser_ms.mean()
libero sim vs 真机的实际影响
真机（pi05，inference ~300ms）：


_elapsed ≈ 300ms > inference_latency=33ms → sleep_s = 0
→ inference_latency 完全无效，无偏差问题
libero sim（smolvla，inference ~20ms）：


_elapsed ≈ 20ms < inference_latency=300ms → sleep_s = 280ms
complete_s ≈ 129 + 300 = 429ms  （比预期的300ms 多出 43%）
infer_delay = ceil(429/50) = 9 步  （比目标 6步 高出 50%）
这就是为什么 libero sim 中 infer_delay 会系统性过高——inference_latency 的 timer 不包含 client 端开销，导致 sleep 过补，进而拉高 complete_s 和 infer_delay。
-------

给出优化建议，例如原来的_elapsed = queue_wait + infer + serialize 加上 grpc_send_ms， 其余net_s2c_ms， client_deser_ms 相对小暂时忽略？是否有更好优化方法


分析：用户建议正确，但有更好的方案
用户建议（加入 grpc_send_ms）
修正后：


adjusted_elapsed = _elapsed + grpc_send_client(~87ms)
_sleep_s = max(0, inference_latency - adjusted_elapsed)

complete_s ≈ (129ms overhead) + inference_latency - (87ms correction)
           = inference_latency + 42ms    ← 从偏差 129ms 降到 42ms
结论：有效，修正约 67% 的偏差。 但还差 recv_deser_srv(~27ms) + client_deser(~15ms) ≈ 42ms 无法纠正。

更好的方案：以 obs.timestamp 为锚点
核心思路： 与其从 server 侧计时，不如直接用 client 打下的 obs.timestamp（time.time() 墙钟），让 sleep 结束的时刻与 obs 出发时刻对齐。


# 当前（server 侧计时，有系统偏差）
_elapsed = time.perf_counter() - getactions_starts
_sleep_s = max(0.0, self.config.inference_latency - _elapsed)

# 方案 A：obs.timestamp 锚点（同机/NTP 环境）
_sleep_s = max(0.0, obs.get_timestamp() + self.config.inference_latency - time.time())
为什么精确：


sleep 结束时刻（dispatch_wall）≈ obs.timestamp + inference_latency

complete_s = receive_after_deser - obs.timestamp
           ≈ dispatch_wall + net_s2c(0ms) + client_deser(15ms) - obs.timestamp
           ≈ inference_latency + 15ms   ← 仅剩 s2c+deser 的 15ms 误差
只需 一行代码替换，残差从 129ms 降至 ~15ms。

约束： 要求 client 和 server 的 time.time() 对齐：

同机（libero sim）：误差 < 0.1ms，完美 ✓
跨机（真机 LAN + NTP）：典型偏差 < 1ms，可接受 ✓
跨机（无时钟同步）：不可用
方案 B：client 传 overhead 给 server（跨机无 NTP 安全方案）
在 TimedObservation 中增加字段，client 把已测量的各组件时间传给 server：


# TimedObservation 新增字段（proto + 数据类）
grpc_send_ms:    float = 0.0   # client 测量的 gRPC 发送耗时
serialize_ms:    float = 0.0   # client 序列化耗时（pickle.dumps obs）
Server 端用这些补偿：


_client_overhead_s = (
    getattr(obs, 'grpc_send_ms', 0.0) +
    getattr(obs, 'serialize_ms', 0.0)
) / 1000.0   # ~87ms p50，~385ms p99

_sleep_s = max(0.0, self.config.inference_latency - _elapsed - _client_overhead_s)
残差：recv_deser_srv(~27ms) + net_s2c(0ms) + client_deser(~15ms) ≈ 42ms。

三种方案对比
方案	complete_s 偏差	代码改动	跨机
当前实现	+129ms	—	✓
用户建议（+grpc_send）	+42ms	小（obs 增 1 字段）	✓
方案 A（obs.timestamp 锚点）	+15ms	1 行替换	NTP 需要
方案 B（client 传 overhead）	+42ms	中（增 2 字段）	✓
建议
场景	推荐
libero sim（同机）	方案 A，1 行替换，精度最高
真机 LAN（NTP 同步）	方案 A，偏差 < 2ms
真机 LAN（无 NTP）	方案 B，安全，偏差 ~42ms
实现方案 A 的代码改动（policy_server.py）：


# 替换 _elapsed 和 _sleep_s 的计算
_elapsed = time.perf_counter() - getactions_starts
# Anchor to client wall-time so inference_latency ≈ complete_s.
# Residual: net_s2c + client_deser ≈ 15 ms (same-machine sim: < 1 ms).
_sleep_s = max(0.0, obs.get_timestamp() + self.config.inference_latency - time.time())

----

实现方案 A（obs.timestamp 锚点）
改动总结（policy_server.py:389-419）：


旧逻辑：
  _elapsed = server_queue_wait + server_infer + serialize  (server 侧 perf_counter)
  _sleep_s = max(0, inference_latency - _elapsed)
  → complete_s ≈ inference_latency + 129ms (多出 client/网络开销)

新逻辑：
  _dispatch_target = obs.get_timestamp() + inference_latency  (client 墙钟锚点)
  _sleep_s = max(0, _dispatch_target - time.time())
  → complete_s ≈ inference_latency + 15ms (仅剩 net_s2c + client_deser)
场景	修改前 complete_s	修改后 complete_s	误差改善
libero sim（同机）	inference_latency + 129ms	inference_latency + ~1ms	128ms ✓
真机 LAN + NTP	inference_latency + 129ms	inference_latency + 15ms	114ms ✓
inference_latency=0	0 sleep	0 sleep（else 分支）	无变化 ✓


------

分析：lerobot.async_inference.policy_server 和 lerobot.async_inference.sim_test.run_libero_test 的client-server 测试流程中，测试分析结果如 /data/users/huoyuan/VLA/LeRobot/lerobot_v0.5.2/outputs/eval_thesis/libero_v1.0/libero_object/async_nortc/smolvla/latency_s0/EH20/timing_analysis/目录下的结果所示，如 fig1_budget.png，为什么obs_prep 需要那么长时间？net_s2c 时间反而比net_c2s(传输包含图片，obs.state) 时间还长?


Q1：obs_prep = 133ms 为什么这么长？
直接原因：libero sim 的 obs 采集 = 物理仿真 step + MuJoCo 多相机渲染


obs_capture_ms: mean=133ms, p50=127ms, p95=303ms, max=466ms
payload_kb: 760KB (!)
jpeg_encode_ms: 0ms (未压缩)
对比真机（SO-101）：


obs_capture_ms: ~5ms     (读硬件寄存器 + 摄像头帧)
payload_kb:    ~442KB
Libero sim 的 _capture_raw_obs() 调用链：


env.step()              # MuJoCo 物理求解 (~10ms)
env.render(camera=...)  # OpenGL/MuJoCo 渲染，每个相机独立渲染
                        #   e.g. 3个 256×256 RGB = 3×196KB = 588KB
                        #   或 512×512 更大分辨率
                        # 渲染时间 = 30-200ms/帧，且高度不稳定
760KB 未压缩 payload 说明：

多个高分辨率相机（无 obs_image_resize_hw 或 obs_image_jpeg_quality）
MuJoCo EGL/GLFW 渲染在不同帧有不同开销 → std=96ms 的方差
优化建议： 在 sim client config 设置 obs_image_jpeg_quality: 85 或 obs_image_resize_hw: [224, 224]，obs_prep 不变但 payload 从 760KB 降到 ~30KB，grpc_send_ms 也会下降。

Q2：net_s2c (76.8ms) > net_c2s (7.5ms) 的根本原因
这是一个标签错误（labeling artifact），不是真实的 s2c 网络传输时间。

关键：server_timing 目录不存在

Loaded recv  : (not found)
Loaded infer : (not found)
当 server_infer 数据不可用时，budget 走 else 分支

# analyze_timing.py:291-295 (else 分支)
else:
    rt  = chunk["round_trip_ms"].mean()           # = 350.46ms
    si  = chunk["server_infer_ms"].mean()         # = 273.67ms (从 action_chunk.inference_time_s 来)
    budget["server_infer"] = si
    budget["net_s2c"]      = max(0.0, rt - si)   # = 350.46 - 273.67 = 76.79ms
round_trip_ms 的实际组成

round_trip_ms = server_queue_wait + server_infer + net_s2c_wire
              = 76ms              + 274ms         + ~0ms (loopback)

"net_s2c" = round_trip_ms - server_infer
          = (server_queue_wait + server_infer + net_s2c_wire) - server_infer
          = server_queue_wait + net_s2c_wire
          ≈ server_queue_wait ≈ 76ms
真实的 s2c 网络传输（同机 loopback，50×6×float32=1.2KB）= < 0.1ms，完全被 server_queue_wait 掩盖。

net_c2s (7.5ms) 同样不是纯网络时间

grpc_send_ms = net_c2s_wire + server_recv_deser + gRPC ACK
             ≈ 0ms (760KB loopback ~0.5ms) + 7ms (760KB pickle.loads) + ~0.5ms
             ≈ 7.5ms
两个问题的完整图景
标签	实际含义	真实组成
obs_prep 133ms	MuJoCo 物理 + 多相机渲染	env.step() + glReadPixels × n_cameras
net_c2s 7.5ms	gRPC 发送（loopback）	server_recv_deser(~7ms) + wire(~0ms)
server_infer 274ms	smolvla 模型推理	preprocess + forward + postprocess
net_s2c 76.8ms	server_queue_wait（标签错误）	round_trip - server_infer，loopback 下 wire ≈ 0ms
deser 7.9ms	client pickle.loads chunk	50步 × 6维 action = 1.2KB，但 gRPC 框架开销
根治方法： 启动 server 时加 --timing_output_dir，获取 server_infer_records.jsonl，budget 走 if 分支，server_queue_wait 会单独显示，net_s2c 将还原为真实的 ~0ms (loopback) 或 ~1ms (LAN)

-----

根本原因：SIGTERM 不触发 Python finally 块
policy_server 的 shutdown 流程

# policy_server.py:810-816
try:
    server.wait_for_termination()      # 阻塞等待
except KeyboardInterrupt:              # 只捕获 SIGINT (Ctrl+C)
    server.stop(grace=2)
finally:
    policy_server.save_timing()        # ← 只有 try/except 正常退出才执行
脚本 stop_server() 的实际信号

# eval-scripts/.../async_libero_smolvla_eval-nortc-sm.sh:128-133
kill "$pid"           # ← SIGTERM (默认)
for i in $(seq 1 10):
    kill -0 "$pid" || break; sleep 1
done
kill -9 "$pid"        # ← SIGKILL (兜底)
Python 的信号处理语义
信号	Python 默认行为	finally 执行？	save_timing() 执行？
SIGINT (Ctrl+C)	抛出 KeyboardInterrupt	✅	✅
SIGTERM (kill pid)	立即终止进程（C 层面）	❌	❌
SIGKILL (kill -9)	内核强杀，无回调	❌	❌
SIGTERM 在 Python 默认情况下不经过 Python 解释器，直接在 C 层终止进程，try/except/finally 完全跳过。这就是 server_infer_records.jsonl 未保存的原因。

修复方案：注册 SIGTERM 处理器
在 policy_server.py 的 serve() 函数中添加 SIGTERM → KeyboardInterrupt 的转换：

Read policy_server.py (lines 2-21)
Bash Find import section of policy_server
IN
grep -n "^import\|^from\|^import signal\|^import os" \
  /data/users/huoyuan/VLA/LeRobot/lerobot_v0.5.2/src/lerobot/async_inference/policy_server.py | head -20

OUT
29:import logging
30:import pickle  # nosec
31:import threading
32:import time
33:from concurrent import futures
34:from dataclasses import asdict
35:from pprint import pformat
36:from queue import Empty, Queue
37:from typing import Any
39:import draccus
40:import grpc
41:import torch
43:from lerobot.policies import get_policy_class, make_pre_post_processors
44:from lerobot.policies.rtc import RTCConfig
45:from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
46:from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, RelativeActionsProcessorStep
47:from lerobot.utils.constants import OBS_STATE
48:from lerobot.utils.utils import init_logging
49:from lerobot.transport import (
53:from lerobot.transport.utils import receive_bytes_in_chunks
Edit policy_server.py
Added 1 line
Edit policy_server.py
Added 8 lines
验证：

Read policy_server.py (lines 809-833)
修改后的完整信号处理流程：


脚本: kill "$pid"
  → Python 收到 SIGTERM
  → _sigterm_handler() 执行: raise KeyboardInterrupt
  → except KeyboardInterrupt: server.stop(grace=2)
  → finally: policy_server.save_timing()          ← 现在能执行了
  → server_infer_records.jsonl 等文件写入磁盘 ✓

脚本: kill -9 "$pid" (如果 10s 内未退出)
  → SIGKILL 无法捕获，save_timing() 仍不执行
  → 但由于 save_timing() 很快（~ms），SIGTERM 路径应在 10s 内完成
注： SIGKILL 永远无法被捕获（内核直接杀进程），但脚本给了 10s 宽限期。save_timing() 写文件通常 < 1s，所以正常情况下 SIGTERM → save → 进程退出会在 10s 内完成，不会触发 SIGKILL。


--------------

Q1：数据来源是日志还是推算？
图	数据	来源
rtc_fig1 Panel 1 (n_overlap/n_new)	n_overlap, n_new	✅ 直接日志 client_aggregate_records.jsonl
fig8 sent infer_delay	sent["infer_delay"]	✅ 直接日志 client_obs_sent_records.jsonl
fig8 actual_steps	(rt + deser + grpc + ser) / dt	⚙️ 推算，从两个 JSONL join 后计算
Q2：绘制方式是否正确？
actual_steps 的计算公式重建了 complete_s = receive_after_deser - obs.timestamp，完全正确，与 LatencyTracker.add() 的输入一致。

Q3：为什么 fig8 没有 leftover_steps？
leftover_steps（obs 发送时 buffer 剩余步数）在 client_obs_sent_records.jsonl 中已记录但未绘制。已修复：fig8 左图加入橙色散点 pre-leftover_steps，三条线的理想关系：


leftover_steps ≈ sent infer_delay ≈ actual_steps   ← 三者重合 = 完美校准
sent >> actual                                       ← LatencyTracker 过估计
leftover < actual                                    ← buffer 耗尽前推理未到 = 饥饿风险
Q4-Q5：n_overlap 与 old/transition/freed 的关系及可视化
它们是不同层面的概念：

n_overlap（queue merge 层面）：新 chunk timestep 与 client 当前 queue 重叠步数
old/freed（RTC guidance 层面）：old = infer_delay_used，freed = chunk_size - old
理想校准下：n_overlap ≈ infer_delay_used（buffer 剩余步恰等于推理延迟估计）
已新增 Panel 3（rtc_fig1 底部）：

堆叠柱状图：old（惰引导区） + freed（新策略输出区），蓝色散点叠加 n_overlap
关键标注：n_overlap − old 偏差说明校准质量
≈0 → 校准良好
>0 → chunk 比预期早到（buffer 充裕，浪费 GPU）
<0 → chunk 比预期晚到（饥饿风险）


------


根本原因：recv_deser_srv 包含网络传输等待时间，不只是反序列化
关键数字

payload_kb = 2701 KB ≈ 2.7 MB  ← 640×480×3通道×3相机 ≈ 2.76MB（未压缩）
serialize_ms   = 8.5ms   (client: pickle.dumps 纯内存操作)
net_c2s        = 62.8ms  (网络传输第一字节到达时间)
recv_deser_srv = 193.2ms (server: 等待剩余字节到达 + pickle.loads)
grpc_send_ms   = 256ms   = net_c2s + recv_deser_srv = 62.8 + 193.2 ✓
recv_deser_srv 计时器实际测量的是什么

# policy_server.py:261-269
start_deserialize = time.perf_counter()          # ← 计时开始

received_bytes = receive_bytes_in_chunks(        # ① 循环读 gRPC 流直到全部到达
    request_iterator, None, ...)                 #   等待 2.7MB 剩余字节通过网络
timed_observation = pickle.loads(received_bytes) # ② pickle.loads(2.7MB)

deserialize_time = time.perf_counter() - start  # ← 计时结束 = ①+②
两个阶段加在一起：


recv_deser_srv = ① 剩余 2.7MB 网络传输  (~140ms)
               + ② pickle.loads(2.7MB)   (~50ms)
               = 193ms
而 net_c2s = grpc_send_ms - recv_deser_srv = 62.8ms 只是第一字节到达前的握手/RTT 时间。

不是日志 bug，而是 2.7MB payload 的物理后果
组件	含义	时长	为何快/慢
serialize 8.5ms	client pickle.dumps(obs)	快	纯内存操作，无 I/O
net_c2s 62.8ms	第一字节到达 server 的时间	中	TCP 握手 + 第一段
recv_deser_srv 193ms	server 收完全部字节 + pickle.loads	慢	需等待剩余 ~2.7MB 传输完毕
对比标准 SO-101（payload=442KB，小 6×）：


recv_deser_srv ≈ 27ms  (442KB)
recv_deser_srv = 193ms (2701KB)   比例 ≈ 2701/442 ≈ 6× payload → ≈7× 时间 ✓
smolvla_v1.0-640-480-nolog 为何 payload 这么大
模型名称 640-480 说明输入分辨率为 640×480：


3 相机 × 640×480×3 = 2,764,800 bytes ≈ 2.7MB（未压缩 uint8 RGB）
未设置 obs_image_jpeg_quality 或 obs_image_resize_hw → raw numpy 直接 pickle → 最大 payload。

解决方案

# async_client_sm.yaml 添加以下任一
obs_image_jpeg_quality: 85          # 2.7MB → ~200KB, recv_deser_srv ≈ 15ms
# 或
obs_image_resize_hw: [224, 224]     # 640×480 → 224×224, 2.7MB → 452KB
# 或组合使用（最优）
obs_image_resize_hw: [224, 224]
obs_image_jpeg_quality: 85          # → ~30KB, recv_deser_srv < 5ms


-----------------

SO-101 client 端 inference_delay 获取的完整流程
总体结构：延迟反馈闭环

上一轮 chunk 到达
  → 测量 complete_s → LatencyTracker.add()
  → 当前轮 control_loop_observation()
      → 从 LatencyTracker 计算 infer_delay
      → 写入 obs.inference_delay 随 obs 发出
Step 1：complete_s 的测量（base_client.py）
每次 chunk 到达时，接收线程记录：


# base_client.py:465-483
receive_after_deser = time.time()           # ← pickle.loads 完成后

complete_s = receive_after_deser - timed_actions[0].get_timestamp()
#            ↑ 当前时刻             ↑ 发送此 obs 时由 _build_timed_observation 打的 time.time()
# = obs_build + serialize + grpc_send + server_queue_wait
#   + server_infer + net_s2c + client_deser
#   ≈ 全链路延迟

if 0 < complete_s <= chunk_lifetime:
    self.latency_tracker.add(complete_s)    # 存入滑动窗口 (maxlen=30)
Step 2：infer_delay 计算（base_client.py:785-818，已修改）
每次 control_loop_observation() 的 Stage 2：


dt = self.config.environment_dt            # = 1/20 = 0.05s (SO-101 20Hz)

_SPIKE_BUFFER_STEPS = max(1, math.ceil(self.config.spike_buffer_s / dt))
# 默认 spike_buffer_s=0.15 → ceil(0.15/0.05) = 3 steps

if len(self.latency_tracker) >= 3:
    _p50_lat = self.latency_tracker.percentile(0.5) or 0.0
    _p90_lat = self.latency_tracker.percentile(0.9) or 0.0
    _infer_delay = min(
        _to_delay_steps(_p50_lat) + _SPIKE_BUFFER_STEPS,  # 典型延迟 + 安全裕量
        _to_delay_steps(_p90_lat),                         # p90 上限 cap
    )
    _infer_delay_low = _to_delay_steps(_p50_lat)           # 仅用于 multi-candidate
else:
    # 样本不足：直接用 p99 保守估计
    _infer_delay = min(_to_delay_steps(_p50_lat) + _SPIKE_BUFFER_STEPS,
                       _to_delay_steps(_p90_lat))
    _infer_delay_low = 0
SO-101 典型数值（async_rtc 方法）：


complete_s p50 ≈ 400ms → p50_steps = ceil(400/50) = 8
spike_buffer_steps = 3
complete_s p90 ≈ 600ms → p90_steps = ceil(600/50) = 12

infer_delay = min(8+3, 12) = 11 steps
Step 3：写入 obs（base_client.py:816-819）

observation = self._build_timed_observation(
    processed, latest_action + 1, _infer_delay, _leftover
)
observation.inference_delay_low = _infer_delay_low   # 仅 multi-candidate 使用
在 _build_timed_observation 内部：


obs.inference_delay = infer_delay    # ← 写入，随 obs 发给 server
obs.timestamp = time.time()          # ← 打时间戳，用于下一轮 complete_s 测量
Step 4：server 端使用（policy_server.py → modeling_rtc.py）

# policy_server._get_action_chunk()
if self._rtc_enabled and policy_type in _RTC_CAPABLE_POLICIES:
    chunk = policy.predict_action_chunk(
        observation,
        inference_delay=obs.inference_delay,   # ← 用于 RTC guidance 权重
        prev_chunk_left_over=leftover,
    )

# modeling_rtc.get_prefix_weights(start=infer_delay, end=exec_horizon, total=50)
# → weights: [1...1 | 1→0 | 0...0]
#             ↑ old  ↑ 过渡  ↑ 新策略
完整时序图

t=0:   obs 发出 (obs.timestamp=T0, inference_delay=11)
       ↓ grpc_send(~67ms) + queue_wait + server_infer(~300ms) + net_s2c
t=370: chunk 到达
       complete_s = time.time() - T0 ≈ 390ms → LatencyTracker.add(0.39)
       ↓ 控制循环继续执行 11步(旧chunk) × 50ms = 550ms
t=920: next obs 发出
       p50 ≈ 390ms → p50_steps=8, +3buffer=11, cap=p90_steps=12
       inference_delay = 11  (下一轮)

-----
分析这种计算 inference_delay 的方式可能会有什么问题

当前 infer_delay 计算方法的潜在问题
当前公式（已修改后）

# maxlen=30 窗口，每次推理后 add(complete_s)
infer_delay = min(
    ceil(p50_lat / dt) + SPIKE_BUFFER_STEPS,   # 典型延迟 + 固定缓冲
    ceil(p90_lat / dt),                          # p90 上限 cap
)
问题一：引导期（Bootstrap）估计不可靠

_DELAY_MIN_SAMPLES = 3  ← 只需 3 个样本就开始计算
前 3 次推理可能包含：

GPU JIT/warmup（首次推理可能 10-46s）→ complete_s 极大
网络冷启动（TCP slow-start）→ grpc_send 偏高
3 样本的 p50/p90 方差极大，此时的 infer_delay 是随机值。

代码中的不一致：虽然有 latency_tracker.reset() 在 episode start，但 warmup 过滤（complete_s > chunk_lifetime 时丢弃）只过滤超大值，不过滤 warmup 阶段的高延迟正常值。

问题二：complete_s 自相关——infer_delay 影响它自己

infer_delay ↑ → n_new ↑ → 推理频率 λ ↓ → server queue_wait ↓
→ complete_s ↓ → p50 ↓ → infer_delay ↓ → λ ↑ → queue_wait ↑
→ complete_s ↑ → infer_delay ↑ → ...
complete_s = grpc + queue_wait + server_infer + deser，其中 queue_wait 直接受当前 infer_delay 控制的推理频率影响。这导致：

complete_s 不是独立观测，而是系统自身状态的函数
极端情况：infer_delay 过大 → λ 极低 → queue_wait≈0 → complete_s 低估 → infer_delay 骤降 → λ 激增 → queue_wait 飙升 → infer_delay 骤增 → 振荡
问题三：SM 恢复事件污染窗口
SM 恢复（~6-7s）后重启推理时，第一次 complete_s 可能异常（buffer 冷、server 刚结束排队清零等）。这个样本进入 maxlen=30 的窗口，并在接下来 30 次推理（~50s）内 持续影响 p50/p90。

问题四：ceil() 量化导致 1-step 振荡

p50_lat 在 395ms 和 405ms 之间波动：
ceil(395/50) = 8 → infer_delay = 11
ceil(405/50) = 9 → infer_delay = 12

两步之间差 50ms：robot 在这两种情况下执行完全不同数量的旧动作
当 p50 在某个步长边界（k×dt）附近抖动时，infer_delay 以 1-step 为单位来回跳动，产生周期性的 RTC guidance 不一致。

问题五：固定 spike_buffer_s 不适应延迟方差
当前：buffer = fixed 150ms = 3 steps

场景	延迟方差	应有 buffer	实际 buffer	效果
稳定网络	小（std~20ms）	1-2 steps	3 steps	过保守，浪费
抖动网络	大（std~100ms）	6-8 steps	3 steps	不足，饥饿风险
双峰 gRPC	极大（p99/p50≈5×）	自适应	3 steps	依赖 p90 cap
更合理的做法：buffer = k × std_lat / dt（adaptive），但增加了复杂度。

问题六：p90 cap 在高负载时失效
当系统持续高负载（p50 和 p90 差异很小）：


p50 = 14 steps, p90 = 15 steps
min(14+3, 15) = min(17, 15) = 15 ← cap 生效，实际 buffer 只有 1 step
设计意图（避免 p50 过高时用 p90 cap）反而在正常高负载时消除了安全缓冲。

问题七：历史窗口与当前状态不同步
窗口 maxlen=30，推理间隔 ≈ 1650ms → 覆盖最近 49.5s。这段时间内可能经历：

SM 恢复前后（延迟跃变）
网络稳定期与抖动期混合
GPU 热态变化
所有样本权重相等，无法感知近期趋势变化。指数移动平均（EMA）对新样本权重更高，适应更快。

问题八：滞后性——总是用"上一轮"的数据
infer_delay 基于已完成的推理轮的 complete_s 估算下一轮的延迟。如果延迟持续上升（GPU 热积累），估计永远落后实际值。


真实延迟趋势: 300ms → 350ms → 400ms → 450ms
LatencyTracker.p50: ~300ms  ~310ms  ~320ms  ~340ms  ← 始终偏低
infer_delay:         6步     7步      7步     7步     ← 不足以覆盖实际450ms
问题汇总
问题	严重程度	触发条件
自相关振荡	⚠ 中	infer_delay 过大或过小时
SM 恢复污染窗口	⚠ 中	SM 方法每次触发
ceil 量化振荡	△ 轻	p50 在步长边界附近
固定 buffer 不适应方差	⚠ 中	抖动网络环境
p90 cap 高负载失效	△ 轻	持续高延迟时
Bootstrap 不可靠	⚠ 中	每个 episode 前 3 次
历史权重均等（无趋势感知）	△ 轻	延迟趋势变化时
最值得关注的是 自相关问题（问题二） 和 SM 恢复污染（问题三），因为这两者在 SO-101 真机评测中必然发生，且会系统性地偏移 infer_delay 估计。

-----

raw_obs
  ↓ [skip_server_resize 控制此步]
  raw_observation_to_observation()   ← 只是数据预处理的第一步，非最终输入
    skip=False → bilinear resize 到 policy_image_features.shape
    skip=True  → 只做 HWC→CHW，不插值
  ↓
preprocessor (Normalizer + Tokenizer)
    └─ 模型自身的 resize_with_pad()  ← 这才是真正的最终 resize（与训练一致）
  ↓
model forward
结论：use_model_resize=true 之所以更准确，是因为它让客户端做了与训练时完全相同的 pad/resize，server 端跳过 bilinear，模型内部的 resize_with_pad 成为 no-op，整条链路与训练对齐。use_model_resize=false 的 bilinear 路径在分辨率上能凑近，但变换方式不同，存在轻微 distribution shift。

分析：server_resize 不是模型最终输入
Server 端完整流水线（policy_server.py:_predict_action_chunk）：


receive obs
  ↓
JPEG decode (if jpeg_images=True)
  ↓
raw_observation_to_observation()          ← 这里是 skip_server_resize 控制的步骤
  │  skip_resize=False → resize_robot_observation_image()
  │                      bilinear 插值到 policy_image_features.shape (e.g. 224×224 / 512×512)
  │  skip_resize=True  → 只做 HWC→CHW permute，不做插值
  ↓
self.preprocessor(observation)            ← 还有一层！
  │  NormalizerStep        → 归一化
  │  TokenizerStep         → 分词、图像编码
  │    └─ model's resize_with_pad()       ← 模型内部的最终 resize（训练时的变换）
  ↓
policy.predict_action_chunk()             ← 模型 forward
server_resize（resize_robot_observation_image）只是把图像 bilinear 插值到 policy_image_features.shape，之后还会经过 preprocessor 里模型自己的 resize_with_pad。

关键区别：

skip_server_resize=True（客户端用 model-specific pad）：客户端做的变换与模型内部 resize_with_pad 完全一致，所以服务端跳过 bilinear resize 后，模型内部的 resize_with_pad 变成 no-op，图像没有被二次处理过
skip_server_resize=False（服务端做 bilinear resize）：bilinear 把图像拉到 224×224（会改变宽高比，无 padding），然后模型内部的 resize_with_pad 在一张已经 224×224 的图上再跑一遍（目标也是 224×224，几乎 no-op，但使用了错误的 resize 路径，不完全匹配训练时的变换）

--------------

分析：当前so101 的client-server 测试流程中，例如  lerobot.async_inference.robot_client 和 lerobot.async_inference.policy_server 在开启RTC 时，client 端给obs 的 inference_delay 的计算方式存在的问题；有那些可行的优化方案？



SO-101 RTC 中 infer_delay 计算的问题与优化方案(综合)
当前计算方式(已应用的修改后)

# base_client.py, 单一 LatencyTracker(maxlen=30) of complete_s
complete_s = receive_after_deser − obs.timestamp   # 全链路时长
infer_delay = min(ceil(p50/dt) + spike_buffer_steps,   # 典型 + 固定缓冲
                  ceil(p90/dt))                          # p90 上限
complete_s = serialize + grpc_send + queue_wait + server_infer + net_s2c + deser

一、存在的问题
根本问题:单标量混淆了性质截然不同的分量

complete_s = server_infer (稳定, σ/μ≈15%, 单峰)
           + grpc_send    (双峰: 70ms / 365ms, TCP slow-start)
           + queue_wait   (重尾: 0–689ms, 自相关)
           + deser/serialize (稳定, ~35ms)
把它们加成一个标量再取分位数,无法对"必须覆盖的稳定分量"和"可容忍偶发漏掉的波动分量"区别对待。

派生问题(按对 SO-101 真机的影响排序)
#	问题	严重度	触发条件
②	自相关振荡:queue_wait 受 infer_delay→推理频率反向影响,形成反馈环	⚠ 高	infer_delay 偏离稳态时
③	SM 恢复污染:恢复后异常 complete_s 进入窗口,影响后续 ~50s	⚠ 高	SM 方法每次触发
⑤	固定 buffer 不适应方差:0.15s 在抖动网络下不足,稳定网络下浪费	⚠ 中	网络抖动
⑥	p90 cap 高负载失效:p50≈p90 时 min(p50+3, p90) 反而消除缓冲	△ 中	持续高延迟
①	Bootstrap 不可靠:仅 3 样本即计算,含 warmup/冷启动	⚠ 中	每 episode 前 3 次
⑧	滞后 + 无趋势感知:等权窗口,延迟上升趋势永远落后	△ 低	GPU 热积累
④	ceil 量化振荡:p50 在步长边界抖动 → infer_delay ±1 跳变	△ 低	p50 ≈ k×dt
二、可行的优化方案(分层)
Tier 0 — 治本:减小 overhead 方差的根源
JPEG 压缩 / resize(obs_image_jpeg_quality: 85)

payload 从 442KB(或 2.7MB)降到 ~30KB → 消除 grpc_send 双峰(slow-start 不触发)→ overhead 方差骤降。这是上游修复,让任何估计器都更容易工作,优先级最高且零风险。

Tier 1 — 已实现(估计器基础改进)
max() → p50+buffer / cap p90:消除单尖刺主导
maxlen 100→30:尖刺更快淡出
spike_buffer_s 可配置(跨频率自适应步数)
Tier 2 — 推荐:分量切分 + 非对称分位数

overhead_s = max(0, complete_s − raw.inference_time_s)  # 跨机无需 NTP(纯时长相减)
infer_tracker.add(raw.inference_time_s)      # 稳定分量, 大窗口
overhead_tracker.add(overhead_s)             # 重尾分量, 小窗口

infer_delay = ceil((infer_tracker.p90 + overhead_tracker.p75) / dt)
稳定的 server_infer 取 p90(覆盖代价小)
重尾的 overhead 取 p75(尖刺由 force_must_go 兜底)
降低过估计(对最大波动源不取高分位),同时保证稳定延迟安全覆盖
修复问题 ②⑤⑥;无需改 server / proto / timing schema
⚠ 注意:不要对 overhead 用 EMA+σ(重尾非高斯,μ+kσ 失真);σ-margin 仅适用于稳定的 server_infer 分量。

Tier 3 — 架构级(更高精度,但成本高)
方案	机制	代价
server 时间戳回报	ActionChunk 携带 obs_received_time + dispatch_wall,client 精确分解三段	改 server+proto;需 NTP 同步
AIMD 闭环	按"是否饥饿"自适应增减,不依赖延迟分布假设	收敛慢;直接对齐 SLO
正交修复(与上面任一组合)
SM reset 联动:恢复后清空所有 tracker(否则污染未清除)
Bootstrap fallback:样本不足时退回旧公式,稳态后切换
ceil 滞回:infer_delay 仅在变化 ≥2 步时更新,消除边界 ±1 抖动
三、推荐落地路径

1. Tier 0 (JPEG/resize)         — 治本, 零风险, 先做
2. Tier 2 (split B + 非对称分位) — 估计器核心改进, 仅改 client
3. 正交修复 (SM reset + fallback) — 配合 Tier 2 必须做
4. Tier 3                        — 仅当 Tier 0+2 仍不达标时考虑
最高性价比组合:Tier 0 + Tier 2 + SM reset 联动 —— 从源头降方差、按分量性质分别估计、消除 SM 污染,且全部限于 client 端改动,不影响 server、proto、timing log 与现有可视化分析。

----
 Tier 2 + 正交修复

 实现总结:Tier 2 + 正交修复
改动文件
文件	改动
configs.py	新增 infer_latency_quantile(默认0.90)、overhead_latency_quantile(默认0.75) + (0,1) 验证
base_client.py	双 tracker、split-B 喂数、非对称分位数公式、bootstrap fallback、SM-reset 联动、非对称滞回、日志
核心公式(Stage 2)

infer_delay = ceil( (infer_tracker.p90 + overhead_tracker.p75) / dt )
# 稳定的 server_infer 取高分位(覆盖便宜) + 重尾 overhead 取中分位(尖刺由 force_must_go 兜底)
切分:overhead_s = max(0, complete_s − raw.inference_time_s)(两个纯时长相减,无需 NTP)。

三项正交修复
SM/episode reset 联动 — _reset_loop_state() 同时 reset 三个 tracker + 滞回状态,清除恢复后污染
Bootstrap fallback — 两个 split tracker 各满 3 样本前,退回旧 p50+buffer/cap-p90 公式,启动行为不变
非对称滞回 — 上升立即采纳(安全,防饥饿),下降需 ≥2 步才缩小(抑制 ceil 边界抖动),避免了对称死区的"缓升锁死"
仿真验证结果
场景	结果
稳态(overhead 双峰 70/370ms)	infer_delay 稳定 10 步,不被双峰主导 ✓
单次 1.4s 尖刺	10→10,被 p75 完全吸收(旧 max() 会飙到 28)✓
overhead 缓升 0.07→0.50s	8→15 单调上升,无锁死 ✓
下行抖动 0.12↔0.17s	输出近似恒定,抑制振荡 ✓
零侵入确认
server / proto / timing.py schema 完全不变
fig8 的 actual_steps(基于 complete_s)仍有效;print_diagnosis 的 overcorr 比值仍有效,且预期下降
三个 client(robot/smart/multicand)经继承链自动获得新配置
[CLIENT←SERVER] 日志改为输出 infer_p90 / overhead_p75,便于直接读校准
调参入口

--infer_latency_quantile=0.95     # 更保守覆盖推理抖动
--overhead_latency_quantile=0.50  # 更激进(更新鲜动作),靠 force_must_go 兜底
--overhead_latency_quantile=0.90  # 更保守(减少饥饿)

----

分析：Tier 2 + 正交修复后的方案会计算一些完整链路上的时间？

Tier 2 后的链路时间覆盖分析
核心结论:链路组件零遗漏(逐样本恒等)
实现用的是切分 B:


complete_s   = receive_after_deser − obs.timestamp   # 全链路
server_infer = raw.inference_time_s
overhead_s   = max(0, complete_s − server_infer)
# ⟹ server_infer + overhead_s ≡ complete_s   (逐样本严格相等)
因为 overhead = complete_s − server_infer,两个 tracker 的每个样本之和精确等于 complete_s。这与早先讨论的切分 A(round_trip − inference_time,会漏掉 grpc_send)本质不同——这里没有任何链路分量被丢弃。

各分量归属:

链路分量	时长(SO-101典型)	归属 tracker
client serialize	~8ms	overhead
grpc_send (net_c2s + recv_deser_srv)	~70-256ms	overhead
server queue_wait	0-689ms	overhead
server_infer	~300ms	infer_tracker
srv_serialize	~2ms	overhead
net_s2c	~0ms	overhead
client deser	~28ms	overhead
合计	= complete_s	infer + overhead
但是:估计值故意不覆盖 overhead 的重尾上部(设计而非遗漏)
这是关键区分。虽然组件全覆盖,但用于 infer_delay 的估计量:


infer_delay = ceil( (infer.p90 + overhead.p75) / dt )
overhead.p75 故意不覆盖 overhead 的 p75~p100 区间(那段重尾/双峰)。这不是 bug,而是降低过估计的核心机制:


对 overhead 取 p99/max → infer_delay 飙到 28 步(fig8 的过估计问题)
对 overhead 取 p75    → infer_delay ≈ 10 步,漏掉的尖刺由 force_must_go 兜底
仿真验证:单次 1.4s overhead 尖刺 → infer_delay 保持 10 不动(被 p75 吸收)。

complete_s 的边界本身:起点是 obs.timestamp
complete_s 起点是 _build_timed_observation 里打的 obs.timestamp,这发生在 Stage 1 obs_capture 之后。所以:


本轮 obs_capture (sim ~133ms / 真机 ~5ms) + preprocess  ← 不在 complete_s 内
obs.timestamp ──────────── complete_s ──────────── receive_after_deser
这是旧方案就有的边界,Tier 2 完全没有改变它。 即相对旧的单 complete_s 方案,覆盖范围完全一致,切分只是把同一个 complete_s 拆成两段分别统计。

边界情况:都不会导致少算
情况	行为	是否少算链路时间
max(0, ...) clamp	仅当 complete_s < server_infer(时钟/测量误差,物理上不可能)时触发,此时反而 over-count	否(偏保守)
warmup 过滤 0 < complete_s ≤ chunk_lifetime	异常样本从所有 tracker 一起剔除	否(稳态样本完整)
bootstrap fallback	用旧 complete_s 公式,直接覆盖全链路	否
总结
维度	结论
链路组件覆盖	✅ 完整无遗漏,infer + overhead ≡ complete_s 逐样本恒等
估计值大小	⚙️ 故意只取 overhead 的 p75(不覆盖重尾),这是降低过估计的设计选择,尖刺由 force_must_go 兜底
相对旧方案的边界	✅ complete_s 边界(起点 obs.timestamp,不含 obs_capture)完全未变
所以回答你的问题:Tier 2 不会少计算任何完整链路上的时间 —— 切分是无损的;唯一"不覆盖"的是 overhead 重尾的上部,那是为降低 infer_delay 过估计而有意留给 force_must_go 兜底的部分,不是测量遗漏。