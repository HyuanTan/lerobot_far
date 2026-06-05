# Experiments

← [Back to README](../README.md)

Eval scripts: [docs/so101_client-server.md](so101_client-server.md) §10–12

---

## Overview

Evaluation is organised in two stages. §5.1 uses the LIBERO benchmark in simulation with a controlled delay-injection setting (§5.1.1) and a full pipeline setting (§5.1.2). §5.2 repeats the pipeline conditions on the physical SO-101 arm. §5.3 examines post-recovery trajectory diversity and execution-gap diagnostics.

Three runtime variants are compared throughout:

| Variant | Description |
|---------|-------------|
| **Sync** | Zero delay; client waits for server response before advancing. Upper-bound reference only. |
| **Baseline** (No-RTC) | Naive async — switches to a new chunk as soon as the server returns one; no timing correction. |
| **RTC** | Freezes the guaranteed execution prefix and blends the remainder with the incoming chunk. |

Each variant is additionally evaluated with the **FAR** (Failure-Aware Recovery) monitor enabled (+FAR): monitors gripper feedback and contact signals, interrupts on empty-grasp or slip, resamples from the recovered state.

**Main findings:**
- FAR raises success rate in simulation across most suites and both policies.
- On the real robot under perturbation: **+68 pp** for π₀.₅, **+42 pp** for SmolVLA.
- False-positive regime at high-confidence positions limits benefit under unperturbed RTC (calibration via grasp-confirmation window T).
- Every success-rate improvement is accompanied by decreased episode length — FAR terminates stalled attempts early.
- RTC and FAR address disjoint failure modes: RTC absorbs chunk-delivery latency tails; FAR detects contact-layer execution failures.

---

## 5.1 Simulated Evaluation (LIBERO)

Simulation runner connects to the same policy-server interface used on the real robot, executes returned action chunks, and writes per-episode and aggregate result files. Two camera views: agent-view (front) + wrist-mounted.

### 5.1.1 Controlled Evaluation

> **Scripts:** `eval_libero_script/eval_pi05.sh`, `eval_libero_script/eval_smolvla.sh`  
> See [so101_client-server.md §10](so101_client-server.md#10-libero-simulation-sweep-eval_libero_script)

**Setup:** Fixed synthetic inference delay `d=2` control steps (~67 ms) at 30 Hz. No pipeline variability.

| Parameter | Baseline | RTC | +FAR |
|-----------|----------|-----|------|
| `async_delay d` | 2 | 2 | 2 |
| `chunk_size K` | 25 | 25 | 25 |
| `execution_horizon H` | — | 20 | 20 |
| `grasp confirm T` | — | — | 20 |
| episodes per suite | 100 | 100 | 100 |
| suites | all 4 LIBERO | all 4 | all 4 |

**Script parameters:**
```bash
# eval_pi05.sh equivalents:
DELAYS=(2)
DELAY_FIXED_S=20        # H for RTC
HORIZONS=(20)
HORIZON_FIXED_D=1
N_EPISODES=100 BATCH_SIZE=10
```

#### Results — π₀.₅

**Baseline vs. Baseline+FAR** (`d=2, K=25`):

| Suite | Baseline SR% | +FAR SR% | ΔSR | Baseline len | +FAR len | Δlen |
|-------|-------------|---------|-----|-------------|---------|------|
| Spatial | 88.0 | 92.0 | **+4.0** | 263.9 | 229.7 | -34.2 |
| Object | 95.0 | 100.0 | **+5.0** | 197.9 | 154.8 | -43.1 |
| Goal | 97.0 | 96.0 | -1.0 | 143.7 | 154.8 | +11.1 |
| LIBERO-10 | 90.0 | 96.0 | **+6.0** | 365.5 | 332.9 | -32.6 |

**RTC vs. RTC+FAR** (`d=2, H=20`):

| Suite | RTC SR% | +FAR SR% | ΔSR | RTC len | +FAR len | Δlen |
|-------|---------|---------|-----|---------|---------|------|
| Spatial | 65.0 | 77.0 | **+12.0** | 443.0 | 356.3 | -86.7 |
| Object | 94.0 | 98.0 | **+4.0** | 190.8 | 157.1 | -33.7 |
| Goal | 92.0 | 93.0 | **+1.0** | 185.8 | 182.3 | -3.5 |
| LIBERO-10 | 72.0 | 86.0 | **+14.0** | 488.9 | 390.1 | -98.9 |

#### Results — SmolVLA

**Baseline vs. Baseline+FAR**:

| Suite | Baseline SR% | +FAR SR% | ΔSR | Baseline len | +FAR len | Δlen |
|-------|-------------|---------|-----|-------------|---------|------|
| Spatial | 61.0 | 72.0 | **+11.0** | 465.3 | 400.3 | -65.0 |
| Object | 83.0 | 68.0 | **-15.0** | 300.3 | 419.3 | +119.0 |
| Goal | 88.0 | 93.0 | **+5.0** | 228.2 | 200.4 | -27.8 |
| LIBERO-10 | 51.0 | 54.0 | **+3.0** | 640.6 | 619.8 | -20.8 |

**RTC vs. RTC+FAR**:

| Suite | RTC SR% | +FAR SR% | ΔSR | RTC len | +FAR len | Δlen |
|-------|---------|---------|-----|---------|---------|------|
| Spatial | 50.0 | 55.0 | **+5.0** | 561.0 | 526.3 | -34.7 |
| Object | 63.0 | 73.0 | **+10.0** | 456.2 | 370.3 | -85.9 |
| Goal | 88.0 | 86.0 | -2.0 | 218.0 | 240.7 | +22.7 |
| LIBERO-10 | 48.0 | 53.0 | **+5.0** | 641.9 | 616.9 | -25.1 |

#### Key observations

- FAR reduces episode length whenever it improves SR — confirms early termination of stalled attempts, not just additional overhead.
- π₀.₅ benefits more consistently than SmolVLA. Largest gains on hard long-horizon suites: RTC+FAR improves Spatial +12 pp and LIBERO-10 +14 pp.
- **SmolVLA false-positive regime on Object (Baseline condition):** -15 pp SR + 119-step longer episodes — FAR interrupts viable grasps. Adding RTC removes this regression (+10 pp); RTC's chunk-continuity suppresses contact transients that trigger FAR incorrectly.
- LIBERO-Goal least affected: shortest episodes, lowest grasp-failure prevalence.

---

### 5.1.2 Pipeline Evaluation

> **Scripts:** `eval-scripts/libero_pi05/async_libero_pi05_eval.sh`, `sync_libero_pi05_eval.sh`,  
> `eval-scripts/libero_smolvla/async_libero_smolvla_eval.sh`, `sync_libero_smolvla_eval.sh`  
> See [so101_client-server.md §11](so101_client-server.md#11-libero-single-point-asyncsync-eval-eval-scriptslibero_)

**Setup:** Real async client–server workflow (no synthetic delay). Inference delay estimated from live round-trip time chain.

| Parameter | Value |
|-----------|-------|
| `actions_per_chunk K` | 50 |
| `rtc_execution_horizon H` (RTC methods) | 16 |
| `grasp confirm T` (+FAR methods) | 20 |
| `chunk_size_threshold` (async) | 0.5 |
| `chunk_size_threshold` (sync) | 0 |
| episodes per suite | 100 |

**Methods compared** (6 async + 2 sync):

| Method | Async | RTC | FAR | Port |
|--------|-------|-----|-----|------|
| Sync | ✗ | ✗ | ✗ | 8081 |
| Sync+FAR | ✗ | ✗ | ✓ | 8082 |
| Baseline (nortc) | ✓ | ✗ | ✗ | 8085 |
| Baseline+FAR (nortc_sm) | ✓ | ✗ | ✓ | 8086 |
| RTC | ✓ | ✓ | ✗ | 8083 |
| RTC+FAR (rtc_sm) | ✓ | ✓ | ✓ | 8084 |

Sync is included as an upper-bound simulation reference only; not viable on physical arm due to round-trip stall.

**Results:** See `outputs/eval_thesis/libero/<suite>/<method>/results/aggregate.json` and the generated plots:
- `single_point_comparison.png` — per-suite SR for all 6 conditions
- `avg_steps_comparison.png` — average episode length
- `sm_retry_stats.png` — FAR intervention and retry statistics

**Generate plots:**
```bash
# After running eval scripts:
uv run python -m lerobot.async_inference.analysis.plot_libero_pipeline \
  outputs/eval_thesis/libero \
  --output-dir outputs/eval_thesis/libero/plots
```

**Key observations:**
- FAR raises SR for Baseline and RTC in most suites for both policies under real pipeline conditions.
- +FAR variants also reduce average episode length: FAR ends stalled attempts early, drains stale queued actions, issues fresh chunk from recovered state.
- Intervention rates are moderate under normal pipeline operation; per-attempt retry SR exceeds baseline SR in suites where FAR helps most.

---

## 5.2 Real-World Evaluation (SO-101)

> **Scripts:** `eval-scripts/so101_pi05/`, `eval-scripts/so101_smolvla/`  
> See [so101_client-server.md §12](so101_client-server.md#12-so-101-real-robot-eval-scripts-eval-scriptsso101_)

**Task:** "Pick up the yellow cube and put it into the box." — SO-101 tabletop pick-and-place.  
**Hardware:** SO-101 arm + Jetson Orin Nano (client) + GPU workstation (server). Top, front, and wrist cameras.  
**Dataset:** 100 episodes at 20 Hz, 5 cube positions × 20 episodes.

**Conditions per policy:** No-RTC (async), RTC (async), RTC with human interference.  
**Interference protocol:** Experimenter lifts the cube clear of the gripper as it closes — creates a reliable empty-grasp event (the exact failure signature FAR monitors).

### 5.2.1 Success Rate and Recovery

**n = 50 trials per row. Δ = +FAR minus No-FAR (pp).**  
**r₁/r₂:** episodes recovered after one/two FAR interventions.  
**P.-att. SR:** per-attempt retry SR = (r₁ + r₂) / (r₁ + 2r₂).  
† No JPEG compression.

| Policy | Config | Interf. | No-FAR SR% | +FAR SR% | Δ | r₁ | r₂ | Retry Rate% | P.-att. SR% |
|--------|--------|---------|-----------|---------|---|----|----|------------|------------|
| SmolVLA | No-RTC | No | 60 | 68 | **+8** | 9 | 3 | 24.0 | 80.0 |
| SmolVLA | RTC | No | 64 | 68 | **+4** | 7 | 6 | 26.0 | 68.4 |
| SmolVLA | RTC | **Yes** | 40 | 82 | **+42** | 34 | 7 | 82.0 | 85.4 |
| π₀.₅ | No-RTC | No | 72 | 82 | **+10** | 4 | 4 | 16.0 | 66.7 |
| π₀.₅ | RTC | No | 68 | 68 | 0 | 8 | 3 | 22.0 | 78.6 |
| π₀.₅ | RTC | **Yes** | 32 | 100 | **+68** | 44 | 3 | 94.0 | 94.0 |
| π₀.₅ | RTC† | No | 58 | — | — | — | — | — | — |

**Key observations:**

- **No-RTC:** FAR provides consistent gains. SmolVLA +8 pp (24% retry, 80% per-attempt SR), π₀.₅ +10 pp (16% retry). Consistent with controlled simulation findings.
- **RTC, no interference, π₀.₅:** Net-zero (+0 pp) despite active recoveries (r₁=8, r₂=3). FAR recovers ~11 trials but causes equal false-positive interruptions at high-confidence positions.
- **With interference:** Large improvements. π₀.₅ RTC: 32%→100% (+68 pp), 94% of trials recovered. SmolVLA RTC: 40%→82% (+42 pp), 85.4% per-attempt SR. Interference creates the exact empty-grasp signature FAR monitors.
- **No-compression baseline (π₀.₅ RTC†):** 58% vs. 68% compressed — JPEG compression reduces latency enough to benefit RTC chunk timing.

#### Per-position breakdown — π₀.₅ RTC (n=10 per position)

| Pos | Without Interf. No-FAR% | +FAR% | Δ | r₁ | r₂ | With Interf. No-FAR% | +FAR% | Δ | r₁ | r₂ |
|-----|------------------------|------|---|----|----|----------------------|------|---|----|----|
| 1 | 70 | 60 | **-10** | 3 | 0 | 50 | 100 | **+50** | 9 | 1 |
| 2 | 80 | 60 | **-20** | 0 | 0 | 30 | 100 | **+70** | 5 | 2 |
| 3 | 90 | 100 | **+10** | 4 | 0 | 10 | 100 | **+90** | 10 | 0 |
| 4 | 40 | 40 | 0 | 1 | 0 | 50 | 100 | **+50** | 10 | 0 |
| 5 | 60 | 80 | **+20** | 0 | 3 | 20 | 100 | **+80** | 10 | 0 |
| **Avg** | **68** | **68** | **0** | 8 | 3 | **32** | **100** | **+68** | 44 | 3 |

False-positive regime at high-confidence positions: P1 (baseline 70%) and P2 (baseline 80%) degrade by -10 and -20 pp. P2: r₁=r₂=0 yet SR drops 80%→60% — FAR interrupts viable grasps and every retry also fails.

Lower-confidence positions benefit: P3 (+10 pp, r₁=4) and P5 (+20 pp, r₂=3 — second retry needed, same failure mode on first retry).

Under interference: all 5 positions reach 100%. P3 collapses from 90% to 10% without FAR under interference, fully restored to 100% with FAR (r₁=10).

**Calibration implication:** Lengthening the grasp-confirmation window T (requiring more consecutive confirming steps) would suppress false positives at P1/P2 without sacrificing recovery at P3–P5 and under interference.

---

### 5.2.2 Latency and Timing Analysis

> Results: `outputs/eval_thesis/so101_*/<method>/<model>/H<H>/`  
> Timing logs: `client_timing/`, `server_timing/`  
> Plots: `fig5_latency_violin.png` per condition

**Key observations:**

- π₀.₅ uses 224×224 inputs (resized + JPEG); small per-observation payload.
- SmolVLA requires ≥512×512 for its VL encoder; larger payload → more network jitter sensitivity.
- Server inference is stable; tail dominated by network RTT and queue wait.
- Fixed RTC horizon H=16 interacts with delay estimate accuracy. When network jitter is high (SmolVLA), the estimated delay lags the true delay and effective horizon drifts out of calibration.
- **+FAR adds negligible latency overhead** — FAR predicate runs within a single control step on the client.

**Generate latency plots:**
```bash
uv run python -m lerobot.async_inference.analyze_timing \
  outputs/eval_thesis/so101/<method>/pi05/H16 \
  --out_dir outputs/eval_thesis/so101/<method>/pi05/H16/timing_analysis
```

---

## 5.3 Trajectory Analysis

### 5.3.1 Stochastic Resampling Diversity

> Scripts: `eval-scripts/so101_pi05/async_so101_pi05_client-rtc-sm-multican.sh`  
> Analysis: `python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory`

Both π₀.₅ and SmolVLA are flow-matching policies. Two independent sources of post-recovery divergence:
1. **Fresh noise vector** per inference call → structurally distinct output trajectories from the same observation.
2. **Physical state change** from FAR recovery motion (lift + rewind) → observation shifts, further changing the conditional distribution.

LIBERO-Spatial: candidate fan widens at the failure point — flow field uncertainty increases when state is unusual (denoising paths diverge).  
SO-101: candidates from a single request span geometrically distinct approach paths; score spread across them is non-zero.

**Current limitation:** Flow-matching generation carries no explicit awareness of the failed trajectory. Each candidate is an independent draw with no mechanism for avoiding the specific approach geometry that caused failure. A principled selector incorporating contact history, failure signature, or task-progress estimates is future work.

**Multi-candidate server:**
```bash
# LIBERO multi-candidate
bash eval-scripts/libero_pi05/async_libero_pi05_eval.sh --method=rtc_multicand

# SO-101 multi-candidate
bash eval-scripts/so101_pi05/async_so101_pi05_server-rtc-sm-multican.sh  # server
bash eval-scripts/so101_pi05/async_so101_pi05_client-rtc-sm-multican.sh  # client
```

---

### 5.3.2 Execution Gap and Failure Detection

Three signals overlaid for a representative SO-101 episode with FAR-triggered recovery:
1. Action chunks returned by policy server
2. Actions executed by client after RTC blending
3. Motor feedback from joint encoders (`Present_Position`)

**Key observations:**

- **Free-space tracking is reliable.** Executed action and motor feedback follow chunk geometry closely in shoulder/elbow joints. At contact transitions, wrist channels develop 1–2 step phase lag.
- **Gripper provides a low-noise binary detection anchor.** Step-shaped trace; rising-edge misalignment between commanded close and motor-feedback close = empty-grasp signature. Low false-alarm floor once grasp-confirmation window T is calibrated.
- **Recovery motion visible as Cartesian back-track.** Short reversals in EE trajectory correspond to lift + rewind motions that return arm to a configuration where the resampled chunk executes without the same geometric obstruction.

**Generate trajectory plots:**
```bash
# EE-space 3D trajectory
python -m lerobot.async_inference.sim_test.analyze_multicand_trajectory \
  --traj_dir=outputs/eval_thesis/so101/<method>/pi05/H16/mc_trajectories \
  --out_dir=outputs/eval_thesis/so101/<method>/pi05/H16/mc_viz \
  --action_dim_names=shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper \
  --robot_type=so101 --viz_mode=ee

# Joint-space + finite-difference analysis
python src/lerobot/async_inference/analyze_trajectory.py \
  outputs/eval_thesis/so101/<method>/pi05/H16/trajectories
```

**Summary:** RTC and FAR address disjoint failure modes and are complementary: RTC operates at chunk-delivery level (absorbs latency tails, reduces velocity discontinuities), FAR operates at execution level (detects contact-layer divergence via gripper signal). Neither subsumes the other; their combination covers the full failure space observed in deployment.
