# Train
```bash
uv pip install -e .
```


```bash
conda activate vlash-libero
cd ~/VLA/vla_asyn_arena_ori_repo/vlash/vlash_main

tmux attach -t vlash_so101_pi05_train_abs
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Async 训练（推荐）
stdbuf -oL -eL vlash train_so101 examples/train/pi05/so101_async_lora.yaml \
2>&1 | tee ./logs/so101_pi05_async/pick-place-v2.4-16b_train_abs_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=0 vlash train examples/train/pi05/so101_async_lora.yaml

# 多卡（例如 4 GPU）
accelerate launch --num_processes=4 -m vlash.train_so101 examples/train/pi05/so101_async.yaml

# Sync 基线对比
vlash train examples/train/pi05/so101_sync.yaml


# Upload
uv run hf upload HollyTan/pi05_vlash_so101_2.4-8b_async \
  ~/VLA/vla_asyn_arena_ori_repo/vlash/vlash_main/outputs/train/so101/pi05_so101_2.4-8b_async/checkpoints/015000/pretrained_model \
  --repo-type model
```
**关键参数说明**
max_delay_steps: 6
SO101 数据集是 20fps（每步 50ms），PI0.5 在 SO101 上推理约需 100-300ms → 2-6 步。设为 6 代表训练时随机模拟 0~300ms 的异步延迟，覆盖实际部署时的推理 latency。

shared_observation: true
核心加速优化：VLM（PaliGemma）的视觉+语言编码只做 1 次，然后对 7 个 offset（0..6）共享该嵌入，让 action expert 分别预测各 offset 的动作序列。相比逐个 offset 前向，约有 7 倍训练吞吐提升。

state_cond: true
VLASH 的关键：将"未来状态"（或其代理值：前一步 action）注入 PI0.5 的 action expert 作为条件，而不是当前状态。这就是让模型学会处理"观测已过时"场景的核心机制。

------
pi05 架构或默认参数有点不匹配：
pretrained_path: HollyTan/pi05_so101_pick_place-v2.4_abs_nofreeze_8b