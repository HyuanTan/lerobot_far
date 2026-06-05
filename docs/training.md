# Model Training

← [Back to README](../README.md)

Fine-tuning runs on the **GPU workstation** (server side). All commands use `uv run lerobot-train`.

---

## Setup

### Download a dataset

```bash
# Using hf download (conda/pip)
hf download ${HF_USER}/so101_pick-place-v2.4 \
  --local-dir ~/.cache/huggingface/lerobot/${HF_USER}/so101_pick-place-v2.4 \
  --repo-type dataset
```

### Log directory

Create log directories before training:

```bash
mkdir -p logs/so101_smolvla logs/so101_pi05 logs/so101_pi0
```

---

## Fine-tune SmolVLA

```bash
tmux attach -t so101_smolvla_train
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/so101_pick-place-v2.0 \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --steps=50000 \
  --output_dir=outputs/train/so101/smolvla_pick-place-v2.0 \
  --job_name=so101_smolvla_pick-place-v2.0_training \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=so101_smolvla_pick-place \
  --policy.load_vlm_weights=true \
  --save_checkpoint=true \
  --save_freq=5000 \
  --policy.input_features='{
    "observation.state":        {"type":"STATE",  "shape":[6]},
    "observation.images.top":   {"type":"VISUAL", "shape":[3,256,256]},
    "observation.images.wrist": {"type":"VISUAL", "shape":[3,256,256]},
    "observation.images.front": {"type":"VISUAL", "shape":[3,256,256]}
  }' \
  --policy.output_features='{"action": {"type":"ACTION","shape":[6]}}' \
  2>&1 | tee ./logs/so101_smolvla/pick-place-v2.0_train_$(date +%Y%m%d_%H%M%S).log
```

---

## Full fine-tune PI05 (no VLM freeze)

Use this when the task scene differs significantly from the base model's training distribution.

> **Note:** With `freeze_vision_encoder=false` and `train_expert_only=false`, use `batch_size=8`. `batch_size=16` on a single GPU tends to diverge late in training; use multi-GPU to increase batch size.

```bash
stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/pi05_pick-place-v2.4_abs_nofreeze_8b \
  --job_name=so101_pi05_pick-place-v2.4_abs_nofreeze_8b \
  --policy.repo_id=${HF_USER}/pi05_so101_pick_place-v2.4_abs_nofreeze_8b \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.input_features='{
    "observation.images.top":   {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state":        {"type":"STATE", "shape":[6]}
  }' \
  --policy.output_features='{"action": {"type":"ACTION","shape":[6]}}' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.use_relative_actions=false \
  --batch_size=8 \
  --steps=20000 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_betas="[0.9,0.95]" \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=1e-6 \
  --wandb.enable=true \
  --wandb.project=so101_pi05_pick-place \
  --save_checkpoint=true \
  --save_freq=2500 \
  2>&1 | tee ./logs/so101_pi05/pick-place-v2.4_abs_nofreeze_$(date +%Y%m%d_%H%M%S).log
```

---

## Fine-tune Pi0

```bash
export CUDA_VISIBLE_DEVICES=0

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi0 \
  --output_dir=outputs/train/so101/pi0/so101_pick-place-v2.4_abs_nofreeze \
  --job_name=so101_pi0_pick-place-v2.4_abs_nofreeze \
  --policy.repo_id=${HF_USER}/pi0_so101_pick_place-v2.4_abs_nofreeze \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.input_features='{
    "observation.images.top":   {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state":        {"type":"STATE", "shape":[6]}
  }' \
  --policy.output_features='{"action": {"type":"ACTION","shape":[6]}}' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.use_relative_actions=false \
  --batch_size=8 \
  --steps=20000 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_betas="[0.9,0.95]" \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=1e-6 \
  --wandb.enable=true \
  --wandb.project=so101_pi0_pick-place \
  --save_checkpoint=true \
  --save_freq=2500 \
  2>&1 | tee ./logs/so101_pi0/pick-place-v2.4_abs_nofreeze_$(date +%Y%m%d_%H%M%S).log
```

---

## Mixed Training Strategy (Pi0.5)

When combining multiple data versions (e.g. v2.2 + v2.3 + v2.4), a two-stage approach works well:

1. **Stage 1** — Train on the merged dataset (broad coverage):

```bash
--policy.pretrained_path=lerobot/pi05_base \
--dataset.repo_id=${HF_USER}/so101_pick-place-merge-v2.2-v2.3-v2.4_20hz \
--steps=20000
```

2. **Stage 2** — Fine-tune on the target dataset from the Stage 1 checkpoint:

```bash
--policy.pretrained_path=${HF_USER}/pi05_so101_pick_place-merge-v2.2-v2.3-v2.4_abs_nofreeze_8b \
--dataset.repo_id=${HF_USER}/so101_pick-place-v2.4 \
--steps=10000
```

---

## Dataset Subset Split

To train on a subset of episodes (e.g., first 50):

```bash
uv run lerobot-edit-dataset \
  --repo_id ${HF_USER}/so101_pick-place-v2.0 \
  --operation.type split \
  --operation.splits '{"train_subset_50": [0,1,2,...,49]}'
```

---

## Upload Checkpoint to HuggingFace Hub

```bash
# Create repo first
hf repo create ${HF_USER}/so101_smolvla_pick_place-v2.0 --type model --public

# Upload last checkpoint
hf upload ${HF_USER}/so101_smolvla_pick_place-v2.0 \
  outputs/train/so101/smolvla_pick-place-v2.0/checkpoints/last/pretrained_model \
  --repo-type model

# Or a specific step checkpoint
hf upload ${HF_USER}/pi05_so101_pick_place-v2.1_20k \
  outputs/train/so101/pi05_pick-place-v2.1/checkpoints/020000/pretrained_model/ \
  --repo-type model
```

### Sync offline WandB runs

```bash
uv run wandb sync outputs/train/so101/<run-dir>/wandb/<run-id>
```

---

## Training Tips

| Situation | Recommendation |
|-----------|---------------|
| Task scene differs from base model | `freeze_vision_encoder=false`, `train_expert_only=false` |
| Single GPU, no freeze | `batch_size=8`; `batch_size=16` tends to diverge late in training |
| Pi0.5 base model mismatch (e.g. Aloha-based) | Use full fine-tune; frozen VLM may hurt if visual domain differs greatly |
| Compile errors on Server aync-inference | Disable CUDA graph: set `compile_model=false` in `config.json` or use `compile_mode="reduce-overhead"` in `configuration_pi05.py` |
