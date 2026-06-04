# Fine tune VLA on SO101
## Fine tune smolvla on SO101
https://huggingface.co/docs/lerobot/smolvla
```BASH
~/VLA/LeRobot/lerobot

uv run lerobot-train --help
```

```bash
conda activate vlash-libero
hf download HollyTan/so101_pick-place-v2.1 \
  --local-dir /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.1 \
  --repo-type dataset
```

```BASH
tmux attach -t so101_smolvla_train
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.0 \
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
    "observation.state": {"type":"STATE","shape":[6]},
    "observation.images.top": {"type":"VISUAL","shape":[3,256,256]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,256,256]},
    "observation.images.front": {"type":"VISUAL","shape":[3,256,256]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
  2>&1 | tee ./logs/so101_smolvla/pick-place-v2.0_train_$(date +%Y%m%d_%H%M%S).log
```


## Fine tune PI05 on SO101
```bash
cd ~/VLA/LeRobot/lerobot
tmux attach -t so101_pi05_train
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.1 \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/pi05_pick-place-v2.1 \
  --job_name=so101_pi05_pick-place-v2.1_training \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-v2.1 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,224,224]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,224,224]},
    "observation.images.front": {"type":"VISUAL","shape":[3,224,224]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --batch_size=32 \
  --steps=20000 \
  --wandb.enable=true \
  --wandb.project=so101_pi05_pick-place \
  --save_checkpoint=true \
  --save_freq=5000 \
2>&1 | tee ./logs/so101_pi05/pick-place-v2.1_train_$(date +%Y%m%d_%H%M%S).log
```

```bash
lerobot-push \
  --model.path=outputs/train/so101/smolvla_pick-place/checkpoints/last/pretrained_model \
  --repo_id=${HF_USER}/so101_smolvla_pick-place

-------
hf repo create HollyTan/so101_smolvla_pick_place-v2.0 \
  --type model \
  --public

hf upload HollyTan/so101_smolvla_pick_place-v2.0 \
  outputs/train/so101/smolvla_pick-place-v2.0/checkpoints/last/pretrained_model \
  --repo-type model
```

### Fine tune PI05 on SO101 use 500 dataset
#### Split a new dataset
```bash
cd ~/VLA/LeRobot/lerobot
uv run lerobot-edit-dataset --help

uv run lerobot-edit-dataset \
  --repo_id HollyTan/so101_pick-place-v2.0 \
  --operation.type split \
  --operation.splits '{"train_subset_50": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]}'
```



#### train PI05 using new dataset
```bash
tmux attach -t so101_pi05_train
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/so101_pick-place-v2.4_abs \
  --job_name=so101_pi05_pick-place-v2.4_train_abs \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-v2.4_abs \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.use_relative_actions=false \
  --batch_size=32 \
  --steps=20000 \
  --wandb.enable=true \
  --wandb.project=so101_pi05_pick-place \
  --save_checkpoint=true \
  --save_freq=5000 \
2>&1 | tee ./logs/so101_pi05/pick-place-v2.4_train_abs_$(date +%Y%m%d_%H%M%S).log
```



```bash
uv run wandb sync outputs/train/so101/pi05_pick-place-v2.0_subset_50/wandb/run-20260420_131016-321b8fv6


hf upload HollyTan/pi05_so101_pick_place-v2.1_20k \
  outputs/train/so101/pi05_pick-place-v2.1/checkpoints/020000/pretrained_model/ \
  --repo-type model
```


# Finetune PI05(Update)
- SO101 自采任务如果视觉环境和 base model 差别较大，完全冻结 VLM 可能不如全量 finetune。显存够的情况下: freeze_vision_encoder=false, train_expert_only=false, `lerobot/pi05_base`可能是基于双臂aloha的数据训练的，差别较大

- PI05/PI0 在不free VLM 情况下使用 batch_size=8 比 batch_size=16 震荡发散一些，单卡batch_size=16到后期训练会crash， 需要考虑多卡增加batch size

```bash
tmux attach -t so101_pi05_train_abs
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/so101_pick-place-v2.4_abs_nofreeze_8b \
  --job_name=so101_pi05_pick-place-v2.4_train_abs_nofreeze_8b \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-v2.4_abs_nofreeze_8b \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
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
2>&1 | tee ./logs/so101_pi05/pick-place-v2.4-8b_train_abs_nofreeze_$(date +%Y%m%d_%H%M%S).log

```
-----------------
```bash
tmux attach -t so101_pi05_train_delta
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# 只重新计算metadata/stats 文件，但通常不会改变原始 episode 数据、视频、action 数值本身
uv run lerobot-edit-dataset \
  --root /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.4_relative \
  --repo_id HollyTan/so101_pick-place-v2.4_relative \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 50 \
  --push_to_hub true \
  --operation.relative_exclude_joints "['gripper']"


conda activate vlash-libero
hf download HollyTan/so101_pick-place-v2.2-100eps \
  --local-dir /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps \
  --repo-type dataset
cp -rf /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps_relative

uv run hf upload HollyTan/so101_pick-place-v2.2-100eps_relative /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps_relative --repo-type dataset

rm -rf /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps_relative


uv run lerobot-edit-dataset \
  --root /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps_relative \
  --repo_id HollyTan/so101_pick-place-v2.2-100eps_relative \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 50 \
  --push_to_hub false \
  --operation.relative_exclude_joints "['gripper']"


uv run hf upload HollyTan/so101_pick-place-v2.2-100eps_relative /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps_relative --repo-type dataset


# 验证 new stats.json 和 old stats.json 是否有变化
uv run python compare_stats.py old_stats.json new_stats.json

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.2-100eps_relative \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/pi05_so101_pick-place-v2.2-100eps_delta_nofreeze-8b \
  --job_name=so101_pi05_pick-place-v2.2-100eps_train_delta_nofreeze-8b \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-v2.2-100eps_delta_nofreeze-8b \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --batch_size=8 \
  --steps=20000 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=1e-6 \
  --wandb.enable=true \
  --wandb.project=so101_pi05_pick-place \
  --save_checkpoint=true \
  --save_freq=2500 \
2>&1 | tee ./logs/so101_pi05/pick-place-v2.2-100eps_train_delta_nofreeze-8b_$(date +%Y%m%d_%H%M%S).log
  
```

# Finetune PI0
```bash
tmux attach -t so101_pi05_train_abs
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi0 \
  --output_dir=outputs/train/so101/pi0/so101_pick-place-v2.4_abs_nofreeze \
  --job_name=so101_pi0_pick-place-v2.4_train_abs_nofreeze \
  --policy.repo_id=HollyTan/pi0_so101_pick_place-v2.4_abs_nofreeze \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
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
2>&1 | tee ./logs/so101_pi0/pick-place-v2.4_train_abs_nofreeze_$(date +%Y%m%d_%H%M%S).log
```
# pi05 混合训练
```bash
tmux attach -t so101_pi05_train_abs2
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/so101_pick-place-v2.4basev2.2_abs_nofreeze_8b \
  --job_name=so101_pi05_pick-place-v2.4basev2.2_train_abs_nofreeze_8b \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-v2.4basev2.2_abs_nofreeze_8b \
  --policy.pretrained_path=HollyTan/pi05_so101_pick_place-v2.2-100eps_abs_nofreeze \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.use_relative_actions=false \
  --batch_size=8 \
  --steps=15000 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_betas="[0.9,0.95]" \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=15000 \
  --policy.scheduler_decay_lr=1e-6 \
  --wandb.enable=true \
  --wandb.project=so101_pi05_pick-place \
  --save_checkpoint=true \
  --save_freq=2500 \
2>&1 | tee ./logs/so101_pi05/pick-place-v2.4basev2.2-8b_train_abs_nofreeze_$(date +%Y%m%d_%H%M%S).log

```

todo: 
- mix with freeze_vision_encoder=true and train_expert_only=true
- 把之前所有采集的数据都混合先训练，然后选择比较好的其中一个fintune?
- 混合所有数据
------
#### 混合数据
observation.state.shape = (6,)
action.shape = (6,)
最后一维 index = -1 是 gripper
observation.state/action 都是 position
arm action: linear
arm state: linear
gripper action: previous-hold
gripper state: nearest
image: nearest
其他低维数值: linear
```bash
uv run python scripts/tools/resample_and_merge_lerobot.py \
  --root ~/.cache/huggingface/lerobot \
  --target-fps 20 \
  --overwrite

 --push-to-hub \
 --private

uv run python scripts/tools/resample_and_merge_lerobot.py \
  --root ~/.cache/huggingface/lerobot \
  --target-fps 20 \
  --skip-resample \
  --overwrite


# 新行为：max-compliance，保留硬物接触信号；默认：原始 nearest 行为，无时序偏移
  --max-compliance-gripper

 # check
uv run python scripts/tools/test_merge_dataset.py

uv run hf upload HollyTan/so101_pick-place-merge-v2.2-v2.3-v2.4_20hz /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan__so101_pick-place-merge-v2.2-v2.3-v2.4_20hz --repo-type dataset

uv run hf upload HollyTan/so101_pick-place-merge-v2.3_20hz /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan__so101_pick-place-v2.3_20hz --repo-type dataset

uv run hf upload HollyTan/so101_pick-place-merge-v2.2_20hz /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan__so101_pick-place-v2.2_20hz --repo-type dataset
```

-----
```bash
tmux attach -t so101_pi05_train_abs2
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1


stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-merge-v2.2-v2.3-v2.4_20hz \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/pi05_so101_pick-place-merge-v2.2-v2.3-v2.4_abs_nofreeze_8b \
  --job_name=so101_pi05_pick-place-merge-v2.2-v2.3-v2.4_abs_nofreeze_8b \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-merge-v2.2-v2.3-v2.4_abs_nofreeze_8b \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
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
2>&1 | tee ./logs/so101_pi05/pick-place-merge-v2.2-v2.3-v2.4-8b_train_abs_nofreeze_$(date +%Y%m%d_%H%M%S).log

```
----
```bash
tmux attach -t so101_pi05_train_abs2
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# uv run lerobot-edit-dataset \
#   --new_repo_id HollyTan/so101_pick-place-merge-v2.2-v2.4_20hz \
#   --operation.type merge \
#   --operation.repo_ids "['HollyTan/so101_pick-place-merge-v2.2_20hz', 'HollyTan/so101_pick-place-v2.4']" \
#   --push_to_hub true


stdbuf -oL -eL uv run lerobot-train \
  --dataset.repo_id=HollyTan/so101_pick-place-v2.4 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/so101/pi05_so101_pick-place-v2.4-base-mixtrain_abs_nofreeze_8b \
  --job_name=so101_pi05_pick-place-v2.4-base-mixtrain_abs_nofreeze_8b \
  --policy.repo_id=HollyTan/pi05_so101_pick_place-v2.4-base-mixtrain_abs_nofreeze_8b \
  --policy.pretrained_path=HollyTan/pi05_so101_pick_place-merge-v2.2-v2.3-v2.4_abs_nofreeze_8b \
  --policy.input_features='{
    "observation.images.top": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.wrist": {"type":"VISUAL","shape":[3,600,800]},
    "observation.images.front": {"type":"VISUAL","shape":[3,480,640]},
    "observation.state": {"type":"STATE","shape":[6]}
  }' \
  --policy.output_features='{
    "action": {"type":"ACTION","shape":[6]}
  }' \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.use_relative_actions=false \
  --batch_size=8 \
  --steps=10000 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_betas="[0.9,0.95]" \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=10000 \
  --policy.scheduler_decay_lr=1e-6 \
  --wandb.enable=true \
  --wandb.project=so101_pi05_pick-place \
  --save_checkpoint=true \
  --save_freq=2500 \
2>&1 | tee ./logs/so101_pi05/pick-place-v2.4-base-mixtrain_abs_nofreeze_8b_$(date +%Y%m%d_%H%M%S).log
```