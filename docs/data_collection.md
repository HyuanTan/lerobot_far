# Data Collection

← [Back to README](../README.md)

Data collection runs on the **Jetson Nano** inside the `lerobot_far` container, using the SO-101 leader/follower arms and USB cameras.

---

## Setup

### HuggingFace login

```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

### Teleoperation check (before recording)

Verify the arms and cameras work before starting a collection session:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --robot.cameras="{
        top:   {type: opencv, index_or_path: '/dev/videotop',   width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
        wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
        front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader \
    --display_data=true \
    --display_async=true \
    --display_image_interval_s=0.5
```

---

## Record a Dataset

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --robot.cameras="{
        top:   {type: opencv, index_or_path: '/dev/videotop',   width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
        wrist: {type: opencv, index_or_path: '/dev/videowrist', width: 800, height: 600, fps: 30, backend: 200, fourcc: MJPG},
        front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader \
    --dataset.root=/data/hf/lerobot/${HF_USER}/so101_pick-place-v2.4 \
    --dataset.repo_id=${HF_USER}/so101_pick-place-v2.4 \
    --dataset.single_task="Pick up the yellow cube and place it in the box." \
    --dataset.fps=20 \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=8 \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --display_data=false \
    --play_sounds=false \
    --display_async=true \
    --display_image_interval_s=0.2 \
    --display_worker_poll_interval_s=0.2 \
    --resume=true
```

> **Resuming a session:** Add `--resume=true` and set `--dataset.num_episodes` to the **number of new episodes to add** (not the cumulative total). The dataset at `--dataset.root` must already exist; new episodes are appended starting from the last recorded index.

### Key recording parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset.fps` | 30 | Recording frame rate (20 recommended for Jetson) |
| `--dataset.num_episodes` | — | Number of episodes to collect (new episodes when `--resume=true`) |
| `--dataset.episode_time_s` | 60 | Max duration per episode (s) |
| `--dataset.reset_time_s` | 8 | Reset time between episodes (s) |
| `--dataset.streaming_encoding` | false | Encode video on-the-fly (reduces disk I/O on Jetson) |
| `--dataset.encoder_threads` | 4 | Encoder thread count |
| `--resume` | false | Append to an existing dataset |

### Video codec on Jetson

```
--dataset.vcodec=auto    # recommended — uses hardware encoder (best for Jetson)
--dataset.vcodec=h264    # fallback — stable, fast, larger files
# default (libsvtav1)    # highest compression but CPU-heavy, causes teleop lag on Jetson
```

---

## Replay an Episode

Verify recorded data by replaying on the real robot:

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --dataset.repo_id=${HF_USER}/so101_pick-place-v2.4 \
    --dataset.root=/data/hf/lerobot/${HF_USER}/so101_pick-place-v2.4 \
    --dataset.fps=20 \
    --play_sounds=false \
    --dataset.episode=0
```

---

## Visualize a Dataset

```bash
lerobot-dataset-viz \
  --repo-id=${HF_USER}/so101_pick-place-v2.4 \
  --root=/data/hf/lerobot/${HF_USER}/so101_pick-place-v2.4 \
  --episode-index=0
```

---

## Upload to HuggingFace Hub

```bash
hf upload ${HF_USER}/so101_pick-place-v2.4 \
  /data/hf/lerobot/${HF_USER}/so101_pick-place-v2.4 \
  --repo-type dataset
```

After uploading, browse and play back episodes in the browser via the LeRobot dataset visualizer:

> [https://huggingface.co/spaces/lerobot/visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset)

Enter your `repo_id` (e.g. `HollyTan/so101_pick-place-v2.4`) to inspect frames, action trajectories, and camera streams for each episode.

---

## Dataset Merging and Resampling

When combining datasets recorded at different fps or from different sessions, use the merge/resample script:

```bash
# Resample all datasets under --root to a common fps and merge
uv run python scripts/tools/resample_and_merge_lerobot.py \
  --root ~/.cache/huggingface/lerobot \
  --target-fps 20 \
  --overwrite

# Skip resampling (merge only, assuming same fps)
uv run python scripts/tools/resample_and_merge_lerobot.py \
  --root ~/.cache/huggingface/lerobot \
  --target-fps 20 \
  --skip-resample \
  --overwrite

# Verify merged dataset
uv run python scripts/tools/test_merge_dataset.py
```

> The last joint dimension (`index = -1`) is the gripper. Action interpolation policy: arm joints → linear; gripper action → previous-hold; gripper state → nearest; images → nearest.

### Upload merged dataset

```bash
uv run hf upload ${HF_USER}/so101_pick-place-merge-v2.4_20hz \
  /data/users/huoyuan/.cache/huggingface/lerobot/${HF_USER}__so101_pick-place-merge-v2.4_20hz \
  --repo-type dataset
```
