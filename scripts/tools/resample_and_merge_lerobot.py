#!/usr/bin/env python3
"""
resample_and_merge_lerobot.py

For SO101 LeRobot datasets:

1. Resample HollyTan/so101_pick-place-v2.3 from ~30 Hz to 20 Hz.
2. Resample HollyTan/so101_pick-place-v2.2-100eps from ~10 Hz to 20 Hz.
3. Merge:
   - HollyTan/so101_pick-place-v2.3_20hz
   - HollyTan/so101_pick-place-v2.2-100eps_20hz
   - HollyTan/so101_pick-place-v2.4
   into one new dataset.

SO101-specific rule:
- observation.state.shape == (6,)
- action.shape == (6,)
- last dimension is gripper

Interpolation policy:
- arm action: linear
- arm state: linear
- gripper action: previous-hold
- gripper state: nearest
- image: nearest frame
- other numeric values: linear

Run example:

python resample_and_merge_lerobot.py \
  --root ~/lerobot_resampled_datasets \
  --target-fps 20 \
  --overwrite

With upload:

python resample_and_merge_lerobot.py \
  --root ~/lerobot_resampled_datasets \
  --target-fps 20 \
  --overwrite \
  --push-to-hub \
  --private
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _resolve_src_root(root: Path, repo_id: str) -> Path | None:
    """Return the local dataset root if it exists, else None (fall back to Hub/HF_LEROBOT_HOME).

    Tries both storage conventions in order:
    1. root/HollyTan__repo-name  (double-underscore: script output from resample step)
    2. root/HollyTan/repo-name   (slash: standard HF_LEROBOT_HOME cache)
    """
    for candidate in (
        root / repo_id.replace("/", "__"),
        root / repo_id,
    ):
        if candidate.exists():
            return candidate
    return None


AUTO_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "next.done",
    "done",
}


SO101_STATE_KEY = "observation.state"
SO101_ACTION_KEY = "action"
SO101_GRIPPER_INDEX = -1


def to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensor / numpy array / scalar / list to numpy."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        return x

    return np.asarray(x)


def to_python_value(x: Any) -> Any:
    """Convert a value to a type accepted by LeRobotDataset.add_frame."""
    if isinstance(x, Image.Image):
        return x

    arr = to_numpy(x)

    if arr.shape == ():
        val = arr.item()
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        return val

    return arr


def normalize_image_for_add_frame(x: Any) -> Any:
    """
    Convert image value to a format accepted by LeRobotDataset.add_frame.

    Usually add_frame accepts PIL.Image or np.ndarray.
    This keeps PIL as PIL, and converts tensor CHW to HWC if needed.
    """
    if isinstance(x, Image.Image):
        return x

    arr = to_numpy(x)

    if arr.ndim != 3:
        return arr

    # CHW -> HWC if likely channel-first.
    # Require H and W (shape[1], shape[2]) both > 4 to avoid misidentifying
    # small square arrays like (3, 3, W) as CHW.
    if (
        arr.shape[0] in (1, 3, 4)
        and arr.shape[1] > 4
        and arr.shape[2] > 4
        and arr.shape[-1] not in (1, 3, 4)
    ):
        arr = np.transpose(arr, (1, 2, 0))

    # If float image in [0, 1], convert to uint8.
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def get_features_for_create(src: LeRobotDataset) -> Dict[str, Any]:
    """
    Reuse source features but remove fields usually managed by LeRobot writer.
    """
    features = dict(src.meta.features)

    cleaned = {}
    for key, value in features.items():
        if key in AUTO_KEYS:
            continue
        cleaned[key] = value

    return cleaned


def infer_image_keys_from_features(src: LeRobotDataset) -> List[str]:
    """
    Infer image/video feature keys.
    """
    image_keys = []

    for key, spec in src.meta.features.items():
        if key in AUTO_KEYS:
            continue

        dtype = None
        if isinstance(spec, dict):
            dtype = spec.get("dtype", None)

        if dtype in ("image", "video"):
            image_keys.append(key)
            continue

        # Fallback for common LeRobot image feature names.
        lowered = key.lower()
        if "image" in lowered or "images" in lowered or "camera" in lowered:
            image_keys.append(key)

    return sorted(set(image_keys))


def infer_episode_indices(src: LeRobotDataset) -> List[int]:
    """
    Infer available episode indices from the dataset.
    """
    if hasattr(src, "num_episodes"):
        try:
            return list(range(int(src.num_episodes)))
        except Exception:
            pass

    values = []
    for i in range(len(src)):
        item = src[i]
        if "episode_index" in item:
            values.append(int(to_numpy(item["episode_index"]).item()))

    return sorted(set(values))


def get_episode_global_indices(src: LeRobotDataset, episode_index: int) -> List[int]:
    """
    Return global row indices for one episode.

    Supports metadata lookup and brute-force fallback.
    """
    if hasattr(src, "episode_data_index"):
        try:
            edi = src.episode_data_index
            start = int(to_numpy(edi["from"][episode_index]).item())
            end = int(to_numpy(edi["to"][episode_index]).item())
            return list(range(start, end))
        except Exception:
            pass

    if hasattr(src, "meta") and hasattr(src.meta, "episodes"):
        try:
            ep = src.meta.episodes[episode_index]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            return list(range(start, end))
        except Exception:
            pass

    out = []
    for i in range(len(src)):
        item = src[i]
        if "episode_index" in item:
            ep = int(to_numpy(item["episode_index"]).item())
            if ep == episode_index:
                out.append(i)

    return out


def nearest_index(times: np.ndarray, t: float) -> int:
    """
    Index of nearest timestamp.
    """
    j = int(np.searchsorted(times, t))

    if j <= 0:
        return 0

    if j >= len(times):
        return len(times) - 1

    if abs(times[j] - t) < abs(times[j - 1] - t):
        return j

    return j - 1


def previous_index(times: np.ndarray, t: float) -> int:
    """
    Index of previous timestamp, used for previous-hold command.
    """
    j = int(np.searchsorted(times, t, side="right")) - 1
    return int(np.clip(j, 0, len(times) - 1))


def interp_numeric_linear(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation for scalar or vector numeric values.

    values shape:
    - [T]
    - [T, D]
    - [T, ...]
    """
    values = np.asarray(values)

    if len(values) == 1:
        return np.repeat(values, len(target_times), axis=0)

    flat = values.reshape(len(values), -1)
    out = np.empty((len(target_times), flat.shape[1]), dtype=np.float32)

    for d in range(flat.shape[1]):
        out[:, d] = np.interp(target_times, times, flat[:, d])

    return out.reshape((len(target_times),) + values.shape[1:])


def _max_compliance_gripper_ids(
    times: np.ndarray,
    state_gripper: np.ndarray,
    action_gripper: np.ndarray,
    target_times: np.ndarray,
) -> List[int]:
    """
    For each target time, find the source frame index with the largest
    |state - action| (compliance gap) within a half-window of the target interval.

    Used during downsampling so that brief contact events in dropped source frames
    are still visible in the output.  Falls back to nearest when no source frame
    falls inside the window (e.g. during upsampling).
    """
    half_win = (float(target_times[1] - target_times[0]) * 0.5) if len(target_times) > 1 else np.inf
    ids: List[int] = []
    for t in target_times:
        lo = int(np.searchsorted(times, t - half_win, side="left"))
        hi = int(np.searchsorted(times, t + half_win, side="right"))
        if lo >= hi:
            ids.append(nearest_index(times, float(t)))
        else:
            compliance = np.abs(state_gripper[lo:hi] - action_gripper[lo:hi])
            ids.append(lo + int(np.argmax(compliance)))
    return ids


def interp_so101_state(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    action_values: np.ndarray | None = None,
    gripper_index: int = SO101_GRIPPER_INDEX,
) -> np.ndarray:
    """
    SO101 observation.state interpolation.

    Rule:
    - arm state: linear
    - gripper state: max-compliance window pooling when action_values is provided
      (preserves contact events during downsampling); falls back to nearest otherwise.

    Assumes state shape is [T, 6].
    """
    values = np.asarray(values)

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(
            f"Expected observation.state shape [T, 6], got {values.shape}. "
            "If your state dimension is not 6, modify interp_so101_state()."
        )

    out = interp_numeric_linear(times, values, target_times)

    gi = gripper_index if gripper_index >= 0 else values.shape[1] + gripper_index

    if action_values is not None:
        gripper_ids = _max_compliance_gripper_ids(
            times, values[:, gi], action_values[:, gi], target_times
        )
    else:
        gripper_ids = [nearest_index(times, float(t)) for t in target_times]

    out[:, gi] = values[gripper_ids, gi]

    return out


def interp_so101_action(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    gripper_index: int = SO101_GRIPPER_INDEX,
) -> np.ndarray:
    """
    SO101 absolute action interpolation.

    Rule:
    - arm action: linear
    - gripper action: previous-hold

    Assumes action shape is [T, 6].
    """
    values = np.asarray(values)

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(
            f"Expected action shape [T, 6], got {values.shape}. "
            "If your action dimension is not 6, modify interp_so101_action()."
        )

    out = interp_numeric_linear(times, values, target_times)

    gi = gripper_index if gripper_index >= 0 else values.shape[1] + gripper_index
    previous_ids = [previous_index(times, float(t)) for t in target_times]
    out[:, gi] = values[previous_ids, gi]

    return out


def make_target_times(
    original_abs_times: np.ndarray,
    target_fps: float,
) -> np.ndarray:
    """
    Create target timestamps relative to episode start:
    0.00, 0.05, 0.10, ...
    """
    if len(original_abs_times) == 0:
        return np.asarray([], dtype=np.float64)

    rel_times = original_abs_times - original_abs_times[0]
    duration = float(rel_times[-1])

    if duration <= 0:
        return np.asarray([0.0], dtype=np.float64)

    dt = 1.0 / float(target_fps)
    n = int(math.floor(duration / dt)) + 1

    return np.arange(n, dtype=np.float64) * dt


def collect_episode(src: LeRobotDataset, global_indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [src[int(i)] for i in global_indices]


def get_task_value(frames: List[Dict[str, Any]], fallback: str = "pick place") -> str:
    """
    Get task string if available.
    """
    if not frames:
        return fallback

    item = frames[0]

    if "task" in item:
        task = item["task"]
        if isinstance(task, str):
            return task
        return str(task)

    return fallback


def get_original_times(
    frames: List[Dict[str, Any]],
    fallback_fps: float | None = None,
) -> np.ndarray:
    """
    Read timestamp from frames.

    If timestamp is missing, use fallback_fps.
    """
    if frames and "timestamp" in frames[0]:
        times = np.asarray(
            [float(to_numpy(f["timestamp"]).item()) for f in frames],
            dtype=np.float64,
        )
        return times

    if fallback_fps is None:
        raise ValueError("No timestamp field found and no fallback_fps was provided.")

    return np.arange(len(frames), dtype=np.float64) / float(fallback_fps)


def numeric_keys_from_frames(
    frames: List[Dict[str, Any]],
    image_keys: Sequence[str],
) -> List[str]:
    """
    Numeric keys to resample.

    Include:
    - action
    - observation.state
    - motor current/load/velocity/status if numeric

    Exclude:
    - image keys
    - auto metadata
    - task string
    """
    if not frames:
        return []

    image_key_set = set(image_keys)
    keys = []

    for key, value in frames[0].items():
        if key in AUTO_KEYS:
            continue
        if key in image_key_set:
            continue
        if key == "task":
            continue

        arr = to_numpy(value)

        if np.issubdtype(arr.dtype, np.number):
            keys.append(key)

    return sorted(set(keys))


def validate_so101_frame_shapes(frames: List[Dict[str, Any]], repo_id: str, ep_idx: int) -> None:
    """
    Validate SO101 state/action shape for every frame in the episode.
    """
    if not frames:
        return

    for i, frame in enumerate(frames):
        if SO101_STATE_KEY not in frame:
            raise KeyError(f"{repo_id} episode {ep_idx} frame {i}: missing {SO101_STATE_KEY}")

        if SO101_ACTION_KEY not in frame:
            raise KeyError(f"{repo_id} episode {ep_idx} frame {i}: missing {SO101_ACTION_KEY}")

        state = to_numpy(frame[SO101_STATE_KEY])
        action = to_numpy(frame[SO101_ACTION_KEY])

        if state.shape != (6,):
            raise ValueError(
                f"{repo_id} episode {ep_idx} frame {i}: "
                f"expected {SO101_STATE_KEY}.shape == (6,), got {state.shape}"
            )

        if action.shape != (6,):
            raise ValueError(
                f"{repo_id} episode {ep_idx} frame {i}: "
                f"expected {SO101_ACTION_KEY}.shape == (6,), got {action.shape}"
            )


def resample_numeric_series(
    key: str,
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    action_values: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply SO101-specific resampling policy.

    action_values: pre-computed action series for the same episode.
    When provided, gripper state uses max-compliance window pooling instead of nearest.
    """
    if key == SO101_STATE_KEY:
        return interp_so101_state(times, values, target_times, action_values=action_values)

    if key == SO101_ACTION_KEY:
        return interp_so101_action(times, values, target_times)

    # Other numeric fields: linear by default.
    return interp_numeric_linear(times, values, target_times)


def resample_dataset(
    src_repo_id: str,
    dst_repo_id: str,
    root: str | Path,
    target_fps: float,
    source_fps_hint: float | None = None,
    use_videos: bool = True,
    push_to_hub: bool = False,
    private: bool = True,
    overwrite: bool = False,
    robot_type: str | None = None,
    max_compliance_gripper: bool = False,
) -> LeRobotDataset:
    """
    Resample a LeRobot dataset into a new dataset.

    max_compliance_gripper: when True, gripper state uses max-compliance window
    pooling during downsampling — picks the source frame with the largest
    |state − action| gap within each target window, preserving hard-object contact
    events that would otherwise be lost by temporal aliasing.  The trade-off is a
    slight temporal offset (≤ half a target frame) between arm joints (linearly
    interpolated) and the chosen gripper state frame.  Default False to keep the
    original nearest-frame behaviour.

    This rewrites the dataset; it does not only change meta/info.json fps.
    """
    root = Path(root).expanduser()
    dst_root = root / dst_repo_id.replace("/", "__")

    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"Destination already exists: {dst_root}. Use --overwrite.")

    print(f"\nLoading source dataset: {src_repo_id}")
    _src_root = _resolve_src_root(root, src_repo_id)
    src = LeRobotDataset(src_repo_id, root=_src_root)

    image_keys = infer_image_keys_from_features(src)
    features = get_features_for_create(src)

    if robot_type is None:
        robot_type = getattr(src.meta, "robot_type", None) or getattr(src.meta, "robot", None) or "so101"

    print(f"Creating destination dataset: {dst_repo_id}")
    print(f"  root       : {dst_root}")
    print(f"  target fps : {target_fps}")
    print(f"  image keys : {image_keys}")
    print("  policy     : arm linear, gripper action previous-hold, gripper state nearest")

    dst = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=target_fps,
        root=dst_root,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_threads=8,
    )

    episode_indices = infer_episode_indices(src)

    for ep_idx in tqdm(episode_indices, desc=f"Resampling {src_repo_id}"):
        global_indices = get_episode_global_indices(src, ep_idx)

        if not global_indices:
            continue

        frames = collect_episode(src, global_indices)
        validate_so101_frame_shapes(frames, src_repo_id, ep_idx)

        old_abs_times = get_original_times(frames, fallback_fps=source_fps_hint)
        old_times = old_abs_times - old_abs_times[0]
        target_times = make_target_times(old_abs_times, target_fps=target_fps)

        numeric_keys = numeric_keys_from_frames(frames, image_keys)

        numeric_series: Dict[str, np.ndarray] = {}
        for key in numeric_keys:
            try:
                numeric_series[key] = np.stack([to_numpy(f[key]) for f in frames], axis=0)
            except Exception:
                numeric_series[key] = np.asarray([to_numpy(f[key]) for f in frames])

        # Pre-interpolate action first so gripper state can reference it when
        # max_compliance_gripper is enabled.
        raw_action = numeric_series.get(SO101_ACTION_KEY)
        interp_action = (
            interp_so101_action(old_times, raw_action, target_times)
            if raw_action is not None
            else None
        )

        interpolated: Dict[str, np.ndarray] = {}
        if interp_action is not None:
            interpolated[SO101_ACTION_KEY] = interp_action

        for key, values in numeric_series.items():
            if key == SO101_ACTION_KEY:
                continue  # already done above
            interpolated[key] = resample_numeric_series(
                key=key,
                times=old_times,
                values=values,
                target_times=target_times,
                action_values=raw_action if max_compliance_gripper else None,
            )

        task = get_task_value(frames)

        for k, t in enumerate(target_times):
            nearest_id = nearest_index(old_times, float(t))
            src_frame = frames[nearest_id]

            new_frame: Dict[str, Any] = {}

            # Images: nearest real frame.
            for key in image_keys:
                if key in src_frame:
                    new_frame[key] = normalize_image_for_add_frame(src_frame[key])

            # Numeric values.
            for key in numeric_keys:
                if key in interpolated:
                    new_frame[key] = interpolated[key][k]

            # Task string.
            new_frame["task"] = task

            dst.add_frame(new_frame)

        dst.save_episode()

    if push_to_hub:
        print(f"Pushing {dst_repo_id} to Hugging Face Hub...")
        dst.push_to_hub(private=private)

    return dst


def validate_merge_sources(
    src_repo_ids: Sequence[str],
    template_features: Dict[str, Any],
    template_repo_id: str,
    target_fps: float,
    root: Path | None = None,
) -> None:
    """
    Pre-flight checks before merging:
    1. Source fps must match target_fps (within 0.5 Hz tolerance).
    2. Each source must provide all feature keys present in the template.
       Extra keys in a source are warned about (they will be dropped).

    root: when provided, looks for each source at root/<repo_id with __> before hitting the Hub.
    """
    template_keys = set(template_features.keys())

    for repo_id in src_repo_ids:
        _src_root = _resolve_src_root(root, repo_id) if root else None
        src = LeRobotDataset(repo_id, root=_src_root)

        src_fps = getattr(src.meta, "fps", None)
        if src_fps is not None and abs(float(src_fps) - float(target_fps)) > 0.5:
            raise ValueError(
                f"{repo_id}: fps={src_fps} does not match target fps={target_fps}. "
                "Resample the dataset first."
            )

        src_features = get_features_for_create(src)
        src_keys = set(src_features.keys())

        missing = template_keys - src_keys
        if missing:
            raise ValueError(
                f"{repo_id} is missing features required by template '{template_repo_id}': "
                f"{sorted(missing)}"
            )

        extra = src_keys - template_keys
        if extra:
            print(
                f"  Warning: {repo_id} has extra features not in template "
                f"(will be dropped): {sorted(extra)}"
            )


def copy_dataset_into_writer(
    src_repo_id: str,
    writer: LeRobotDataset,
    target_fps: float | None = None,
    src_root: Path | None = None,
) -> None:
    """
    Copy one dataset into an existing LeRobotDataset writer.

    This does not resample. It copies frames as-is.
    Used for merging already normalized datasets.

    target_fps: when provided, raises ValueError if source fps does not match.
    src_root: local directory for the source dataset; skips Hub lookup when set.
    """
    print(f"\nCopying into merged dataset: {src_repo_id}")
    src = LeRobotDataset(src_repo_id, root=src_root)

    if target_fps is not None:
        src_fps = getattr(src.meta, "fps", None)
        if src_fps is not None and abs(float(src_fps) - float(target_fps)) > 0.5:
            raise ValueError(
                f"{src_repo_id}: fps={src_fps} does not match target fps={target_fps}. "
                "Resample the dataset first."
            )
    image_keys = infer_image_keys_from_features(src)
    episode_indices = infer_episode_indices(src)

    for ep_idx in tqdm(episode_indices, desc=f"Merging {src_repo_id}"):
        global_indices = get_episode_global_indices(src, ep_idx)

        if not global_indices:
            continue

        frames = collect_episode(src, global_indices)
        task = get_task_value(frames)

        for f in frames:
            new_frame: Dict[str, Any] = {}

            for key, value in f.items():
                if key in AUTO_KEYS:
                    continue
                if key == "task":
                    continue

                if key in image_keys:
                    new_frame[key] = normalize_image_for_add_frame(value)
                else:
                    new_frame[key] = to_python_value(value)

            new_frame["task"] = task
            writer.add_frame(new_frame)

        writer.save_episode()


def merge_datasets(
    src_repo_ids: Sequence[str],
    dst_repo_id: str,
    root: str | Path,
    target_fps: float,
    template_repo_id: str,
    use_videos: bool = True,
    push_to_hub: bool = False,
    private: bool = True,
    overwrite: bool = False,
    robot_type: str | None = None,
) -> LeRobotDataset:
    """
    Merge multiple compatible LeRobot datasets into one new dataset.

    Assumption:
    - all datasets have compatible features
    - all datasets should be treated as target_fps
    """
    root = Path(root).expanduser()
    dst_root = root / dst_repo_id.replace("/", "__")

    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"Destination already exists: {dst_root}. Use --overwrite.")

    _template_root = _resolve_src_root(root, template_repo_id)
    template = LeRobotDataset(template_repo_id, root=_template_root)
    features = get_features_for_create(template)

    if robot_type is None:
        robot_type = getattr(template.meta, "robot_type", None) or getattr(template.meta, "robot", None) or "so101"

    print(f"\nCreating merged dataset: {dst_repo_id}")
    print(f"  root       : {dst_root}")
    print(f"  target fps : {target_fps}")
    print(f"  sources    :")
    for src in src_repo_ids:
        print(f"    - {src}")

    print("Validating source datasets (fps + feature compatibility)...")
    validate_merge_sources(
        src_repo_ids=src_repo_ids,
        template_features=features,
        template_repo_id=template_repo_id,
        target_fps=target_fps,
        root=root,
    )
    print("Validation passed.")

    merged = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=target_fps,
        root=dst_root,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_threads=8,
    )

    for src_repo_id in src_repo_ids:
        _src_root = _resolve_src_root(root, src_repo_id)
        copy_dataset_into_writer(src_repo_id, merged, target_fps=target_fps, src_root=_src_root)

    if push_to_hub:
        print(f"Pushing merged dataset {dst_repo_id} to Hugging Face Hub...")
        merged.push_to_hub(private=private)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="~/lerobot_resampled_datasets")
    parser.add_argument("--target-fps", type=float, default=20.0)

    parser.add_argument("--v23-src", type=str, default="HollyTan/so101_pick-place-v2.3")
    parser.add_argument("--v23-dst", type=str, default="HollyTan/so101_pick-place-v2.3_20hz")

    parser.add_argument("--v22-src", type=str, default="HollyTan/so101_pick-place-v2.2-100eps")
    parser.add_argument("--v22-dst", type=str, default="HollyTan/so101_pick-place-v2.2-100eps_20hz")

    parser.add_argument("--v24-src", type=str, default="HollyTan/so101_pick-place-v2.4")
    parser.add_argument(
        "--merged-dst",
        type=str,
        default="HollyTan/so101_pick-place-merge-v2.2-v2.3-v2.4_20hz",
    )

    parser.add_argument("--use-videos", action="store_true", default=True)
    parser.add_argument("--no-videos", dest="use_videos", action="store_false")

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")

    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--max-compliance-gripper",
        action="store_true",
        default=False,
        help=(
            "During downsampling, select the gripper state frame with the largest "
            "|state - action| gap within each target window instead of the nearest frame. "
            "Preserves hard-object contact events at the cost of a slight temporal offset "
            "(≤ half a target frame) between arm joints and gripper state. "
            "Default: off (original nearest-frame behaviour)."
        ),
    )

    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Skip resampling and only merge existing processed repos.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only generate the two 20 Hz datasets; do not merge.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_resample:
        # v2.3: 30 Hz -> 20 Hz
        resample_dataset(
            src_repo_id=args.v23_src,
            dst_repo_id=args.v23_dst,
            root=root,
            target_fps=args.target_fps,
            source_fps_hint=30.0,
            use_videos=args.use_videos,
            push_to_hub=args.push_to_hub,
            private=args.private,
            overwrite=args.overwrite,
            max_compliance_gripper=args.max_compliance_gripper,
        )

        # v2.2: 10 Hz -> 20 Hz
        resample_dataset(
            src_repo_id=args.v22_src,
            dst_repo_id=args.v22_dst,
            root=root,
            target_fps=args.target_fps,
            source_fps_hint=10.0,
            use_videos=args.use_videos,
            push_to_hub=args.push_to_hub,
            private=args.private,
            overwrite=args.overwrite,
            max_compliance_gripper=args.max_compliance_gripper,
        )

    if not args.skip_merge:
        merge_datasets(
            src_repo_ids=[
                args.v23_dst,
                args.v22_dst,
                args.v24_src,
            ],
            dst_repo_id=args.merged_dst,
            root=root,
            target_fps=args.target_fps,
            template_repo_id=args.v23_dst,
            use_videos=args.use_videos,
            push_to_hub=args.push_to_hub,
            private=args.private,
            overwrite=args.overwrite,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()