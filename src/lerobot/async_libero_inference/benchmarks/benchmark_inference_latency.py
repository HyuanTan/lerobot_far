#!/usr/bin/env python
"""Inference Latency Benchmark — adapted from bt-libero/benchmarks/benchmark_inference_latency.py.

Measures wall-clock latency of policy.predict_action_chunk() on a LeRobot dataset.
The preprocessor pipeline (tokenization + normalization) is applied once per batch
to warm up and excluded from the timing loop, so only model inference is measured.

Usage::

    python -m lerobot.async_libero_inference.benchmarks.benchmark_inference_latency \\
        --policy.path=<checkpoint_dir> \\
        --dataset.repo_id=<dataset_id> \\
        --num_samples=100 --warmup_steps=10 \\
        --policy.device=cuda \\
        --output_file=outputs/benchmark/results.json
"""

import json
import logging
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from .benchmark_config import BenchmarkConfig


def load_dataset(cfg: BenchmarkConfig) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
    )
    logging.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    return dataset, ds_meta


def load_policy(cfg: BenchmarkConfig, ds_meta: LeRobotDatasetMetadata) -> PreTrainedPolicy:
    logging.info(f"Loading policy type: {cfg.policy.type}")
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    policy.eval()
    device = get_safe_torch_device(cfg.policy.device)
    logging.info(f"Policy loaded on device: {device}")
    return policy


def prepare_batch(batch: dict, device: torch.device) -> dict:
    """Move batch to device; convert language_instruction → task if present."""
    prepared = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            prepared[k] = v.to(device)
        else:
            prepared[k] = v
    if "language_instruction" in batch and "task" not in batch:
        prepared["task"] = batch["language_instruction"]
    return prepared


def apply_preprocessor_to_batch(batch: dict, preprocessor, device: torch.device) -> dict:
    """Apply policy preprocessor to a dataset batch.

    Dataset batches already have the batch dimension; we apply the preprocessor
    to each sample individually then re-stack so the TokenizerProcessorStep
    (designed for single observations) works correctly.
    """
    if preprocessor is None:
        return batch

    batch_size = next(
        (v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor)), 1
    )
    processed_list = []
    for i in range(batch_size):
        sample = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v[i]
            elif isinstance(v, (list, tuple)):
                sample[k] = v[i] if len(v) > i else v[0]
            else:
                sample[k] = v
        try:
            sample_proc = preprocessor(sample)
            processed_list.append(sample_proc)
        except Exception:
            # Preprocessor failed on this sample; fall back to raw batch
            return batch

    if not processed_list:
        return batch

    # Re-stack into a single batch
    stacked: dict = {}
    for k in processed_list[0]:
        vals = [p[k] for p in processed_list]
        if isinstance(vals[0], torch.Tensor):
            stacked[k] = torch.stack(vals, dim=0)
        else:
            stacked[k] = vals
    return stacked


def warmup_model(policy: PreTrainedPolicy, dataloader: DataLoader, cfg: BenchmarkConfig, preprocessor=None):
    if cfg.warmup_steps <= 0:
        return
    logging.info(f"Warming up model for {cfg.warmup_steps} steps...")
    device = get_safe_torch_device(cfg.policy.device)
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.warmup_steps:
                break
            batch = prepare_batch(batch, device)
            batch = apply_preprocessor_to_batch(batch, preprocessor, device)
            _ = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logging.info("Warmup complete")


def benchmark_inference_latency_impl(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
    preprocessor=None,
) -> dict:
    device = get_safe_torch_device(cfg.policy.device)
    latencies = []
    logging.info(f"Benchmarking {cfg.num_samples} samples...")
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.num_samples:
                break
            batch = prepare_batch(batch, device)
            batch = apply_preprocessor_to_batch(batch, preprocessor, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = policy.predict_action_chunk(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{cfg.num_samples}")

    arr = np.array(latencies)
    return {
        "num_samples": len(arr),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / np.mean(arr)),
    }


def print_results(results: dict, cfg: BenchmarkConfig):
    pretrained_path = getattr(cfg.policy, "pretrained_path", None) or "N/A"
    print("\n" + "=" * 80)
    print("INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Policy Type   : {cfg.policy.type}")
    print(f"Checkpoint    : {pretrained_path}")
    print(f"Dataset       : {cfg.dataset.repo_id}")
    print(f"Device        : {cfg.policy.device}")
    print(f"Batch Size    : {cfg.batch_size}")
    print(f"Num Samples   : {results['num_samples']}")
    print(f"\nLatency (ms):")
    print(f"  Mean   : {results['mean_ms']:.2f}")
    print(f"  Median : {results['median_ms']:.2f}")
    print(f"  Std    : {results['std_ms']:.2f}")
    print(f"  P90    : {results['p90_ms']:.2f}")
    print(f"  P95    : {results['p95_ms']:.2f}")
    print(f"  P99    : {results['p99_ms']:.2f}")
    print(f"\nThroughput: {results['fps']:.2f} FPS")
    print("=" * 80 + "\n")


def save_results(results: dict, cfg: BenchmarkConfig):
    if cfg.output_file is None:
        return
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    output_data = {
        "config": {
            "benchmark_type": "inference_latency",
            "policy_type": cfg.policy.type,
            "policy_path": str(pretrained_path) if pretrained_path else None,
            "dataset_repo_id": cfg.dataset.repo_id,
            "device": cfg.policy.device,
            "batch_size": cfg.batch_size,
            "num_samples": cfg.num_samples,
            "warmup_steps": cfg.warmup_steps,
            "seed": cfg.seed,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Results saved to: {output_path}")


@parser.wrap()
def benchmark_inference_latency(cfg: BenchmarkConfig):
    init_logging()
    logging.info("Starting inference latency benchmark")
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    set_seed(cfg.seed)

    dataset, ds_meta = load_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.policy.device == "cuda"),
    )

    policy = load_policy(cfg, ds_meta)

    # Load preprocessor pipeline (tokenization + normalization)
    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    try:
        preprocessor, _ = make_pre_post_processors(
            cfg.policy,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
        )
    except Exception as e:
        logging.warning(f"Could not load preprocessor ({e}); timing raw predict_action_chunk")
        preprocessor = None

    warmup_model(policy, dataloader, cfg, preprocessor)

    # Reset dataloader for actual benchmarking
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.policy.device == "cuda"),
    )

    results = benchmark_inference_latency_impl(policy, dataloader, cfg, preprocessor)
    print_results(results, cfg)
    save_results(results, cfg)
    logging.info("Benchmark complete!")


def main():
    benchmark_inference_latency()


if __name__ == "__main__":
    main()
