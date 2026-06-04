"""Benchmark configuration (adapted from bt-libero/benchmarks/benchmark_config.py).

Uses LeRobot's TrainPipelineConfig as the base; training-specific fields are
ignored during benchmarking.
"""

from dataclasses import dataclass
from typing import Union

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class BenchmarkConfig(TrainPipelineConfig):
    """Configuration for benchmarking a pretrained policy.

    Reuses TrainPipelineConfig's dataset and policy configuration.
    Training-specific fields are ignored during benchmarking.

    Use policy.compile_model=true to enable torch.compile optimization.
    """

    type: str = "inference_latency"
    num_samples: int = 100
    warmup_steps: int = 10
    output_file: Union[str, None] = None

    def validate(self) -> None:
        if self.type not in ["inference_latency"]:
            raise ValueError(f"Invalid benchmark type: {self.type}.")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
