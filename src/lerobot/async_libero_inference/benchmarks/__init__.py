"""Inference latency benchmarks (Naive-Async baseline)."""
from .benchmark_config import BenchmarkConfig
from .benchmark_inference_latency import benchmark_inference_latency

__all__ = ["BenchmarkConfig", "benchmark_inference_latency"]
