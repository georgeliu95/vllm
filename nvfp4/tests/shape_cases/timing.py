# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# Run: imported by the layer harness in the retained B300 vLLM container.
"""Warmed component timing with explicit GPU-span versus host-time semantics."""

import os
from contextlib import nullcontext
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Timing:
    name: str
    iterations: int
    repeats: int
    gpu_span_us_median: float
    gpu_span_us_min: float
    gpu_span_us_max: float
    wall_us_median: float
    host_submit_us_median: float


def measure(
    name: str,
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int = 10,
    iterations: int = 30,
    repeats: int = 5,
) -> Timing:
    """Time a callable; CUDA event spans can include host starvation gaps."""
    profiling = os.environ.get("B300_PROFILE") == "1"
    if profiling:
        import nvtx

        warmup, iterations, repeats = 3, 1, 3
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    spans: list[float] = []
    walls: list[float] = []
    submissions: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        for _ in range(iterations):
            label = os.environ.get("B300_SHAPE_LABEL", "shape") + "/" + name
            with (
                nvtx.annotate(label, domain="shape-breakdown")
                if profiling
                else nullcontext()
            ):
                operation()
        end.record()
        submit_end = time.perf_counter_ns()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        spans.append(start.elapsed_time(end) * 1000 / iterations)
        walls.append((wall_end - wall_start) / (1000 * iterations))
        submissions.append((submit_end - wall_start) / (1000 * iterations))
    return Timing(
        name,
        iterations,
        repeats,
        statistics.median(spans),
        min(spans),
        max(spans),
        statistics.median(walls),
        statistics.median(submissions),
    )


def measure_graph(
    name: str,
    operation: Callable[[], torch.Tensor],
    *,
    valid_mask: torch.Tensor | None = None,
    nmse_limit: float = 0.0,
) -> Timing:
    """Capture the same warmed callable to test host-submission overhead."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output = operation()

    def replay() -> torch.Tensor:
        graph.replay()
        return captured_output

    eager_output = operation()
    expected = (
        eager_output if valid_mask is None else eager_output[valid_mask]
    ).clone()
    replay()
    torch.cuda.synchronize()
    actual = captured_output if valid_mask is None else captured_output[valid_mask]
    if nmse_limit:
        error = float(
            (actual.float() - expected.float()).square().sum()
            / expected.float().square().sum().clamp_min(1.0e-20)
        )
        assert error < nmse_limit, error
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return measure(name, replay)
