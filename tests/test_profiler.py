import time

import torch

from profiler import FlopTracker, MemoryTracker, TimerRegistry, flops_to_tflops_per_second, overlap_efficiency


def test_overlap_efficiency_bounds() -> None:
    # no overlap
    assert overlap_efficiency(step_ms=15.0, compute_ms=10.0, communication_ms=5.0) == 0.0
    # full overlap
    assert overlap_efficiency(step_ms=10.0, compute_ms=10.0, communication_ms=5.0) == 1.0


def test_timer_registry_records_samples() -> None:
    registry = TimerRegistry(device=None)
    timer = registry.timer("sleep")

    timer.start()
    time.sleep(0.01)
    timer.stop()

    summary = registry.summarize()["sleep"]
    assert summary.count == 1
    assert summary.total_ms > 0.0


def test_memory_tracker_records_snapshots() -> None:
    tracker = MemoryTracker(device=None)
    tracker.record("before")
    tracker.record("after")

    snapshots = tracker.as_dicts()
    assert len(snapshots) == 2
    assert snapshots[0]["label"] == "before"
    assert snapshots[1]["label"] == "after"


def test_flop_tracker_measures_linear_step() -> None:
    tracker = FlopTracker(device=None)
    layer = torch.nn.Linear(16, 8)
    x = torch.randn(4, 16)

    def run_step() -> torch.Tensor:
        layer.zero_grad(set_to_none=True)
        y = layer(x)
        loss = y.square().mean()
        loss.backward()
        return loss

    _loss, total_flops = tracker.measure("linear_step", run_step)

    assert total_flops is not None
    assert total_flops > 0.0
    assert tracker.as_dicts()[0]["label"] == "linear_step"
    assert flops_to_tflops_per_second(total_flops=total_flops, step_time_s=0.5) > 0.0
