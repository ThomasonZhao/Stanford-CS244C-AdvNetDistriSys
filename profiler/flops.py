from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Generic, List, Optional, TypeVar

import torch


T = TypeVar("T")


@dataclass
class FlopSnapshot:
    label: str
    total_flops: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def flops_to_tflops_per_second(total_flops: float, step_time_s: float) -> float:
    if total_flops < 0.0 or step_time_s <= 0.0:
        raise ValueError("total_flops must be >= 0 and step_time_s must be > 0")
    return float(total_flops) / 1e12 / float(step_time_s)


@dataclass
class FlopTracker(Generic[T]):
    device: Optional[torch.device] = None
    snapshots: List[FlopSnapshot] = None

    def __post_init__(self) -> None:
        if self.snapshots is None:
            self.snapshots = []

    def _activities(self) -> List[torch.profiler.ProfilerActivity]:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities

    def measure(self, label: str, fn: Callable[[], T]) -> tuple[T, float | None]:
        with torch.profiler.profile(
            activities=self._activities(),
            with_flops=True,
            acc_events=True,
        ) as prof:
            result = fn()
            if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        total_flops = float(sum(float(getattr(evt, "flops", 0.0) or 0.0) for evt in prof.key_averages()))
        if total_flops <= 0.0:
            return result, None

        snapshot = FlopSnapshot(label=label, total_flops=total_flops)
        self.snapshots.append(snapshot)
        return result, total_flops

    def as_dicts(self) -> List[Dict[str, float | str]]:
        return [snapshot.to_dict() for snapshot in self.snapshots]