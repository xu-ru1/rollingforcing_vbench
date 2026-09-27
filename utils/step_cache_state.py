"""State contract for a future RollingForcing layer-residual step cache.

The module stores no scheduler or model global state.  The model integration in
R2 will supply first-layer features and layer residual tensors to this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


CacheKey = Tuple[int, int, str]


@dataclass(frozen=True)
class MetricObservation:
    distance: Optional[float]
    accumulator_previous: float
    accumulator_test: float
    valid: bool
    reason: str


@dataclass
class BlockCacheState:
    previous_feature: Optional[torch.Tensor] = None
    accumulator: float = 0.0
    residuals: Dict[int, torch.Tensor] = field(default_factory=dict)


def make_cache_key(*, sample_id: int, global_video_block_id: int, branch: str) -> CacheKey:
    if sample_id < 0 or global_video_block_id < 0:
        raise ValueError("sample_id and global_video_block_id must be non-negative")
    if not branch:
        raise ValueError("branch must be non-empty")
    return (int(sample_id), int(global_video_block_id), str(branch))


def fp32_relative_l1(current: torch.Tensor, previous: torch.Tensor, *, epsilon: float = 1e-6) -> MetricObservation:
    """Compute the frozen R1 candidate metric without modifying either tensor."""
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if current.shape != previous.shape:
        return MetricObservation(None, 0.0, 0.0, False, "feature_shape_mismatch")
    if current.device != previous.device:
        return MetricObservation(None, 0.0, 0.0, False, "feature_device_mismatch")
    current_fp32 = current.detach().to(dtype=torch.float32)
    previous_fp32 = previous.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(current_fp32).all()) or not bool(torch.isfinite(previous_fp32).all()):
        return MetricObservation(None, 0.0, 0.0, False, "nonfinite_feature")
    numerator = torch.mean(torch.abs(current_fp32 - previous_fp32))
    denominator = torch.clamp(torch.mean(torch.abs(previous_fp32)), min=epsilon)
    distance = float((numerator / denominator).item())
    if not math.isfinite(distance):
        return MetricObservation(None, 0.0, 0.0, False, "nonfinite_distance")
    return MetricObservation(distance, 0.0, distance, True, "ok")


class StepCacheState:
    """Per-sample/block/branch state with explicit main-access lifecycle."""

    def __init__(self) -> None:
        self._blocks: Dict[CacheKey, BlockCacheState] = {}

    def prepare_main_observation(self, key: CacheKey, current_feature: torch.Tensor) -> MetricObservation:
        """Read metric state; caller must commit exactly once after the decision."""
        state = self._blocks.setdefault(key, BlockCacheState())
        if state.previous_feature is None:
            return MetricObservation(None, state.accumulator, state.accumulator, False, "no_previous_feature")
        metric = fp32_relative_l1(current_feature, state.previous_feature)
        if not metric.valid or metric.distance is None:
            return MetricObservation(None, state.accumulator, state.accumulator, False, metric.reason)
        test_value = state.accumulator + metric.distance
        if not math.isfinite(test_value):
            return MetricObservation(None, state.accumulator, state.accumulator, False, "nonfinite_accumulator")
        return MetricObservation(metric.distance, state.accumulator, test_value, True, "ok")

    def commit_main_observation(
        self,
        key: CacheKey,
        current_feature: torch.Tensor,
        observation: MetricObservation,
        *,
        would_reuse: bool,
    ) -> None:
        """Update feature every main access; reset/retain accumulator by virtual decision.

        Observe-only calls pass their virtual `would_reuse`, so their decision
        trace remains comparable to Fixed although execution fully recomputes.
        Clean-refresh calls must never invoke this method.
        """
        state = self._blocks.setdefault(key, BlockCacheState())
        state.previous_feature = current_feature.detach().clone()
        if observation.valid and would_reuse:
            state.accumulator = observation.accumulator_test
        else:
            state.accumulator = 0.0

    def update_residual(
        self, key: CacheKey, layer_index: int, residual: torch.Tensor, *, validate_finite: bool = True
    ) -> None:
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if validate_finite and not bool(torch.isfinite(residual.detach()).all()):
            raise ValueError("residual must be finite")
        self._blocks.setdefault(key, BlockCacheState()).residuals[int(layer_index)] = residual.detach().clone()

    def has_required_residuals(self, key: CacheKey, layer_indices: Iterable[int]) -> bool:
        residuals = self._blocks.get(key, BlockCacheState()).residuals
        return all(int(layer_index) in residuals for layer_index in layer_indices)

    def get_residual(self, key: CacheKey, layer_index: int) -> torch.Tensor:
        try:
            return self._blocks[key].residuals[int(layer_index)]
        except KeyError as exc:
            raise KeyError(f"missing residual for key={key}, layer={layer_index}") from exc

    def release_block(self, key: CacheKey) -> None:
        self._blocks.pop(key, None)

    def clear_sample(self, sample_id: int) -> None:
        for key in [key for key in self._blocks if key[0] == sample_id]:
            del self._blocks[key]

    def stats(self) -> dict:
        residual_bytes = 0
        residual_count = 0
        for state in self._blocks.values():
            for residual in state.residuals.values():
                residual_count += 1
                residual_bytes += residual.numel() * residual.element_size()
        return {
            "active_block_count": len(self._blocks),
            "residual_count": residual_count,
            "residual_bytes": residual_bytes,
        }
