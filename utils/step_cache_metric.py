"""Read-only first-layer feature metric recorder for R1 calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any, Optional, Sequence

import torch

from utils.step_cache_state import fp32_relative_l1, make_cache_key


@dataclass(frozen=True)
class MetricObserverConfig:
    enabled: bool = False
    output_path: Optional[str] = None
    spatial_grid_rows: int = 2
    spatial_grid_cols: int = 2

    @classmethod
    def from_args(cls, args: Any) -> "MetricObserverConfig":
        raw = args.get("step_cache") if isinstance(args, dict) else getattr(args, "step_cache", None)
        if raw is None:
            return cls()
        get = raw.get if isinstance(raw, dict) else lambda key, default: getattr(raw, key, default)
        shape = get("metric_spatial_grid", (2, 2)) or (2, 2)
        return cls(
            enabled=bool(get("metric_observer_enabled", False)),
            output_path=_optional_string(get("metric_observer_output_path", None)),
            spatial_grid_rows=int(shape[0]),
            spatial_grid_cols=int(shape[1]),
        )

    def validate(self) -> None:
        if self.enabled and not self.output_path:
            raise ValueError("metric_observer_output_path is required when metric observation is enabled")
        if self.spatial_grid_rows <= 0 or self.spatial_grid_cols <= 0:
            raise ValueError("metric spatial grid dimensions must be positive")


@dataclass(frozen=True)
class MetricBlockContext:
    sample_id: int
    window_index: int
    global_video_block_id: int
    local_block_index: int
    local_stage_index: int
    actual_timestep: float
    branch: str


def pool_block_grid_features(
    first_layer_modulated_input: torch.Tensor,
    grid_sizes: torch.Tensor,
    *,
    num_frame_per_block: int,
    spatial_grid_rows: int,
    spatial_grid_cols: int,
) -> torch.Tensor:
    """Pool [B, F*H*W, C] features to [B, blocks, cells, C] in FP32."""
    if first_layer_modulated_input.ndim != 3:
        raise ValueError("expected first-layer input with shape [B, L, C]")
    if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
        raise ValueError("expected grid_sizes with shape [B, 3]")
    if first_layer_modulated_input.shape[0] != grid_sizes.shape[0]:
        raise ValueError("feature batch and grid batch differ")
    if num_frame_per_block <= 0 or spatial_grid_rows <= 0 or spatial_grid_cols <= 0:
        raise ValueError("pooling dimensions must be positive")
    pooled_samples = []
    for batch_index, (frames, height, width) in enumerate(grid_sizes.detach().cpu().tolist()):
        if frames % num_frame_per_block:
            raise ValueError("grid frame count is not a multiple of num_frame_per_block")
        if first_layer_modulated_input.shape[1] != frames * height * width:
            raise ValueError("feature sequence length does not match grid size")
        feature = first_layer_modulated_input[batch_index].detach().to(dtype=torch.float32)
        feature = feature.reshape(frames, height, width, feature.shape[-1])
        blocks = []
        for block_start in range(0, frames, num_frame_per_block):
            cells = []
            for row in range(spatial_grid_rows):
                row_start = row * height // spatial_grid_rows
                row_end = (row + 1) * height // spatial_grid_rows
                for col in range(spatial_grid_cols):
                    col_start = col * width // spatial_grid_cols
                    col_end = (col + 1) * width // spatial_grid_cols
                    cells.append(feature[block_start:block_start + num_frame_per_block, row_start:row_end, col_start:col_end].mean(dim=(0, 1, 2)))
            blocks.append(torch.stack(cells))
        pooled_samples.append(torch.stack(blocks))
    return torch.stack(pooled_samples)


class StepCacheMetricRecorder:
    """Captures scalar distances from the first main forward layer only."""

    def __init__(self, config: MetricObserverConfig) -> None:
        config.validate()
        self.config = config
        self._previous_features: dict[tuple[int, int, str], torch.Tensor] = {}
        self._active_context: Optional[tuple[Sequence[MetricBlockContext], int]] = None

    @classmethod
    def from_args(cls, args: Any) -> "StepCacheMetricRecorder":
        return cls(MetricObserverConfig.from_args(args))

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def begin_main_forward(self, contexts: Sequence[MetricBlockContext], *, num_frame_per_block: int) -> None:
        if not self.enabled:
            return
        if self._active_context is not None:
            raise RuntimeError("previous metric observation was not closed")
        if not contexts:
            raise ValueError("metric observation requires at least one block context")
        self._active_context = (tuple(contexts), int(num_frame_per_block))

    def record_first_layer_modulated_input(self, first_layer_modulated_input: torch.Tensor, grid_sizes: torch.Tensor) -> None:
        if not self.enabled or self._active_context is None:
            return
        contexts, num_frame_per_block = self._active_context
        pooled = pool_block_grid_features(
            first_layer_modulated_input,
            grid_sizes,
            num_frame_per_block=num_frame_per_block,
            spatial_grid_rows=self.config.spatial_grid_rows,
            spatial_grid_cols=self.config.spatial_grid_cols,
        )
        if pooled.shape[0] != 1 or pooled.shape[1] != len(contexts):
            raise ValueError("R1 metric observer currently requires batch=1 and one feature block per context")
        for context, feature in zip(contexts, pooled[0]):
            key = make_cache_key(
                sample_id=context.sample_id,
                global_video_block_id=context.global_video_block_id,
                branch=context.branch,
            )
            previous = self._previous_features.get(key)
            if previous is None:
                valid, distance, reason = False, None, "no_previous_feature"
            else:
                metric = fp32_relative_l1(feature, previous)
                valid, distance, reason = metric.valid, metric.distance, metric.reason
            self._write({
                "event": "r1_first_layer_metric",
                "timestamp": time.time(),
                "sample_id": context.sample_id,
                "window_index": context.window_index,
                "global_video_block_id": context.global_video_block_id,
                "local_block_index": context.local_block_index,
                "local_stage_index": context.local_stage_index,
                "actual_timestep": context.actual_timestep,
                "branch": context.branch,
                "metric": "first_layer_modulated_grid_l1rel_v1",
                "metric_valid": valid,
                "distance": distance,
                "metric_reason": reason,
                "spatial_grid": [self.config.spatial_grid_rows, self.config.spatial_grid_cols],
                "feature_shape": list(feature.shape),
            })
            self._previous_features[key] = feature.detach().clone()

    def end_main_forward(self) -> None:
        self._active_context = None

    def clear_sample(self, sample_id: int) -> None:
        for key in [key for key in self._previous_features if key[0] == sample_id]:
            del self._previous_features[key]

    def _write(self, payload: dict) -> None:
        if not self.config.output_path:
            raise ValueError("metric observer output path is missing")
        output_dir = os.path.dirname(self.config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.config.output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _optional_string(value: Any) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None
    return str(value)
