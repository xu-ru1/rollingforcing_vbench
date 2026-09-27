"""RollingForcing layer-residual step-cache runtime.

R2 established the partial-compute path. R3 adds the online first-layer metric
handshake used by fixed/dynamic/observe policies while keeping diagnostic
injection and force policies available as deterministic validation paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Dict, Optional, Sequence

import torch

from utils.step_cache_metric import pool_block_grid_features
from utils.step_cache_policy import (
    DecisionIdentity,
    PolicyInput,
    StepCachePolicyConfig,
    decide_reuse,
)
from utils.step_cache_state import StepCacheState, make_cache_key


_METRIC_POLICIES = frozenset({"fixed", "observe_only", "dynamic_threshold"})


@dataclass(frozen=True)
class StepCacheRuntimeConfig:
    adapter: str = "layer_residual_current_kv_v1"
    implementation: str = "sparse"  # sparse | dense_reference
    decision_log_path: Optional[str] = None
    execution_log_path: Optional[str] = None
    summary_output_path: Optional[str] = None
    debug_validate_residual_finite: bool = False
    metric_spatial_grid_rows: int = 2
    metric_spatial_grid_cols: int = 2

    @classmethod
    def from_args(cls, args: Any) -> "StepCacheRuntimeConfig":
        raw = args.get("step_cache") if isinstance(args, dict) else getattr(args, "step_cache", None)
        if raw is None:
            return cls()
        get = raw.get if isinstance(raw, dict) else lambda key, default: getattr(raw, key, default)
        shape = get("metric_spatial_grid", (2, 2)) or (2, 2)
        return cls(
            adapter=str(get("adapter", "layer_residual_current_kv_v1")),
            implementation=str(get("implementation", "sparse")),
            decision_log_path=_optional_string(get("decision_log_path", None)),
            execution_log_path=_optional_string(get("execution_log_path", None)),
            summary_output_path=_optional_string(get("summary_output_path", None)),
            debug_validate_residual_finite=bool(get("debug_validate_residual_finite", False)),
            metric_spatial_grid_rows=int(shape[0]),
            metric_spatial_grid_cols=int(shape[1]),
        )

    def validate(self) -> None:
        if self.adapter != "layer_residual_current_kv_v1":
            raise ValueError(f"unsupported step-cache adapter: {self.adapter}")
        if self.implementation not in {"sparse", "dense_reference"}:
            raise ValueError(f"unsupported step-cache implementation: {self.implementation}")
        if self.metric_spatial_grid_rows <= 0 or self.metric_spatial_grid_cols <= 0:
            raise ValueError("metric spatial grid dimensions must be positive")


class StepCacheRuntime:
    """Main-forward decisions plus persistent per-block metric/residual state."""

    def __init__(self, args: Any) -> None:
        self.policy = StepCachePolicyConfig.from_args(args)
        self.config = StepCacheRuntimeConfig.from_args(args)
        self.policy.validate()
        self.config.validate()
        self.state = StepCacheState()
        self._contexts: tuple[Any, ...] = ()
        self._decisions: Dict[int, Any] = {}
        self._num_layers: int = 0
        self._tokens_per_block: int = 0
        self._num_frame_per_block: int = 0
        self._final_stage_keys: list[tuple[int, int, str]] = []
        self._operator_totals = {
            "main_forwards": 0,
            "layer_calls": 0,
            "full_window_tokens": 0,
            "k_tokens": 0,
            "v_tokens": 0,
            "q_tokens": 0,
            "attention_output_tokens": 0,
            "cross_attention_tokens": 0,
            "mlp_tokens": 0,
            "reuse_tokens": 0,
            "recompute_block_events": 0,
            "reuse_block_events": 0,
        }
        self._peak_residual_bytes = 0

    @property
    def enabled(self) -> bool:
        return bool(self.policy.enabled)

    @property
    def implementation(self) -> str:
        return self.config.implementation

    @property
    def active(self) -> bool:
        return self.enabled and bool(self._contexts)

    @property
    def current_block_count(self) -> int:
        return len(self._contexts)

    @property
    def tokens_per_block(self) -> int:
        return self._tokens_per_block

    @property
    def num_frame_per_block(self) -> int:
        return self._num_frame_per_block

    @property
    def decisions_ready(self) -> bool:
        return bool(self._contexts) and len(self._decisions) == len(self._contexts)

    @property
    def needs_first_layer_metric(self) -> bool:
        return self.active and self.policy.policy in _METRIC_POLICIES and not self.decisions_ready

    def begin_main_forward(
        self,
        contexts: Sequence[Any],
        *,
        num_layers: int,
        tokens_per_block: int,
        num_frame_per_block: int,
    ) -> None:
        if self._contexts:
            raise RuntimeError("previous step-cache main forward was not closed")
        if not contexts:
            raise ValueError("step-cache runtime requires at least one block context")
        self._contexts = tuple(contexts)
        self._num_layers = int(num_layers)
        self._tokens_per_block = int(tokens_per_block)
        self._num_frame_per_block = int(num_frame_per_block)
        self._decisions = {}
        self._final_stage_keys = []
        if self._num_layers <= 0 or self._tokens_per_block <= 0 or self._num_frame_per_block <= 0:
            raise ValueError("num_layers, tokens_per_block and num_frame_per_block must be positive")

        # Force/diagnostic policies do not depend on a content metric, so freeze
        # their mask before entering the transformer. Metric policies are frozen
        # immediately after layer-0 norm/modulation and before any skip occurs.
        if self.policy.policy not in _METRIC_POLICIES:
            for context in self._contexts:
                self._decide_without_metric(context)
        for context in self._contexts:
            if int(context.local_stage_index) == self.policy.total_local_stages - 1:
                self._final_stage_keys.append(self._key_from_context(context))
        self._operator_totals["main_forwards"] += 1

    def prepare_decisions_from_first_layer(
        self,
        first_layer_modulated_input: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> None:
        """Freeze one mask from the current layer-0 content feature.

        The method is called after full-window norm/modulation and before the
        layer-0 self-attention path branches into sparse or dense execution.
        Thus the feature itself is always computed and decisions cannot depend
        on a post-skip tensor from the same layer.
        """
        if not self.active:
            return
        if self.decisions_ready:
            return
        if self.policy.policy not in _METRIC_POLICIES:
            raise RuntimeError("non-metric policy entered the metric decision path")
        pooled = pool_block_grid_features(
            first_layer_modulated_input,
            grid_sizes,
            num_frame_per_block=self._num_frame_per_block,
            spatial_grid_rows=self.config.metric_spatial_grid_rows,
            spatial_grid_cols=self.config.metric_spatial_grid_cols,
        )
        if pooled.shape[0] != 1 or pooled.shape[1] != len(self._contexts):
            raise ValueError("step-cache online metric currently requires batch=1 and one feature per context")

        for context, feature in zip(self._contexts, pooled[0]):
            local_index = int(context.local_block_index)
            key = self._key_from_context(context)
            observation = self.state.prepare_main_observation(key, feature)
            residuals_valid = self.state.has_required_residuals(key, range(self._num_layers))
            identity = self._identity_from_context(context)
            decision = decide_reuse(
                self.policy,
                PolicyInput(
                    identity=identity,
                    local_block_index=local_index,
                    all_required_residuals_valid=residuals_valid,
                    metric_valid=observation.valid,
                    accumulator_previous=observation.accumulator_previous,
                    accumulator_test=observation.accumulator_test,
                ),
            )
            self.state.commit_main_observation(
                key,
                feature,
                observation,
                would_reuse=decision.would_reuse,
            )
            self._decisions[local_index] = decision
            accumulator_after = (
                observation.accumulator_test
                if observation.valid and decision.would_reuse
                else 0.0
            )
            self._write_decision(
                context=context,
                decision=decision,
                residuals_valid=residuals_valid,
                distance=observation.distance,
                metric_valid=observation.valid,
                metric_reason=observation.reason,
                accumulator_after=accumulator_after,
            )
        if not self.decisions_ready:
            raise RuntimeError("step-cache failed to freeze a decision for every current block")

    def _decide_without_metric(self, context: Any) -> None:
        local_index = int(context.local_block_index)
        key = self._key_from_context(context)
        residuals_valid = self.state.has_required_residuals(key, range(self._num_layers))
        identity = self._identity_from_context(context)
        decision = decide_reuse(
            self.policy,
            PolicyInput(
                identity=identity,
                local_block_index=local_index,
                all_required_residuals_valid=residuals_valid,
                metric_valid=False,
                accumulator_previous=0.0,
                accumulator_test=0.0,
            ),
        )
        self._decisions[local_index] = decision
        self._write_decision(
            context=context,
            decision=decision,
            residuals_valid=residuals_valid,
            distance=None,
            metric_valid=False,
            metric_reason="not_required_by_policy",
            accumulator_after=0.0,
        )

    def _write_decision(
        self,
        *,
        context: Any,
        decision: Any,
        residuals_valid: bool,
        distance: Optional[float],
        metric_valid: bool,
        metric_reason: str,
        accumulator_after: float,
    ) -> None:
        identity = decision.identity
        self._write(self.config.decision_log_path, {
            "event": "step_cache_decision",
            "timestamp": time.time(),
            "sample_id": identity.sample_id,
            "window_index": identity.window_index,
            "global_video_block_id": identity.global_video_block_id,
            "local_block_index": int(context.local_block_index),
            "local_stage_index": identity.local_stage_index,
            "actual_timestep": float(getattr(context, "actual_timestep", 0.0)),
            "branch": identity.branch,
            "eligible": decision.eligible,
            "would_reuse": decision.would_reuse,
            "execute_reuse": decision.execute_reuse,
            "reason": decision.reason,
            "threshold": decision.threshold,
            "distance": distance,
            "metric_valid": bool(metric_valid),
            "metric_reason": metric_reason,
            "accumulator_previous": decision.accumulator_previous,
            "accumulator_test": decision.accumulator_test,
            "accumulator_after": float(accumulator_after),
            "all_required_residuals_valid": residuals_valid,
            "implementation": self.config.implementation,
            "policy": self.policy.policy,
            "schedule": self.policy.schedule,
        })

    def recompute_local_block_indices(self, layer_index: int) -> tuple[int, ...]:
        self._validate_layer(layer_index)
        self._require_decisions_ready()
        result = []
        for local_index in range(len(self._contexts)):
            decision = self._decisions[local_index]
            if not decision.execute_reuse:
                result.append(local_index)
                continue
            key = self._key_for_local(local_index)
            if not self.state.has_required_residuals(key, [layer_index]):
                raise RuntimeError(
                    f"decision requested reuse without residual: local={local_index} layer={layer_index}")
        return tuple(result)

    def reuse_local_block_indices(self, layer_index: int) -> tuple[int, ...]:
        self._validate_layer(layer_index)
        recompute = set(self.recompute_local_block_indices(layer_index))
        return tuple(index for index in range(len(self._contexts)) if index not in recompute)

    def get_residual(self, local_block_index: int, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        return self.state.get_residual(self._key_for_local(local_block_index), layer_index)

    def update_residual(self, local_block_index: int, layer_index: int, residual: torch.Tensor) -> None:
        self._validate_layer(layer_index)
        self.state.update_residual(
            self._key_for_local(local_block_index),
            layer_index,
            residual,
            validate_finite=self.config.debug_validate_residual_finite,
        )
        stats = self.state.stats()
        self._peak_residual_bytes = max(self._peak_residual_bytes, int(stats["residual_bytes"]))

    def record_layer_execution(
        self,
        *,
        layer_index: int,
        full_blocks: int,
        recompute_blocks: Sequence[int],
        reuse_blocks: Sequence[int],
    ) -> None:
        full_tokens = int(full_blocks) * self._tokens_per_block
        active_tokens = len(recompute_blocks) * self._tokens_per_block
        reuse_tokens = len(reuse_blocks) * self._tokens_per_block
        actual_compute_tokens = full_tokens if self.config.implementation == "dense_reference" else active_tokens
        self._operator_totals["layer_calls"] += 1
        self._operator_totals["full_window_tokens"] += full_tokens
        self._operator_totals["k_tokens"] += full_tokens
        self._operator_totals["v_tokens"] += full_tokens
        self._operator_totals["q_tokens"] += actual_compute_tokens
        self._operator_totals["attention_output_tokens"] += actual_compute_tokens
        self._operator_totals["cross_attention_tokens"] += actual_compute_tokens
        self._operator_totals["mlp_tokens"] += actual_compute_tokens
        self._operator_totals["reuse_tokens"] += reuse_tokens
        self._operator_totals["recompute_block_events"] += len(recompute_blocks)
        self._operator_totals["reuse_block_events"] += len(reuse_blocks)
        self._write(self.config.execution_log_path, {
            "event": "step_cache_layer_execution",
            "timestamp": time.time(),
            "window_index": int(self._contexts[0].window_index),
            "layer_index": int(layer_index),
            "implementation": self.config.implementation,
            "full_blocks": int(full_blocks),
            "recompute_blocks": list(map(int, recompute_blocks)),
            "reuse_blocks": list(map(int, reuse_blocks)),
            "full_window_tokens": full_tokens,
            "k_tokens": full_tokens,
            "v_tokens": full_tokens,
            "q_tokens": actual_compute_tokens,
            "attention_output_tokens": actual_compute_tokens,
            "cross_attention_tokens": actual_compute_tokens,
            "mlp_tokens": actual_compute_tokens,
            "reuse_tokens": reuse_tokens,
            "residual_bytes": int(self.state.stats()["residual_bytes"]),
        })

    def end_main_forward(self) -> None:
        self._contexts = ()
        self._decisions = {}
        self._num_layers = 0
        self._tokens_per_block = 0
        self._num_frame_per_block = 0

    def finish_window_after_clean(self) -> None:
        for key in self._final_stage_keys:
            self.state.release_block(key)
        self._final_stage_keys = []

    def clear_sample(self, sample_id: int) -> None:
        sample_id = int(sample_id)
        self.state.clear_sample(sample_id)
        # This also makes explicit cleanup safe if an exception prevents the
        # corresponding clean-refresh release from being reached.
        self._final_stage_keys = [
            key for key in self._final_stage_keys if key[0] != sample_id
        ]

    def summary(self) -> dict:
        stats = self.state.stats()
        return {
            "adapter": self.config.adapter,
            "implementation": self.config.implementation,
            "enabled": self.enabled,
            "policy": self.policy.policy,
            "schedule": self.policy.schedule,
            "base_threshold": self.policy.base_threshold,
            "operator_totals": dict(self._operator_totals),
            "state": stats,
            "peak_residual_bytes": self._peak_residual_bytes,
        }

    def write_summary(self) -> None:
        if not self.config.summary_output_path:
            return
        directory = os.path.dirname(self.config.summary_output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.config.summary_output_path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _require_decisions_ready(self) -> None:
        if not self.decisions_ready:
            raise RuntimeError("step-cache decisions are not ready; layer-0 metric handshake was not executed")

    def _validate_layer(self, layer_index: int) -> None:
        if not self._contexts:
            raise RuntimeError("step-cache runtime has no active main-forward context")
        if not 0 <= int(layer_index) < self._num_layers:
            raise ValueError(f"layer index out of range: {layer_index}")

    def _identity_from_context(self, context: Any) -> DecisionIdentity:
        return DecisionIdentity(
            sample_id=int(context.sample_id),
            window_index=int(context.window_index),
            global_video_block_id=int(context.global_video_block_id),
            local_stage_index=int(context.local_stage_index),
            branch=str(context.branch),
        )

    def _key_from_context(self, context: Any):
        return make_cache_key(
            sample_id=int(context.sample_id),
            global_video_block_id=int(context.global_video_block_id),
            branch=str(context.branch),
        )

    def _key_for_local(self, local_block_index: int):
        return self._key_from_context(self._contexts[int(local_block_index)])

    @staticmethod
    def _write(path: Optional[str], payload: dict) -> None:
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _optional_string(value: Any) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None
    return str(value)
