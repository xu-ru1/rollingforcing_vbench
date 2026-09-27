"""Pure policy decisions for the RollingForcing step-position cache.

This module deliberately contains no model, scheduler, cache, or CUDA calls.
It makes an auditable reuse decision from scalar state supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, FrozenSet, Optional, Tuple


POLICIES = frozenset({
    "fixed", "observe_only", "force_stage", "force_window",
    "dynamic_threshold", "diagnostic_injection",
})
SCHEDULES = frozenset({"front_protect", "u_shape_protect"})


@dataclass(frozen=True)
class DecisionIdentity:
    """Stable identity for one main-denoise block decision."""

    sample_id: int
    window_index: int
    global_video_block_id: int
    local_stage_index: int
    branch: str = "conditional"


@dataclass(frozen=True)
class StepCachePolicyConfig:
    enabled: bool = False
    policy: str = "fixed"
    schedule: Optional[str] = None
    base_threshold: float = 0.0
    total_local_stages: int = 5
    warmup_local_stages: int = 1
    protect_last_stage: bool = True
    protect_window_first_block: bool = True
    force_stage_index: Optional[int] = None
    force_window_indices: FrozenSet[int] = field(default_factory=frozenset)
    diagnostic_injection_points: FrozenSet[Tuple[int, int]] = field(
        default_factory=frozenset)

    @classmethod
    def from_args(cls, args: Any) -> "StepCachePolicyConfig":
        raw = _value(args, "step_cache", None)
        if raw is None:
            return cls()
        points = _value(raw, "diagnostic_injection_points", ()) or ()
        return cls(
            enabled=bool(_value(raw, "enabled", False)),
            policy=str(_value(raw, "policy", "fixed")),
            schedule=_optional_string(_value(raw, "schedule", None)),
            base_threshold=float(_value(raw, "base_threshold", 0.0)),
            total_local_stages=int(_value(raw, "total_local_stages", 5)),
            warmup_local_stages=int(_value(raw, "warmup_local_stages", 1)),
            protect_last_stage=bool(_value(raw, "protect_last_stage", True)),
            protect_window_first_block=bool(_value(raw, "protect_window_first_block", True)),
            force_stage_index=_optional_int(_value(raw, "force_stage_index", None)),
            force_window_indices=frozenset(int(item) for item in (_value(raw, "force_window_indices", ()) or ())),
            diagnostic_injection_points=frozenset((int(point[0]), int(point[1])) for point in points),
        )

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unsupported policy: {self.policy}")
        if self.schedule is not None and self.schedule not in SCHEDULES:
            raise ValueError(f"unsupported schedule: {self.schedule}")
        if self.policy == "dynamic_threshold" and self.schedule is None:
            raise ValueError("dynamic_threshold requires a schedule")
        if self.base_threshold < 0 or not math.isfinite(self.base_threshold):
            raise ValueError("base_threshold must be finite and non-negative")
        if self.total_local_stages < 2:
            raise ValueError("at least two local stages are required")
        if not 0 <= self.warmup_local_stages < self.total_local_stages:
            raise ValueError("warmup_local_stages is outside the stage range")
        if self.force_stage_index is not None and not (
            0 <= self.force_stage_index < self.total_local_stages):
            raise ValueError("force_stage_index is outside the stage range")
        if self.policy == "force_stage" and self.force_stage_index is None:
            raise ValueError("force_stage requires force_stage_index")
        if self.policy == "diagnostic_injection" and not self.diagnostic_injection_points:
            raise ValueError("diagnostic_injection requires injection points")
        for block_id, stage_index in self.diagnostic_injection_points:
            if block_id < 0 or not 0 <= stage_index < self.total_local_stages:
                raise ValueError("diagnostic injection point is outside the valid range")


@dataclass(frozen=True)
class PolicyInput:
    identity: DecisionIdentity
    local_block_index: int
    all_required_residuals_valid: bool
    metric_valid: bool
    accumulator_previous: float
    accumulator_test: float


@dataclass(frozen=True)
class PolicyDecision:
    identity: DecisionIdentity
    eligible: bool
    would_reuse: bool
    execute_reuse: bool
    reason: str
    threshold: Optional[float]
    accumulator_previous: float
    accumulator_test: float


def threshold_multiplier(*, schedule: Optional[str], stage_index: int) -> float:
    """Return the frozen stage multiplier for an eligible internal stage."""
    if schedule is None:
        return 1.0
    if schedule == "front_protect":
        return {1: 0.5, 2: 1.0, 3: 1.3}.get(stage_index, 1.0)
    if schedule == "u_shape_protect":
        return {1: 0.5, 2: 1.4, 3: 0.7}.get(stage_index, 1.0)
    raise ValueError(f"unsupported schedule: {schedule}")


def decide_reuse(config: StepCachePolicyConfig, value: PolicyInput) -> PolicyDecision:
    """Evaluate one decision without changing policy or cache state.

    `would_reuse` is the virtual decision used by observe-only mode to preserve
    its decision trace. `execute_reuse` controls whether a caller may skip work.
    """
    config.validate()
    identity = value.identity
    if not 0 <= identity.local_stage_index < config.total_local_stages:
        raise ValueError("identity local_stage_index is outside the configured stage range")
    if value.local_block_index < 0:
        raise ValueError("local_block_index must be non-negative")
    if not math.isfinite(value.accumulator_previous) or not math.isfinite(value.accumulator_test):
        return _decision(value, False, False, "nonfinite_accumulator", None)
    if not config.enabled:
        return _decision(value, False, False, "disabled", None)
    if identity.local_stage_index < config.warmup_local_stages:
        return _decision(value, False, False, "protected_warmup_stage", None)
    if config.protect_last_stage and identity.local_stage_index == config.total_local_stages - 1:
        return _decision(value, False, False, "protected_last_stage", None)
    if config.protect_window_first_block and value.local_block_index == 0:
        return _decision(value, False, False, "protected_window_first_block", None)
    if not value.all_required_residuals_valid:
        return _decision(value, False, False, "missing_required_residual", None)

    eligible = True
    if config.policy == "force_stage":
        reuse = identity.local_stage_index == config.force_stage_index
        return _decision(value, reuse, reuse, "force_stage" if reuse else "force_stage_miss", None, eligible)
    if config.policy == "force_window":
        reuse = identity.window_index in config.force_window_indices
        return _decision(value, reuse, reuse, "force_window" if reuse else "force_window_miss", None, eligible)
    if config.policy == "diagnostic_injection":
        point = (identity.global_video_block_id, identity.local_stage_index)
        reuse = point in config.diagnostic_injection_points
        return _decision(value, reuse, reuse, "diagnostic_injection" if reuse else "diagnostic_injection_miss", None, eligible)
    if not value.metric_valid:
        return _decision(value, False, False, "invalid_metric", None, eligible)

    schedule = config.schedule if config.policy == "dynamic_threshold" else None
    threshold = config.base_threshold * threshold_multiplier(
        schedule=schedule, stage_index=identity.local_stage_index)
    reuse = value.accumulator_test < threshold
    if config.policy == "observe_only":
        return _decision(value, reuse, False, "observe_only_reuse" if reuse else "observe_only_recompute", threshold, eligible)
    return _decision(value, reuse, reuse, "below_threshold" if reuse else "at_or_above_threshold", threshold, eligible)


def _decision(
    value: PolicyInput,
    would_reuse: bool,
    execute_reuse: bool,
    reason: str,
    threshold: Optional[float],
    eligible: bool = False,
) -> PolicyDecision:
    return PolicyDecision(
        identity=value.identity,
        eligible=eligible,
        would_reuse=would_reuse,
        execute_reuse=execute_reuse,
        reason=reason,
        threshold=threshold,
        accumulator_previous=value.accumulator_previous,
        accumulator_test=value.accumulator_test,
    )


def _value(raw: Any, key: str, default: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _optional_string(value: Any) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "null", "None"):
        return None
    return int(value)
