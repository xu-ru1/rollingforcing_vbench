"""Pure helpers for the R0 RollingForcing stage-trace audit."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def resolve_stage_index(
    actual_timestep: float,
    stage_values: Sequence[float],
    tolerance: float,
) -> int:
    matches = [
        index for index, value in enumerate(stage_values)
        if abs(float(value) - float(actual_timestep)) <= float(tolerance)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one local stage match: "
            f"actual_timestep={actual_timestep}, matches={matches}, "
            f"stage_values={[float(item) for item in stage_values]}, "
            f"tolerance={tolerance}")
    return matches[0]


def validate_block_timestep_values(
    values: Iterable[float],
    tolerance: float,
) -> float:
    flattened = [float(value) for value in values]
    if not flattened:
        raise ValueError("cannot infer a local stage from an empty timestep slice")
    minimum, maximum = min(flattened), max(flattened)
    if maximum - minimum > float(tolerance):
        raise ValueError(
            "a global video block carries mixed timesteps: "
            f"min={minimum}, max={maximum}, tolerance={tolerance}")
    return sum(flattened) / len(flattened)


def stage_bin(stage_index: int, total_stages: int) -> str:
    if total_stages <= 0:
        raise ValueError(f"total_stages must be positive, got {total_stages}")
    if not 0 <= int(stage_index) < int(total_stages):
        raise ValueError(
            f"stage_index={stage_index} is outside [0, {total_stages})")
    ratio = int(stage_index) / max(int(total_stages) - 1, 1)
    if ratio < 1.0 / 3.0:
        return "early"
    if ratio < 2.0 / 3.0:
        return "middle"
    return "late"


def summarize_stage_trace(records: Iterable[dict]) -> dict:
    trace = [record for record in records if record.get("event") == "r0_stage_trace"]
    stage_counts = Counter(int(record["local_stage_index"]) for record in trace)
    block_to_stages: dict[int, list[int]] = {}
    duplicate_identities: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for record in trace:
        identity = (int(record["global_video_block_id"]), int(record["local_stage_index"]))
        if identity in seen:
            duplicate_identities.append(identity)
        seen.add(identity)
        block_to_stages.setdefault(identity[0], []).append(identity[1])

    total_stages = (
        int(trace[0]["total_local_stages"]) if trace else None)
    expected = list(range(total_stages)) if total_stages is not None else []
    invalid_blocks = {
        block_id: sorted(stages)
        for block_id, stages in block_to_stages.items()
        if sorted(stages) != expected
    }
    return {
        "trace_event_count": len(trace),
        "stage_counts": {str(key): stage_counts[key] for key in sorted(stage_counts)},
        "unique_global_blocks": len(block_to_stages),
        "duplicate_identities": duplicate_identities,
        "invalid_block_stage_sequences": invalid_blocks,
        "total_local_stages": total_stages,
    }
