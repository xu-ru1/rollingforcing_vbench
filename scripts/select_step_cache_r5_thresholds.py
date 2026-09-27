#!/usr/bin/env python3
"""Select one compute-matched threshold per online step-cache schedule.

The input is a dense observe-only decision trace.  Selection uses decision
metadata and distances only; it deliberately never reads latents or videos.
The selected thresholds are provisional because real reuse can change later
features.  R5A therefore validates the actual online token counts afterwards.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

# Allow direct execution as ``python scripts/select_step_cache_r5_thresholds.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.step_cache_policy import (
    DecisionIdentity,
    PolicyInput,
    StepCachePolicyConfig,
    decide_reuse,
)


METHODS = {
    "fixed": ("fixed", None),
    "front": ("dynamic_threshold", "front_protect"),
    "u_shape": ("dynamic_threshold", "u_shape_protect"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay(rows: list[dict[str, Any]], *, policy: str, schedule: str | None,
           threshold: float) -> dict[str, Any]:
    config = StepCachePolicyConfig(
        enabled=True,
        policy=policy,
        schedule=schedule,
        base_threshold=threshold,
        total_local_stages=5,
        warmup_local_stages=1,
        protect_last_stage=True,
        protect_window_first_block=True,
    )
    accumulators: dict[tuple[int, int, str], float] = defaultdict(float)
    stage_counts: Counter[int] = Counter()
    prompt_counts: Counter[int] = Counter()
    identities: list[tuple[int, int, int]] = []
    for row in rows:
        key = (
            int(row["sample_id"]),
            int(row["global_video_block_id"]),
            str(row.get("branch", "conditional")),
        )
        metric_valid = bool(row.get("metric_valid"))
        distance = row.get("distance")
        previous = float(accumulators[key])
        test = previous + float(distance) if metric_valid and distance is not None else 0.0
        identity = DecisionIdentity(
            sample_id=key[0],
            window_index=int(row["window_index"]),
            global_video_block_id=key[1],
            local_stage_index=int(row["local_stage_index"]),
            branch=key[2],
        )
        decision = decide_reuse(
            config,
            PolicyInput(
                identity=identity,
                local_block_index=int(row["local_block_index"]),
                all_required_residuals_valid=bool(row["all_required_residuals_valid"]),
                metric_valid=metric_valid,
                accumulator_previous=previous,
                accumulator_test=test,
            ),
        )
        accumulators[key] = test if metric_valid and decision.would_reuse else 0.0
        if decision.execute_reuse:
            stage_counts[identity.local_stage_index] += 1
            prompt_counts[identity.sample_id] += 1
            identities.append((identity.sample_id, identity.global_video_block_id,
                               identity.local_stage_index))
    return {
        "reuse_count": len(identities),
        "reuse_count_by_stage": {str(k): stage_counts[k] for k in sorted(stage_counts)},
        "reuse_count_by_prompt": {str(k): prompt_counts[k] for k in sorted(prompt_counts)},
        "reuse_identities": identities,
    }


def choose(rows: list[dict[str, Any]], *, policy: str, schedule: str | None,
           target: int, grid_step: float, grid_max: float) -> dict[str, Any]:
    candidates = []
    steps = int(round(grid_max / grid_step))
    for index in range(steps + 1):
        threshold = index * grid_step
        result = replay(rows, policy=policy, schedule=schedule, threshold=threshold)
        # Prefer a conservative undershoot when equally far from the target,
        # then the lower threshold for deterministic selection.
        count = int(result["reuse_count"])
        candidates.append((abs(count - target), count > target, threshold, result))
    _, _, threshold, result = min(candidates, key=lambda item: item[:3])
    return {"base_threshold": threshold, "predicted": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-reuse-decisions", type=int, default=45)
    parser.add_argument("--grid-step", type=float, default=0.0005)
    parser.add_argument("--grid-max", type=float, default=2.0)
    args = parser.parse_args()
    rows = load_jsonl(args.trace)
    if len(rows) != 225:
        raise SystemExit(f"expected 225 observer decisions, found {len(rows)}")
    if args.target_reuse_decisions <= 0:
        raise SystemExit("target reuse decisions must be positive")
    if args.grid_step <= 0 or args.grid_max <= 0:
        raise SystemExit("threshold grid bounds must be positive")

    selected = {}
    for name, (policy, schedule) in METHODS.items():
        payload = choose(
            rows,
            policy=policy,
            schedule=schedule,
            target=args.target_reuse_decisions,
            grid_step=args.grid_step,
            grid_max=args.grid_max,
        )
        payload.update({"policy": policy, "schedule": schedule})
        selected[name] = payload
    output = {
        "protocol": "r5a_dense_trace_compute_match_v1",
        "quality_blind_selection": True,
        "observer_decision_count": len(rows),
        "target_reuse_decisions": args.target_reuse_decisions,
        "target_reuse_tokens": args.target_reuse_decisions * 3 * 1560 * 30,
        "grid_step": args.grid_step,
        "grid_max": args.grid_max,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        name: {
            "base_threshold": value["base_threshold"],
            "predicted_reuse": value["predicted"]["reuse_count"],
        }
        for name, value in selected.items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
