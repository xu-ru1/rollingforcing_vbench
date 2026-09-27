#!/usr/bin/env python3
"""Assess the compressed R3 long-lifecycle and multi-prompt integration run."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKENS_PER_BLOCK = 3 * 1560
KV_CAPACITY_TOKENS = 24 * 1560
EXPECTED_THRESHOLDS = {
    "front_protect": {1: 0.13, 2: 0.26, 3: 0.338},
    "u_shape_protect": {1: 0.13, 2: 0.364, 3: 0.182},
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_schedule(rows: list[dict[str, Any]], schedule: str, errors: list[str], label: str) -> dict[str, Any]:
    observed: dict[int, set[float]] = defaultdict(set)
    protected_violations = 0
    for row in rows:
        stage = int(row["local_stage_index"])
        if (stage in (0, 4) or int(row["local_block_index"]) == 0) and row.get("execute_reuse"):
            protected_violations += 1
        if row.get("threshold") is not None:
            observed[stage].add(float(row["threshold"]))
    if protected_violations:
        errors.append(f"protected_reuse:{label}:{protected_violations}")
    for stage, expected in EXPECTED_THRESHOLDS[schedule].items():
        values = observed.get(stage, set())
        if not values or any(not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9) for value in values):
            errors.append(f"threshold_mismatch:{label}:stage={stage}:values={sorted(values)}:expected={expected}")
    return {
        "schedule": schedule,
        "thresholds_by_stage": {str(stage): sorted(values) for stage, values in sorted(observed.items())},
        "protected_reuse_violations": protected_violations,
    }


def check_operator_contract(summary: dict[str, Any], errors: list[str], label: str, *, main_forwards: int) -> dict[str, Any]:
    totals = summary.get("operator_totals", {})
    if totals.get("main_forwards") != main_forwards:
        errors.append(f"main_forward_count:{label}:{totals.get('main_forwards')}:{main_forwards}")
    expected_layer_calls = main_forwards * 30
    if totals.get("layer_calls") != expected_layer_calls:
        errors.append(f"layer_call_count:{label}:{totals.get('layer_calls')}:{expected_layer_calls}")
    full = totals.get("full_window_tokens")
    if totals.get("k_tokens") != full or totals.get("v_tokens") != full:
        errors.append(f"kv_not_full:{label}")
    if totals.get("q_tokens", 0) + totals.get("reuse_tokens", 0) != full:
        errors.append(f"q_reuse_conservation:{label}")
    if totals.get("reuse_block_events", 0) <= 0 or totals.get("q_tokens", 0) >= full:
        errors.append(f"no_real_sparse_reuse:{label}")
    state = summary.get("state", {})
    if any(int(state.get(key, -1)) != 0 for key in ("active_block_count", "residual_count", "residual_bytes")):
        errors.append(f"nonempty_terminal_state:{label}:{state}")
    return totals


def eval_metrics(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"\[EvalMetrics\]\s+runtime_sec=([0-9.]+)\s+peak_cuda_allocated_gb=([^\s]+)\s+"
        r"peak_cuda_reserved_gb=([^\s]+)\s+saved_videos=(\d+)", text,
    )
    if not matches:
        return {"available": False}
    runtime, allocated, reserved, videos = matches[-1]
    return {
        "available": True,
        "runtime_sec": float(runtime),
        "peak_cuda_allocated_gb": None if allocated == "none" else float(allocated),
        "peak_cuda_reserved_gb": None if reserved == "none" else float(reserved),
        "saved_videos": int(videos),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = args.run_root / "reports"
    logs = args.run_root / "logs"
    errors: list[str] = []
    result: dict[str, Any] = {}

    front_rows = load_jsonl(reports / "front_81_decisions.jsonl")
    front_summary = load_json(reports / "front_81_runtime_summary.json")
    front_video = load_json(reports / "front_81_video_check.json")
    if len(front_rows) != 135:
        errors.append(f"front_decision_count:{len(front_rows)}:135")
    front_stage_counts = Counter(int(row["local_stage_index"]) for row in front_rows)
    if front_stage_counts != Counter({stage: 27 for stage in range(5)}):
        errors.append(f"front_stage_counts:{dict(front_stage_counts)}")
    front_reuse = sum(bool(row.get("execute_reuse")) for row in front_rows)
    if front_reuse <= 0:
        errors.append("front_has_no_reuse")
    front_totals = check_operator_contract(front_summary, errors, "front_81", main_forwards=31)
    front_schedule = check_schedule(front_rows, "front_protect", errors, "front_81")
    if front_video.get("status") != "ok":
        errors.append("front_video_failed")

    front_kv = load_jsonl(reports / "front_81_kv_metadata.jsonl")
    front_layer0 = [row for row in front_kv if int(row.get("layer_idx", -1)) == 0]
    max_global = max((int(row["global_end_index"]) for row in front_layer0 if row.get("global_end_index") is not None), default=-1)
    max_local = max((int(row["local_end_index"]) for row in front_layer0 if row.get("local_end_index") is not None), default=-1)
    if max_global < 27 * TOKENS_PER_BLOCK:
        errors.append(f"long_global_end_not_reached:{max_global}")
    if max_local > KV_CAPACITY_TOKENS:
        errors.append(f"local_kv_capacity_exceeded:{max_local}")
    if max_global <= max_local:
        errors.append(f"kv_eviction_not_observed:global={max_global}:local={max_local}")

    u_rows = load_jsonl(reports / "u_reset_decisions.jsonl")
    u_summary = load_json(reports / "u_reset_runtime_summary.json")
    u_video = load_json(reports / "u_reset_video_check.json")
    sample_counts = Counter(int(row["sample_id"]) for row in u_rows)
    if sample_counts != Counter({0: 35, 1: 35}):
        errors.append(f"u_sample_counts:{dict(sample_counts)}")
    u_reuse = sum(bool(row.get("execute_reuse")) for row in u_rows)
    if u_reuse <= 0:
        errors.append("u_has_no_reuse")
    u_totals = check_operator_contract(u_summary, errors, "u_reset", main_forwards=22)
    u_schedule = check_schedule(u_rows, "u_shape_protect", errors, "u_reset")
    if u_video.get("status") != "ok" or len(u_video.get("videos", [])) != 2:
        errors.append("u_two_video_check_failed")

    u_kv = load_jsonl(reports / "u_reset_kv_metadata.jsonl")
    reset_rows = [
        row for row in u_kv
        if row.get("event") == "before_denoise"
        and int(row.get("window_index", -1)) == 0
        and int(row.get("layer_idx", -1)) == 0
    ]
    if len(reset_rows) != 2 or any(int(row.get("global_end_index", -1)) != 0 or int(row.get("local_end_index", -1)) != 0 for row in reset_rows):
        errors.append(f"kv_sample_reset_failed:{reset_rows}")

    result["front_81"] = {
        "decision_count": len(front_rows),
        "stage_counts": dict(sorted(front_stage_counts.items())),
        "execute_reuse_count": front_reuse,
        "operator_totals": front_totals,
        "peak_residual_bytes": front_summary.get("peak_residual_bytes"),
        "schedule_check": front_schedule,
        "kv_lifecycle": {
            "metadata_records": len(front_kv),
            "max_global_end_index": max_global,
            "max_local_end_index": max_local,
            "capacity_tokens": KV_CAPACITY_TOKENS,
            "eviction_observed": max_global > max_local and max_local <= KV_CAPACITY_TOKENS,
        },
        "eval_metrics": eval_metrics(logs / "front_81.log"),
    }
    result["u_reset"] = {
        "decision_count": len(u_rows),
        "decision_count_by_sample": dict(sorted(sample_counts.items())),
        "execute_reuse_count": u_reuse,
        "operator_totals": u_totals,
        "peak_residual_bytes": u_summary.get("peak_residual_bytes"),
        "schedule_check": u_schedule,
        "kv_reset_rows": reset_rows,
        "eval_metrics": eval_metrics(logs / "u_reset.log"),
    }
    payload = {"status": "ok" if not errors else "error", "errors": errors, "cases": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
