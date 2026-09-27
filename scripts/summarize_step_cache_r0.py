#!/usr/bin/env python3
"""Validate and summarize scalar-only R0 stage/RNG traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.step_cache_r0 import summarize_stage_trace


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(record)
    return records


def summarize_rng(records: list[dict], expected_windows: int) -> dict:
    rows = [record for record in records if record.get("event") == "r0_rng_trace"]
    phases_by_window: dict[int, dict[str, dict]] = {}
    duplicate_pairs = []
    for row in rows:
        window = int(row["window_index"])
        phase = str(row["phase"])
        bucket = phases_by_window.setdefault(window, {})
        if phase in bucket:
            duplicate_pairs.append([window, phase])
        bucket[phase] = row

    required = {
        "before_main_forward",
        "after_main_forward",
        "after_noisy_cache_update",
    }
    missing = {
        str(window): sorted(required - set(phases))
        for window, phases in sorted(phases_by_window.items())
        if required - set(phases)
    }
    absent_windows = [
        window for window in range(expected_windows)
        if window not in phases_by_window
    ]
    main_changed = [
        window for window, phases in sorted(phases_by_window.items())
        if required <= set(phases)
        and phases["before_main_forward"]["cpu_rng_sha256"]
        != phases["after_main_forward"]["cpu_rng_sha256"]
    ]
    cuda_main_changed = [
        window for window, phases in sorted(phases_by_window.items())
        if required <= set(phases)
        and phases["before_main_forward"]["cuda_rng_sha256"]
        != phases["after_main_forward"]["cuda_rng_sha256"]
    ]
    return {
        "rng_event_count": len(rows),
        "expected_rng_event_count": expected_windows * len(required),
        "duplicate_window_phase_pairs": duplicate_pairs,
        "missing_phases_by_window": missing,
        "absent_windows": absent_windows,
        "cpu_rng_changed_during_main_windows": main_changed,
        "cuda_rng_changed_during_main_windows": cuda_main_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--num-latent-frames", required=True, type=int)
    parser.add_argument("--num-frame-per-block", default=3, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-main-rng-unchanged", action="store_true")
    args = parser.parse_args()

    if args.num_latent_frames <= 0 or args.num_latent_frames % args.num_frame_per_block:
        parser.error("num-latent-frames must be a positive multiple of num-frame-per-block")
    records = read_jsonl(args.trace)
    num_blocks = args.num_latent_frames // args.num_frame_per_block
    expected_windows = num_blocks + 4
    stage_summary = summarize_stage_trace(records)
    expected_stage_counts = {str(index): num_blocks for index in range(5)}
    errors = []
    if stage_summary["trace_event_count"] != num_blocks * 5:
        errors.append("unexpected_stage_trace_event_count")
    if stage_summary["stage_counts"] != expected_stage_counts:
        errors.append("unexpected_stage_counts")
    if stage_summary["duplicate_identities"]:
        errors.append("duplicate_stage_identity")
    if stage_summary["invalid_block_stage_sequences"]:
        errors.append("invalid_block_stage_sequence")
    if stage_summary["total_local_stages"] != 5:
        errors.append("unexpected_total_local_stages")

    rng_summary = summarize_rng(records, expected_windows)
    if rng_summary["rng_event_count"] != rng_summary["expected_rng_event_count"]:
        errors.append("unexpected_rng_event_count")
    if rng_summary["duplicate_window_phase_pairs"]:
        errors.append("duplicate_rng_window_phase")
    if rng_summary["missing_phases_by_window"] or rng_summary["absent_windows"]:
        errors.append("missing_rng_trace")
    if args.require_main_rng_unchanged and (
        rng_summary["cpu_rng_changed_during_main_windows"]
        or rng_summary["cuda_rng_changed_during_main_windows"]
    ):
        errors.append("rng_changed_during_main_forward")

    payload = {
        "status": "ok" if not errors else "failed",
        "trace": str(args.trace),
        "num_latent_frames": args.num_latent_frames,
        "num_frame_per_block": args.num_frame_per_block,
        "num_video_blocks": num_blocks,
        "expected_windows": expected_windows,
        "stage": stage_summary,
        "rng": rng_summary,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
