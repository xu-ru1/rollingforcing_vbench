#!/usr/bin/env python3
"""Create one reviewable R0 acceptance report from an R0 output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "path": str(path), "error": str(exc)}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "error": "expected JSON object"}
    return payload


def check_trace(summary: dict, *, latent_frames: int) -> dict:
    expected_blocks = latent_frames // 3
    expected_windows = expected_blocks + 4
    return {
        "summary_status": summary.get("status"),
        "expected_video_blocks": expected_blocks,
        "reported_video_blocks": summary.get("num_video_blocks"),
        "expected_windows": expected_windows,
        "reported_windows": summary.get("expected_windows"),
        "stage_events": summary.get("stage", {}).get("trace_event_count"),
        "rng_events": summary.get("rng", {}).get("rng_event_count"),
        "rng_changed_during_main_windows": {
            "cpu": summary.get("rng", {}).get("cpu_rng_changed_during_main_windows", []),
            "cuda": summary.get("rng", {}).get("cuda_rng_changed_during_main_windows", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = args.run_root / "reports"
    preflight = load_json(reports / "preflight_manifest.json")
    trace_21 = load_json(reports / "trace_21_trace_summary.json")
    trace_81 = load_json(reports / "trace_81_trace_summary.json")
    b0_video = load_json(reports / "b0_video_check.json")
    trace_21_video = load_json(reports / "trace_21_video_check.json")
    trace_81_video = load_json(reports / "trace_81_video_check.json")

    required_reports = [preflight, trace_21, trace_81, b0_video, trace_21_video, trace_81_video]
    errors = []
    if any(item.get("status") != "ok" for item in required_reports):
        errors.append("missing_or_failed_required_report")
    if b0_video.get("comparison", {}).get("same_decoded_frame_count") is not True:
        errors.append("baseline_repeat_frame_count_mismatch")
    for label, video_check, expected_frames in (
        ("trace_21", trace_21_video, 81),
        ("trace_81", trace_81_video, 321),
    ):
        counts = [row.get("decoded_frame_count") for row in video_check.get("videos", [])]
        if counts != [expected_frames]:
            errors.append(f"{label}_decoded_frame_count_mismatch")

    payload = {
        "status": "ok" if not errors else "failed",
        "stage": "R0",
        "run_root": str(args.run_root),
        "preflight_status": preflight.get("status"),
        "trace_21": check_trace(trace_21, latent_frames=21),
        "trace_81": check_trace(trace_81, latent_frames=81),
        "baseline_repeat": {
            "same_decoded_frame_count": b0_video.get("comparison", {}).get("same_decoded_frame_count"),
            "same_decoded_frames_sha256": b0_video.get("comparison", {}).get("same_decoded_frames_sha256"),
        },
        "errors": errors,
        "limitations": [
            "Decoded MP4 equality is diagnostic only; R0 does not yet capture latent-level numerical equivalence.",
            "A successful report confirms baseline observability, not step-cache correctness or acceleration.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
