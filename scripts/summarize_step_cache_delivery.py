#!/usr/bin/env python3
"""Summarize generation coverage and separate performance reports without changing metrics."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import run_step_cache_delivery as delivery


def optional_json(path: Path):
    return delivery.read_json(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    protocol, records = delivery.static_contract()
    methods = {}
    for method in protocol["methods"]:
        covered: set[int] = set()
        overlaps: set[int] = set()
        shards = []
        for status_path in sorted((root / "shards" / method).glob("*/status.json")):
            status = delivery.read_json(status_path)
            if status.get("status") != "ok" or status.get("method") != method:
                continue
            start, count = int(status["start"]), int(status["count"])
            for index in range(start, start + count):
                if index in covered:
                    overlaps.add(index)
                covered.add(index)
            shards.append({"start": start, "count": count, "status_path": str(status_path)})
        expected = set(range(len(records)))
        standard = root / "videos" / method / "vbench30_standard"
        expected_names = {row["vbench_filename"] for row in records}
        actual_names = {path.name for path in standard.glob("*.mp4")}
        latency = optional_json(root / "latency" / method / "latency_report.json")
        pflops = optional_json(root / "pflops" / method / "pflops_report.json")
        score = optional_json(root / "vbench_results" / method / "status.json")
        methods[method] = {
            "completed_prompt_count": len(covered), "expected_prompt_count": len(records),
            "missing_indices": sorted(expected - covered), "overlap_indices": sorted(overlaps),
            "standard_video_count": len(actual_names),
            "missing_standard_video_count": len(expected_names - actual_names),
            "unexpected_standard_video_count": len(actual_names - expected_names),
            "shards": shards,
            "latency_mean_diffusion_ms": latency.get("mean_diffusion_ms") if latency else None,
            "latency_raw_log": latency.get("raw_log") if latency else None,
            "mean_dit_pflops": pflops.get("mean_pflops") if pflops else None,
            "pflops_raw_json": pflops.get("raw_flops_json") if pflops else None,
            "vbench_score_status": score,
        }
    output = root / "reports" / "delivery_summary.json"
    delivery.write_json(output, {
        "schema": "rollingforcing_step_cache_delivery_summary_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": delivery.commit(),
        "formal_generation_complete": all(
            value["completed_prompt_count"] == len(records) and not value["overlap_indices"]
            and value["missing_standard_video_count"] == 0 and value["unexpected_standard_video_count"] == 0
            for value in methods.values()),
        "methods": methods,
    })
    print(json.dumps({"output": str(output), "completed_per_method":
                      {method: value["completed_prompt_count"] for method, value in methods.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
