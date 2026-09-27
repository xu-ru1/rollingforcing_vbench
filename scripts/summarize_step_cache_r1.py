#!/usr/bin/env python3
"""Validate the scalar-only first-layer metric trace produced by R1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--num-latent-frames", required=True, type=int)
    parser.add_argument("--num-frame-per-block", default=3, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.num_latent_frames <= 0 or args.num_latent_frames % args.num_frame_per_block:
        parser.error("num-latent-frames must be a positive multiple of num-frame-per-block")
    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    blocks = args.num_latent_frames // args.num_frame_per_block
    stage_counts = Counter(int(row["local_stage_index"]) for row in rows)
    identities = [(int(row["global_video_block_id"]), int(row["local_stage_index"])) for row in rows]
    expected_stage_counts = {str(index): blocks for index in range(5)}
    errors = []
    if len(rows) != blocks * 5:
        errors.append("unexpected_metric_record_count")
    if {str(key): stage_counts[key] for key in sorted(stage_counts)} != expected_stage_counts:
        errors.append("unexpected_stage_counts")
    if len(set(identities)) != len(identities):
        errors.append("duplicate_block_stage_metric")
    if any(row.get("metric") != "first_layer_modulated_grid_l1rel_v1" for row in rows):
        errors.append("unexpected_metric_name")
    if any("feature" in row for row in rows):
        errors.append("raw_feature_serialized")
    valid_count = sum(bool(row.get("metric_valid")) for row in rows)
    expected_valid_count = blocks * 4
    if valid_count != expected_valid_count:
        errors.append("unexpected_valid_metric_count")
    payload = {
        "status": "ok" if not errors else "failed",
        "metrics": str(args.metrics),
        "num_latent_frames": args.num_latent_frames,
        "num_video_blocks": blocks,
        "record_count": len(rows),
        "stage_counts": {str(key): stage_counts[key] for key in sorted(stage_counts)},
        "valid_metric_count": valid_count,
        "expected_valid_metric_count": expected_valid_count,
        "feature_shapes": sorted({tuple(row.get("feature_shape", ())) for row in rows}),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
