#!/usr/bin/env python3
"""Decode R0 videos and compare decoded-frame digests for paired baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio


def inspect_video(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = imageio.get_reader(str(path))
    digest = hashlib.sha256()
    frame_count = 0
    frame_shape = None
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            frame_count += 1
            frame_shape = list(frame.shape)
            digest.update(frame.tobytes())
    finally:
        reader.close()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "decoded_frame_count": frame_count,
        "decoded_frame_shape": frame_shape,
        "fps": metadata.get("fps"),
        "decoded_frames_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--expect-decoded-frames", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = [inspect_video(path) for path in args.video]
    comparison = None
    if args.compare:
        left, right = (inspect_video(path) for path in args.compare)
        comparison = {
            "left": left["path"],
            "right": right["path"],
            "same_decoded_frame_count": left["decoded_frame_count"] == right["decoded_frame_count"],
            "same_decoded_frames_sha256": left["decoded_frames_sha256"] == right["decoded_frames_sha256"],
        }
    errors = []
    if args.expect_decoded_frames is not None:
        errors.extend(
            f"unexpected_frame_count:{record['path']}={record['decoded_frame_count']}"
            for record in records
            if record["decoded_frame_count"] != args.expect_decoded_frames
        )
    payload = {
        "status": "ok" if not errors else "failed",
        "videos": records,
        "comparison": comparison,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
