#!/usr/bin/env python3
"""Run official VBench-1 scoring after all 944 cropped videos exist."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import run_step_cache_delivery as delivery


DIMENSIONS = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "object_class", "multiple_objects", "color",
    "spatial_relationship", "scene", "temporal_style", "overall_consistency",
    "human_action", "temporal_flickering", "motion_smoothness",
    "dynamic_degree", "appearance_style",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    protocol, records = delivery.static_contract()
    delivery.require(args.method in protocol["methods"], "unknown frozen method")
    output = args.output_root.resolve()
    vbench = args.vbench_root.resolve()
    video_dir = output / "videos" / args.method / "vbench30_standard"
    score_dir = output / "vbench_results" / args.method
    delivery.require((vbench / "evaluate.py").is_file() and (vbench / "static_filter.py").is_file(),
                     f"official VBench checkout is incomplete: {vbench}")
    delivery.require(delivery.digest(vbench / "vbench/VBench_full_info.json") ==
                     protocol["vbench_metadata_sha256"], "VBench checkout metadata differs from frozen R8 metadata")
    expected = {row["vbench_filename"] for row in records}
    actual = {path.name for path in video_dir.glob("*.mp4")}
    delivery.require(actual == expected, f"VBench videos missing={len(expected-actual)} extra={len(actual-expected)}")
    delivery.require(not score_dir.exists(), f"scoring output exists: {score_dir}")
    score_dir.mkdir(parents=True)
    try:
        head = subprocess.run(["git", "-C", str(vbench), "rev-parse", "HEAD"],
                              text=True, capture_output=True, check=False)
        vbench_commit = head.stdout.strip() if head.returncode == 0 else None
        delivery.require(vbench_commit is not None, "VBench git commit unavailable; NEED_SERVER_CHECK before scoring")
        frozen_commit = protocol.get("vbench_git_commit_frozen")
        if frozen_commit:
            delivery.require(vbench_commit == frozen_commit, "VBench git commit differs from frozen protocol")
        environment = subprocess.run(
            [args.vbench_python, "-c",
             "import json,platform,sys,importlib.metadata as m; "
             "names=('torch','flash-attn','diffusers','transformers','vbench'); "
             "versions={n:(m.version(n) if n in {d.metadata['Name'].lower() for d in m.distributions()} else None) for n in names}; "
             "print(json.dumps({'python_executable':sys.executable,'python_version':sys.version,'platform':platform.platform(),'packages':versions}))"],
            text=True, capture_output=True, check=False)
        delivery.require(environment.returncode == 0, f"cannot record VBench Python environment: {environment.stderr}")
        environment_record = json.loads(environment.stdout)
        cuda = subprocess.run([args.vbench_python, "-c",
                               "import torch; print(torch.version.cuda or '')"],
                              text=True, capture_output=True, check=False)
        environment_record["torch_cuda_version"] = cuda.stdout.strip() if cuda.returncode == 0 else None
        freeze = subprocess.run([args.vbench_python, "-m", "pip", "freeze"],
                                text=True, capture_output=True, check=False)
        delivery.require(freeze.returncode == 0, f"cannot record VBench pip environment: {freeze.stderr}")
        (score_dir / "vbench_pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        delivery.write_json(score_dir / "vbench_environment.json", environment_record)
        delivery.write_json(score_dir / "run_manifest.json", {
            "schema": "rollingforcing_step_cache_vbench_score_v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "rollingforcing_git_commit": delivery.commit(),
            "vbench_git_commit": vbench_commit,
            "vbench_git_commit_frozen": frozen_commit,
            "vbench_git_commit_status": "verified_frozen" if frozen_commit else "NEED_SERVER_CHECK",
            "method": args.method, "video_count": len(actual), "metadata_records": 946,
            "seed": 0, "latent_frames_generated": 126, "decoded_frames_scored": 480,
            "vbench_root": str(vbench), "vbench_python": args.vbench_python,
            "vbench_environment": environment_record,
            "vbench_metadata_sha256": protocol["vbench_metadata_sha256"],
            "video_directory": str(video_dir), "gpu": args.gpu,
            "dimension_list": DIMENSIONS,
        })
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        non_temporal = [dimension for dimension in DIMENSIONS if dimension != "temporal_flickering"]
        delivery.run_logged([args.vbench_python, str(vbench / "evaluate.py"),
                             "--videos_path", str(video_dir), "--output_path", str(score_dir / "non_temporal"),
                             "--full_json_dir", str(vbench / "vbench/VBench_full_info.json"),
                             "--mode", "vbench_standard", "--dimension", *non_temporal],
                            score_dir / "non_temporal.log", env)
        filter_dir = score_dir / "static_filter"
        delivery.run_logged([args.vbench_python, str(vbench / "static_filter.py"),
                             "--videos_path", str(video_dir), "--result_path", str(filter_dir),
                             "--filter_scope", "temporal_flickering"],
                            score_dir / "static_filter.log", env)
        filtered = filter_dir / "filtered_videos"
        retained = list(filtered.glob("*.mp4"))
        delivery.require(retained, "static filter retained no videos; inspect its raw log")
        delivery.run_logged([args.vbench_python, str(vbench / "evaluate.py"),
                             "--videos_path", str(filtered), "--output_path", str(score_dir / "temporal"),
                             "--full_json_dir", str(vbench / "vbench/VBench_full_info.json"),
                             "--mode", "vbench_standard", "--dimension", "temporal_flickering"],
                            score_dir / "temporal.log", env)
        delivery.write_json(score_dir / "status.json", {
            "status": "ok", "method": args.method, "input_videos": len(actual),
            "temporal_filter_retained_videos": len(retained), "dimensions_requested": DIMENSIONS,
        })
        print(f"[delivery] VBENCH_SCORE_OK {score_dir}")
    except Exception as exc:
        delivery.write_json(score_dir / "failure.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[delivery][error] {error}", file=sys.stderr)
        raise SystemExit(1)
