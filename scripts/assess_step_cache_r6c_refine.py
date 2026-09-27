#!/usr/bin/env python3
"""Assess direct latency matching candidates for Front-slow and U-shape-fast."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


DIFFUSION_RE = re.compile(r"Diffusion generation time:\s*([0-9.]+)\s*ms")
ALLOCATED_RE = re.compile(r"Peak CUDA allocated:\s*([0-9.]+)\s*GB")
RESERVED_RE = re.compile(r"Peak CUDA reserved:\s*([0-9.]+)\s*GB")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_profile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    diffusion = [float(value) for value in DIFFUSION_RE.findall(text)]
    allocated = [float(value) for value in ALLOCATED_RE.findall(text)]
    reserved = [float(value) for value in RESERVED_RE.findall(text)]
    if len(diffusion) != 4:
        raise ValueError(f"expected four diffusion timings, found {len(diffusion)}")
    measured = diffusion[1:]
    mean = statistics.fmean(measured)
    stdev = statistics.stdev(measured)
    return {
        "warmup_diffusion_ms": diffusion[0],
        "measured_diffusion_ms": measured,
        "mean_diffusion_ms": mean,
        "median_diffusion_ms": statistics.median(measured),
        "stdev_diffusion_ms": stdev,
        "cv": stdev / mean if mean else None,
        "peak_cuda_allocated_gb": max(allocated) if allocated else None,
        "peak_cuda_reserved_gb": max(reserved) if reserved else None,
    }


def relative_error(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference <= 0:
        return None
    return (candidate - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()
    reports, logs = args.run_root / "reports", args.run_root / "logs"
    cases = load_json(reports / "scan_manifest.json")["cases"]
    errors: list[str] = []
    results: dict[str, dict[str, Any]] = {}
    audits: dict[str, list[dict[str, Any]]] = {}

    for case in cases:
        label = case["label"]
        try:
            timing = parse_profile(logs / f"{label}.log")
        except Exception as exc:
            errors.append(f"profile:{label}:{exc}")
            timing = {}
        video = load_json(reports / f"{label}_video_check.json")
        if video.get("status") != "ok" or len(video.get("videos", [])) != 4:
            errors.append(f"video_check:{label}")
        for item in video.get("videos", []):
            if item.get("decoded_frame_count") != 501:
                errors.append(f"decoded_frames:{label}:{item.get('decoded_frame_count')}")
            if item.get("fps") is None or not math.isclose(float(item["fps"]), 16.0, abs_tol=1e-3):
                errors.append(f"fps:{label}:{item.get('fps')}")
            if item.get("decoded_frame_shape") != [480, 832, 3]:
                errors.append(f"shape:{label}:{item.get('decoded_frame_shape')}")
        audit = load_jsonl(reports / f"{label}_audit_hashes.jsonl")
        audits[label] = audit
        if len(audit) != 4 or [row.get("prompt_idx") for row in audit] != [0, 1, 2, 3]:
            errors.append(f"audit_rows:{label}")
        for row in audit:
            if row.get("seed") != 0 or row.get("num_output_frames") != 126:
                errors.append(f"audit_protocol:{label}:{row.get('prompt_idx')}")
            if row.get("initial_noise_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"noise_shape:{label}:{row.get('prompt_idx')}")
            if row.get("final_latent_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"latent_shape:{label}:{row.get('prompt_idx')}")

        summary = load_json(reports / f"{label}_runtime_summary.json")
        totals = summary.get("operator_totals", {})
        full = int(totals.get("full_window_tokens", -1))
        reuse = int(totals.get("reuse_tokens", -1))
        reuse_layers = int(totals.get("reuse_block_events", -1))
        if totals.get("main_forwards") != 184 or totals.get("layer_calls") != 5520:
            errors.append(f"forward_counts:{label}:{totals}")
        if int(totals.get("k_tokens", -2)) != full or int(totals.get("v_tokens", -2)) != full:
            errors.append(f"kv_not_full:{label}")
        if int(totals.get("q_tokens", -2)) + reuse != full:
            errors.append(f"q_reuse_conservation:{label}")
        if reuse_layers < 0 or reuse_layers % 30:
            errors.append(f"reuse_layer_multiple:{label}:{reuse_layers}")
        state = summary.get("state", {})
        if any(int(state.get(key, -1)) != 0 for key in (
                "active_block_count", "residual_count", "residual_bytes")):
            errors.append(f"terminal_state:{label}:{state}")
        results[label] = {
            "kind": case["kind"],
            "schedule": case.get("schedule"),
            "threshold": case["threshold"],
            "timing": timing,
            "operator": {
                "policy": summary.get("policy"),
                "schedule": summary.get("schedule"),
                "base_threshold": summary.get("base_threshold"),
                "reuse_decisions": reuse_layers // 30 if reuse_layers >= 0 else None,
                "partial_operator_token_saving_fraction": reuse / full if full > 0 else None,
                "peak_residual_bytes": summary.get("peak_residual_bytes"),
                "operator_totals": totals,
            },
        }

    for prompt_idx in range(4):
        for field in (
            "initial_noise_sha256", "cpu_rng_after_noise_sha256",
            "cuda_rng_after_noise_sha256", "cpu_rng_after_inference_sha256",
            "cuda_rng_after_inference_sha256",
        ):
            values = {rows[prompt_idx].get(field) for rows in audits.values() if len(rows) == 4}
            if len(values) != 1:
                errors.append(f"alignment:{field}:prompt={prompt_idx}:unique={len(values)}")

    slow_ref = results.get("fixed_slow", {}).get("timing", {}).get("mean_diffusion_ms")
    fast_ref = results.get("fixed_fast", {}).get("timing", {}).get("mean_diffusion_ms")
    candidates = []
    for case in cases:
        label = case["label"]
        if label in {"fixed_slow", "fixed_fast"}:
            target, target_name = (slow_ref, "slow") if label == "fixed_slow" else (fast_ref, "fast")
        elif case["kind"] == "front":
            target, target_name = slow_ref, "slow"
        else:
            target, target_name = fast_ref, "fast"
        mean = results[label]["timing"].get("mean_diffusion_ms")
        mismatch = relative_error(target, mean)
        candidates.append({
            "label": label,
            "kind": case["kind"],
            "schedule": case.get("schedule"),
            "threshold": case["threshold"],
            "target": target_name,
            "mean_diffusion_ms": mean,
            "stdev_diffusion_ms": results[label]["timing"].get("stdev_diffusion_ms"),
            "cv": results[label]["timing"].get("cv"),
            "relative_latency_error_vs_target": mismatch,
            "within_two_percent": mismatch is not None and abs(mismatch) <= 0.02,
            **results[label]["operator"],
        })
    candidates.sort(key=lambda item: (item["target"], abs(item["relative_latency_error_vs_target"] or float("inf"))))
    payload = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "name": "R6C 126-latent direct latency-match refinement",
            "seed": 0,
            "latent_frames": 126,
            "decoded_frames": 501,
            "measured_prompt_indices": [1, 2, 3],
            "quality_metrics_read": False,
            "automatic_configuration_freeze": False,
        },
        "references": {"fixed_slow_ms": slow_ref, "fixed_fast_ms": fast_ref},
        "candidates": candidates,
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = [
        "label", "kind", "schedule", "threshold", "target", "mean_diffusion_ms",
        "stdev_diffusion_ms", "cv", "relative_latency_error_vs_target", "within_two_percent",
        "reuse_decisions", "partial_operator_token_saving_fraction", "peak_residual_bytes",
    ]
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps({
        "status": payload["status"], "errors": errors,
        "references": payload["references"], "candidates": candidates,
    }, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
