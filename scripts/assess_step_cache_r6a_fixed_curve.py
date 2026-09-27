#!/usr/bin/env python3
"""Assess the 126-latent, quality-blind RF+Fixed cost-curve scan."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()

    reports = args.run_root / "reports"
    logs = args.run_root / "logs"
    manifest = load_json(reports / "scan_manifest.json")
    cases = manifest["cases"]
    labels = [item["label"] for item in cases]
    errors: list[str] = []
    results: dict[str, dict[str, Any]] = {}

    audit_by_label: dict[str, list[dict[str, Any]]] = {}
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
        audit_by_label[label] = audit
        if len(audit) != 4 or [row.get("prompt_idx") for row in audit] != [0, 1, 2, 3]:
            errors.append(f"audit_rows:{label}")
        for row in audit:
            if row.get("seed") != 0 or row.get("num_output_frames") != 126:
                errors.append(f"audit_protocol:{label}:{row.get('prompt_idx')}")
            if row.get("initial_noise_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"noise_shape:{label}:{row.get('prompt_idx')}")
            if row.get("final_latent_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"latent_shape:{label}:{row.get('prompt_idx')}")

        operator = None
        if case["kind"] == "fixed":
            summary = load_json(reports / f"{label}_runtime_summary.json")
            totals = summary.get("operator_totals", {})
            full = int(totals.get("full_window_tokens", -1))
            reuse_tokens = int(totals.get("reuse_tokens", -1))
            if totals.get("main_forwards") != 184 or totals.get("layer_calls") != 5520:
                errors.append(f"forward_counts:{label}:{totals}")
            if int(totals.get("k_tokens", -2)) != full or int(totals.get("v_tokens", -2)) != full:
                errors.append(f"kv_not_full:{label}")
            if int(totals.get("q_tokens", -2)) + reuse_tokens != full:
                errors.append(f"q_reuse_conservation:{label}")
            reuse_block_events = int(totals.get("reuse_block_events", -1))
            if reuse_block_events < 0 or reuse_block_events % 30:
                errors.append(f"reuse_layer_multiple:{label}:{reuse_block_events}")
            state = summary.get("state", {})
            if any(int(state.get(key, -1)) != 0 for key in (
                    "active_block_count", "residual_count", "residual_bytes")):
                errors.append(f"terminal_state:{label}:{state}")
            operator = {
                "base_threshold": summary.get("base_threshold"),
                "reuse_decisions": reuse_block_events // 30 if reuse_block_events >= 0 else None,
                "reuse_tokens": reuse_tokens,
                "full_window_tokens": full,
                "partial_operator_token_saving_fraction": (
                    reuse_tokens / full if full > 0 else None),
                "peak_residual_bytes": summary.get("peak_residual_bytes"),
                "operator_totals": totals,
            }
        results[label] = {
            "kind": case["kind"],
            "threshold": case.get("threshold"),
            "timing": timing,
            "operator": operator,
        }

    # Every method is a new process with the same seed and prompt order.  The
    # initial noise and post-inference RNG states must therefore match exactly.
    for prompt_idx in range(4):
        for field in (
            "initial_noise_sha256",
            "cpu_rng_after_noise_sha256",
            "cuda_rng_after_noise_sha256",
            "cpu_rng_after_inference_sha256",
            "cuda_rng_after_inference_sha256",
        ):
            values = {
                rows[prompt_idx].get(field)
                for rows in audit_by_label.values()
                if len(rows) == 4
            }
            if len(values) != 1:
                errors.append(f"alignment:{field}:prompt={prompt_idx}:unique={len(values)}")

    baseline_labels = [item["label"] for item in cases if item["kind"] == "baseline"]
    baseline_means = [
        results[label]["timing"].get("mean_diffusion_ms")
        for label in baseline_labels
        if results[label]["timing"].get("mean_diffusion_ms") is not None
    ]
    baseline_reference = statistics.fmean(baseline_means) if baseline_means else None
    baseline_drift = None
    if len(baseline_means) == 2:
        baseline_drift = abs(baseline_means[1] - baseline_means[0]) / statistics.fmean(baseline_means)

    curve = []
    for case in cases:
        if case["kind"] != "fixed":
            continue
        item = results[case["label"]]
        mean = item["timing"].get("mean_diffusion_ms")
        speedup = baseline_reference / mean - 1.0 if baseline_reference and mean else None
        curve.append({
            "label": case["label"],
            "threshold": case["threshold"],
            "mean_diffusion_ms": mean,
            "stdev_diffusion_ms": item["timing"].get("stdev_diffusion_ms"),
            "cv": item["timing"].get("cv"),
            "speedup_fraction_vs_baseline_reference": speedup,
            **(item["operator"] or {}),
        })
    curve.sort(key=lambda item: item["threshold"])

    payload = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "name": "R6A 126-latent RF+Fixed quality-blind cost curve",
            "seed": 0,
            "latent_frames": 126,
            "decoded_frames": 501,
            "fps": 16,
            "warmup_prompt_index": 0,
            "measured_prompt_indices": [1, 2, 3],
            "quality_metrics_read": False,
            "automatic_slow_fast_selection": False,
        },
        "baseline": {
            "labels": baseline_labels,
            "mean_diffusion_ms_reference": baseline_reference,
            "start_end_relative_drift": baseline_drift,
            "runs": {label: results[label]["timing"] for label in baseline_labels},
        },
        "fixed_curve": curve,
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "label", "threshold", "mean_diffusion_ms", "stdev_diffusion_ms", "cv",
        "speedup_fraction_vs_baseline_reference", "reuse_decisions",
        "partial_operator_token_saving_fraction", "peak_residual_bytes",
    ]
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps({
        "status": payload["status"],
        "errors": errors,
        "baseline_drift": baseline_drift,
        "fixed_curve": curve,
    }, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
