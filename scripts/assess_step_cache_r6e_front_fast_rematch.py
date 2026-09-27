#!/usr/bin/env python3
"""Assess R6E Front-fast rematching with repeated fixed-seed workloads."""

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
REPEATS = {"cat": [1, 4, 7], "rainy_car": [2, 5, 8], "robot": [3, 6, 9]}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def coefficient_of_variation(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.fmean(values)


def relative_error(reference: float, candidate: float) -> float:
    return (candidate - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    reports, logs = args.run_root / "reports", args.run_root / "logs"
    cases = read_json(reports / "scan_manifest.json")["cases"]
    errors: list[str] = []
    audits: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {}

    for case in cases:
        label = case["label"]
        times = [float(value) for value in DIFFUSION_RE.findall((logs / f"{label}.log").read_text(encoding="utf-8", errors="replace"))]
        if len(times) != 10:
            errors.append(f"profile_rows:{label}:{len(times)}")
            times = [float("nan")] * 10
        video = read_json(reports / f"{label}_video_check.json")
        if video.get("status") != "ok" or len(video.get("videos", [])) != 10:
            errors.append(f"video_check:{label}")
        for item in video.get("videos", []):
            if item.get("decoded_frame_count") != 501 or item.get("decoded_frame_shape") != [480, 832, 3]:
                errors.append(f"video_geometry:{label}")
            fps = item.get("fps")
            if fps is None or not math.isclose(float(fps), 16.0, abs_tol=1e-3):
                errors.append(f"video_fps:{label}:{fps}")

        audit = read_jsonl(reports / f"{label}_audit_hashes.jsonl")
        audits[label] = audit
        if len(audit) != 10 or [row.get("prompt_idx") for row in audit] != list(range(10)):
            errors.append(f"audit_rows:{label}")
        for row in audit:
            if row.get("seed") != 0 or row.get("initial_noise_shape") != [1, 126, 16, 60, 104] or row.get("final_latent_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"audit_protocol:{label}:{row.get('prompt_idx')}")

        summary = read_json(reports / f"{label}_runtime_summary.json")
        totals = summary.get("operator_totals", {})
        full = int(totals.get("full_window_tokens", -1))
        reuse = int(totals.get("reuse_tokens", -1))
        reuse_layers = int(totals.get("reuse_block_events", -1))
        if totals.get("main_forwards") != 460 or totals.get("layer_calls") != 13800:
            errors.append(f"forward_counts:{label}")
        if int(totals.get("k_tokens", -2)) != full or int(totals.get("v_tokens", -2)) != full:
            errors.append(f"kv_not_full:{label}")
        if int(totals.get("q_tokens", -2)) + reuse != full:
            errors.append(f"q_reuse_conservation:{label}")
        if reuse_layers < 0 or reuse_layers % 30:
            errors.append(f"reuse_layers:{label}")
        state = summary.get("state", {})
        if any(int(state.get(key, -1)) != 0 for key in ("active_block_count", "residual_count", "residual_bytes")):
            errors.append(f"terminal_state:{label}")

        by_prompt = {}
        for name, indices in REPEATS.items():
            values = [times[index] for index in indices]
            by_prompt[name] = {"diffusion_ms": values, "mean_diffusion_ms": statistics.fmean(values), "repeat_cv": coefficient_of_variation(values)}
        results[label] = {
            "policy": case["policy"], "schedule": case["schedule"], "threshold": case["threshold"],
            "warmup_diffusion_ms": times[0], "by_prompt": by_prompt,
            "prompt_balanced_mean_diffusion_ms": statistics.fmean(item["mean_diffusion_ms"] for item in by_prompt.values()),
            "all_repeat_cv": coefficient_of_variation(times[1:]),
            "reuse_decisions": reuse_layers // 30,
            "partial_operator_token_saving_fraction": reuse / full if full > 0 else None,
            "peak_residual_bytes": summary.get("peak_residual_bytes"),
            "operator_totals": totals,
        }

    for index in range(10):
        for field in ("initial_noise_sha256", "cpu_rng_after_noise_sha256", "cuda_rng_after_noise_sha256"):
            if len({records[index].get(field) for records in audits.values() if len(records) == 10}) != 1:
                errors.append(f"cross_case_rng_alignment:{field}:{index}")
    for label, records in audits.items():
        if len(records) != 10:
            continue
        for name, indices in REPEATS.items():
            for field in ("initial_noise_sha256", "final_latent_sha256", "cpu_rng_after_inference_sha256", "cuda_rng_after_inference_sha256"):
                if len({records[index].get(field) for index in indices}) != 1:
                    errors.append(f"repeat_determinism:{label}:{name}:{field}")

    reference = results.get("fixed_fast")
    if reference is None:
        errors.append("missing_fixed_fast")
        reference = {"prompt_balanced_mean_diffusion_ms": float("nan"), "by_prompt": {}}
    candidates = []
    for label, result in results.items():
        if label == "fixed_fast":
            continue
        prompt_errors = {name: relative_error(reference["by_prompt"][name]["mean_diffusion_ms"], result["by_prompt"][name]["mean_diffusion_ms"]) for name in REPEATS}
        mean_error = relative_error(reference["prompt_balanced_mean_diffusion_ms"], result["prompt_balanced_mean_diffusion_ms"])
        candidate = {
            "label": label, "threshold": result["threshold"],
            "prompt_balanced_mean_diffusion_ms": result["prompt_balanced_mean_diffusion_ms"],
            "relative_latency_error_vs_fixed_fast": mean_error,
            "prompt_relative_latency_errors": prompt_errors,
            "within_two_percent": abs(mean_error) <= 0.02,
            "candidate_max_repeat_cv": max(result["by_prompt"][name]["repeat_cv"] for name in REPEATS),
            "fixed_fast_max_repeat_cv": max(reference["by_prompt"][name]["repeat_cv"] for name in REPEATS),
            "reuse_decisions": result["reuse_decisions"],
            "partial_operator_token_saving_fraction": result["partial_operator_token_saving_fraction"],
            "peak_residual_bytes": result["peak_residual_bytes"],
        }
        candidates.append(candidate)
    candidates.sort(key=lambda item: abs(item["relative_latency_error_vs_fixed_fast"]))
    payload = {
        "status": "ok" if not errors else "error", "errors": errors,
        "protocol": {"name": "R6E Front-fast rematch", "seed": 0, "latent_frames": 126, "decoded_frames": 501, "quality_metrics_read": False, "automatic_configuration_freeze": False},
        "fixed_fast": results.get("fixed_fast"), "candidates": candidates, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = ["label", "threshold", "prompt_balanced_mean_diffusion_ms", "relative_latency_error_vs_fixed_fast", "within_two_percent", "candidate_max_repeat_cv", "fixed_fast_max_repeat_cv", "reuse_decisions", "partial_operator_token_saving_fraction", "peak_residual_bytes"]
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps({"status": payload["status"], "errors": errors, "candidates": candidates}, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
