#!/usr/bin/env python3
"""Assess repeated, quality-blind R6D latency measurements."""

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
PROMPT_REPEATS = {"cat": [1, 4, 7], "rainy_car": [2, 5, 8], "robot": [3, 6, 9]}
LATENCY_PAIRS = {
    "front_slow": "fixed_slow",
    "front_fast": "fixed_fast",
    "u_shape_fast": "fixed_fast",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


def relative_error(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference <= 0:
        return None
    return (candidate - reference) / reference


def parse_timings(path: Path) -> list[float]:
    values = [float(value) for value in DIFFUSION_RE.findall(path.read_text(encoding="utf-8", errors="replace"))]
    if len(values) != 10:
        raise ValueError(f"expected ten diffusion timings (warmup + 9), found {len(values)}")
    return values


def assert_video_report(path: Path, label: str, errors: list[str]) -> None:
    report = load_json(path)
    videos = report.get("videos", [])
    if report.get("status") != "ok" or len(videos) != 10:
        errors.append(f"video_check:{label}")
    for video in videos:
        if video.get("decoded_frame_count") != 501:
            errors.append(f"decoded_frames:{label}:{video.get('decoded_frame_count')}")
        if video.get("decoded_frame_shape") != [480, 832, 3]:
            errors.append(f"shape:{label}:{video.get('decoded_frame_shape')}")
        fps = video.get("fps")
        if fps is None or not math.isclose(float(fps), 16.0, abs_tol=1e-3):
            errors.append(f"fps:{label}:{fps}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    reports, logs = args.run_root / "reports", args.run_root / "logs"
    manifest = load_json(reports / "scan_manifest.json")
    cases = manifest["cases"]
    errors: list[str] = []
    audits: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {}

    for case in cases:
        label = case["label"]
        try:
            timings = parse_timings(logs / f"{label}.log")
        except Exception as exc:
            errors.append(f"profile:{label}:{exc}")
            timings = []
        assert_video_report(reports / f"{label}_video_check.json", label, errors)
        audit = load_jsonl(reports / f"{label}_audit_hashes.jsonl")
        audits[label] = audit
        if len(audit) != 10 or [row.get("prompt_idx") for row in audit] != list(range(10)):
            errors.append(f"audit_rows:{label}")
        for row in audit:
            if row.get("seed") != 0 or row.get("num_output_frames") != 126:
                errors.append(f"audit_protocol:{label}:{row.get('prompt_idx')}")
            if row.get("initial_noise_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"noise_shape:{label}:{row.get('prompt_idx')}")
            if row.get("final_latent_shape") != [1, 126, 16, 60, 104]:
                errors.append(f"latent_shape:{label}:{row.get('prompt_idx')}")

        by_prompt: dict[str, dict[str, Any]] = {}
        for prompt_name, indices in PROMPT_REPEATS.items():
            values = [timings[index] for index in indices] if len(timings) == 10 else []
            by_prompt[prompt_name] = {
                "indices": indices,
                "diffusion_ms": values,
                "mean_diffusion_ms": statistics.fmean(values) if values else None,
                "median_diffusion_ms": statistics.median(values) if values else None,
                "repeat_cv": cv(values),
            }
        measured = timings[1:] if len(timings) == 10 else []
        results[label] = {
            "enabled": bool(case["enabled"]), "policy": case["policy"],
            "schedule": case["schedule"], "threshold": case["threshold"],
            "warmup_diffusion_ms": timings[0] if timings else None,
            "measured_diffusion_ms": measured,
            "prompt_balanced_mean_diffusion_ms": statistics.fmean(item["mean_diffusion_ms"] for item in by_prompt.values()) if measured else None,
            "all_repeat_cv": cv(measured), "by_prompt": by_prompt,
        }

        if case["enabled"]:
            summary = load_json(reports / f"{label}_runtime_summary.json")
            totals = summary.get("operator_totals", {})
            full = int(totals.get("full_window_tokens", -1))
            reuse = int(totals.get("reuse_tokens", -1))
            reuse_layers = int(totals.get("reuse_block_events", -1))
            if totals.get("main_forwards") != 460 or totals.get("layer_calls") != 13800:
                errors.append(f"forward_counts:{label}:{totals}")
            if int(totals.get("k_tokens", -2)) != full or int(totals.get("v_tokens", -2)) != full:
                errors.append(f"kv_not_full:{label}")
            if int(totals.get("q_tokens", -2)) + reuse != full:
                errors.append(f"q_reuse_conservation:{label}")
            if reuse_layers < 0 or reuse_layers % 30:
                errors.append(f"reuse_layer_multiple:{label}:{reuse_layers}")
            state = summary.get("state", {})
            if any(int(state.get(key, -1)) != 0 for key in ("active_block_count", "residual_count", "residual_bytes")):
                errors.append(f"terminal_state:{label}:{state}")
            results[label]["operator"] = {
                "reuse_decisions": reuse_layers // 30 if reuse_layers >= 0 else None,
                "partial_operator_token_saving_fraction": reuse / full if full > 0 else None,
                "peak_residual_bytes": summary.get("peak_residual_bytes"),
                "operator_totals": totals,
            }

    rng_fields = ("initial_noise_sha256", "cpu_rng_after_noise_sha256", "cuda_rng_after_noise_sha256")
    for index in range(10):
        for field in rng_fields:
            values = {rows[index].get(field) for rows in audits.values() if len(rows) == 10}
            if len(values) != 1:
                errors.append(f"cross_case_alignment:{field}:prompt={index}:unique={len(values)}")
    for label, rows in audits.items():
        if len(rows) != 10:
            continue
        for prompt_name, indices in PROMPT_REPEATS.items():
            for field in (*rng_fields, "final_latent_sha256", "cpu_rng_after_inference_sha256", "cuda_rng_after_inference_sha256"):
                if len({rows[index].get(field) for index in indices}) != 1:
                    errors.append(f"repeat_determinism:{label}:{prompt_name}:{field}")

    matching: dict[str, Any] = {}
    for candidate, reference in LATENCY_PAIRS.items():
        candidate_mean = results[candidate]["prompt_balanced_mean_diffusion_ms"]
        reference_mean = results[reference]["prompt_balanced_mean_diffusion_ms"]
        error = relative_error(reference_mean, candidate_mean)
        prompt_errors = {
            name: relative_error(results[reference]["by_prompt"][name]["mean_diffusion_ms"], results[candidate]["by_prompt"][name]["mean_diffusion_ms"])
            for name in PROMPT_REPEATS
        }
        candidate_cvs = [results[candidate]["by_prompt"][name]["repeat_cv"] for name in PROMPT_REPEATS]
        reference_cvs = [results[reference]["by_prompt"][name]["repeat_cv"] for name in PROMPT_REPEATS]
        max_repeat_cv = max(value if value is not None else float("inf") for value in candidate_cvs)
        max_reference_cv = max(value if value is not None else float("inf") for value in reference_cvs)
        matching[candidate] = {
            "reference": reference,
            "prompt_balanced_relative_latency_error": error,
            "prompt_relative_latency_errors": prompt_errors,
            "mean_within_two_percent": error is not None and abs(error) <= 0.02,
            "candidate_max_repeat_cv": max_repeat_cv,
            "reference_max_repeat_cv": max_reference_cv,
            "repeat_stable_at_two_percent": max(max_repeat_cv, max_reference_cv) <= 0.02,
        }

    rows = []
    for label, result in results.items():
        item = {
            "label": label, "enabled": result["enabled"], "policy": result["policy"],
            "schedule": result["schedule"], "threshold": result["threshold"],
            "prompt_balanced_mean_diffusion_ms": result["prompt_balanced_mean_diffusion_ms"],
            "all_repeat_cv": result["all_repeat_cv"],
        }
        for name in PROMPT_REPEATS:
            item[f"{name}_mean_ms"] = result["by_prompt"][name]["mean_diffusion_ms"]
            item[f"{name}_repeat_cv"] = result["by_prompt"][name]["repeat_cv"]
        if label in matching:
            item.update(matching[label])
        if "operator" in result:
            item.update(result["operator"])
        rows.append(item)
    payload = {
        "status": "ok" if not errors else "error", "errors": errors,
        "protocol": {
            "name": "R6D repeated final timing confirmation", "seed": 0,
            "latent_frames": 126, "decoded_frames": 501, "warmup_prompt_index": 0,
            "prompt_repeats": PROMPT_REPEATS, "quality_metrics_read": False,
            "automatic_configuration_freeze": False,
        },
        "matching": matching, "results": results, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = ["label", "enabled", "policy", "schedule", "threshold", "prompt_balanced_mean_diffusion_ms", "all_repeat_cv", "cat_mean_ms", "cat_repeat_cv", "rainy_car_mean_ms", "rainy_car_repeat_cv", "robot_mean_ms", "robot_repeat_cv", "reference", "prompt_balanced_relative_latency_error", "mean_within_two_percent", "candidate_max_repeat_cv", "reference_max_repeat_cv", "repeat_stable_at_two_percent", "reuse_decisions", "partial_operator_token_saving_fraction", "peak_residual_bytes"]
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": payload["status"], "errors": errors, "matching": matching}, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
