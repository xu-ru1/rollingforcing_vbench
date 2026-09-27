#!/usr/bin/env python3
"""Assess the compact R5A online policy screen against the R4 baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


METHODS = ("fixed", "front", "u_shape")
PROMPTS = ("cat", "rainy_car", "robot")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | None]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"latent mismatch: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}")
    a, b = left.float(), right.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("non-finite latent")
    delta = (b - a).abs()
    mse = float(delta.square().mean().item())
    dynamic_range = float((a.max() - a.min()).item())
    return {
        "mse": mse,
        "mae": float(delta.mean().item()),
        "relative_l1": float(delta.mean().div(a.abs().mean().clamp_min(1e-6)).item()),
        "max_abs": float(delta.max().item()),
        "dynamic_range_psnr_db": None if mse == 0 else (
            20.0 * math.log10(max(dynamic_range, 1e-6) / math.sqrt(mse))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--r4-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()
    reports = args.run_root / "reports"
    errors: list[str] = []

    r4_assessment = load_json(args.r4_root / "reports" / "r4_position_assessment.json")
    if r4_assessment.get("status") != "ok":
        errors.append("r4_baseline_assessment_not_ok")
    selection = load_json(reports / "threshold_selection.json")
    observer_rows = load_jsonl(reports / "observer_decisions.jsonl")
    if len(observer_rows) != 225 or any(row.get("execute_reuse") for row in observer_rows):
        errors.append("observer_trace_not_dense_225")
    observer_summary = load_json(reports / "observer_runtime_summary.json")
    observer_video = load_json(reports / "observer_video_check.json")
    observer_totals = observer_summary.get("operator_totals", {})
    observer_full = int(observer_totals.get("full_window_tokens", -1))
    if observer_video.get("status") != "ok" or len(observer_video.get("videos", [])) != 3:
        errors.append("observer_video_check")
    if observer_totals.get("main_forwards") != 57 or observer_totals.get("layer_calls") != 1710:
        errors.append("observer_forward_counts")
    if any(int(observer_totals.get(field, -2)) != observer_full for field in (
            "k_tokens", "v_tokens", "q_tokens", "attention_output_tokens",
            "cross_attention_tokens", "mlp_tokens")):
        errors.append("observer_not_full_compute")

    baseline_noise = {}
    baseline_latents = {}
    for prompt_index in range(3):
        noise = torch.load(
            args.r4_root / "tensors" / "baseline" / f"{prompt_index}_initial_noise.pt",
            map_location="cpu")
        latent = torch.load(
            args.r4_root / "tensors" / "baseline" / f"{prompt_index}_latents.pt",
            map_location="cpu")
        baseline_noise[prompt_index] = tensor_hash(noise)
        baseline_latents[prompt_index] = latent

    method_reports: dict[str, Any] = {}
    csv_rows = []
    actual_reuses = []
    for method in METHODS:
        rows = load_jsonl(reports / f"{method}_decisions.jsonl")
        summary = load_json(reports / f"{method}_runtime_summary.json")
        video = load_json(reports / f"{method}_video_check.json")
        selected = selection["selected"][method]
        if len(rows) != 225:
            errors.append(f"decision_count:{method}:{len(rows)}")
        if video.get("status") != "ok" or len(video.get("videos", [])) != 3:
            errors.append(f"video_check:{method}")
        reuse_rows = [row for row in rows if row.get("execute_reuse")]
        reuse_by_stage = Counter(int(row["local_stage_index"]) for row in reuse_rows)
        reuse_by_prompt = Counter(int(row["sample_id"]) for row in reuse_rows)
        if any(stage not in (1, 2, 3) for stage in reuse_by_stage):
            errors.append(f"protected_stage_reuse:{method}:{dict(reuse_by_stage)}")
        totals = summary.get("operator_totals", {})
        if summary.get("policy") != selected["policy"] or summary.get("schedule") != selected["schedule"]:
            errors.append(f"policy_schedule:{method}")
        if abs(float(summary.get("base_threshold", -1.0)) - float(selected["base_threshold"])) > 1e-12:
            errors.append(f"base_threshold:{method}")
        if totals.get("main_forwards") != 57 or totals.get("layer_calls") != 1710:
            errors.append(f"forward_counts:{method}")
        full = int(totals.get("full_window_tokens", -1))
        reuse_tokens = int(totals.get("reuse_tokens", -1))
        if totals.get("k_tokens") != full or totals.get("v_tokens") != full:
            errors.append(f"kv_not_full:{method}")
        if int(totals.get("q_tokens", -2)) + reuse_tokens != full:
            errors.append(f"q_reuse_conservation:{method}")
        if int(totals.get("reuse_block_events", -1)) != len(reuse_rows) * 30:
            errors.append(f"layer_reuse_count:{method}")
        state = summary.get("state", {})
        if any(int(state.get(key, -1)) != 0 for key in (
                "active_block_count", "residual_count", "residual_bytes")):
            errors.append(f"terminal_state:{method}:{state}")
        actual_reuses.append(len(reuse_rows))

        metrics = {}
        for prompt_index, prompt_name in enumerate(PROMPTS):
            noise = torch.load(
                args.run_root / "tensors" / method / f"{prompt_index}_initial_noise.pt",
                map_location="cpu")
            latent = torch.load(
                args.run_root / "tensors" / method / f"{prompt_index}_latents.pt",
                map_location="cpu")
            if tensor_hash(noise) != baseline_noise[prompt_index]:
                errors.append(f"initial_noise_mismatch:{method}:{prompt_index}")
            if list(latent.shape) != [1, 45, 16, 60, 104]:
                errors.append(f"latent_shape:{method}:{prompt_index}:{list(latent.shape)}")
            values = compare(baseline_latents[prompt_index], latent)
            metrics[prompt_name] = values
            csv_rows.append({"method": method, "prompt": prompt_name, **values})
        mean_metrics = {}
        for field in ("mse", "mae", "relative_l1", "max_abs", "dynamic_range_psnr_db"):
            values = [float(item[field]) for item in metrics.values() if item[field] is not None]
            mean_metrics[field] = sum(values) / len(values) if values else None
        method_reports[method] = {
            "base_threshold": selected["base_threshold"],
            "predicted_reuse_count": selected["predicted"]["reuse_count"],
            "actual_reuse_count": len(reuse_rows),
            "actual_reuse_count_by_stage": {
                str(key): reuse_by_stage[key] for key in sorted(reuse_by_stage)},
            "actual_reuse_count_by_prompt": {
                str(key): reuse_by_prompt[key] for key in sorted(reuse_by_prompt)},
            "partial_operator_token_saving_fraction": reuse_tokens / full,
            "operator_totals": totals,
            "peak_residual_bytes": summary.get("peak_residual_bytes"),
            "paired_latent_metrics": metrics,
            "mean_across_prompts": mean_metrics,
        }

    target = int(selection["target_reuse_decisions"])
    max_target_error = max(abs(count - target) for count in actual_reuses)
    spread = max(actual_reuses) - min(actual_reuses)
    payload = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "name": "R5A compact quality-blind online calibration",
            "latent_frames": 45,
            "prompts": list(PROMPTS),
            "seed": 0,
            "r4_baseline_reused": str(args.r4_root),
            "target_reuse_decisions": target,
        },
        "selection": selection,
        "methods": method_reports,
        "actual_budget_match": {
            "reuse_counts": dict(zip(METHODS, actual_reuses)),
            "max_absolute_target_error": max_target_error,
            "max_pairwise_reuse_count_spread": spread,
            "within_two_decisions": spread <= 2 and max_target_error <= 2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        fields = ("method", "prompt", "mse", "mae", "relative_l1", "max_abs",
                  "dynamic_range_psnr_db")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({
        "status": payload["status"],
        "errors": errors,
        "reuse_counts": payload["actual_budget_match"]["reuse_counts"],
        "within_two_decisions": payload["actual_budget_match"]["within_two_decisions"],
    }, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
