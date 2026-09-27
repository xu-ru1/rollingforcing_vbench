#!/usr/bin/env python3
"""Assess frozen R6 generalization and paired diffusion timing."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import torch


METHODS = ("baseline", "fixed", "front")
MEASURED_PROMPTS = (1, 2, 3)
DIFFUSION_RE = re.compile(r"Diffusion generation time:\s*([0-9.]+)\s*ms")
ALLOCATED_RE = re.compile(r"Peak CUDA allocated:\s*([0-9.]+)\s*GB")
RESERVED_RE = re.compile(r"Peak CUDA reserved:\s*([0-9.]+)\s*GB")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | None]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"latent mismatch: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}")
    a, b = left.float(), right.float()
    delta = (b - a).abs()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("non-finite latent")
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


def parse_profile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    diffusion = [float(value) for value in DIFFUSION_RE.findall(text)]
    allocated = [float(value) for value in ALLOCATED_RE.findall(text)]
    reserved = [float(value) for value in RESERVED_RE.findall(text)]
    if len(diffusion) != 4:
        raise ValueError(f"expected four diffusion timings in {path}, found {len(diffusion)}")
    measured = diffusion[1:]
    return {
        "warmup_diffusion_ms": diffusion[0],
        "measured_diffusion_ms": measured,
        "mean_diffusion_ms": statistics.fmean(measured),
        "median_diffusion_ms": statistics.median(measured),
        "stdev_diffusion_ms": statistics.stdev(measured),
        "peak_cuda_allocated_gb": max(allocated) if allocated else None,
        "peak_cuda_reserved_gb": max(reserved) if reserved else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = args.run_root / "reports"
    tensors = args.run_root / "tensors"
    logs = args.run_root / "logs"
    errors: list[str] = []

    profiles = {}
    for method in METHODS:
        try:
            profiles[method] = parse_profile(logs / f"{method}.log")
        except Exception as exc:
            errors.append(f"profile:{method}:{exc}")
        video = load_json(reports / f"{method}_video_check.json")
        if video.get("status") != "ok" or len(video.get("videos", [])) != 4:
            errors.append(f"video_check:{method}")

    noise_hashes: dict[str, dict[int, str]] = {method: {} for method in METHODS}
    latent_data: dict[str, dict[int, torch.Tensor]] = {method: {} for method in METHODS}
    for method in METHODS:
        for prompt_index in range(4):
            noise = load_tensor(tensors / method / f"{prompt_index}_initial_noise.pt")
            latent = load_tensor(tensors / method / f"{prompt_index}_latents.pt")
            noise_hashes[method][prompt_index] = tensor_hash(noise)
            latent_data[method][prompt_index] = latent
            if list(latent.shape) != [1, 81, 16, 60, 104]:
                errors.append(f"latent_shape:{method}:{prompt_index}:{list(latent.shape)}")
    for prompt_index in range(4):
        hashes = {noise_hashes[method][prompt_index] for method in METHODS}
        if len(hashes) != 1:
            errors.append(f"initial_noise_mismatch:{prompt_index}")

    candidate_reports = {}
    for method in ("fixed", "front"):
        rows = load_jsonl(reports / f"{method}_decisions.jsonl")
        summary = load_json(reports / f"{method}_runtime_summary.json")
        if len(rows) != 540:
            errors.append(f"decision_count:{method}:{len(rows)}")
        prompt_counts = Counter(int(row["sample_id"]) for row in rows)
        if prompt_counts != Counter({0: 135, 1: 135, 2: 135, 3: 135}):
            errors.append(f"sample_decisions:{method}:{dict(prompt_counts)}")
        reuse_rows = [row for row in rows if row.get("execute_reuse")]
        reuse_by_prompt = Counter(int(row["sample_id"]) for row in reuse_rows)
        reuse_by_stage = Counter(int(row["local_stage_index"]) for row in reuse_rows)
        totals = summary.get("operator_totals", {})
        full = int(totals.get("full_window_tokens", -1))
        reuse_tokens = int(totals.get("reuse_tokens", -1))
        if totals.get("main_forwards") != 124 or totals.get("layer_calls") != 3720:
            errors.append(f"forward_counts:{method}:{totals}")
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
        candidate_reports[method] = {
            "base_threshold": summary.get("base_threshold"),
            "reuse_count": len(reuse_rows),
            "reuse_count_by_prompt": {str(key): reuse_by_prompt[key] for key in sorted(reuse_by_prompt)},
            "reuse_count_by_stage": {str(key): reuse_by_stage[key] for key in sorted(reuse_by_stage)},
            "partial_operator_token_saving_fraction": reuse_tokens / full,
            "operator_totals": totals,
            "peak_residual_bytes": summary.get("peak_residual_bytes"),
        }

    latent_metrics = {"fixed": {}, "front": {}}
    for method in ("fixed", "front"):
        for prompt_index in MEASURED_PROMPTS:
            latent_metrics[method][str(prompt_index)] = compare(
                latent_data["baseline"][prompt_index], latent_data[method][prompt_index])
        for field in ("mse", "mae", "relative_l1", "max_abs", "dynamic_range_psnr_db"):
            values = [item[field] for item in latent_metrics[method].values() if item[field] is not None]
            latent_metrics[method][f"mean_{field}"] = statistics.fmean(values) if values else None

    timing = {method: profiles.get(method) for method in METHODS}
    if all(method in profiles for method in METHODS):
        base = profiles["baseline"]["measured_diffusion_ms"]
        for method in ("fixed", "front"):
            current = profiles[method]["measured_diffusion_ms"]
            timing[method]["paired_speedup_fraction"] = [
                (left / right) - 1.0 for left, right in zip(base, current)]
            timing[method]["mean_speedup_from_mean"] = (
                profiles["baseline"]["mean_diffusion_ms"] /
                profiles[method]["mean_diffusion_ms"] - 1.0)

    front_timing = timing.get("front") or {}
    front_speedup = front_timing.get("mean_speedup_from_mean")
    practical_gate = {
        "minimum_required_mean_diffusion_speedup": 0.05,
        "front_mean_diffusion_speedup": front_speedup,
        "pass": front_speedup is not None and front_speedup >= 0.05,
    }
    front_quality_better = (
        latent_metrics["front"]["mean_mse"] <= latent_metrics["fixed"]["mean_mse"])
    payload = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "name": "R6 frozen 81-latent generalization and timing",
            "seed": 1,
            "prompt_0_role": "warmup_excluded_from_timing_and_quality",
            "measured_prompt_indices": list(MEASURED_PROMPTS),
            "decoded_frames_per_video": 321,
        },
        "noise_sha256": noise_hashes,
        "candidate_execution": candidate_reports,
        "timing": timing,
        "latent_metrics_vs_baseline": latent_metrics,
        "front_vs_fixed_generalization": {
            "front_mean_mse_no_worse": front_quality_better,
            "fixed_mean_mse": latent_metrics["fixed"]["mean_mse"],
            "front_mean_mse": latent_metrics["front"]["mean_mse"],
            "front_mse_reduction_fraction": (
                1.0 - latent_metrics["front"]["mean_mse"] /
                latent_metrics["fixed"]["mean_mse"]),
        },
        "practical_speed_gate": practical_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "errors": errors,
        "front_speed_gate": practical_gate,
        "front_vs_fixed": payload["front_vs_fixed_generalization"],
    }, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
