#!/usr/bin/env python3
"""Assess equal-dose stage-position injections against a disabled baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch


LABELS = ("baseline", "stage1", "stage2", "stage3")
PROMPT_NAMES = ("cat", "rainy_car", "robot")
INJECTION_BLOCKS = tuple(range(4, 12))
TOKENS_PER_BLOCK = 3 * 1560


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def compare_latents(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.dtype != candidate.dtype:
        raise ValueError(
            f"latent shape/dtype mismatch: baseline={baseline.shape}/{baseline.dtype} "
            f"candidate={candidate.shape}/{candidate.dtype}")
    left = baseline.float()
    right = candidate.float()
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("non-finite latent tensor")
    difference = (right - left).abs()
    squared = difference.square()
    mse = float(squared.mean().item())
    mae = float(difference.mean().item())
    relative_l1 = float(difference.mean().div(left.abs().mean().clamp_min(1e-6)).item())
    max_abs = float(difference.max().item())
    dynamic_range = float((left.max() - left.min()).item())
    psnr = None if mse == 0.0 else 20.0 * math.log10(max(dynamic_range, 1e-6) / math.sqrt(mse))
    per_block_mse = []
    for start in range(0, left.shape[1], 3):
        per_block_mse.append(float(squared[:, start:start + 3].mean().item()))
    return {
        "mse": mse,
        "mae": mae,
        "relative_l1": relative_l1,
        "max_abs": max_abs,
        "dynamic_range_psnr_db": psnr,
        "per_video_block_mse": per_block_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()
    reports = args.run_root / "reports"
    tensors_root = args.run_root / "tensors"
    errors: list[str] = []

    tensors: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    noise_hashes: dict[str, dict[int, str]] = {}
    latent_hashes: dict[str, dict[int, str]] = {}
    for label in LABELS:
        tensors[label] = {}
        noise_hashes[label] = {}
        latent_hashes[label] = {}
        for prompt_index in range(3):
            noise_path = tensors_root / label / f"{prompt_index}_initial_noise.pt"
            latent_path = tensors_root / label / f"{prompt_index}_latents.pt"
            if not noise_path.is_file() or not latent_path.is_file():
                errors.append(f"missing_tensor:{label}:{prompt_index}")
                continue
            noise, latent = load_tensor(noise_path), load_tensor(latent_path)
            tensors[label][prompt_index] = (noise, latent)
            noise_hashes[label][prompt_index] = tensor_hash(noise)
            latent_hashes[label][prompt_index] = tensor_hash(latent)
            if list(latent.shape) != [1, 45, 16, 60, 104]:
                errors.append(f"latent_shape:{label}:{prompt_index}:{list(latent.shape)}")
        video_path = reports / f"{label}_video_check.json"
        if not video_path.is_file():
            errors.append(f"missing_video_check:{label}")
        else:
            video = load_json(video_path)
            if video.get("status") != "ok" or len(video.get("videos", [])) != 3:
                errors.append(f"video_check_failed:{label}")

    for prompt_index in range(3):
        hashes = {noise_hashes[label].get(prompt_index) for label in LABELS}
        if len(hashes) != 1 or None in hashes:
            errors.append(f"initial_noise_mismatch:prompt={prompt_index}:{sorted(str(value) for value in hashes)}")

    case_reports: dict[str, Any] = {}
    comparable_operator_totals = None
    operator_fields = (
        "full_window_tokens", "k_tokens", "v_tokens", "q_tokens",
        "attention_output_tokens", "cross_attention_tokens", "mlp_tokens",
        "reuse_tokens", "recompute_block_events", "reuse_block_events",
    )
    for stage in (1, 2, 3):
        label = f"stage{stage}"
        rows = load_jsonl(reports / f"{label}_decisions.jsonl")
        summary = load_json(reports / f"{label}_runtime_summary.json")
        if len(rows) != 225:
            errors.append(f"decision_count:{label}:{len(rows)}:225")
        sample_counts = Counter(int(row["sample_id"]) for row in rows)
        if sample_counts != Counter({0: 75, 1: 75, 2: 75}):
            errors.append(f"sample_decision_counts:{label}:{dict(sample_counts)}")
        actual_points = Counter(
            (int(row["sample_id"]), int(row["global_video_block_id"]), int(row["local_stage_index"]))
            for row in rows if row.get("execute_reuse")
        )
        expected_points = Counter(
            (sample_id, block_id, stage)
            for sample_id in range(3) for block_id in INJECTION_BLOCKS
        )
        if actual_points != expected_points:
            errors.append(f"injection_points:{label}:actual={sorted(actual_points)}")
        totals = summary.get("operator_totals", {})
        if totals.get("main_forwards") != 57 or totals.get("layer_calls") != 1710:
            errors.append(f"forward_layer_counts:{label}:{totals}")
        full = totals.get("full_window_tokens")
        if totals.get("k_tokens") != full or totals.get("v_tokens") != full:
            errors.append(f"kv_not_full:{label}")
        if totals.get("q_tokens", 0) + totals.get("reuse_tokens", 0) != full:
            errors.append(f"q_reuse_conservation:{label}")
        if totals.get("reuse_block_events") != 24 * 30:
            errors.append(f"layer_reuse_count:{label}:{totals.get('reuse_block_events')}")
        state = summary.get("state", {})
        if any(int(state.get(key, -1)) != 0 for key in ("active_block_count", "residual_count", "residual_bytes")):
            errors.append(f"terminal_state:{label}:{state}")
        selected_totals = {field: totals.get(field) for field in operator_fields}
        if comparable_operator_totals is None:
            comparable_operator_totals = selected_totals
        elif selected_totals != comparable_operator_totals:
            errors.append(f"unequal_operator_dose:{label}")
        case_reports[label] = {
            "decision_count": len(rows),
            "decision_count_by_sample": dict(sorted(sample_counts.items())),
            "execute_reuse_count": sum(actual_points.values()),
            "injection_blocks": list(INJECTION_BLOCKS),
            "injection_stage": stage,
            "operator_totals": totals,
            "peak_residual_bytes": summary.get("peak_residual_bytes"),
        }

    metrics: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for stage in (1, 2, 3):
        label = f"stage{stage}"
        metrics[label] = {}
        for prompt_index, prompt_name in enumerate(PROMPT_NAMES):
            if prompt_index not in tensors["baseline"] or prompt_index not in tensors[label]:
                continue
            comparison = compare_latents(
                tensors["baseline"][prompt_index][1], tensors[label][prompt_index][1])
            metrics[label][prompt_name] = comparison
            csv_rows.append({
                "stage": stage,
                "prompt_index": prompt_index,
                "prompt_name": prompt_name,
                "mse": comparison["mse"],
                "mae": comparison["mae"],
                "relative_l1": comparison["relative_l1"],
                "max_abs": comparison["max_abs"],
                "dynamic_range_psnr_db": comparison["dynamic_range_psnr_db"],
            })

    aggregates: dict[str, Any] = {}
    for stage in (1, 2, 3):
        label = f"stage{stage}"
        rows = [row for row in csv_rows if row["stage"] == stage]
        aggregates[label] = {}
        if rows:
            for field in ("mse", "mae", "relative_l1", "max_abs", "dynamic_range_psnr_db"):
                values = [float(row[field]) for row in rows if row[field] is not None]
                aggregates[label][field] = sum(values) / len(values) if values else None
    ranking = sorted(
        (payload.get("mse", math.inf), label) for label, payload in aggregates.items())

    payload = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "latent_frames": 45,
            "video_blocks": 15,
            "prompts": list(PROMPT_NAMES),
            "seed": 0,
            "injection_blocks": list(INJECTION_BLOCKS),
            "injections_per_prompt": len(INJECTION_BLOCKS),
            "quality_claim": "none; equal-dose latent sensitivity diagnostic",
        },
        "noise_sha256": noise_hashes,
        "latent_sha256": latent_hashes,
        "cases": case_reports,
        "paired_latent_metrics": metrics,
        "mean_across_prompts": aggregates,
        "mse_ranking_low_to_high": [label for _, label in ranking],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "stage", "prompt_index", "prompt_name", "mse", "mae",
            "relative_l1", "max_abs", "dynamic_range_psnr_db"))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": payload["status"], "errors": errors, "ranking": payload["mse_ranking_low_to_high"]}, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
