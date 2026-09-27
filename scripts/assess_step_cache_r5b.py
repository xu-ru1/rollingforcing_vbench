#!/usr/bin/env python3
"""Assess the two-run R5B budget refinement with the retained U-shape run."""

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


METHOD_SOURCES = {
    "fixed": "refined",
    "front": "refined",
    "u_shape": "r5a",
}
PROMPTS = ("cat", "rainy_car", "robot")


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
    parser.add_argument("--r5a-root", required=True, type=Path)
    parser.add_argument("--r4-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    reports = args.run_root / "reports"
    r5a_reports = args.r5a_root / "reports"
    r5a_assessment = load_json(r5a_reports / "r5a_assessment.json")
    if r5a_assessment.get("status") != "ok":
        errors.append("r5a_assessment_not_ok")
    r4_assessment = load_json(args.r4_root / "reports" / "r4_position_assessment.json")
    if r4_assessment.get("status") != "ok":
        errors.append("r4_assessment_not_ok")
    selection = load_json(reports / "refinement_selection.json")
    if selection.get("target_reuse_decisions") != 43:
        errors.append("refinement_target_not_43")

    baseline_noise, baseline_latents = {}, {}
    for index in range(3):
        baseline_noise[index] = tensor_hash(load_tensor(
            args.r4_root / "tensors" / "baseline" / f"{index}_initial_noise.pt"))
        baseline_latents[index] = load_tensor(
            args.r4_root / "tensors" / "baseline" / f"{index}_latents.pt")

    methods: dict[str, Any] = {}
    identities: dict[str, set[tuple[int, int, int]]] = {}
    csv_rows = []
    for method, source in METHOD_SOURCES.items():
        root = args.run_root if source == "refined" else args.r5a_root
        source_reports = root / "reports"
        rows = load_jsonl(source_reports / f"{method}_decisions.jsonl")
        summary = load_json(source_reports / f"{method}_runtime_summary.json")
        if len(rows) != 225:
            errors.append(f"decision_count:{method}:{len(rows)}")
        reuse_rows = [row for row in rows if row.get("execute_reuse")]
        identities[method] = {
            (int(row["sample_id"]), int(row["global_video_block_id"]),
             int(row["local_stage_index"])) for row in reuse_rows
        }
        by_stage = Counter(int(row["local_stage_index"]) for row in reuse_rows)
        by_prompt = Counter(int(row["sample_id"]) for row in reuse_rows)
        if any(stage not in (1, 2, 3) for stage in by_stage):
            errors.append(f"protected_stage_reuse:{method}:{dict(by_stage)}")
        totals = summary.get("operator_totals", {})
        full = int(totals.get("full_window_tokens", -1))
        reuse_tokens = int(totals.get("reuse_tokens", -1))
        if totals.get("main_forwards") != 57 or totals.get("layer_calls") != 1710:
            errors.append(f"forward_counts:{method}")
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
        if source == "refined":
            video = load_json(source_reports / f"{method}_video_check.json")
            if video.get("status") != "ok" or len(video.get("videos", [])) != 3:
                errors.append(f"video_check:{method}")

        metrics = {}
        for index, prompt in enumerate(PROMPTS):
            tensor_root = root / "tensors" / method
            if source == "refined":
                noise = load_tensor(tensor_root / f"{index}_initial_noise.pt")
                if tensor_hash(noise) != baseline_noise[index]:
                    errors.append(f"initial_noise_mismatch:{method}:{index}")
            latent = load_tensor(tensor_root / f"{index}_latents.pt")
            values = compare(baseline_latents[index], latent)
            metrics[prompt] = values
            csv_rows.append({"method": method, "prompt": prompt, **values})
        means = {}
        for field in ("mse", "mae", "relative_l1", "max_abs", "dynamic_range_psnr_db"):
            values = [float(item[field]) for item in metrics.values() if item[field] is not None]
            means[field] = sum(values) / len(values) if values else None
        methods[method] = {
            "source": source,
            "base_threshold": summary.get("base_threshold"),
            "reuse_count": len(reuse_rows),
            "reuse_count_by_stage": {str(key): by_stage[key] for key in sorted(by_stage)},
            "reuse_count_by_prompt": {str(key): by_prompt[key] for key in sorted(by_prompt)},
            "partial_operator_token_saving_fraction": reuse_tokens / full,
            "operator_totals": totals,
            "peak_residual_bytes": summary.get("peak_residual_bytes"),
            "paired_latent_metrics": metrics,
            "mean_across_prompts": means,
        }

    counts = {name: value["reuse_count"] for name, value in methods.items()}
    spread = max(counts.values()) - min(counts.values())
    target_error = max(abs(count - 43) for count in counts.values())
    overlaps = {}
    for left, right in (("fixed", "front"), ("fixed", "u_shape"), ("front", "u_shape")):
        union = identities[left] | identities[right]
        overlaps[f"{left}_vs_{right}"] = {
            "intersection": len(identities[left] & identities[right]),
            "union": len(union),
            "jaccard": len(identities[left] & identities[right]) / len(union) if union else 1.0,
        }

    budget_ok = spread <= 2 and target_error <= 2
    payload = {
        "status": "ok" if not errors and budget_ok else "error",
        "errors": errors + ([] if budget_ok else [f"budget_mismatch:{counts}"]),
        "protocol": {
            "name": "R5B closed-loop budget refinement",
            "latent_frames": 45,
            "prompts": list(PROMPTS),
            "seed": 0,
            "target_reuse_decisions": 43,
            "retained_u_shape_from_r5a": True,
        },
        "refinement_selection": selection,
        "methods": methods,
        "budget_match": {
            "reuse_counts": counts,
            "max_pairwise_spread": spread,
            "max_absolute_target_error": target_error,
            "within_two_decisions": budget_ok,
        },
        "reuse_mask_overlap": overlaps,
        "latent_mse_ranking_low_to_high": sorted(
            methods, key=lambda name: methods[name]["mean_across_prompts"]["mse"]),
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
        "errors": payload["errors"],
        "reuse_counts": counts,
        "ranking": payload["latent_mse_ranking_low_to_high"],
    }, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
