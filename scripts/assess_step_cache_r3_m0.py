#!/usr/bin/env python3
"""Assess the frozen R3/M0 latent-level numerical seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


CASES = (
    "disabled_a",
    "disabled_b",
    "all_recompute",
    "mixed_sparse",
    "mixed_dense_reference",
    "fixed_quiet",
    "fixed_observed",
)
STRICT_MAX_ABS = 0.0
STRICT_RELATIVE_L1 = 0.0
SPARSE_DENSE_MAX_ABS = 0.05
SPARSE_DENSE_RELATIVE_L1 = 0.005


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_comparison(left: torch.Tensor, right: torch.Tensor, *, max_abs_gate: float, relative_l1_gate: float) -> dict[str, Any]:
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    if not shape_equal:
        return {
            "status": "error", "shape_equal": False, "dtype_equal": dtype_equal,
            "error": f"shape_mismatch:{list(left.shape)}:{list(right.shape)}",
        }
    difference = (left.float() - right.float()).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    denominator = left.float().abs().mean().clamp_min(1e-6)
    relative_l1 = float(difference.mean().div(denominator).item()) if difference.numel() else 0.0
    exact_equal = bool(torch.equal(left, right))
    passed = max_abs <= max_abs_gate and relative_l1 <= relative_l1_gate
    return {
        "status": "ok" if passed else "error",
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "exact_equal": exact_equal,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l1": relative_l1,
        "max_abs_gate": max_abs_gate,
        "relative_l1_gate": relative_l1_gate,
        "sha256_a": tensor_hash(left),
        "sha256_b": tensor_hash(right),
    }


def decision_mask(rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    fields = (
        "sample_id", "window_index", "global_video_block_id", "local_block_index",
        "local_stage_index", "branch", "eligible", "would_reuse", "execute_reuse", "reason",
    )
    compact = [{field: row.get(field) for field in fields} for row in rows]
    compact.sort(key=lambda row: (
        row["sample_id"], row["window_index"], row["local_block_index"], row["local_stage_index"], row["branch"]
    ))
    encoded = json.dumps(compact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), compact


def case_paths(root: Path, label: str) -> dict[str, Path]:
    report_root = root / "reports"
    return {
        "noise": root / "tensors" / label / "0_initial_noise.pt",
        "latent": root / "tensors" / label / "0_latents.pt",
        "video": report_root / f"{label}_video_check.json",
        "decision": report_root / f"{label}_decisions.jsonl",
        "execution": report_root / f"{label}_execution.jsonl",
        "summary": report_root / f"{label}_runtime_summary.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    masks: dict[str, str] = {}
    for label in CASES:
        paths = case_paths(args.run_root, label)
        required = ("noise", "latent", "video") if label.startswith("disabled") else tuple(paths)
        missing = [name for name in required if not paths[name].is_file()]
        if missing:
            errors.append(f"missing:{label}:{','.join(missing)}")
            continue
        noise = torch.load(paths["noise"], map_location="cpu")
        latent = torch.load(paths["latent"], map_location="cpu")
        tensors[label] = (noise, latent)
        record: dict[str, Any] = {
            "noise_sha256": tensor_hash(noise),
            "latent_sha256": tensor_hash(latent),
            "noise_shape": list(noise.shape),
            "latent_shape": list(latent.shape),
            "noise_dtype": str(noise.dtype),
            "latent_dtype": str(latent.dtype),
        }
        video = load_json(paths["video"])
        record["video"] = video
        if video.get("status") != "ok":
            errors.append(f"video_failed:{label}")
        if not label.startswith("disabled"):
            decisions = load_jsonl(paths["decision"])
            mask_hash, _ = decision_mask(decisions)
            record["decision_count"] = len(decisions)
            record["mask_hash"] = mask_hash
            record["execute_reuse_count"] = sum(bool(row.get("execute_reuse")) for row in decisions)
            record["runtime_summary"] = load_json(paths["summary"])
            if len(decisions) != 35:
                errors.append(f"unexpected_decision_count:{label}:{len(decisions)}")
            masks[label] = mask_hash
        cases[label] = record

    comparisons: dict[str, dict[str, Any]] = {}
    def compare(name: str, left_label: str, right_label: str, *, max_abs: float, relative_l1: float) -> None:
        if left_label not in tensors or right_label not in tensors:
            errors.append(f"comparison_inputs_missing:{name}")
            return
        left_noise, left_latent = tensors[left_label]
        right_noise, right_latent = tensors[right_label]
        if tensor_hash(left_noise) != tensor_hash(right_noise):
            errors.append(f"initial_noise_mismatch:{name}")
        comparison = tensor_comparison(left_latent, right_latent, max_abs_gate=max_abs, relative_l1_gate=relative_l1)
        comparisons[name] = comparison
        if comparison["status"] != "ok":
            errors.append(f"latent_gate_failed:{name}")

    compare("disabled_repeat", "disabled_a", "disabled_b", max_abs=STRICT_MAX_ABS, relative_l1=STRICT_RELATIVE_L1)
    compare("disabled_vs_all_recompute", "disabled_a", "all_recompute", max_abs=STRICT_MAX_ABS, relative_l1=STRICT_RELATIVE_L1)
    compare("mixed_sparse_vs_dense_reference", "mixed_sparse", "mixed_dense_reference", max_abs=SPARSE_DENSE_MAX_ABS, relative_l1=SPARSE_DENSE_RELATIVE_L1)
    compare("fixed_quiet_vs_observed", "fixed_quiet", "fixed_observed", max_abs=STRICT_MAX_ABS, relative_l1=STRICT_RELATIVE_L1)

    for left_label, right_label, name in (
        ("mixed_sparse", "mixed_dense_reference", "mixed_mask"),
        ("fixed_quiet", "fixed_observed", "observer_mask"),
    ):
        if left_label in masks and right_label in masks and masks[left_label] != masks[right_label]:
            errors.append(f"mask_mismatch:{name}")

    if "all_recompute" in cases:
        totals = cases["all_recompute"]["runtime_summary"].get("operator_totals", {})
        if cases["all_recompute"].get("execute_reuse_count") != 0:
            errors.append("all_recompute_has_reuse")
        if totals.get("q_tokens") != totals.get("k_tokens") or totals.get("k_tokens") != totals.get("v_tokens"):
            errors.append("all_recompute_kvq_contract_failed")
    for label, implementation in (("mixed_sparse", "sparse"), ("mixed_dense_reference", "dense_reference")):
        if label not in cases:
            continue
        totals = cases[label]["runtime_summary"].get("operator_totals", {})
        if cases[label].get("execute_reuse_count", 0) <= 0:
            errors.append(f"mixed_case_has_no_reuse:{label}")
        if totals.get("k_tokens") != totals.get("full_window_tokens") or totals.get("v_tokens") != totals.get("full_window_tokens"):
            errors.append(f"kv_not_full:{label}")
        if implementation == "sparse" and not totals.get("q_tokens", 0) < totals.get("k_tokens", 0):
            errors.append("sparse_q_not_reduced")
        if implementation == "dense_reference" and totals.get("q_tokens") != totals.get("k_tokens"):
            errors.append("dense_reference_q_not_full")
    if "fixed_observed" in cases:
        metric_path = args.run_root / "reports" / "fixed_observed_metric.jsonl"
        if not metric_path.is_file():
            errors.append("observer_metric_missing")
        else:
            metric_rows = load_jsonl(metric_path)
            if len(metric_rows) != 35:
                errors.append(f"observer_metric_count:{len(metric_rows)}")
            cases["fixed_observed"]["metric_record_count"] = len(metric_rows)

    payload = {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "frozen_gates": {
            "strict_max_abs": STRICT_MAX_ABS,
            "strict_relative_l1": STRICT_RELATIVE_L1,
            "mixed_sparse_dense_max_abs": SPARSE_DENSE_MAX_ABS,
            "mixed_sparse_dense_relative_l1": SPARSE_DENSE_RELATIVE_L1,
        },
        "cases": cases,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
