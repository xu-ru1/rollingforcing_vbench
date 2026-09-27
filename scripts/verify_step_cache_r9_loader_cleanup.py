#!/usr/bin/env python3
"""Verify that the loader-only memory cleanup preserves formal inference exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


AUDIT_FIELDS = (
    "event",
    "seed",
    "num_output_frames",
    "initial_noise_shape",
    "initial_noise_dtype",
    "initial_noise_sha256",
    "final_latent_shape",
    "final_latent_dtype",
    "final_latent_sha256",
    "cpu_rng_after_noise_sha256",
    "cuda_rng_after_noise_sha256",
    "cpu_rng_after_inference_sha256",
    "cuda_rng_after_inference_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"no JSONL record in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--r8-manifest", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = first_jsonl(args.reference_audit)
    candidate = first_jsonl(args.candidate_audit)
    manifest = json.loads(args.r8_manifest.read_text(encoding="utf-8"))
    old_hash = manifest["rollingforcing"]["r7_key_source_file_sha256"]["inference.py"]
    new_hash = sha256(args.inference)

    comparisons = {
        field: {
            "reference": reference.get(field),
            "candidate": candidate.get(field),
            "equal": reference.get(field) == candidate.get(field),
        }
        for field in AUDIT_FIELDS
    }
    errors = [field for field, result in comparisons.items() if not result["equal"]]
    if old_hash == new_hash:
        errors.append("inference_hash_did_not_change")

    payload = {
        "schema": "rollingforcing.r9.loader_cleanup_source_amendment.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok" if not errors else "error",
        "scope": "loader_only_cpu_checkpoint_reference_cleanup",
        "method_semantics_changed": False,
        "generation_protocol_changed": False,
        "old_inference_sha256": old_hash,
        "new_inference_sha256": new_hash,
        "reference_audit": str(args.reference_audit),
        "candidate_audit": str(args.candidate_audit),
        "memory_report": str(args.memory_report),
        "audit_comparisons": comparisons,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "errors": errors, "output": str(args.output)}))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
