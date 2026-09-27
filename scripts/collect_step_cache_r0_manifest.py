#!/usr/bin/env python3
"""Capture the R0 source, environment, and local asset inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "inference.py",
    "pipeline/rolling_forcing_inference.py",
    "utils/flowcache.py",
    "utils/step_cache_r0.py",
    "utils/wan_wrapper.py",
    "wan/modules/causal_model.py",
    "wan/modules/attention.py",
    "configs/default_config.yaml",
    "configs/rolling_forcing_dmd.yaml",
    "configs/rolling_forcing_dmd_step_cache_r0_trace.yaml",
    "configs/step_cache_r0_protocol_v2.json",
    "prompts/step_cache_r0_single.txt",
    "scripts/run_step_cache_r0.sh",
    "scripts/collect_step_cache_r0_manifest.py",
    "scripts/summarize_step_cache_r0.py",
    "scripts/inspect_step_cache_r0_videos.py",
    "scripts/assess_step_cache_r0.py",
    "requirements.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, hash_file: bool) -> dict:
    record = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["bytes"] = path.stat().st_size
        if hash_file:
            record["sha256"] = sha256_file(path)
    return record


def run_optional(command: Iterable[str]) -> dict:
    try:
        result = subprocess.run(
            list(command), text=True, capture_output=True, check=False, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def model_inventory(model: Path, *, hash_weights: bool) -> dict:
    required = ("config.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth")
    files = []
    if model.is_dir():
        for path in sorted(item for item in model.rglob("*") if item.is_file()):
            is_weight = path.suffix.lower() in {".bin", ".pt", ".pth", ".safetensors"}
            files.append(file_record(path, hash_file=hash_weights or not is_weight))
    return {
        "path": str(model),
        "exists": model.is_dir(),
        "required_files": {name: (model / name).is_file() for name in required},
        "files": files,
        "weights_hashed": bool(hash_weights),
    }


def inventory_is_complete(inventory: dict) -> bool:
    return bool(inventory["exists"]) and all(inventory["required_files"].values())


def read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wan-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-model-weights", action="store_true")
    args = parser.parse_args()

    missing = [str(ROOT / relative) for relative in SOURCE_FILES if not (ROOT / relative).is_file()]
    source = [file_record(ROOT / relative, hash_file=True) for relative in SOURCE_FILES]
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    wan_inventory = model_inventory(args.wan_model, hash_weights=args.hash_model_weights)
    payload = {
        "status": "ok" if not missing and args.checkpoint.is_file() and inventory_is_complete(wan_inventory) else "failed",
        "repo_root": str(ROOT),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "platform": platform.platform(),
        "torch": torch_info,
        "git": run_optional(["git", "rev-parse", "HEAD"]),
        "git_status": run_optional(["git", "status", "--short"]),
        "nvidia_smi": run_optional(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "cgroup": {
            "memory_max": read_optional_text(Path("/sys/fs/cgroup/memory.max")),
            "memory_high": read_optional_text(Path("/sys/fs/cgroup/memory.high")),
            "memory_current": read_optional_text(Path("/sys/fs/cgroup/memory.current")),
        },
        "missing_source_files": missing,
        "source": source,
        "checkpoint": file_record(args.checkpoint, hash_file=True),
        "wan_model": wan_inventory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
