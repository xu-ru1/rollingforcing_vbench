#!/usr/bin/env bash
# Capture the exact server environment without installing or changing packages.
set -euo pipefail
OUT_DIR="${1:?usage: bash scripts/record_step_cache_delivery_env.sh OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" --version > "$OUT_DIR/python_version.txt" 2>&1
"$PYTHON_BIN" -m pip freeze > "$OUT_DIR/pip_freeze.txt"
"$PYTHON_BIN" - <<'PY' > "$OUT_DIR/runtime_versions.txt" 2>&1
import importlib
for name in ("torch", "torchvision", "diffusers", "transformers", "flash_attn", "xformers", "omegaconf", "imageio", "av"):
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', 'UNKNOWN')}")
        if name == "torch":
            print(f"torch_cuda={module.version.cuda}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{type(exc).__name__}:{exc}")
PY
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi > "$OUT_DIR/nvidia_smi.txt"; fi
if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -version > "$OUT_DIR/ffmpeg_version.txt" 2>&1; fi
git rev-parse HEAD > "$OUT_DIR/git_commit.txt" 2>/dev/null || printf 'UNKNOWN_NO_GIT_METADATA\n' > "$OUT_DIR/git_commit.txt"
printf '[delivery] environment captured in %s\n' "$OUT_DIR"
