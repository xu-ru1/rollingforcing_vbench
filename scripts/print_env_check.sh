#!/usr/bin/env bash

echo "== Basic paths =="
pwd
echo

echo "== Git =="
if command -v git >/dev/null 2>&1; then
  git rev-parse --show-toplevel 2>/dev/null || echo "not a git repository"
  git rev-parse HEAD 2>/dev/null || echo "no git HEAD"
else
  echo "git not found"
fi
echo

echo "== Python =="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
  python - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("torch_cuda", torch.version.cuda)
    if torch.cuda.is_available():
        print("cuda_device_count", torch.cuda.device_count())
        print("current_device", torch.cuda.current_device())
        print("device_name", torch.cuda.get_device_name(torch.cuda.current_device()))
except Exception as exc:
    print("torch_check_error", repr(exc))
PY
else
  echo "python not found"
fi
echo

echo "== Optional Python packages =="
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
for name in ["flash_attn", "diffusers", "transformers", "omegaconf", "einops", "torchvision"]:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}: ok ({version})")
    except Exception as exc:
        print(f"{name}: missing_or_error ({exc!r})")
PY
fi
echo

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo

echo "== Repo files =="
ls -la
echo
echo "-- configs --"
ls -la configs 2>/dev/null || echo "configs not found"
echo
echo "-- checkpoints --"
ls -la checkpoints 2>/dev/null || echo "checkpoints not found"
echo
echo "-- wan_models --"
ls -la wan_models 2>/dev/null || echo "wan_models not found"
echo
echo "-- prompts --"
ls -la prompts 2>/dev/null || echo "prompts not found"
