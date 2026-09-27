#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R6_RUN_ID="${R6_RUN_ID:-r6_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r6/$R6_RUN_ID}"
PROMPTS="$RF_ROOT/prompts/step_cache_r6_four_prompts.txt"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r6][error] $*" >&2; exit 1; }

if [ -z "${R5B_ROOT:-}" ]; then
  R5B_ASSESSMENT="$(find "$RF_ROOT/outputs/step_cache_r5" -path '*/reports/r5b_assessment.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [ -n "$R5B_ASSESSMENT" ] || die "R5B assessment not found; set R5B_ROOT"
  R5B_ROOT="$(dirname "$(dirname "$R5B_ASSESSMENT")")"
fi

for file in "$RF_ROOT/inference.py" "$PROMPTS" "$CKPT" \
  "$RF_ROOT/scripts/assess_step_cache_r6.py" \
  "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py" \
  "$R5B_ROOT/reports/r5b_assessment.json" \
  "$R5B_ROOT/configs/fixed.yaml" "$R5B_ROOT/configs/front.yaml"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ "$(wc -l < "$PROMPTS")" -eq 4 ] || die "R6 prompt file must contain exactly four lines"
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{configs,logs,reports,tensors,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  prompts/step_cache_r6_four_prompts.txt scripts/run_step_cache_r6.sh
  scripts/assess_step_cache_r6.py scripts/inspect_step_cache_r0_videos.py
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"
sha256sum "$R5B_ROOT/reports/r5b_assessment.json" \
  "$R5B_ROOT/configs/fixed.yaml" "$R5B_ROOT/configs/front.yaml" \
  > "$OUT_ROOT/reports/input_manifest.sha256"

"$PYTHON_BIN" -m py_compile scripts/assess_step_cache_r6.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"

STEP_CACHE_R5_DECISION_LOG="$OUT_ROOT/reports/config_probe.jsonl" \
STEP_CACHE_R5_SUMMARY="$OUT_ROOT/reports/config_probe.json" \
"$PYTHON_BIN" - "$R5B_ROOT" "$OUT_ROOT/configs" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

r5b_root, output_dir = map(Path, sys.argv[1:])
fixed = OmegaConf.load(r5b_root / "configs" / "fixed.yaml")
front = OmegaConf.load(r5b_root / "configs" / "front.yaml")
if abs(float(fixed.step_cache.base_threshold) - 0.2785) > 1e-12:
    raise SystemExit(f"unexpected frozen Fixed threshold: {fixed.step_cache.base_threshold}")
if abs(float(front.step_cache.base_threshold) - 0.361) > 1e-12:
    raise SystemExit(f"unexpected frozen Front threshold: {front.step_cache.base_threshold}")
for name, config in (("fixed", fixed), ("front", front)):
    config.image_or_video_shape = [1, 81, 16, 60, 104]
    config.data_path = "prompts/step_cache_r6_four_prompts.txt"
    config.step_cache.decision_log_path = "${oc.env:STEP_CACHE_R6_DECISION_LOG}"
    config.step_cache.summary_output_path = "${oc.env:STEP_CACHE_R6_SUMMARY}"
    config.step_cache.execution_log_path = None
    config.step_cache.debug_validate_residual_finite = False
    OmegaConf.save(config, f=str(output_dir / f"{name}.yaml"))
baseline = OmegaConf.create(OmegaConf.to_container(fixed, resolve=False))
baseline.step_cache.enabled = False
baseline.step_cache.policy = "fixed"
baseline.step_cache.schedule = None
OmegaConf.save(baseline, f=str(output_dir / "baseline.yaml"))
print("[r6] frozen-configs-ok")
PY
sha256sum "$OUT_ROOT"/configs/*.yaml > "$OUT_ROOT/reports/effective_config_manifest.sha256"

run_case() {
  local label="$1"
  local tensor_dir="$OUT_ROOT/tensors/$label"
  mkdir -p "$tensor_dir"
  STEP_CACHE_R6_DECISION_LOG="$OUT_ROOT/reports/${label}_decisions.jsonl" \
  STEP_CACHE_R6_SUMMARY="$OUT_ROOT/reports/${label}_runtime_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$OUT_ROOT/configs/${label}.yaml" --checkpoint_path "$CKPT" \
    --data_path "$PROMPTS" --output_folder "$OUT_ROOT/videos/$label" \
    --save_step_cache_tensors_folder "$tensor_dir" \
    --num_output_frames 81 --num_samples 1 --seed 1 --use_ema --save_with_index --profile \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/1-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/2-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/3-0_ema.mp4" \
    --expect-decoded-frames 321 --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
}

# One process per method. Prompt 0 warms CUDA/model state; prompts 1..3 are
# paired measurements and unseen-prompt generalization cases.
run_case baseline
run_case fixed
run_case front

"$PYTHON_BIN" scripts/assess_step_cache_r6.py \
  --run-root "$OUT_ROOT" --output "$OUT_ROOT/reports/r6_assessment.json" \
  2>&1 | tee "$OUT_ROOT/logs/r6_assessment.log"

# Assessment has frozen all hashes and metrics. Retain one paired latent triplet
# for audit and discard the remaining large tensors from the return bundle.
rm -f \
  "$OUT_ROOT"/tensors/baseline/0_initial_noise.pt "$OUT_ROOT"/tensors/baseline/0_latents.pt \
  "$OUT_ROOT"/tensors/fixed/0_initial_noise.pt "$OUT_ROOT"/tensors/fixed/0_latents.pt \
  "$OUT_ROOT"/tensors/front/0_initial_noise.pt "$OUT_ROOT"/tensors/front/0_latents.pt \
  "$OUT_ROOT"/tensors/fixed/1_initial_noise.pt "$OUT_ROOT"/tensors/front/1_initial_noise.pt \
  "$OUT_ROOT"/tensors/baseline/2_initial_noise.pt "$OUT_ROOT"/tensors/baseline/2_latents.pt \
  "$OUT_ROOT"/tensors/fixed/2_initial_noise.pt "$OUT_ROOT"/tensors/fixed/2_latents.pt \
  "$OUT_ROOT"/tensors/front/2_initial_noise.pt "$OUT_ROOT"/tensors/front/2_latents.pt \
  "$OUT_ROOT"/tensors/baseline/3_initial_noise.pt "$OUT_ROOT"/tensors/baseline/3_latents.pt \
  "$OUT_ROOT"/tensors/fixed/3_initial_noise.pt "$OUT_ROOT"/tensors/fixed/3_latents.pt \
  "$OUT_ROOT"/tensors/front/3_initial_noise.pt "$OUT_ROOT"/tensors/front/3_latents.pt

echo "[r6] completed output=$OUT_ROOT r5b_root=$R5B_ROOT"
