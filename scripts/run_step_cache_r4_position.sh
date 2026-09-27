#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R4_RUN_ID="${R4_RUN_ID:-r4_position_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r4/$R4_RUN_ID}"
PROMPTS="$RF_ROOT/prompts/step_cache_r4_three_prompts.txt"
TEMPLATE="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r4_injection_template.yaml"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r4][error] $*" >&2; exit 1; }
for file in "$RF_ROOT/inference.py" "$PROMPTS" "$TEMPLATE" "$CKPT" \
  "$RF_ROOT/scripts/assess_step_cache_r4_position.py" "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{configs,logs,reports,tensors,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  configs/rolling_forcing_dmd_step_cache_r4_injection_template.yaml
  prompts/step_cache_r4_three_prompts.txt scripts/run_step_cache_r4_position.sh
  scripts/assess_step_cache_r4_position.py scripts/inspect_step_cache_r0_videos.py
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"

STEP_CACHE_R4_DECISION_LOG="$OUT_ROOT/reports/config_probe.jsonl" \
STEP_CACHE_R4_SUMMARY="$OUT_ROOT/reports/config_probe.json" \
"$PYTHON_BIN" - "$TEMPLATE" "$OUT_ROOT/configs" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

template_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
template = OmegaConf.load(template_path)
required = ("num_train_timestep", "denoising_step_list", "image_or_video_shape", "num_frame_per_block")
missing = [key for key in required if OmegaConf.select(template, key, default=None) is None]
if missing:
    raise SystemExit(f"incomplete R4 template: {missing}")

baseline = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
baseline.step_cache.enabled = False
baseline.step_cache.policy = "fixed"
baseline.step_cache.diagnostic_injection_points = []
OmegaConf.save(baseline, f=str(output_dir / "baseline.yaml"))

for stage in (1, 2, 3):
    config = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
    config.step_cache.diagnostic_injection_points = [[block, stage] for block in range(4, 12)]
    OmegaConf.save(config, f=str(output_dir / f"stage{stage}.yaml"))
    if len(config.step_cache.diagnostic_injection_points) != 8:
        raise SystemExit(f"bad injection count for stage {stage}")
print("[r4] effective-configs-ok")
PY

sha256sum "$OUT_ROOT"/configs/*.yaml > "$OUT_ROOT/reports/effective_config_manifest.sha256"
"$PYTHON_BIN" -m py_compile scripts/assess_step_cache_r4_position.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"

run_case() {
  local label="$1"
  local config="$OUT_ROOT/configs/${label}.yaml"
  local tensor_dir="$OUT_ROOT/tensors/$label"
  mkdir -p "$tensor_dir"
  STEP_CACHE_R4_DECISION_LOG="$OUT_ROOT/reports/${label}_decisions.jsonl" \
  STEP_CACHE_R4_SUMMARY="$OUT_ROOT/reports/${label}_runtime_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$config" --checkpoint_path "$CKPT" --data_path "$PROMPTS" \
    --output_folder "$OUT_ROOT/videos/$label" --save_step_cache_tensors_folder "$tensor_dir" \
    --num_output_frames 45 --num_samples 1 --seed 0 --use_ema --save_with_index \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"

  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/1-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/2-0_ema.mp4" \
    --expect-decoded-frames 177 --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
}

run_case baseline
run_case stage1
run_case stage2
run_case stage3

"$PYTHON_BIN" scripts/assess_step_cache_r4_position.py \
  --run-root "$OUT_ROOT" --output "$OUT_ROOT/reports/r4_position_assessment.json" \
  --csv-output "$OUT_ROOT/reports/r4_position_metrics.csv" \
  2>&1 | tee "$OUT_ROOT/logs/r4_position_assessment.log"

# Hash equality has been frozen in the assessment. Keep one noise tensor per
# prompt and all final latents, removing nine byte-identical noise duplicates
# from the return bundle.
rm -f \
  "$OUT_ROOT"/tensors/stage1/*_initial_noise.pt \
  "$OUT_ROOT"/tensors/stage2/*_initial_noise.pt \
  "$OUT_ROOT"/tensors/stage3/*_initial_noise.pt

echo "[r4] completed output=$OUT_ROOT"
