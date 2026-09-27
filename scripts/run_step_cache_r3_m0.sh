#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R3_RUN_ID="${R3_RUN_ID:-r3_m0_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r3/$R3_RUN_ID}"
PROMPT="${PROMPT:-$RF_ROOT/prompts/step_cache_r0_single.txt}"
VIDEO_VARIANT="${VIDEO_VARIANT:-ema}"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true

die() { echo "[r3-m0][error] $*" >&2; exit 1; }
for file in "$RF_ROOT/inference.py" "$PROMPT" "$CKPT" \
  "$RF_ROOT/scripts/assess_step_cache_r3_m0.py" "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{logs,reports,tensors,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  configs/rolling_forcing_dmd_step_cache_r3_disabled.yaml
  configs/rolling_forcing_dmd_step_cache_r3_all_recompute.yaml
  configs/rolling_forcing_dmd_step_cache_r3_mixed.yaml
  configs/rolling_forcing_dmd_step_cache_r3_fixed_quiet.yaml
  configs/rolling_forcing_dmd_step_cache_r3_fixed_observed.yaml
  scripts/run_step_cache_r3_m0.sh scripts/assess_step_cache_r3_m0.py
  scripts/inspect_step_cache_r0_videos.py
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"

# R3 configs must remain self-contained because default_config.yaml is an
# auxiliary FlowCache config, not a full diffusion-model config.
"$PYTHON_BIN" -c '
from omegaconf import OmegaConf
import sys
required = ("num_train_timestep", "denoising_step_list", "image_or_video_shape", "num_frame_per_block")
for path in sys.argv[1:]:
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(path))
    missing = [key for key in required if OmegaConf.select(config, key, default=None) is None]
    if missing:
        raise SystemExit(f"R3 config is incomplete: {path}: missing={missing}")
    print(f"[r3-m0] config-ok {path}")
' \
  configs/rolling_forcing_dmd_step_cache_r3_disabled.yaml \
  configs/rolling_forcing_dmd_step_cache_r3_all_recompute.yaml \
  configs/rolling_forcing_dmd_step_cache_r3_mixed.yaml \
  configs/rolling_forcing_dmd_step_cache_r3_fixed_quiet.yaml \
  configs/rolling_forcing_dmd_step_cache_r3_fixed_observed.yaml \
  2>&1 | tee "$OUT_ROOT/logs/config_preflight.log"

"$PYTHON_BIN" -m py_compile \
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py wan/modules/causal_model.py \
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py \
  utils/step_cache_runtime.py utils/step_cache_sparse.py scripts/assess_step_cache_r3_m0.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_*.py' -v \
  2>&1 | tee "$OUT_ROOT/logs/unit_tests.log"

run_case() {
  local label="$1" config="$2" implementation="$3" observer="$4"
  local decision="$OUT_ROOT/reports/${label}_decisions.jsonl"
  local execution="$OUT_ROOT/reports/${label}_execution.jsonl"
  local summary="$OUT_ROOT/reports/${label}_runtime_summary.json"
  local metric="$OUT_ROOT/reports/${label}_metric.jsonl"
  local tensors="$OUT_ROOT/tensors/$label"
  mkdir -p "$tensors"
  STEP_CACHE_R3_IMPLEMENTATION="$implementation" \
  STEP_CACHE_R3_DECISION_LOG="$decision" \
  STEP_CACHE_R3_EXECUTION_LOG="$execution" \
  STEP_CACHE_R3_SUMMARY="$summary" \
  STEP_CACHE_R3_METRIC_OUTPUT="$metric" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$RF_ROOT/configs/$config" --checkpoint_path "$CKPT" --data_path "$PROMPT" \
    --output_folder "$OUT_ROOT/videos/$label" --save_step_cache_tensors_folder "$tensors" \
    --num_output_frames 21 --num_samples 1 --seed 0 --use_ema --save_with_index --eval_metrics \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"

  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_${VIDEO_VARIANT}.mp4" \
    --expect-decoded-frames 81 --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
  if [ "$observer" = "false" ]; then
    [ ! -e "$metric" ] || die "observer output unexpectedly exists for $label"
  fi
}

run_case disabled_a rolling_forcing_dmd_step_cache_r3_disabled.yaml sparse false
run_case disabled_b rolling_forcing_dmd_step_cache_r3_disabled.yaml sparse false
run_case all_recompute rolling_forcing_dmd_step_cache_r3_all_recompute.yaml sparse false
run_case mixed_sparse rolling_forcing_dmd_step_cache_r3_mixed.yaml sparse false
run_case mixed_dense_reference rolling_forcing_dmd_step_cache_r3_mixed.yaml dense_reference false
run_case fixed_quiet rolling_forcing_dmd_step_cache_r3_fixed_quiet.yaml sparse false
run_case fixed_observed rolling_forcing_dmd_step_cache_r3_fixed_observed.yaml sparse true

"$PYTHON_BIN" scripts/assess_step_cache_r3_m0.py \
  --run-root "$OUT_ROOT" --output "$OUT_ROOT/reports/r3_m0_assessment.json" \
  2>&1 | tee "$OUT_ROOT/logs/r3_m0_assessment.log"

echo "[r3-m0] completed output=$OUT_ROOT"
