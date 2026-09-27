#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R3_FINAL_RUN_ID="${R3_FINAL_RUN_ID:-r3_final_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r3/$R3_FINAL_RUN_ID}"
SINGLE_PROMPT="$RF_ROOT/prompts/step_cache_r0_single.txt"
TWO_PROMPTS="$RF_ROOT/prompts/step_cache_r3_two_prompts.txt"
CONFIG="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r3_dynamic.yaml"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r3-final][error] $*" >&2; exit 1; }
for file in "$RF_ROOT/inference.py" "$SINGLE_PROMPT" "$TWO_PROMPTS" "$CONFIG" "$CKPT" \
  "$RF_ROOT/scripts/assess_step_cache_r3_final.py" "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{logs,reports,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  configs/rolling_forcing_dmd_step_cache_r3_dynamic.yaml
  prompts/step_cache_r0_single.txt prompts/step_cache_r3_two_prompts.txt
  scripts/run_step_cache_r3_final.sh scripts/assess_step_cache_r3_final.py
  scripts/inspect_step_cache_r0_videos.py
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"

STEP_CACHE_R3_SCHEDULE=front_protect \
STEP_CACHE_R3_DECISION_LOG="$OUT_ROOT/reports/config_probe_decisions.jsonl" \
STEP_CACHE_R3_EXECUTION_LOG="$OUT_ROOT/reports/config_probe_execution.jsonl" \
STEP_CACHE_R3_SUMMARY="$OUT_ROOT/reports/config_probe_summary.json" \
STEP_CACHE_R3_KV_METADATA="$OUT_ROOT/reports/config_probe_kv.jsonl" \
"$PYTHON_BIN" -c '
from omegaconf import OmegaConf
config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/rolling_forcing_dmd_step_cache_r3_dynamic.yaml"))
required = ("num_train_timestep", "denoising_step_list", "image_or_video_shape", "num_frame_per_block")
missing = [key for key in required if OmegaConf.select(config, key, default=None) is None]
if missing:
    raise SystemExit(f"incomplete R3 final config: {missing}")
if config.step_cache.schedule != "front_protect":
    raise SystemExit(f"schedule resolver failed: {config.step_cache.schedule}")
print("[r3-final] config-ok")
' 2>&1 | tee "$OUT_ROOT/logs/config_preflight.log"

"$PYTHON_BIN" -m py_compile scripts/assess_step_cache_r3_final.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"

run_case() {
  local label="$1" schedule="$2" latent_frames="$3" prompt_path="$4"
  STEP_CACHE_R3_SCHEDULE="$schedule" \
  STEP_CACHE_R3_DECISION_LOG="$OUT_ROOT/reports/${label}_decisions.jsonl" \
  STEP_CACHE_R3_EXECUTION_LOG="$OUT_ROOT/reports/${label}_execution.jsonl" \
  STEP_CACHE_R3_SUMMARY="$OUT_ROOT/reports/${label}_runtime_summary.json" \
  STEP_CACHE_R3_KV_METADATA="$OUT_ROOT/reports/${label}_kv_metadata.jsonl" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$CONFIG" --checkpoint_path "$CKPT" --data_path "$prompt_path" \
    --output_folder "$OUT_ROOT/videos/$label" --num_output_frames "$latent_frames" \
    --num_samples 1 --seed 0 --use_ema --save_with_index --eval_metrics \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"
}

run_case front_81 front_protect 81 "$SINGLE_PROMPT"
"$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
  --video "$OUT_ROOT/videos/front_81/0-0_ema.mp4" --expect-decoded-frames 321 \
  --output "$OUT_ROOT/reports/front_81_video_check.json" \
  2>&1 | tee "$OUT_ROOT/logs/front_81_video_check.log"

run_case u_reset u_shape_protect 21 "$TWO_PROMPTS"
"$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
  --video "$OUT_ROOT/videos/u_reset/0-0_ema.mp4" \
  --video "$OUT_ROOT/videos/u_reset/1-0_ema.mp4" --expect-decoded-frames 81 \
  --output "$OUT_ROOT/reports/u_reset_video_check.json" \
  2>&1 | tee "$OUT_ROOT/logs/u_reset_video_check.log"

"$PYTHON_BIN" scripts/assess_step_cache_r3_final.py \
  --run-root "$OUT_ROOT" --output "$OUT_ROOT/reports/r3_final_assessment.json" \
  2>&1 | tee "$OUT_ROOT/logs/r3_final_assessment.log"

echo "[r3-final] completed output=$OUT_ROOT"
