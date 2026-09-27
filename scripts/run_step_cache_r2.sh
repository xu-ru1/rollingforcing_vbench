#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R2_RUN_ID="${R2_RUN_ID:-r2_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r2/$R2_RUN_ID}"
PROMPT="$RF_ROOT/prompts/step_cache_r0_single.txt"
VIDEO_VARIANT="${VIDEO_VARIANT:-ema}"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true

die() { echo "[r2][error] $*" >&2; exit 1; }
for file in "$RF_ROOT/inference.py" "$PROMPT" "$CKPT"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/reports" "$OUT_ROOT/videos"
cd "$RF_ROOT"

"$PYTHON_BIN" -m py_compile \
  pipeline/rolling_forcing_inference.py utils/wan_wrapper.py wan/modules/causal_model.py \
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py \
  utils/step_cache_runtime.py utils/step_cache_sparse.py \
  scripts/validate_step_cache_r2_attention.py scripts/summarize_step_cache_r2.py scripts/assess_step_cache_r2.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_*.py' -v \
  2>&1 | tee "$OUT_ROOT/logs/unit_tests.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/validate_step_cache_r2_attention.py \
  --output "$OUT_ROOT/reports/attention_oracle.json" \
  2>&1 | tee "$OUT_ROOT/logs/attention_oracle.log"

run_case() {
  local label="$1" config="$2" expected_reuse="$3"
  local decision="$OUT_ROOT/reports/${label}_decisions.jsonl"
  local execution="$OUT_ROOT/reports/${label}_execution.jsonl"
  local runtime_summary="$OUT_ROOT/reports/${label}_runtime_summary.json"
  STEP_CACHE_R2_IMPLEMENTATION=sparse \
  STEP_CACHE_R2_DECISION_LOG="$decision" \
  STEP_CACHE_R2_EXECUTION_LOG="$execution" \
  STEP_CACHE_R2_SUMMARY="$runtime_summary" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$RF_ROOT/configs/$config" --checkpoint_path "$CKPT" --data_path "$PROMPT" \
    --output_folder "$OUT_ROOT/videos/$label" --num_output_frames 21 --num_samples 1 \
    --use_ema --save_with_index --eval_metrics \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"

  "$PYTHON_BIN" scripts/summarize_step_cache_r2.py \
    --decisions "$decision" --executions "$execution" --runtime-summary "$runtime_summary" \
    --expected-reuse "$expected_reuse" --output "$OUT_ROOT/reports/${label}_summary.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_summary.log"

  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_${VIDEO_VARIANT}.mp4" \
    --expect-decoded-frames 81 --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
}

run_case all_recompute rolling_forcing_dmd_step_cache_r2_all_recompute.yaml 0
run_case few rolling_forcing_dmd_step_cache_r2_few.yaml 2
run_case more rolling_forcing_dmd_step_cache_r2_more.yaml 9

"$PYTHON_BIN" scripts/assess_step_cache_r2.py \
  --run-root "$OUT_ROOT" --output "$OUT_ROOT/reports/r2_assessment.json" \
  2>&1 | tee "$OUT_ROOT/logs/r2_assessment.log"

echo "[r2] completed output=$OUT_ROOT"
