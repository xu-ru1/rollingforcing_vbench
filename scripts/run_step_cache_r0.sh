#!/usr/bin/env bash
set -euo pipefail

# R0 is read-only with respect to generation semantics: it records stage/RNG
# hashes and profiling metadata only. It never enables step cache or KV compression.
# Usage from any directory:
#   RF_ROOT=/path/RollingForcing-main GPU=0 CKPT=/path/rolling_forcing_dmd.pt \
#     bash /path/RollingForcing-main/scripts/run_step_cache_r0.sh all

STAGE="${1:-all}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
WAN_MODEL="${WAN_MODEL:-$RF_ROOT/wan_models/Wan2.1-T2V-1.3B}"
R0_RUN_ID="${R0_RUN_ID:-r0_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r0/$R0_RUN_ID}"
PROMPT="$RF_ROOT/prompts/step_cache_r0_single.txt"
CONFIG_BASE="$RF_ROOT/configs/rolling_forcing_dmd.yaml"
CONFIG_TRACE="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r0_trace.yaml"
if [[ "${USE_EMA:-1}" == "1" ]]; then
  EMA_ARGS=(--use_ema)
  VIDEO_VARIANT="ema"
else
  EMA_ARGS=()
  VIDEO_VARIANT="regular"
fi
if [[ "${HASH_MODEL_WEIGHTS:-0}" == "1" ]]; then
  HASH_ARGS=(--hash-model-weights)
else
  HASH_ARGS=()
fi

die() {
  echo "[r0][error] $*" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || die "required file is missing: $1"
}

require_dir() {
  [ -d "$1" ] || die "required directory is missing: $1"
}

prepare() {
  require_dir "$RF_ROOT"
  require_file "$RF_ROOT/inference.py"
  require_file "$CONFIG_BASE"
  require_file "$CONFIG_TRACE"
  require_file "$PROMPT"
  require_file "$CKPT"
  require_dir "$WAN_MODEL"
  [ ! -e "$OUT_ROOT" ] || die "output already exists; choose a new R0_RUN_ID or OUT_ROOT: $OUT_ROOT"
  mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/videos" "$OUT_ROOT/reports"
  cd "$RF_ROOT"
}

compile_and_test() {
  "$PYTHON_BIN" -m py_compile \
    inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py \
    utils/step_cache_r0.py utils/wan_wrapper.py wan/modules/causal_model.py \
    scripts/collect_step_cache_r0_manifest.py scripts/summarize_step_cache_r0.py \
    scripts/inspect_step_cache_r0_videos.py scripts/assess_step_cache_r0.py \
    2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_r0.py' -v \
    2>&1 | tee "$OUT_ROOT/logs/unit_tests.log"
}

preflight() {
  "$PYTHON_BIN" scripts/collect_step_cache_r0_manifest.py \
    --checkpoint "$CKPT" --wan-model "$WAN_MODEL" \
    --output "$OUT_ROOT/reports/preflight_manifest.json" \
    "${HASH_ARGS[@]}" \
    2>&1 | tee "$OUT_ROOT/logs/preflight.log"
}

run_b0() {
  local label="$1"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$CONFIG_BASE" --checkpoint_path "$CKPT" --data_path "$PROMPT" \
    --output_folder "$OUT_ROOT/videos/$label" --num_output_frames 21 --num_samples 1 \
    "${EMA_ARGS[@]}" --save_with_index --eval_metrics \
    2>&1 | tee "$OUT_ROOT/logs/$label.log"
}

run_trace() {
  local label="$1"
  local latent_frames="$2"
  FLOWCACHE_R0_TRACE_OUTPUT_PATH="$OUT_ROOT/reports/${label}_trace.jsonl" \
  FLOWCACHE_R0_PROFILER_OUTPUT_PATH="$OUT_ROOT/reports/${label}_profile.jsonl" \
  FLOWCACHE_R0_PROFILER_SUMMARY_PATH="$OUT_ROOT/reports/${label}_profile_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$CONFIG_TRACE" --checkpoint_path "$CKPT" --data_path "$PROMPT" \
    --output_folder "$OUT_ROOT/videos/$label" --num_output_frames "$latent_frames" --num_samples 1 \
    "${EMA_ARGS[@]}" --save_with_index --profile --eval_metrics \
    2>&1 | tee "$OUT_ROOT/logs/$label.log"
  "$PYTHON_BIN" scripts/summarize_step_cache_r0.py \
    --trace "$OUT_ROOT/reports/${label}_trace.jsonl" \
    --num-latent-frames "$latent_frames" --num-frame-per-block 3 \
    --output "$OUT_ROOT/reports/${label}_trace_summary.json" \
    --require-main-rng-unchanged \
    2>&1 | tee "$OUT_ROOT/logs/${label}_trace_summary.log"
}

inspect_b0_outputs() {
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/b0_a/0-0_${VIDEO_VARIANT}.mp4" \
    --video "$OUT_ROOT/videos/b0_b/0-0_${VIDEO_VARIANT}.mp4" \
    --compare "$OUT_ROOT/videos/b0_a/0-0_${VIDEO_VARIANT}.mp4" "$OUT_ROOT/videos/b0_b/0-0_${VIDEO_VARIANT}.mp4" \
    --expect-decoded-frames 81 \
    --output "$OUT_ROOT/reports/b0_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/b0_video_check.log"
}

inspect_trace_output() {
  local label="$1"
  local decoded_frames="$2"
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_${VIDEO_VARIANT}.mp4" \
    --expect-decoded-frames "$decoded_frames" \
    --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
}

assess_all() {
  "$PYTHON_BIN" scripts/assess_step_cache_r0.py \
    --run-root "$OUT_ROOT" \
    --output "$OUT_ROOT/reports/r0_assessment.json" \
    2>&1 | tee "$OUT_ROOT/logs/r0_assessment.log"
}

write_run_info() {
  "$PYTHON_BIN" - "$OUT_ROOT/reports/run_info.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "stage": "R0",
    "run_id": os.environ["R0_RUN_ID"],
    "gpu": os.environ["GPU"],
    "checkpoint": os.environ["CKPT"],
    "wan_model": os.environ["WAN_MODEL"],
    "output_root": os.environ["OUT_ROOT"],
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

prepare
export R0_RUN_ID GPU CKPT WAN_MODEL OUT_ROOT
write_run_info

case "$STAGE" in
  preflight)
    compile_and_test
    preflight
    ;;
  b0)
    compile_and_test
    preflight
    run_b0 b0_a
    run_b0 b0_b
    inspect_b0_outputs
    ;;
  trace21)
    compile_and_test
    preflight
    run_trace trace_21 21
    inspect_trace_output trace_21 81
    ;;
  trace81)
    compile_and_test
    preflight
    run_trace trace_81 81
    inspect_trace_output trace_81 321
    ;;
  all)
    compile_and_test
    preflight
    run_b0 b0_a
    run_b0 b0_b
    run_trace trace_21 21
    run_trace trace_81 81
    inspect_b0_outputs
    inspect_trace_output trace_21 81
    inspect_trace_output trace_81 321
    assess_all
    ;;
  *)
    die "unknown stage '$STAGE'; use preflight, b0, trace21, trace81, or all"
    ;;
esac

echo "[r0] completed stage=$STAGE output=$OUT_ROOT"
