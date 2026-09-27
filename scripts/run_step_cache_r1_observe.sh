#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R1_RUN_ID="${R1_RUN_ID:-r1_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r1/$R1_RUN_ID}"
CONFIG="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r1_observe.yaml"
PROMPT="$RF_ROOT/prompts/step_cache_r0_single.txt"

die() { echo "[r1][error] $*" >&2; exit 1; }
for file in "$RF_ROOT/inference.py" "$CONFIG" "$PROMPT" "$CKPT"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/reports" "$OUT_ROOT/videos"
cd "$RF_ROOT"

"$PYTHON_BIN" -m py_compile pipeline/rolling_forcing_inference.py utils/wan_wrapper.py \
  wan/modules/causal_model.py utils/step_cache_policy.py utils/step_cache_state.py \
  utils/step_cache_metric.py scripts/summarize_step_cache_r1.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_*.py' -v \
  2>&1 | tee "$OUT_ROOT/logs/unit_tests.log"

STEP_CACHE_R1_METRIC_OUTPUT_PATH="$OUT_ROOT/reports/r1_metrics.jsonl" \
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
  --config_path "$CONFIG" --checkpoint_path "$CKPT" --data_path "$PROMPT" \
  --output_folder "$OUT_ROOT/videos/observe_81" --num_output_frames 81 --num_samples 1 \
  --use_ema --save_with_index --eval_metrics \
  2>&1 | tee "$OUT_ROOT/logs/observe_81.log"

"$PYTHON_BIN" scripts/summarize_step_cache_r1.py \
  --metrics "$OUT_ROOT/reports/r1_metrics.jsonl" --num-latent-frames 81 \
  --output "$OUT_ROOT/reports/r1_metric_summary.json" \
  2>&1 | tee "$OUT_ROOT/logs/r1_metric_summary.log"

echo "[r1] completed output=$OUT_ROOT"
