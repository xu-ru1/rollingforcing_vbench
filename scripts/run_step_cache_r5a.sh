#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R5_RUN_ID="${R5_RUN_ID:-r5a_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r5/$R5_RUN_ID}"
PROMPTS="$RF_ROOT/prompts/step_cache_r4_three_prompts.txt"
TEMPLATE="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r5a_template.yaml"
TARGET_REUSE_DECISIONS="${TARGET_REUSE_DECISIONS:-45}"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r5a][error] $*" >&2; exit 1; }
for file in "$RF_ROOT/inference.py" "$PROMPTS" "$TEMPLATE" "$CKPT" \
  "$RF_ROOT/scripts/select_step_cache_r5_thresholds.py" \
  "$RF_ROOT/scripts/assess_step_cache_r5a.py" \
  "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py"; do
  [ -f "$file" ] || die "required file is missing: $file"
done

if [ -z "${R4_ROOT:-}" ]; then
  R4_ASSESSMENT="$(find "$RF_ROOT/outputs/step_cache_r4" -path '*/reports/r4_position_assessment.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [ -n "$R4_ASSESSMENT" ] || die "R4 assessment not found; set R4_ROOT explicitly"
  R4_ROOT="$(dirname "$(dirname "$R4_ASSESSMENT")")"
fi
[ -f "$R4_ROOT/reports/r4_position_assessment.json" ] || die "invalid R4_ROOT: $R4_ROOT"
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{configs,logs,reports,tensors,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  configs/rolling_forcing_dmd_step_cache_r5a_template.yaml
  prompts/step_cache_r4_three_prompts.txt scripts/run_step_cache_r5a.sh
  scripts/select_step_cache_r5_thresholds.py scripts/assess_step_cache_r5a.py
  scripts/inspect_step_cache_r0_videos.py
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"

STEP_CACHE_R5_DECISION_LOG="$OUT_ROOT/reports/config_probe.jsonl" \
STEP_CACHE_R5_SUMMARY="$OUT_ROOT/reports/config_probe.json" \
"$PYTHON_BIN" - "$TEMPLATE" "$OUT_ROOT/configs/observer.yaml" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

source, output = map(Path, sys.argv[1:])
config = OmegaConf.load(source)
required = ("num_train_timestep", "denoising_step_list", "image_or_video_shape", "num_frame_per_block")
missing = [key for key in required if OmegaConf.select(config, key, default=None) is None]
if missing:
    raise SystemExit(f"incomplete R5A template: {missing}")
if list(config.image_or_video_shape) != [1, 45, 16, 60, 104]:
    raise SystemExit(f"unexpected R5A shape: {list(config.image_or_video_shape)}")
OmegaConf.save(config, f=str(output))
print("[r5a] observer-config-ok")
PY

"$PYTHON_BIN" -m py_compile \
  scripts/select_step_cache_r5_thresholds.py scripts/assess_step_cache_r5a.py \
  2>&1 | tee "$OUT_ROOT/logs/py_compile.log"

run_observer() {
  STEP_CACHE_R5_DECISION_LOG="$OUT_ROOT/reports/observer_decisions.jsonl" \
  STEP_CACHE_R5_SUMMARY="$OUT_ROOT/reports/observer_runtime_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$OUT_ROOT/configs/observer.yaml" --checkpoint_path "$CKPT" \
    --data_path "$PROMPTS" --output_folder "$OUT_ROOT/videos/observer" \
    --num_output_frames 45 --num_samples 1 --seed 0 --use_ema --save_with_index \
    2>&1 | tee "$OUT_ROOT/logs/observer.log"
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/observer/0-0_ema.mp4" \
    --video "$OUT_ROOT/videos/observer/1-0_ema.mp4" \
    --video "$OUT_ROOT/videos/observer/2-0_ema.mp4" \
    --expect-decoded-frames 177 --output "$OUT_ROOT/reports/observer_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/observer_video_check.log"
}

run_observer

"$PYTHON_BIN" scripts/select_step_cache_r5_thresholds.py \
  --trace "$OUT_ROOT/reports/observer_decisions.jsonl" \
  --target-reuse-decisions "$TARGET_REUSE_DECISIONS" \
  --output "$OUT_ROOT/reports/threshold_selection.json" \
  2>&1 | tee "$OUT_ROOT/logs/threshold_selection.log"

STEP_CACHE_R5_DECISION_LOG="$OUT_ROOT/reports/config_probe.jsonl" \
STEP_CACHE_R5_SUMMARY="$OUT_ROOT/reports/config_probe.json" \
"$PYTHON_BIN" - "$TEMPLATE" "$OUT_ROOT/reports/threshold_selection.json" "$OUT_ROOT/configs" <<'PY'
from pathlib import Path
import json
import sys
from omegaconf import OmegaConf

template_path, selection_path, output_dir = map(Path, sys.argv[1:])
template = OmegaConf.load(template_path)
selection = json.loads(selection_path.read_text(encoding="utf-8"))
for name in ("fixed", "front", "u_shape"):
    selected = selection["selected"][name]
    config = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
    config.step_cache.policy = selected["policy"]
    config.step_cache.schedule = selected["schedule"]
    config.step_cache.base_threshold = float(selected["base_threshold"])
    OmegaConf.save(config, f=str(output_dir / f"{name}.yaml"))
print("[r5a] candidate-configs-ok")
PY
sha256sum "$OUT_ROOT"/configs/*.yaml > "$OUT_ROOT/reports/effective_config_manifest.sha256"

run_candidate() {
  local label="$1"
  local tensor_dir="$OUT_ROOT/tensors/$label"
  mkdir -p "$tensor_dir"
  STEP_CACHE_R5_DECISION_LOG="$OUT_ROOT/reports/${label}_decisions.jsonl" \
  STEP_CACHE_R5_SUMMARY="$OUT_ROOT/reports/${label}_runtime_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$OUT_ROOT/configs/${label}.yaml" --checkpoint_path "$CKPT" \
    --data_path "$PROMPTS" --output_folder "$OUT_ROOT/videos/$label" \
    --save_step_cache_tensors_folder "$tensor_dir" \
    --num_output_frames 45 --num_samples 1 --seed 0 --use_ema --save_with_index \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/1-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/2-0_ema.mp4" \
    --expect-decoded-frames 177 --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
}

run_candidate fixed
run_candidate front
run_candidate u_shape

"$PYTHON_BIN" scripts/assess_step_cache_r5a.py \
  --run-root "$OUT_ROOT" --r4-root "$R4_ROOT" \
  --output "$OUT_ROOT/reports/r5a_assessment.json" \
  --csv-output "$OUT_ROOT/reports/r5a_latent_metrics.csv" \
  2>&1 | tee "$OUT_ROOT/logs/r5a_assessment.log"

# The assessor has frozen noise equality against R4.  Keep final latents and
# remove redundant candidate noise tensors from the compact return bundle.
rm -f "$OUT_ROOT"/tensors/*/*_initial_noise.pt

echo "[r5a] completed output=$OUT_ROOT r4_root=$R4_ROOT"
