#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R6B_RUN_ID="${R6B_RUN_ID:-r6b_schedule_curve_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r6b/$R6B_RUN_ID}"
PROMPTS="$RF_ROOT/prompts/step_cache_r6a_warmup_and_calibration.txt"
TEMPLATE="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r5a_template.yaml"

FRONT_THRESHOLDS=(0.28 0.34 0.40 0.46)
U_THRESHOLDS=(0.38 0.44 0.50 0.56)
FIXED_SLOW=0.26
FIXED_FAST=0.40

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r6b][error] $*" >&2; exit 1; }

for file in "$RF_ROOT/inference.py" "$PROMPTS" "$TEMPLATE" "$CKPT" \
  "$RF_ROOT/scripts/assess_step_cache_r6b_schedule_curve.py" \
  "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ "$(wc -l < "$PROMPTS")" -eq 4 ] || die "prompt file must contain warmup + three calibration prompts"
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{configs,logs,reports,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  configs/rolling_forcing_dmd_step_cache_r5a_template.yaml
  prompts/step_cache_r6a_warmup_and_calibration.txt
  scripts/run_step_cache_r6b_schedule_curve.sh
  scripts/assess_step_cache_r6b_schedule_curve.py
  scripts/inspect_step_cache_r0_videos.py
  docs/formal_vbench_protocol_freeze_20260919.md
  docs/r6a_assessment_20260920.md
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"

"$PYTHON_BIN" -m py_compile inference.py scripts/assess_step_cache_r6b_schedule_curve.py \
  scripts/inspect_step_cache_r0_videos.py 2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_*.py' -v \
  2>&1 | tee "$OUT_ROOT/logs/cpu_tests.log"

"$PYTHON_BIN" - "$TEMPLATE" "$OUT_ROOT/configs" "$OUT_ROOT/reports/scan_manifest.json" <<'PY'
from pathlib import Path
import json
import sys
from omegaconf import OmegaConf

template_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
template = OmegaConf.load(template_path)
if int(template.num_frame_per_block) != 3:
    raise SystemExit(f"expected num_frame_per_block=3, got {template.num_frame_per_block}")

def common(config):
    config.image_or_video_shape = [1, 126, 16, 60, 104]
    config.data_path = "prompts/step_cache_r6a_warmup_and_calibration.txt"
    config.seed = 0
    config.step_cache.decision_log_path = None
    config.step_cache.execution_log_path = None
    config.step_cache.summary_output_path = None
    config.step_cache.debug_validate_residual_finite = False
    config.step_cache.metric_observer_enabled = False
    return config

def fresh():
    return common(OmegaConf.create(OmegaConf.to_container(template, resolve=False)))

cases = [{"label": "baseline", "kind": "baseline", "threshold": None, "schedule": None}]
baseline = fresh()
baseline.step_cache.enabled = False
OmegaConf.save(baseline, f=str(output_dir / "baseline.yaml"))

fixed = [("fixed_slow", 0.26), ("fixed_fast", 0.40)]
for label, threshold in fixed:
    config = fresh()
    config.step_cache.enabled = True
    config.step_cache.policy = "fixed"
    config.step_cache.schedule = None
    config.step_cache.base_threshold = threshold
    config.step_cache.summary_output_path = "${oc.env:STEP_CACHE_R6B_SUMMARY}"
    OmegaConf.save(config, f=str(output_dir / f"{label}.yaml"))
    cases.append({"label": label, "kind": "fixed", "threshold": threshold, "schedule": None})

for name, schedule, thresholds in (
        ("front", "front_protect", (0.28, 0.34, 0.40, 0.46)),
        ("u_shape", "u_shape_protect", (0.38, 0.44, 0.50, 0.56))):
    for threshold in thresholds:
        label = f"{name}_t{int(round(threshold * 1000)):04d}"
        config = fresh()
        config.step_cache.enabled = True
        config.step_cache.policy = "dynamic_threshold"
        config.step_cache.schedule = schedule
        config.step_cache.base_threshold = threshold
        config.step_cache.summary_output_path = "${oc.env:STEP_CACHE_R6B_SUMMARY}"
        OmegaConf.save(config, f=str(output_dir / f"{label}.yaml"))
        cases.append({"label": label, "kind": name, "threshold": threshold, "schedule": schedule})

manifest_path.write_text(json.dumps({
    "protocol": "r6b_schedule_curve_126_latent_v1",
    "cases": cases,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"cases": cases}, sort_keys=True))
PY

STEP_CACHE_R6B_SUMMARY="$OUT_ROOT/reports/config_probe_runtime_summary.json" \
"$PYTHON_BIN" - "$OUT_ROOT/configs" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

config_dir = Path(sys.argv[1])
paths = sorted(config_dir.glob("*.yaml"))
if len(paths) != 11:
    raise SystemExit(f"expected 11 generated configs, found {len(paths)}")
for path in paths:
    config = OmegaConf.load(path)
    OmegaConf.to_container(config, resolve=True)
    if list(config.image_or_video_shape) != [1, 126, 16, 60, 104]:
        raise SystemExit(f"unexpected shape in {path}")
    if int(config.num_frame_per_block) != 3 or int(config.seed) != 0:
        raise SystemExit(f"protocol mismatch in {path}")
print("[r6b] generated-config-resolution-ok")
PY
sha256sum "$OUT_ROOT"/configs/*.yaml > "$OUT_ROOT/reports/effective_config_manifest.sha256"

run_case() {
  local label="$1"
  mkdir -p "$OUT_ROOT/videos/$label"
  STEP_CACHE_R6B_SUMMARY="$OUT_ROOT/reports/${label}_runtime_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$OUT_ROOT/configs/${label}.yaml" \
    --checkpoint_path "$CKPT" --data_path "$PROMPTS" \
    --output_folder "$OUT_ROOT/videos/$label" \
    --audit_hash_log "$OUT_ROOT/reports/${label}_audit_hashes.jsonl" \
    --num_output_frames 126 --num_samples 1 --seed 0 --use_ema \
    --save_with_index --profile \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    --video "$OUT_ROOT/videos/$label/0-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/1-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/2-0_ema.mp4" \
    --video "$OUT_ROOT/videos/$label/3-0_ema.mp4" \
    --expect-decoded-frames 501 \
    --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
  rm -rf -- "$OUT_ROOT/videos/$label"
}

run_case baseline
run_case fixed_slow
run_case fixed_fast
for threshold in "${FRONT_THRESHOLDS[@]}"; do
  label="front_t$("$PYTHON_BIN" -c 'import sys; print(f"{round(float(sys.argv[1])*1000):04d}")' "$threshold")"
  run_case "$label"
done
for threshold in "${U_THRESHOLDS[@]}"; do
  label="u_shape_t$("$PYTHON_BIN" -c 'import sys; print(f"{round(float(sys.argv[1])*1000):04d}")' "$threshold")"
  run_case "$label"
done

"$PYTHON_BIN" scripts/assess_step_cache_r6b_schedule_curve.py \
  --run-root "$OUT_ROOT" \
  --output "$OUT_ROOT/reports/r6b_schedule_curve_assessment.json" \
  --csv-output "$OUT_ROOT/reports/r6b_schedule_curve.csv" \
  2>&1 | tee "$OUT_ROOT/logs/r6b_assessment.log"

echo "[r6b] completed output=$OUT_ROOT"
