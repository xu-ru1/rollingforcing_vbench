#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R6D_RUN_ID="${R6D_RUN_ID:-r6d_final_timing_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r6d/$R6D_RUN_ID}"
PROMPTS="$RF_ROOT/prompts/step_cache_r6d_warmup_and_repeats.txt"
TEMPLATE="$RF_ROOT/configs/rolling_forcing_dmd_step_cache_r5a_template.yaml"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r6d][error] $*" >&2; exit 1; }

for file in "$RF_ROOT/inference.py" "$PROMPTS" "$TEMPLATE" "$CKPT" \
  "$RF_ROOT/scripts/assess_step_cache_r6d_final_timing.py" \
  "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ "$(wc -l < "$PROMPTS")" -eq 10 ] || die "prompt file must contain one warmup and nine timed prompts"
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{configs,logs,reports,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  configs/rolling_forcing_dmd_step_cache_r5a_template.yaml
  prompts/step_cache_r6d_warmup_and_repeats.txt
  scripts/run_step_cache_r6d_final_timing.sh scripts/assess_step_cache_r6d_final_timing.py
  scripts/inspect_step_cache_r0_videos.py docs/r6c_assessment_20260920.md
  docs/r6d_final_timing_protocol_20260920.md
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"

"$PYTHON_BIN" -m py_compile inference.py scripts/assess_step_cache_r6d_final_timing.py \
  scripts/inspect_step_cache_r0_videos.py 2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_*.py' -v \
  2>&1 | tee "$OUT_ROOT/logs/cpu_tests.log"

"$PYTHON_BIN" - "$TEMPLATE" "$OUT_ROOT/configs" "$OUT_ROOT/reports/scan_manifest.json" <<'PY'
from pathlib import Path
import json
import sys
from omegaconf import OmegaConf

template_path, output_dir, manifest_path = map(Path, sys.argv[1:])
template = OmegaConf.load(template_path)
if int(template.num_frame_per_block) != 3:
    raise SystemExit(f"expected num_frame_per_block=3, got {template.num_frame_per_block}")

def fresh():
    config = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
    config.image_or_video_shape = [1, 126, 16, 60, 104]
    config.data_path = "prompts/step_cache_r6d_warmup_and_repeats.txt"
    config.seed = 0
    config.step_cache.decision_log_path = None
    config.step_cache.execution_log_path = None
    config.step_cache.summary_output_path = "${oc.env:STEP_CACHE_R6D_SUMMARY}"
    config.step_cache.debug_validate_residual_finite = False
    config.step_cache.metric_observer_enabled = False
    return config

specs = [
    ("vanilla", False, "fixed", None, 0.0),
    ("fixed_slow", True, "fixed", None, 0.26),
    ("front_slow", True, "dynamic_threshold", "front_protect", 0.24),
    ("fixed_fast", True, "fixed", None, 0.40),
    ("front_fast", True, "dynamic_threshold", "front_protect", 0.46),
    ("u_shape_fast", True, "dynamic_threshold", "u_shape_protect", 0.58),
]
cases = []
for label, enabled, policy, schedule, threshold in specs:
    config = fresh()
    config.step_cache.enabled = enabled
    config.step_cache.policy = policy
    config.step_cache.schedule = schedule
    config.step_cache.base_threshold = threshold
    OmegaConf.save(config, f=str(output_dir / f"{label}.yaml"))
    cases.append({
        "label": label, "enabled": enabled, "policy": policy,
        "schedule": schedule, "threshold": threshold,
    })
manifest_path.write_text(json.dumps({
    "protocol": "r6d_final_timing_126_latent_v1", "cases": cases,
    "warmup_prompt_index": 0,
    "timed_prompt_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "prompt_repeats": {"cat": [1, 4, 7], "rainy_car": [2, 5, 8], "robot": [3, 6, 9]},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"cases": cases}, sort_keys=True))
PY

STEP_CACHE_R6D_SUMMARY="$OUT_ROOT/reports/config_probe_runtime_summary.json" \
"$PYTHON_BIN" - "$OUT_ROOT/configs" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

paths = sorted(Path(sys.argv[1]).glob("*.yaml"))
if len(paths) != 6:
    raise SystemExit(f"expected six generated configs, found {len(paths)}")
for path in paths:
    config = OmegaConf.load(path)
    OmegaConf.to_container(config, resolve=True)
    if list(config.image_or_video_shape) != [1, 126, 16, 60, 104]:
        raise SystemExit(f"unexpected shape in {path}")
    if int(config.num_frame_per_block) != 3 or int(config.seed) != 0:
        raise SystemExit(f"protocol mismatch in {path}")
print("[r6d] generated-config-resolution-ok")
PY
sha256sum "$OUT_ROOT"/configs/*.yaml > "$OUT_ROOT/reports/effective_config_manifest.sha256"

run_case() {
  local label="$1"
  local video_args=()
  local index
  mkdir -p "$OUT_ROOT/videos/$label"
  STEP_CACHE_R6D_SUMMARY="$OUT_ROOT/reports/${label}_runtime_summary.json" \
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py \
    --config_path "$OUT_ROOT/configs/${label}.yaml" \
    --checkpoint_path "$CKPT" --data_path "$PROMPTS" \
    --output_folder "$OUT_ROOT/videos/$label" \
    --audit_hash_log "$OUT_ROOT/reports/${label}_audit_hashes.jsonl" \
    --num_output_frames 126 --num_samples 1 --seed 0 --use_ema \
    --reset_seed_per_prompt --save_with_index --profile \
    2>&1 | tee "$OUT_ROOT/logs/${label}.log"
  for index in {0..9}; do
    video_args+=(--video "$OUT_ROOT/videos/$label/${index}-0_ema.mp4")
  done
  "$PYTHON_BIN" scripts/inspect_step_cache_r0_videos.py \
    "${video_args[@]}" --expect-decoded-frames 501 \
    --output "$OUT_ROOT/reports/${label}_video_check.json" \
    2>&1 | tee "$OUT_ROOT/logs/${label}_video_check.log"
  rm -rf -- "$OUT_ROOT/videos/$label"
}

run_case vanilla
run_case fixed_slow
run_case front_slow
run_case fixed_fast
run_case front_fast
run_case u_shape_fast

"$PYTHON_BIN" scripts/assess_step_cache_r6d_final_timing.py \
  --run-root "$OUT_ROOT" \
  --output "$OUT_ROOT/reports/r6d_final_timing_assessment.json" \
  --csv-output "$OUT_ROOT/reports/r6d_final_timing.csv" \
  2>&1 | tee "$OUT_ROOT/logs/r6d_final_timing_assessment.log"

echo "[r6d] completed output=$OUT_ROOT"
