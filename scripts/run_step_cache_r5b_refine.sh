#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R5B_RUN_ID="${R5B_RUN_ID:-r5b_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r5/$R5B_RUN_ID}"
PROMPTS="$RF_ROOT/prompts/step_cache_r4_three_prompts.txt"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die() { echo "[r5b][error] $*" >&2; exit 1; }

if [ -z "${R5A_ROOT:-}" ]; then
  R5A_ASSESSMENT="$(find "$RF_ROOT/outputs/step_cache_r5" -path '*/reports/r5a_assessment.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [ -n "$R5A_ASSESSMENT" ] || die "R5A assessment not found; set R5A_ROOT"
  R5A_ROOT="$(dirname "$(dirname "$R5A_ASSESSMENT")")"
fi
if [ -z "${R4_ROOT:-}" ]; then
  R4_ASSESSMENT="$(find "$RF_ROOT/outputs/step_cache_r4" -path '*/reports/r4_position_assessment.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [ -n "$R4_ASSESSMENT" ] || die "R4 assessment not found; set R4_ROOT"
  R4_ROOT="$(dirname "$(dirname "$R4_ASSESSMENT")")"
fi

for file in "$RF_ROOT/inference.py" "$PROMPTS" "$CKPT" \
  "$RF_ROOT/scripts/select_step_cache_r5_thresholds.py" \
  "$RF_ROOT/scripts/assess_step_cache_r5b.py" \
  "$RF_ROOT/scripts/inspect_step_cache_r0_videos.py" \
  "$R5A_ROOT/reports/r5a_assessment.json" \
  "$R5A_ROOT/reports/fixed_decisions.jsonl" \
  "$R5A_ROOT/reports/front_decisions.jsonl" \
  "$R5A_ROOT/reports/u_shape_decisions.jsonl" \
  "$R5A_ROOT/configs/fixed.yaml" "$R5A_ROOT/configs/front.yaml" \
  "$R4_ROOT/reports/r4_position_assessment.json"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
[ ! -e "$OUT_ROOT" ] || die "output already exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT"/{configs,logs,reports,tensors,videos}
cd "$RF_ROOT"

SOURCE_FILES=(
  inference.py pipeline/rolling_forcing_inference.py utils/wan_wrapper.py
  utils/step_cache_policy.py utils/step_cache_state.py utils/step_cache_metric.py
  utils/step_cache_runtime.py utils/step_cache_sparse.py wan/modules/causal_model.py
  prompts/step_cache_r4_three_prompts.txt scripts/run_step_cache_r5b_refine.sh
  scripts/select_step_cache_r5_thresholds.py scripts/assess_step_cache_r5b.py
  scripts/inspect_step_cache_r0_videos.py
)
sha256sum "${SOURCE_FILES[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"
sha256sum \
  "$R5A_ROOT/reports/r5a_assessment.json" \
  "$R5A_ROOT/reports/fixed_decisions.jsonl" \
  "$R5A_ROOT/reports/front_decisions.jsonl" \
  "$R5A_ROOT/reports/u_shape_decisions.jsonl" \
  "$R5A_ROOT/configs/fixed.yaml" "$R5A_ROOT/configs/front.yaml" \
  "$R4_ROOT/reports/r4_position_assessment.json" \
  > "$OUT_ROOT/reports/input_manifest.sha256"

"$PYTHON_BIN" -m py_compile scripts/select_step_cache_r5_thresholds.py \
  scripts/assess_step_cache_r5b.py 2>&1 | tee "$OUT_ROOT/logs/py_compile.log"

"$PYTHON_BIN" - "$R5A_ROOT" "$OUT_ROOT/reports/refinement_selection.json" <<'PY'
from pathlib import Path
import json
import sys

from scripts.select_step_cache_r5_thresholds import choose, load_jsonl

r5a_root = Path(sys.argv[1])
output = Path(sys.argv[2])
specs = {
    "fixed": ("fixed", None),
    "front": ("dynamic_threshold", "front_protect"),
}
selected = {}
for name, (policy, schedule) in specs.items():
    rows = load_jsonl(r5a_root / "reports" / f"{name}_decisions.jsonl")
    result = choose(rows, policy=policy, schedule=schedule, target=43,
                    grid_step=0.0005, grid_max=2.0)
    result.update({"policy": policy, "schedule": schedule,
                   "source_trace": str(r5a_root / "reports" / f"{name}_decisions.jsonl")})
    selected[name] = result
payload = {
    "protocol": "r5b_candidate_self_trace_refinement_v1",
    "quality_blind_selection": True,
    "target_reuse_decisions": 43,
    "selected": selected,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({name: {"base_threshold": item["base_threshold"],
                         "predicted_reuse": item["predicted"]["reuse_count"]}
                  for name, item in selected.items()}, sort_keys=True))
PY

STEP_CACHE_R5_DECISION_LOG="$OUT_ROOT/reports/config_probe.jsonl" \
STEP_CACHE_R5_SUMMARY="$OUT_ROOT/reports/config_probe.json" \
"$PYTHON_BIN" - "$R5A_ROOT" "$OUT_ROOT" <<'PY'
from pathlib import Path
import json
import sys
from omegaconf import OmegaConf

r5a_root, out_root = map(Path, sys.argv[1:])
selection = json.loads((out_root / "reports" / "refinement_selection.json").read_text(encoding="utf-8"))
for name in ("fixed", "front"):
    config = OmegaConf.load(r5a_root / "configs" / f"{name}.yaml")
    config.step_cache.base_threshold = float(selection["selected"][name]["base_threshold"])
    config.step_cache.decision_log_path = "${oc.env:STEP_CACHE_R5_DECISION_LOG}"
    config.step_cache.summary_output_path = "${oc.env:STEP_CACHE_R5_SUMMARY}"
    config.step_cache.execution_log_path = None
    OmegaConf.save(config, f=str(out_root / "configs" / f"{name}.yaml"))
print("[r5b] refined-configs-ok")
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

"$PYTHON_BIN" scripts/assess_step_cache_r5b.py \
  --run-root "$OUT_ROOT" --r5a-root "$R5A_ROOT" --r4-root "$R4_ROOT" \
  --output "$OUT_ROOT/reports/r5b_assessment.json" \
  --csv-output "$OUT_ROOT/reports/r5b_latent_metrics.csv" \
  2>&1 | tee "$OUT_ROOT/logs/r5b_assessment.log"

rm -f "$OUT_ROOT"/tensors/*/*_initial_noise.pt
echo "[r5b] completed output=$OUT_ROOT r5a_root=$R5A_ROOT r4_root=$R4_ROOT"
