#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"; PYTHON_BIN="${PYTHON_BIN:-python}"; GPU="${GPU:-7}"; CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
RUN_ID="${R7_RUN_ID:-r7_formal_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"; OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r7/$RUN_ID}"; PROMPTS="$RF_ROOT/prompts/step_cache_r4_three_prompts.txt"
unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die(){ echo "[r7][error] $*" >&2; exit 1; }
for f in inference.py "$CKPT" "$PROMPTS" utils/step_cache_flops.py scripts/build_step_cache_formal_manifest.py scripts/crop_step_cache_vbench_30s.py scripts/assess_step_cache_r7_formal_smoke.py; do [ -f "$RF_ROOT/$f" ] || [ -f "$f" ] || die "missing $f"; done
[ ! -e "$OUT_ROOT" ] || die "output exists: $OUT_ROOT"; mkdir -p "$OUT_ROOT"/{logs,reports,videos}; cd "$RF_ROOT"
SOURCE=(inference.py pipeline/rolling_forcing_inference.py utils/step_cache_flops.py utils/step_cache_runtime.py utils/step_cache_sparse.py utils/wan_wrapper.py wan/modules/attention.py wan/modules/model.py wan/modules/causal_model.py configs/default_config.yaml configs/rolling_forcing_dmd_step_cache_r5a_template.yaml prompts/step_cache_r4_three_prompts.txt scripts/build_step_cache_formal_manifest.py scripts/crop_step_cache_vbench_30s.py scripts/assess_step_cache_r7_formal_smoke.py scripts/run_step_cache_r7_formal_smoke.sh)
sha256sum "${SOURCE[@]}" > "$OUT_ROOT/reports/source_manifest.sha256"
"$PYTHON_BIN" -m py_compile inference.py utils/step_cache_flops.py scripts/build_step_cache_formal_manifest.py scripts/crop_step_cache_vbench_30s.py scripts/assess_step_cache_r7_formal_smoke.py 2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_step_cache_*.py' -v 2>&1 | tee "$OUT_ROOT/logs/cpu_tests.log"
"$PYTHON_BIN" scripts/build_step_cache_formal_manifest.py --repo-root "$RF_ROOT" --output-root "$OUT_ROOT/reports" --checkpoint "$CKPT" 2>&1 | tee "$OUT_ROOT/logs/build_manifest.log"
sha256sum "$OUT_ROOT/reports"/configs/*.yaml > "$OUT_ROOT/reports/effective_config_manifest.sha256"
run_case(){ local slug="$1"; mkdir -p "$OUT_ROOT/videos/$slug"; STEP_CACHE_RUNTIME_SUMMARY="$OUT_ROOT/reports/${slug}_runtime_summary.json" STEP_CACHE_FLOPS_SUMMARY="$OUT_ROOT/reports/${slug}_flops.json" CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py --config_path "$OUT_ROOT/reports/configs/${slug}.yaml" --checkpoint_path "$CKPT" --data_path "$PROMPTS" --output_folder "$OUT_ROOT/videos/$slug" --audit_hash_log "$OUT_ROOT/reports/${slug}_audit_hashes.jsonl" --num_output_frames 126 --num_samples 1 --seed 0 --use_ema --reset_seed_per_prompt --save_with_index 2>&1 | tee "$OUT_ROOT/logs/${slug}.log"; "$PYTHON_BIN" scripts/crop_step_cache_vbench_30s.py --input "$OUT_ROOT/videos/$slug/0-0_ema.mp4" --input "$OUT_ROOT/videos/$slug/1-0_ema.mp4" --input "$OUT_ROOT/videos/$slug/2-0_ema.mp4" --output-dir "$OUT_ROOT/videos/${slug}_vbench30" --report "$OUT_ROOT/reports/${slug}_crop_report.json" 2>&1 | tee "$OUT_ROOT/logs/${slug}_crop.log"; }
for slug in rf_vanilla rf_fixed_slow rf_front_slow rf_fixed_fast rf_front_fast rf_u_shape_fast; do run_case "$slug"; done
"$PYTHON_BIN" scripts/assess_step_cache_r7_formal_smoke.py --run-root "$OUT_ROOT" --output "$OUT_ROOT/reports/r7_formal_smoke_assessment.json" 2>&1 | tee "$OUT_ROOT/logs/r7_assessment.log"
rm -rf -- "$OUT_ROOT/videos"
echo "[r7] completed output=$OUT_ROOT"
