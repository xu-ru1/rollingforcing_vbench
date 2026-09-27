#!/usr/bin/env bash
# Execute one formal, resumable R8-frozen VBench generation shard. Usage: METHOD START COUNT [--execute]
set -euo pipefail
METHOD="${1:?method is required}"; START="${2:?start is required}"; COUNT="${3:?count is required}"; MODE="${4:---plan-only}"
[ "$MODE" = "--plan-only" ] || [ "$MODE" = "--execute" ] || { echo "[r9][error] mode must be --plan-only or --execute" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"; PYTHON_BIN="${PYTHON_BIN:-python}"; GPU="${GPU:-7}"; CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R8_ROOT="${R8_ROOT:-$RF_ROOT/outputs/step_cache_r8/r8_vbench_plan_20260921}"; FORMAL_ROOT="${FORMAL_ROOT:-$RF_ROOT/outputs/step_cache_formal_vbench_20260922}"
unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die(){ echo "[r9][error] $*" >&2; exit 1; }; [ -f "$R8_ROOT/reports/pre_vbench_immutable_manifest.json" ] || die "missing frozen R8 root: $R8_ROOT"; [ -f "$R8_ROOT/configs/$METHOD.yaml" ] || die "missing frozen config for $METHOD"; [ -f "$CKPT" ] || die "missing checkpoint: $CKPT"
SHARD_DIR="$FORMAL_ROOT/shards/$METHOD/${START}_${COUNT}"; METHOD_ROOT="$FORMAL_ROOT/videos/$METHOD"; STATUS="$SHARD_DIR/status.json"
if [ -f "$STATUS" ]; then "$PYTHON_BIN" - <<PY
import json,sys
x=json.load(open("$STATUS")); sys.exit(0 if x.get("status")=="ok" else 1)
PY
 echo "[r9] already completed method=$METHOD start=$START count=$COUNT"; exit 0; fi
if [ -e "$SHARD_DIR" ]; then
  [ -d "$SHARD_DIR" ] && [ -f "$SHARD_DIR/index.json" ] && [ -f "$SHARD_DIR/prompts.txt" ] || die "incomplete shard directory exists; inspect it before retrying: $SHARD_DIR"
else
  mkdir -p "$SHARD_DIR"
fi
mkdir -p "$METHOD_ROOT/raw" "$METHOD_ROOT/vbench30_indexed" "$METHOD_ROOT/vbench30_standard" "$FORMAL_ROOT/freeze"
for relative in reports/pre_vbench_immutable_manifest.json reports/pre_vbench_immutable_manifest.sha256 configs/$METHOD.yaml prompts/vbench_prompt_index.json plans/generation_plan.json; do
  source="$R8_ROOT/$relative"; target="$FORMAL_ROOT/freeze/${relative//\//__}"
  if [ -e "$target" ]; then
    cmp -s "$source" "$target" || die "frozen R8 artifact changed: $relative"
  elif ln "$source" "$target" 2>/dev/null; then
    :
  else
    temp_target="$target.$$.tmp"; cp -p "$source" "$temp_target"
    if ln "$temp_target" "$target" 2>/dev/null; then
      rm -f -- "$temp_target"
    else
      rm -f -- "$temp_target"
      [ -e "$target" ] && cmp -s "$source" "$target" || die "cannot freeze artifact: $relative"
    fi
  fi
done
if [ ! -f "$SHARD_DIR/index.json" ]; then
  "$PYTHON_BIN" scripts/prepare_step_cache_vbench_shard.py --r8-root "$R8_ROOT" --method "$METHOD" --start "$START" --count "$COUNT" --output-dir "$SHARD_DIR" | tee "$SHARD_DIR/prepare.log"
fi
if [ "$MODE" = "--plan-only" ]; then echo "[r9] plan-only method=$METHOD start=$START count=$COUNT formal_root=$FORMAL_ROOT"; exit 0; fi
TEMP="$SHARD_DIR/generated"; mkdir -p "$TEMP"
STEP_CACHE_RUNTIME_SUMMARY="$SHARD_DIR/runtime_summary.json" STEP_CACHE_FLOPS_SUMMARY="$SHARD_DIR/flops.json" CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference.py --config_path "$R8_ROOT/configs/$METHOD.yaml" --checkpoint_path "$CKPT" --data_path "$SHARD_DIR/prompts.txt" --output_folder "$TEMP" --audit_hash_log "$SHARD_DIR/audit_hashes.jsonl" --num_output_frames 126 --num_samples 1 --seed 0 --use_ema --reset_seed_per_prompt --save_with_index 2>&1 | tee "$SHARD_DIR/inference.log"
"$PYTHON_BIN" scripts/commit_step_cache_vbench_shard.py --shard-index "$SHARD_DIR/index.json" --input-dir "$TEMP" --output-dir "$METHOD_ROOT/raw" --report "$SHARD_DIR/raw_commit.json" | tee "$SHARD_DIR/raw_commit.log"
INPUTS=(); while IFS= read -r file; do INPUTS+=(--input "$file"); done < <("$PYTHON_BIN" - <<PY
import json
for x in json.load(open("$SHARD_DIR/index.json"))["records"]: print("$METHOD_ROOT/raw/"+x["raw_filename"])
PY
)
"$PYTHON_BIN" scripts/crop_step_cache_vbench_30s.py "${INPUTS[@]}" --output-dir "$METHOD_ROOT/vbench30_indexed" --report "$SHARD_DIR/crop_report.json" 2>&1 | tee "$SHARD_DIR/crop.log"
"$PYTHON_BIN" scripts/materialize_step_cache_vbench_standard.py --allow-existing --prompt-index "$SHARD_DIR/index.json" --input-dir "$METHOD_ROOT/vbench30_indexed" --output-dir "$METHOD_ROOT/vbench30_standard" --report "$SHARD_DIR/materialize_report.json" | tee "$SHARD_DIR/materialize.log"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
s=Path("$SHARD_DIR"); records=json.load(open(s/"index.json"))["records"]; crop=json.load(open(s/"crop_report.json")); mat=json.load(open(s/"materialize_report.json")); audit=[json.loads(x) for x in open(s/"audit_hashes.jsonl") if x.strip()]
errors=[]
if len(audit)!=len(records) or [x.get("prompt_idx") for x in audit]!=list(range(len(records))): errors.append("audit")
if crop.get("status")!="ok" or len(crop.get("records",[]))!=len(records) or not all(x.get("pixel_exact") for x in crop.get("records",[])): errors.append("crop")
if mat.get("status")!="ok" or len(mat.get("records",[]))!=len(records) or not all(x.get("byte_identical") for x in mat.get("records",[])): errors.append("materialize")
json.dump({"status":"ok" if not errors else "error","method":"$METHOD","start":int("$START"),"count":int("$COUNT"),"errors":errors},open(s/"status.json","w"),indent=2,sort_keys=True)
if errors: raise SystemExit(2)
PY
rm -rf -- "$TEMP"
echo "[r9] completed formal shard method=$METHOD start=$START count=$COUNT formal_root=$FORMAL_ROOT"
