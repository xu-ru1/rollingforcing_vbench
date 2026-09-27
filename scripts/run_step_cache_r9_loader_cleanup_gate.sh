#!/usr/bin/env bash
# One-prompt exact-equivalence gate for the loader-only CPU memory cleanup.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R8_ROOT="${R8_ROOT:-$RF_ROOT/outputs/step_cache_r8/r8_vbench_plan_20260921}"
FORMAL_ROOT="${FORMAL_ROOT:-$RF_ROOT/outputs/step_cache_formal_vbench_20260922}"
RUN_ID="${R9_LOADER_GATE_RUN_ID:-r9_loader_cleanup_gate_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$RF_ROOT/outputs/step_cache_r9_loader_cleanup/$RUN_ID"
REFERENCE_SHARD="$FORMAL_ROOT/shards/rf_vanilla/8_32"

die() { echo "[r9-loader-gate][error] $*" >&2; exit 2; }
for file in \
  "$RF_ROOT/inference.py" \
  "$RF_ROOT/scripts/verify_step_cache_r9_loader_cleanup.py" \
  "$R8_ROOT/reports/pre_vbench_immutable_manifest.json" \
  "$R8_ROOT/configs/rf_vanilla.yaml" \
  "$REFERENCE_SHARD/prompts.txt" \
  "$REFERENCE_SHARD/audit_hashes.jsonl" \
  "$CKPT"; do
  [ -f "$file" ] || die "required file is missing: $file"
done
command -v /usr/bin/time >/dev/null 2>&1 || die "/usr/bin/time is required"

mkdir -p "$OUT/generated" "$OUT/reports"
head -n 1 "$REFERENCE_SHARD/prompts.txt" > "$OUT/prompt.txt"
sha256sum "$RF_ROOT/inference.py" > "$OUT/reports/inference.sha256"

/usr/bin/time -v -o "$OUT/reports/memory.time.txt" \
  env \
  STEP_CACHE_RUNTIME_SUMMARY="$OUT/reports/runtime_summary.json" \
  STEP_CACHE_FLOPS_SUMMARY="$OUT/reports/flops.json" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" "$RF_ROOT/inference.py" \
    --config_path "$R8_ROOT/configs/rf_vanilla.yaml" \
    --checkpoint_path "$CKPT" \
    --data_path "$OUT/prompt.txt" \
    --output_folder "$OUT/generated" \
    --audit_hash_log "$OUT/reports/candidate_audit_hashes.jsonl" \
    --num_output_frames 126 \
    --num_samples 1 \
    --seed 0 \
    --use_ema \
    --reset_seed_per_prompt \
    --save_with_index \
  2>&1 | tee "$OUT/inference.log"

"$PYTHON_BIN" "$RF_ROOT/scripts/verify_step_cache_r9_loader_cleanup.py" \
  --reference-audit "$REFERENCE_SHARD/audit_hashes.jsonl" \
  --candidate-audit "$OUT/reports/candidate_audit_hashes.jsonl" \
  --r8-manifest "$R8_ROOT/reports/pre_vbench_immutable_manifest.json" \
  --inference "$RF_ROOT/inference.py" \
  --memory-report "$OUT/reports/memory.time.txt" \
  --output "$OUT/reports/source_amendment.json"

sha256sum "$OUT/reports/source_amendment.json" > "$OUT/reports/source_amendment.sha256"
echo "[r9-loader-gate] PASS output=$OUT gpu=$GPU"
