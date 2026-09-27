#!/usr/bin/env bash
# R8 only freezes and validates plans; it never starts diffusion or VBench scoring.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-$PYTHON_BIN}"
GPU="${GPU:-7}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
VBENCH_ROOT="${VBENCH_ROOT:-}"
VBENCH_METADATA="${VBENCH_METADATA:-$RF_ROOT/third_party/vbench_reference/VBench_full_info.json}"
R7_MANIFEST="${R7_MANIFEST:-$RF_ROOT/outputs/step_cache_r7/r7_formal_smoke_20260921T051807Z/reports/immutable_manifest.json}"
RUN_ID="${R8_RUN_ID:-r8_vbench_plan_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$RF_ROOT/outputs/step_cache_r8/$RUN_ID}"
unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE || true
die(){ echo "[r8][error] $*" >&2; exit 1; }
[ -f "$CKPT" ] || die "checkpoint is missing: $CKPT"
[ -f "$R7_MANIFEST" ] || die "R7 immutable manifest is missing: $R7_MANIFEST"
[ -f "$VBENCH_METADATA" ] || die "VBench metadata is missing: $VBENCH_METADATA"
if [ -n "$VBENCH_ROOT" ]; then
  [ -d "$VBENCH_ROOT" ] || die "VBench checkout is missing: $VBENCH_ROOT"
  [ -f "$VBENCH_ROOT/vbench/VBench_full_info.json" ] || die "missing VBench_full_info.json under: $VBENCH_ROOT"
fi
[ ! -e "$OUT_ROOT" ] || die "output exists: $OUT_ROOT"
mkdir -p "$OUT_ROOT/logs"
cd "$RF_ROOT"
"$PYTHON_BIN" -m py_compile scripts/prepare_step_cache_r8_vbench_plan.py scripts/materialize_step_cache_vbench_standard.py 2>&1 | tee "$OUT_ROOT/logs/py_compile.log"
PLAN_ARGS=(
  --repo-root "$RF_ROOT"
  --r7-manifest "$R7_MANIFEST"
  --checkpoint "$CKPT"
  --vbench-metadata "$VBENCH_METADATA"
  --output-root "$OUT_ROOT"
)
if [ -n "$VBENCH_ROOT" ]; then PLAN_ARGS+=(--vbench-root "$VBENCH_ROOT"); fi
PYTHONPATH="${VBENCH_ROOT:+$VBENCH_ROOT:}${PYTHONPATH:-}" \
  "$VBENCH_PYTHON" scripts/prepare_step_cache_r8_vbench_plan.py "${PLAN_ARGS[@]}" \
  2>&1 | tee "$OUT_ROOT/logs/r8_preflight.log"
echo "[r8] plan-only preflight completed output=$OUT_ROOT gpu_contract=$GPU"
