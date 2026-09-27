#!/usr/bin/env bash
# Run one formal prompt shard for all frozen methods on GPUs 2..7.
set -euo pipefail
START="${1:?start is required}"; COUNT="${2:?count is required}"; MODE="${3:---plan-only}"
[ "$MODE" = "--plan-only" ] || [ "$MODE" = "--execute" ] || { echo "[r10][error] mode must be --plan-only or --execute" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"; PYTHON_BIN="${PYTHON_BIN:-python}"; CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R8_ROOT="${R8_ROOT:-$RF_ROOT/outputs/step_cache_r8/r8_vbench_plan_20260921}"; FORMAL_ROOT="${FORMAL_ROOT:-$RF_ROOT/outputs/step_cache_formal_vbench_20260922}"
METHODS=(rf_vanilla rf_fixed_slow rf_front_slow rf_fixed_fast rf_front_fast rf_u_shape_fast)
GPUS=(2 3 4 5 6 7)
mkdir -p "$FORMAL_ROOT/wave_logs"
pids=(); labels=()
for i in "${!METHODS[@]}"; do
  method="${METHODS[$i]}"; gpu="${GPUS[$i]}"
  (
    RF_ROOT="$RF_ROOT" PYTHON_BIN="$PYTHON_BIN" GPU="$gpu" CKPT="$CKPT" R8_ROOT="$R8_ROOT" FORMAL_ROOT="$FORMAL_ROOT" \
      bash "$SCRIPT_DIR/run_step_cache_r9_formal_generation_shard.sh" "$method" "$START" "$COUNT" "$MODE"
  ) >"$FORMAL_ROOT/wave_logs/${method}_${START}_${COUNT}.log" 2>&1 &
  pids+=("$!"); labels+=("$method@gpu$gpu")
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[r10] completed ${labels[$i]}"
  else
    echo "[r10][error] failed ${labels[$i]}; see $FORMAL_ROOT/wave_logs/${METHODS[$i]}_${START}_${COUNT}.log" >&2
    failed=1
  fi
done
[ "$failed" -eq 0 ] || exit 1
echo "[r10] wave completed start=$START count=$COUNT mode=$MODE"
