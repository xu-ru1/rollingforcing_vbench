#!/usr/bin/env bash
set -euo pipefail

# Fast three-round decision runner.
# Usage:
#   bash scripts/run_round8_fast_three_rounds.sh round81
#   bash scripts/run_round8_fast_three_rounds.sh round82
#   bash scripts/run_round8_fast_three_rounds.sh round83
#   bash scripts/run_round8_fast_three_rounds.sh summary
#   bash scripts/run_round8_fast_three_rounds.sh all

STAGE="${1:-summary}"
RF_ROOT="${RF_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-0}"
GPU_3PROMPT="${GPU_3PROMPT:-$GPU}"
CKPT="${CKPT:-checkpoints/rolling_forcing_dmd.pt}"
MIN_SPEEDUP="${MIN_SPEEDUP:-0.05}"
if [ -z "${EMA_FLAG+x}" ]; then
  EMA_FLAG="--use_ema"
fi

CONFIG_BASELINE="configs/rolling_forcing_dmd.yaml"
CONFIG_FAST="configs/rolling_forcing_dmd_flowcache_round8_2_fast_compacted.yaml"
PROMPT_SINGLE="logs/round8_fast_single_prompt.txt"
PROMPT_3PROMPT="logs/round8_fast_3prompts.txt"

die() {
  echo "[round8-fast][error] $*" >&2
  exit 1
}

ensure_file() {
  local path="$1"
  [ -f "$path" ] || die "missing required file: $path"
}

run_header() {
  local stage_name="$1"
  local gpu_id="$2"
  echo "[round8-fast] stage=$stage_name"
  echo "[round8-fast] GPU=$gpu_id"
  echo "[round8-fast] CKPT=$CKPT"
  echo "[round8-fast] EMA_FLAG=${EMA_FLAG:-<empty>}"
  echo "[round8-fast] MIN_SPEEDUP=$MIN_SPEEDUP"
}

prepare() {
  cd "$RF_ROOT"
  mkdir -p logs videos

  cat > "$PROMPT_SINGLE" <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
EOF

  cat > "$PROMPT_3PROMPT" <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
A futuristic city street at night, neon lights, slow camera pan.
A small dog running through a flower field, bright daylight, smooth motion.
EOF

  ensure_file inference.py
  ensure_file "$CONFIG_BASELINE"
  ensure_file "$CONFIG_FAST"
  ensure_file "$CKPT"
  ensure_file "$PROMPT_SINGLE"
  ensure_file "$PROMPT_3PROMPT"
}

compile_check() {
  python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py scripts/summarize_round8_fast_three_rounds.py 2>&1 | tee logs/round8_fast_py_compile.log
}

check_log() {
  local log_path="$1"
  ensure_file "$log_path"
  grep -Eq "\[Inference\] Loaded checkpoint weights|state_dict = torch.load\(args.checkpoint_path" "$log_path" || die "checkpoint load confirmation missing in $log_path"
  if grep -Eq "Traceback|RuntimeError|CUDA out of memory|OOM" "$log_path"; then
    die "error pattern found in $log_path"
  fi
  grep -n "\[EvalMetrics\]" "$log_path" || die "EvalMetrics missing in $log_path"
}

run_baseline_81() {
  {
    run_header "round8_1_81_baseline" "$GPU"
    FLOWCACHE_ATTENTION_PROFILER_ENABLED=false CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_BASELINE" --checkpoint_path "$CKPT" --data_path "$PROMPT_SINGLE" --output_folder videos/round8_1_81_baseline --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee logs/round8_1_81_baseline.log
  check_log logs/round8_1_81_baseline.log
  ls -lh videos/round8_1_81_baseline | tee logs/round8_1_81_baseline_ls.txt
}

run_candidate_81() {
  local name="$1"
  local apply_clean="$2"
  local output_folder="videos/round8_2_81_ratio025_${name}"
  local log_path="logs/round8_2_81_ratio025_${name}.log"
  {
    run_header "round8_2_81_ratio025_${name}" "$GPU"
    echo "[round8-fast] FLOWCACHE_COMPACTED_RATIO=0.25"
    echo "[round8-fast] FLOWCACHE_COMPACTED_CLEAN=$apply_clean"
    FLOWCACHE_COMPACTED_RATIO=0.25 FLOWCACHE_COMPACTED_CLEAN="$apply_clean" FLOWCACHE_FAST_JSONL=null CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_FAST" --checkpoint_path "$CKPT" --data_path "$PROMPT_SINGLE" --output_folder "$output_folder" --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee "$log_path"
  check_log "$log_path"
  grep -n "\[FlowCache\]\[compacted_kv_summary\]" "$log_path" || die "compacted_kv_summary missing in $log_path"
  ls -lh "$output_folder" | tee "logs/round8_2_81_ratio025_${name}_ls.txt"
}

run_baseline_3prompt() {
  {
    run_header "round8_3_81_3prompt_baseline" "$GPU_3PROMPT"
    FLOWCACHE_ATTENTION_PROFILER_ENABLED=false CUDA_VISIBLE_DEVICES="$GPU_3PROMPT" python inference.py --config_path "$CONFIG_BASELINE" --checkpoint_path "$CKPT" --data_path "$PROMPT_3PROMPT" --output_folder videos/round8_3_81_3prompt_baseline --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee logs/round8_3_81_3prompt_baseline.log
  check_log logs/round8_3_81_3prompt_baseline.log
  ls -lh videos/round8_3_81_3prompt_baseline | tee logs/round8_3_81_3prompt_baseline_ls.txt
}

run_candidate_3prompt() {
  local name="$1"
  local apply_clean="$2"
  local output_folder="videos/round8_3_81_3prompt_ratio025_${name}"
  local log_path="logs/round8_3_81_3prompt_ratio025_${name}.log"
  {
    run_header "round8_3_81_3prompt_ratio025_${name}" "$GPU_3PROMPT"
    echo "[round8-fast] FLOWCACHE_COMPACTED_RATIO=0.25"
    echo "[round8-fast] FLOWCACHE_COMPACTED_CLEAN=$apply_clean"
    FLOWCACHE_COMPACTED_RATIO=0.25 FLOWCACHE_COMPACTED_CLEAN="$apply_clean" FLOWCACHE_FAST_JSONL=null CUDA_VISIBLE_DEVICES="$GPU_3PROMPT" python inference.py --config_path "$CONFIG_FAST" --checkpoint_path "$CKPT" --data_path "$PROMPT_3PROMPT" --output_folder "$output_folder" --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee "$log_path"
  check_log "$log_path"
  grep -n "\[FlowCache\]\[compacted_kv_summary\]" "$log_path" || die "compacted_kv_summary missing in $log_path"
  ls -lh "$output_folder" | tee "logs/round8_3_81_3prompt_ratio025_${name}_ls.txt"
}

summary() {
  python scripts/summarize_round8_fast_three_rounds.py --min-speedup "$MIN_SPEEDUP" 2>&1 | tee logs/round8_fast_three_rounds_report_stdout.log
  cat logs/round8_fast_three_rounds_report.txt
}

round81() {
  compile_check
  run_baseline_81
  summary
}

round82() {
  run_candidate_81 "denoise" "false"
  run_candidate_81 "clean" "true"
  summary
}

round83() {
  summary
  local best
  best="$(python scripts/summarize_round8_fast_three_rounds.py --min-speedup "$MIN_SPEEDUP" --print-best | tail -n 1)"
  if [ "$best" = "none" ]; then
    die "no 81-frame candidate reached MIN_SPEEDUP=$MIN_SPEEDUP; stop here and use the report as the conclusion"
  fi
  run_baseline_3prompt
  if [ "$best" = "clean" ]; then
    run_candidate_3prompt "clean" "true"
  elif [ "$best" = "denoise" ]; then
    run_candidate_3prompt "denoise" "false"
  else
    die "unknown best candidate: $best"
  fi
  summary
}

prepare

case "$STAGE" in
  round81|81)
    round81
    ;;
  round82|82)
    round82
    ;;
  round83|83)
    round83
    ;;
  summary)
    summary
    ;;
  all)
    round81
    round82
    round83
    ;;
  *)
    die "unknown stage '$STAGE'; use round81, round82, round83, summary, or all"
    ;;
esac
