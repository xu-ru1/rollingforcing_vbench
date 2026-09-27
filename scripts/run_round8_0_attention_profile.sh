#!/usr/bin/env bash
set -euo pipefail

# Copy-safe Round 8.0 server runner.
# Usage:
#   bash scripts/run_round8_0_attention_profile.sh smoke
#   bash scripts/run_round8_0_attention_profile.sh 81
#   bash scripts/run_round8_0_attention_profile.sh 3prompt
#   bash scripts/run_round8_0_attention_profile.sh summary
#
# Optional environment overrides:
#   GPU=0 CKPT=checkpoints/rolling_forcing_dmd.pt EMA_FLAG=--use_ema bash scripts/run_round8_0_attention_profile.sh smoke

STAGE="${1:-smoke}"
RF_ROOT="${RF_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-0}"
GPU_3PROMPT="${GPU_3PROMPT:-$GPU}"
CKPT="${CKPT:-checkpoints/rolling_forcing_dmd.pt}"
if [ -z "${EMA_FLAG+x}" ]; then
  EMA_FLAG="--use_ema"
fi

CONFIG_DISABLED="configs/rolling_forcing_dmd.yaml"
CONFIG_ATTENTION="configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml"
PROMPT_SINGLE="logs/round8_0_single_prompt.txt"
PROMPT_3PROMPT="logs/round8_0_3prompts.txt"

die() {
  echo "[round8-runner][error] $*" >&2
  exit 1
}

ensure_file() {
  local path="$1"
  [ -f "$path" ] || die "missing required file: $path"
}

run_header() {
  local stage_name="$1"
  local gpu_id="$2"
  echo "[round8-runner] stage=$stage_name"
  echo "[round8-runner] GPU=$gpu_id"
  echo "[round8-runner] CKPT=$CKPT"
  echo "[round8-runner] EMA_FLAG=${EMA_FLAG:-<empty>}"
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
  ensure_file "$CONFIG_DISABLED"
  ensure_file "$CONFIG_ATTENTION"
  ensure_file "$CKPT"
  ensure_file "$PROMPT_SINGLE"
  ensure_file "$PROMPT_3PROMPT"
}

compile_check() {
  python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py scripts/summarize_round8_0_attention_profile.py 2>&1 | tee logs/round8_0_py_compile.log
}

disabled_smoke() {
  {
    run_header "disabled_smoke" "$GPU"
    FLOWCACHE_ATTENTION_PROFILER_ENABLED=false CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_DISABLED" --checkpoint_path "$CKPT" --data_path "$PROMPT_SINGLE" --output_folder videos/round8_0_disabled_smoke --num_output_frames 21 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee logs/round8_0_disabled_smoke.log
}

attention_21() {
  : > logs/round8_0_21_attention.jsonl
  {
    run_header "attention_21" "$GPU"
    FLOWCACHE_ATTENTION_PROFILER_OUTPUT_PATH=logs/round8_0_21_attention.jsonl FLOWCACHE_ATTENTION_PROFILER_SUMMARY_PATH=logs/round8_0_21_attention_summary.json CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_ATTENTION" --checkpoint_path "$CKPT" --data_path "$PROMPT_SINGLE" --output_folder videos/round8_0_21_attention --num_output_frames 21 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee logs/round8_0_21_attention.log
}

attention_81_single() {
  : > logs/round8_0_81_single_attention.jsonl
  {
    run_header "attention_81_single" "$GPU"
    FLOWCACHE_ATTENTION_PROFILER_OUTPUT_PATH=logs/round8_0_81_single_attention.jsonl FLOWCACHE_ATTENTION_PROFILER_SUMMARY_PATH=logs/round8_0_81_single_attention_summary.json CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_ATTENTION" --checkpoint_path "$CKPT" --data_path "$PROMPT_SINGLE" --output_folder videos/round8_0_81_single_attention --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee logs/round8_0_81_single_attention.log
}

attention_81_3prompt() {
  : > logs/round8_0_81_3prompt_attention.jsonl
  {
    run_header "attention_81_3prompt" "$GPU_3PROMPT"
    FLOWCACHE_ATTENTION_PROFILER_OUTPUT_PATH=logs/round8_0_81_3prompt_attention.jsonl FLOWCACHE_ATTENTION_PROFILER_SUMMARY_PATH=logs/round8_0_81_3prompt_attention_summary.json CUDA_VISIBLE_DEVICES="$GPU_3PROMPT" python inference.py --config_path "$CONFIG_ATTENTION" --checkpoint_path "$CKPT" --data_path "$PROMPT_3PROMPT" --output_folder videos/round8_0_81_3prompt_attention --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee logs/round8_0_81_3prompt_attention.log
}

check_smoke_outputs() {
  grep -n "\[Inference\] Loaded checkpoint weights" logs/round8_0_disabled_smoke.log || die "checkpoint load confirmation missing in logs/round8_0_disabled_smoke.log"
  grep -n "\[Inference\] Loaded checkpoint weights" logs/round8_0_21_attention.log || die "checkpoint load confirmation missing in logs/round8_0_21_attention.log"
  grep -n "\[FlowCache\]" logs/round8_0_disabled_smoke.log | head -n 40 || true
  grep -n "\[FlowCache\]\[attention_profiler_summary\]" logs/round8_0_21_attention.log || true
  grep -n "\[FlowCache\]\[warning\]" logs/round8_0_21_attention.log | head -n 40 || true
  grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round8_0_disabled_smoke.log logs/round8_0_21_attention.log || true
  ls -lh videos/round8_0_disabled_smoke | tee logs/round8_0_disabled_smoke_ls.txt
  ls -lh videos/round8_0_21_attention | tee logs/round8_0_21_attention_ls.txt
  python - <<'PY' | tee logs/round8_0_21_attention_jsonl_check.txt
import json
path = "logs/round8_0_21_attention.jsonl"
total = parse_errors = tensor_like = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        if "tensor(" in line or "Tensor" in line or "<tensor_like" in line:
            tensor_like += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
print({"path": path, "total": total, "parse_errors": parse_errors, "tensor_like": tensor_like})
PY
}

summary() {
  python scripts/summarize_round8_0_attention_profile.py 2>&1 | tee logs/round8_0_attention_report_stdout.log
}

prepare

case "$STAGE" in
  smoke)
    compile_check
    disabled_smoke
    attention_21
    check_smoke_outputs
    ;;
  81)
    attention_81_single
    summary
    ;;
  3prompt)
    attention_81_3prompt
    summary
    ;;
  summary)
    summary
    ;;
  all)
    compile_check
    disabled_smoke
    attention_21
    check_smoke_outputs
    attention_81_single
    summary
    ;;
  *)
    die "unknown stage '$STAGE'; use smoke, 81, 3prompt, summary, or all"
    ;;
esac
