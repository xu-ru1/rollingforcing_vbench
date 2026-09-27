#!/usr/bin/env bash
set -euo pipefail

# Round 8.4 focused confirmation for the current winning candidate:
# ratio025_clean on 81-frame 3-prompt.
#
# Usage:
#   GPU=1 CKPT=checkpoints/rolling_forcing_dmd.pt EMA_FLAG=--use_ema bash scripts/run_round8_4_clean_repeat.sh all
#   bash scripts/run_round8_4_clean_repeat.sh summary

STAGE="${1:-all}"
RF_ROOT="${RF_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-0}"
CKPT="${CKPT:-checkpoints/rolling_forcing_dmd.pt}"
if [ -z "${EMA_FLAG+x}" ]; then
  EMA_FLAG="--use_ema"
fi

CONFIG_BASELINE="configs/rolling_forcing_dmd.yaml"
CONFIG_CLEAN="configs/rolling_forcing_dmd_flowcache_round8_2_fast_compacted.yaml"
PROMPT_3PROMPT="logs/round8_4_3prompts.txt"

BASELINE_LOG="logs/round8_4_81_3prompt_baseline_repeat.log"
CLEAN_LOG="logs/round8_4_81_3prompt_ratio025_clean_repeat.log"
REPORT_TXT="logs/round8_4_clean_repeat_report.txt"
REPORT_JSON="logs/round8_4_clean_repeat_report.json"

die() {
  echo "[round8.4][error] $*" >&2
  exit 1
}

ensure_file() {
  local path="$1"
  [ -f "$path" ] || die "missing required file: $path"
}

prepare() {
  cd "$RF_ROOT"
  mkdir -p logs videos

  cat > "$PROMPT_3PROMPT" <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
A futuristic city street at night, neon lights, slow camera pan.
A small dog running through a flower field, bright daylight, smooth motion.
EOF

  ensure_file inference.py
  ensure_file "$CONFIG_BASELINE"
  ensure_file "$CONFIG_CLEAN"
  ensure_file "$CKPT"
  ensure_file "$PROMPT_3PROMPT"
}

run_header() {
  local stage_name="$1"
  echo "[round8.4] stage=$stage_name"
  echo "[round8.4] GPU=$GPU"
  echo "[round8.4] CKPT=$CKPT"
  echo "[round8.4] EMA_FLAG=${EMA_FLAG:-<empty>}"
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

compile_check() {
  python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py 2>&1 | tee logs/round8_4_py_compile.log
}

baseline_repeat() {
  {
    run_header "baseline_repeat"
    FLOWCACHE_ATTENTION_PROFILER_ENABLED=false CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_BASELINE" --checkpoint_path "$CKPT" --data_path "$PROMPT_3PROMPT" --output_folder videos/round8_4_81_3prompt_baseline_repeat --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee "$BASELINE_LOG"
  check_log "$BASELINE_LOG"
  ls -lh videos/round8_4_81_3prompt_baseline_repeat | tee logs/round8_4_81_3prompt_baseline_repeat_ls.txt
}

clean_repeat() {
  {
    run_header "ratio025_clean_repeat"
    FLOWCACHE_COMPACTED_RATIO=0.25 FLOWCACHE_COMPACTED_CLEAN=true FLOWCACHE_FAST_JSONL=null CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path "$CONFIG_CLEAN" --checkpoint_path "$CKPT" --data_path "$PROMPT_3PROMPT" --output_folder videos/round8_4_81_3prompt_ratio025_clean_repeat --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics
  } 2>&1 | tee "$CLEAN_LOG"
  check_log "$CLEAN_LOG"
  grep -n "\[FlowCache\]\[compacted_kv_summary\]" "$CLEAN_LOG" || die "compacted_kv_summary missing in $CLEAN_LOG"
  ls -lh videos/round8_4_81_3prompt_ratio025_clean_repeat | tee logs/round8_4_81_3prompt_ratio025_clean_repeat_ls.txt
}

summary() {
  python - <<'PY'
import json
import re
from pathlib import Path

logs = {
    "baseline": Path("logs/round8_4_81_3prompt_baseline_repeat.log"),
    "ratio025_clean": Path("logs/round8_4_81_3prompt_ratio025_clean_repeat.log"),
}
eval_re = re.compile(
    r"\[EvalMetrics\] runtime_sec=([0-9.]+) "
    r"peak_cuda_allocated_gb=([0-9.]+) "
    r"peak_cuda_reserved_gb=([0-9.]+) "
    r"saved_videos=(\d+) output_folder=(\S+)"
)
kv_re = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def parse(log_path):
    result = {"exists": log_path.exists(), "path": str(log_path)}
    if not log_path.exists():
        return result
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result["checkpoint_loaded"] = (
        "[Inference] Loaded checkpoint weights" in text or
        "state_dict = torch.load(args.checkpoint_path" in text
    )
    result["error_lines"] = sum(
        text.count(pattern)
        for pattern in ["Traceback", "RuntimeError", "CUDA out of memory", "OOM"]
    )
    result["flowcache_warning_lines"] = text.count("[FlowCache][warning]")
    for match in eval_re.finditer(text):
        result.update({
            "runtime_sec": float(match.group(1)),
            "peak_cuda_allocated_gb": float(match.group(2)),
            "peak_cuda_reserved_gb": float(match.group(3)),
            "saved_videos": int(match.group(4)),
            "output_folder": match.group(5),
        })
    summaries = [
        line for line in text.splitlines()
        if "[FlowCache][compacted_kv_summary]" in line
    ]
    if summaries:
        summary = {}
        for key, raw in kv_re.findall(summaries[-1]):
            value = raw.rstrip(",")
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value == "none":
                        value = None
            summary[key] = value
        result["compacted_summary"] = summary
    result["valid"] = bool(
        result.get("runtime_sec") and
        result.get("checkpoint_loaded") and
        result.get("error_lines") == 0 and
        result.get("saved_videos", 0) == 3
    )
    return result

report = {name: parse(path) for name, path in logs.items()}
baseline = report["baseline"].get("runtime_sec")
clean = report["ratio025_clean"].get("runtime_sec")
if baseline and clean:
    report["speedup_ratio"] = (baseline - clean) / baseline
    report["runtime_delta_sec"] = clean - baseline
    report["passes_15pct_gate"] = report["speedup_ratio"] >= 0.15
    report["passes_5pct_gate"] = report["speedup_ratio"] >= 0.05
else:
    report["speedup_ratio"] = None
    report["runtime_delta_sec"] = None
    report["passes_15pct_gate"] = False
    report["passes_5pct_gate"] = False

Path("logs/round8_4_clean_repeat_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = ["Round 8.4 clean repeat report", ""]
for name in ["baseline", "ratio025_clean"]:
    row = report[name]
    lines.append(
        f"{name}: exists={row.get('exists')} valid={row.get('valid')} "
        f"runtime={row.get('runtime_sec')} saved={row.get('saved_videos')} "
        f"checkpoint={row.get('checkpoint_loaded')} errors={row.get('error_lines')} "
        f"warnings={row.get('flowcache_warning_lines')}"
    )
    summary = row.get("compacted_summary") or {}
    if summary:
        lines.append(
            "  compacted: "
            f"applied={summary.get('applied_events')} "
            f"clean_applied={summary.get('clean_cache_update_applied')} "
            f"overall_saving={summary.get('overall_weighted_visible_saving_ratio')} "
            f"fallback={summary.get('fallback_count')} warning={summary.get('warning_count')}"
        )
lines.append("")
lines.append(f"speedup_ratio={report['speedup_ratio']}")
lines.append(f"runtime_delta_sec={report['runtime_delta_sec']}")
lines.append(f"passes_15pct_gate={report['passes_15pct_gate']}")
lines.append(f"passes_5pct_gate={report['passes_5pct_gate']}")
Path("logs/round8_4_clean_repeat_report.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
print("\n".join(lines))
PY
}

prepare

case "$STAGE" in
  baseline)
    compile_check
    baseline_repeat
    summary
    ;;
  clean)
    clean_repeat
    summary
    ;;
  summary)
    summary
    ;;
  all)
    compile_check
    baseline_repeat
    clean_repeat
    summary
    ;;
  *)
    die "unknown stage '$STAGE'; use baseline, clean, summary, or all"
    ;;
esac
