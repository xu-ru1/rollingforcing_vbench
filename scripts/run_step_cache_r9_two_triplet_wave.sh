#!/usr/bin/env bash
# Run two formal three-method waves with staggered checkpoint loading.
# Usage: run_step_cache_r9_two_triplet_wave.sh START COUNT --execute
set -euo pipefail

START="${1:?start index is required}"
COUNT="${2:?count is required}"
MODE="${3:---plan-only}"
[ "$MODE" = "--execute" ] || {
  echo "[r9-wave][error] pass --execute to run formal generation" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RF_ROOT="${RF_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-$RF_ROOT/checkpoints/rolling_forcing_dmd.pt}"
R8_ROOT="${R8_ROOT:-$RF_ROOT/outputs/step_cache_r8/r8_vbench_plan_20260921}"
FORMAL_ROOT="${FORMAL_ROOT:-$RF_ROOT/outputs/step_cache_formal_vbench_20260922}"
AMENDMENT="${R9_SOURCE_AMENDMENT:-$FORMAL_ROOT/freeze/source_amendment_loader_cleanup_20260923.json}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-5}"
RUN_ID="${R9_WAVE_RUN_ID:-r9_two_triplet_${START}_${COUNT}_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_NAME="${R9_RESULT_NAME:-${RUN_ID}_result.tar.gz}"
OUT="$FORMAL_ROOT/orchestration/$RUN_ID"
LOG_ROOT="$FORMAL_ROOT/launch_logs"
CGROUP="/sys/fs/cgroup/user.slice/user-$(id -u).slice"
RUNNER="$RF_ROOT/scripts/run_step_cache_r9_formal_generation_shard.sh"

FAST_METHODS=(rf_fixed_fast rf_front_fast rf_u_shape_fast)
SLOW_METHODS=(rf_vanilla rf_fixed_slow rf_front_slow)
GPUS=("$GPU_A" "$GPU_B" "$GPU_C")

die() {
  echo "[r9-wave][error] $*" >&2
  exit 2
}

for file in \
  "$RF_ROOT/inference.py" \
  "$RUNNER" \
  "$AMENDMENT" \
  "$R8_ROOT/reports/pre_vbench_immutable_manifest.json" \
  "$CKPT"; do
  [ -f "$file" ] || die "required file is missing: $file"
done

[ "$GPU_A" != "$GPU_B" ] && [ "$GPU_A" != "$GPU_C" ] && [ "$GPU_B" != "$GPU_C" ] ||
  die "GPU_A, GPU_B, and GPU_C must be distinct"

mkdir -p "$OUT" "$LOG_ROOT" "$RF_ROOT/debug-GPT"

"$PYTHON_BIN" - "$AMENDMENT" "$RF_ROOT/inference.py" <<'PY'
import hashlib
import json
import pathlib
import sys

amendment_path = pathlib.Path(sys.argv[1])
inference_path = pathlib.Path(sys.argv[2])
data = json.loads(amendment_path.read_text(encoding="utf-8"))
observed = hashlib.sha256(inference_path.read_bytes()).hexdigest()
assert data.get("status") == "ok", data
assert not data.get("errors"), data.get("errors")
assert data.get("new_inference_sha256") == observed, (data.get("new_inference_sha256"), observed)
assert all(item.get("equal") for item in data.get("audit_comparisons", {}).values())
print("[r9-wave] source amendment PASS sha256=" + observed)
PY

for gpu in "${GPUS[@]}"; do
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [ "$used" -lt 1000 ] || die "GPU $gpu is occupied: ${used} MiB"
done

host_guard() {
  local current_bytes user_rss_kb
  current_bytes="$(cat "$CGROUP/memory.current")"
  user_rss_kb="$(ps -u "$(id -un)" -o rss= | awk '{sum += $1} END {print sum+0}')"
  awk -v current="$current_bytes" -v rss="$user_rss_kb" 'BEGIN {
    printf "[r9-wave][guard] cgroup=%.2f GiB user_rss=%.2f GiB\n", current/1073741824, rss/1048576
  }'
  [ "$current_bytes" -lt 75161927680 ] || return 1
  [ "$user_rss_kb" -lt 33554432 ] || return 1
}

monitor_wave() {
  local marker="$1" output="$2"
  shift 2
  : > "$output"
  while [ -e "$marker" ]; do
    {
      echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
      ps -u "$(id -un)" -ww -o pid,ppid,etime,pcpu,rss,stat,args |
        grep -E '[i]nference.py|[f]fmpeg|[r]un_step_cache_r9' || true
      ps -u "$(id -un)" -o rss= | awk '{sum += $1} END {printf "user_rss_gib=%.2f\n", sum/1048576}'
      current="$(cat "$CGROUP/memory.current")"
      awk -v value="$current" 'BEGIN {printf "cgroup_current_gib=%.2f\n", value/1073741824}'
      echo "[memory.events]"
      cat "$CGROUP/memory.events"
      for gpu in "$@"; do
        nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
      done
      echo
    } >> "$output" 2>&1
    sleep 20
  done
}

LAST_PID=0
launch_and_wait_stable() {
  local method="$1" gpu="$2" wave="$3"
  local shard="$FORMAL_ROOT/shards/$method/${START}_${COUNT}"
  local status="$shard/status.json"
  local log="$LOG_ROOT/${method}_${START}_${COUNT}.log"
  local used pid child gpu_mem rss_kb deadline ready

  if [ -f "$status" ] && "$PYTHON_BIN" - "$status" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "ok" else 1)
PY
  then
    echo "[r9-wave] already complete method=$method start=$START count=$COUNT"
    LAST_PID=0
    return 0
  fi

  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [ "$used" -lt 1000 ] || {
    echo "[r9-wave][guard] GPU $gpu is occupied: ${used} MiB" >&2
    return 1
  }

  env \
    RF_ROOT="$RF_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    GPU="$gpu" \
    CKPT="$CKPT" \
    R8_ROOT="$R8_ROOT" \
    FORMAL_ROOT="$FORMAL_ROOT" \
    bash "$RUNNER" "$method" "$START" "$COUNT" --execute \
    > "$log" 2>&1 &

  pid=$!
  LAST_PID=$pid
  echo "$pid" > "$log.pid"
  echo "[r9-wave] launched wave=$wave method=$method gpu=$gpu pid=$pid"

  deadline=$((SECONDS + 1200))
  ready=0
  while kill -0 "$pid" 2>/dev/null; do
    child="$(pgrep -P "$pid" -f 'inference.py' | head -n 1 || true)"
    gpu_mem="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    rss_kb=0
    if [ -n "$child" ]; then
      rss_kb="$(ps -o rss= -p "$child" | tr -d ' ')"
      rss_kb="${rss_kb:-0}"
    fi
    printf '[r9-wave][wait] method=%s gpu=%s gpu_mem_mib=%s rss_gib=%.2f\n' \
      "$method" "$gpu" "$gpu_mem" \
      "$(awk -v value="$rss_kb" 'BEGIN {print value/1048576}')"
    if [ "$gpu_mem" -ge 30000 ] && [ "$rss_kb" -gt 0 ] && [ "$rss_kb" -le 18874368 ]; then
      ready=1
      break
    fi
    [ "$SECONDS" -lt "$deadline" ] || break
    sleep 15
  done
  [ "$ready" -eq 1 ] || {
    echo "[r9-wave][guard] method=$method did not reach stable state" >&2
    return 1
  }
}

validate_wave() {
  local wave="$1"
  shift
  "$PYTHON_BIN" - "$FORMAL_ROOT" "$START" "$COUNT" "$wave" "$@" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
start, count, wave = sys.argv[2:5]
methods = sys.argv[5:]
for method in methods:
    status = root / "shards" / method / f"{start}_{count}" / "status.json"
    if not status.is_file():
        raise SystemExit(f"[r9-wave][error] missing {status}")
    data = json.loads(status.read_text(encoding="utf-8"))
    if data.get("status") != "ok" or data.get("count") != int(count):
        raise SystemExit(f"[r9-wave][error] {method}: {data}")
    print(f"[r9-wave] PASS wave={wave} method={method} count={count}")
PY
}

run_wave() {
  local wave="$1"
  shift
  local methods=("$@")
  local marker="$OUT/${wave}.active"
  local monitor="$OUT/${wave}_resource_monitor.log"
  local monitor_pid failed method gpu pid
  local pids=()

  : > "$marker"
  monitor_wave "$marker" "$monitor" "${GPUS[@]}" &
  monitor_pid=$!
  failed=0

  for index in 0 1 2; do
    method="${methods[$index]}"
    gpu="${GPUS[$index]}"
    if ! host_guard; then
      echo "[r9-wave][guard] host memory guard stopped before $method" >&2
      failed=1
      break
    fi
    if ! launch_and_wait_stable "$method" "$gpu" "$wave"; then
      pid="$LAST_PID"
      [ "$pid" -eq 0 ] || pids+=("$pid")
      failed=1
      break
    fi
    pid="$LAST_PID"
    [ "$pid" -eq 0 ] || pids+=("$pid")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[r9-wave][error] child process failed pid=$pid" >&2
      failed=1
    fi
  done

  rm -f -- "$marker"
  wait "$monitor_pid" || true
  [ "$failed" -eq 0 ] || return 1
  validate_wave "$wave" "${methods[@]}"
}

run_wave fast "${FAST_METHODS[@]}"
run_wave slow "${SLOW_METHODS[@]}"

"$PYTHON_BIN" - "$OUT/status.json" "$FORMAL_ROOT" "$START" "$COUNT" \
  "${FAST_METHODS[@]}" "${SLOW_METHODS[@]}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

output = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
start, count = map(int, sys.argv[3:5])
methods = sys.argv[5:]
records = []
for method in methods:
    status_path = root / "shards" / method / f"{start}_{count}" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    records.append({"method": method, "status_path": str(status_path), "status": status})
payload = {
    "schema": "rollingforcing.r9.two_triplet_wave.v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "ok",
    "start": start,
    "count": count,
    "records": records,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

items=(
  "${OUT#$RF_ROOT/}"
  "${AMENDMENT#$RF_ROOT/}"
  "scripts/run_step_cache_r9_two_triplet_wave.sh"
)
for method in "${FAST_METHODS[@]}" "${SLOW_METHODS[@]}"; do
  items+=("outputs/step_cache_formal_vbench_20260922/shards/$method/${START}_${COUNT}")
  items+=("outputs/step_cache_formal_vbench_20260922/launch_logs/${method}_${START}_${COUNT}.log")
done

# Backfill the previously generated vanilla 8_32 reports when they are available.
if [ -d "$FORMAL_ROOT/shards/rf_vanilla/8_32" ]; then
  items+=("outputs/step_cache_formal_vbench_20260922/shards/rf_vanilla/8_32")
fi

archive="$RF_ROOT/debug-GPT/$RESULT_NAME"
tar -czf "$archive" -C "$RF_ROOT" "${items[@]}"
sha256sum "$archive" | tee "$archive.sha256"
echo "[r9-wave] COMPLETE archive=$archive"
