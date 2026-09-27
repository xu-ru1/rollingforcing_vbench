#!/usr/bin/env bash
# Continuous formal generation, two workers; existing successful shards are skipped.
set -euo pipefail
cd "${RF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RF_ROOT="$PWD"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export FORMAL_ROOT="${FORMAL_ROOT:-$PWD/outputs/step_cache_formal_vbench_20260922}"
export R8_ROOT="${R8_ROOT:-$PWD/outputs/step_cache_r8/r8_vbench_plan_20260921}"
export CKPT="${CKPT:-$PWD/checkpoints/rolling_forcing_dmd.pt}"
GPU_A="${GPU_A:-1}"; GPU_B="${GPU_B:-2}"
[ "$GPU_A" != "$GPU_B" ]
mkdir -p debug-GPT "$FORMAL_ROOT/launch_logs"
exec 9>"$FORMAL_ROOT/remaining_pairs.lock"
flock -n 9 || { echo 'Another remaining-pairs driver is active'; exit 2; }
RUN="$FORMAL_ROOT/orchestration/remaining_pairs_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
CG="/sys/fs/cgroup/user.slice/user-$(id -u).slice"
"$PYTHON_BIN" - "$FORMAL_ROOT" "$PWD/inference.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); a=json.loads((r/'freeze/source_amendment_loader_cleanup_20260923.json').read_text())
assert a['status']=='ok' and not a['errors']
assert hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()==a['new_inference_sha256']
PY
done_shard() {
  "$PYTHON_BIN" - "$FORMAL_ROOT/shards/$1/${2}_${3}/status.json" "$3" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists(): sys.exit(1)
d=json.loads(p.read_text()); sys.exit(0 if d.get('status')=='ok' and d.get('count')==int(sys.argv[2]) and not d.get('errors') else 1)
PY
}
monitor() {
  while :; do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    for field in memory.current memory.high memory.max memory.events; do
      echo "[$field]"; cat "$CG/$field"
    done
    ps -u "$(id -un)" -o pid,ppid,pcpu,rss,stat,args | grep -E '[i]nference.py|[f]fmpeg' || true
    sleep 20
  done
}
monitor > "$RUN/resources.log" 2>&1 &
MPID=$!
trap 'kill "$MPID" 2>/dev/null || true' EXIT
launch() {
  local method="$1" gpu="$2" start="$3" count="$4"
  LAST=0
  if done_shard "$method" "$start" "$count"; then return; fi
  # Do not restart a partially generated shard: preserve it for recovery.
  local shard="$FORMAL_ROOT/shards/$method/${start}_${count}"
  if [ -s "$shard/audit_hashes.jsonl" ]; then
    echo "[STOP] incomplete shard has generated samples: $shard"; exit 3
  fi
  local used current
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [ "$used" -lt 1000 ] || { echo "[STOP] GPU $gpu occupied"; exit 4; }
  current="$(cat "$CG/memory.current")"
  [ "$current" -lt 75161927680 ] || { echo '[STOP] account memory >=70 GiB before loading'; exit 5; }
  GPU="$gpu" bash scripts/run_step_cache_r9_formal_generation_shard.sh "$method" "$start" "$count" --execute \
    > "$FORMAL_ROOT/launch_logs/${method}_${start}_${count}.log" 2>&1 &
  LAST=$!
  echo "[START] $method $start $count GPU=$gpu PID=$LAST"
}
methods=(rf_fixed_fast rf_front_fast rf_u_shape_fast rf_vanilla rf_fixed_slow rf_front_slow)
for start in 168 296 424 552 680 808 936; do
  count=128; [ "$start" -ne 936 ] || count=8
  for offset in 0 2 4; do
    a="${methods[$offset]}"; b="${methods[$((offset+1))]}"
    launch "$a" "$GPU_A" "$start" "$count"; pa=$LAST
    # First completed prompt proves initialization and checkpoint transfer finished.
    if [ "$pa" -ne 0 ]; then
      until [ -s "$FORMAL_ROOT/shards/$a/${start}_${count}/audit_hashes.jsonl" ]; do
        kill -0 "$pa" 2>/dev/null || { wait "$pa"; done_shard "$a" "$start" "$count"; break; }
        sleep 15
      done
    fi
    launch "$b" "$GPU_B" "$start" "$count"; pb=$LAST
    [ "$pa" -eq 0 ] || wait "$pa"
    [ "$pb" -eq 0 ] || wait "$pb"
    done_shard "$a" "$start" "$count"
    done_shard "$b" "$start" "$count"
    echo "[PASS] $a + $b start=$start count=$count"
  done
  echo "[BATCH COMPLETE] start=$start count=$count"
  archive="$RF_ROOT/debug-GPT/r9_remaining_through_$((start+count))_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  tar --exclude='generated' --exclude='*.mp4' -czf "$archive" -C "$FORMAL_ROOT" shards freeze orchestration launch_logs
  echo "[RETURN] $archive"
done
echo '[COMPLETE] remaining formal generation finished; review reports before VBench scoring'
