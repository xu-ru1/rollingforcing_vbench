#!/usr/bin/env bash
# Continuously report account RSS, host RAM, and GPU memory for formal runs.
set -euo pipefail
INTERVAL_SEC="${INTERVAL_SEC:-5}"
WARN_USER_RSS_GIB="${WARN_USER_RSS_GIB:-105}"
USER_NAME="${MONITOR_USER:-${USER:-$(id -un)}}"
LOG_PATH="${MONITOR_LOG:-}"
emit() {
  local rss_kib rss_gib available_kib available_gib
  rss_kib="$(ps -u "$USER_NAME" -o rss= 2>/dev/null | awk '{sum+=$1} END {print sum+0}')"
  rss_gib="$(awk -v value="$rss_kib" 'BEGIN {printf "%.2f", value/1024/1024}')"
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  available_gib="$(awk -v value="$available_kib" 'BEGIN {printf "%.2f", value/1024/1024}')"
  printf '\n[%s] user=%s rss_gib=%s warn_gib=%s host_available_gib=%s\n' "$(date '+%F %T')" "$USER_NAME" "$rss_gib" "$WARN_USER_RSS_GIB" "$available_gib"
  awk -v value="$rss_gib" -v warn="$WARN_USER_RSS_GIB" 'BEGIN {if (value >= warn) print "[MEMORY WARNING] user RSS is at or above the configured safety line."}'
  printf '%-8s %-8s %-10s %-10s %s\n' PID %CPU RSS_GiB ELAPSED COMMAND
  ps -u "$USER_NAME" -o pid=,pcpu=,rss=,etime=,args= --sort=-rss 2>/dev/null | head -n 15 | awk '{printf "%-8s %-8s %-10.2f %-10s ", $1,$2,$3/1024/1024,$4; for(i=5;i<=NF;i++) printf "%s%s", $i,(i==NF?"\n":" ")}'
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo '[GPU]'
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  fi
}
if [ -n "$LOG_PATH" ]; then
  mkdir -p "$(dirname -- "$LOG_PATH")"
  while true; do emit; sleep "$INTERVAL_SEC"; done | tee -a "$LOG_PATH"
else
  while true; do emit; sleep "$INTERVAL_SEC"; done
fi
