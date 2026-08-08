#!/bin/bash
# Sequential background job queue for BARAM2026 experiments.
# Usage: nohup bash research/nodes/runq.sh > /tmp/runq.log 2>&1 &
cd /Users/um-yunsang/BARAM2026
Q=research/nodes/queue.txt
STATUS=research/nodes/queue_status.txt
: > "$STATUS"
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  name=$(echo "$line" | awk '{print $1}')
  cmd=$(echo "$line" | cut -d' ' -f2-)
  echo "START $name $(date +%H:%M:%S)" >> "$STATUS"
  start=$(date +%s)
  .venv/bin/python $cmd > "/tmp/job_${name}.log" 2>&1
  rc=$?
  echo "DONE  $name rc=$rc secs=$(( $(date +%s) - start )) $(date +%H:%M:%S)" >> "$STATUS"
done < "$Q"
echo "QUEUE_COMPLETE $(date +%H:%M:%S)" >> "$STATUS"
