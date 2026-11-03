#!/bin/bash
# Event stream for the operator: new worker tracebacks, pod/server deaths, and a queue summary every 15 min.
ROOT=/home/guy.azran/workspace/cable-routing-vlm-rl
cd $ROOT
declare -A seen
for f in runs/study/logs/pod_*/worker_*.log; do seen[$f]=$(grep -c Traceback "$f" 2>/dev/null || echo 0); done
last_summary=0; prev_jobs=""
while true; do
  for f in runs/study/logs/pod_*/worker_*.log; do
    n=$(grep -c Traceback "$f" 2>/dev/null || echo 0)
    if [ "$n" -gt "${seen[$f]:-0}" ]; then
      echo "TRACEBACK in $f: $(grep -A30 Traceback "$f" | grep -E '^[A-Za-z]+(Error|Exception)|Error:' | tail -1 | cut -c1-200)"
      seen[$f]=$n
    fi
  done
  jobs=$(squeue -u $USER -h -o "%i:%T" | grep -v -E "1475614|1475613" | sort | tr '\n' ' ')
  [ "$jobs" != "$prev_jobs" ] && { echo "JOBS: $jobs"; prev_jobs=$jobs; }
  curl -sf -m 10 http://127.0.0.1:8000/v1/models >/dev/null || echo "S1 (clair1:8000) NOT RESPONDING"
  now=$(date +%s)
  if [ $((now - last_summary)) -ge 900 ]; then
    s=""
    for q in dev5 calib2 screen diag; do
      d=runs/study/queue/$q; t=$(ls $d/tasks 2>/dev/null | wc -l); [ "$t" -eq 0 ] && continue
      s="$s $q: $(ls $d/done | wc -l)/$t done, $(( $(ls $d/claims | wc -l) - $(ls $d/done | wc -l) )) running;"
    done
    echo "$(date +%H:%M) QUEUES:$s"; last_summary=$now
  fi
  sleep 60
done
