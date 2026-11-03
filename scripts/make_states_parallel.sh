#!/usr/bin/env bash
# Settle every missing state of a manifest in N parallel Isaac processes, then check and index them.
#
#   scripts/make_states_parallel.sh manifests/shapes_train.json 3 [stall_seconds=180]
#
# Long make-states processes running in parallel occasionally stop making progress while still
# using CPU (seen 2026-11-02, not tied to particular scenes). A watchdog restarts any shard with
# no new state for stall_seconds (Kit start-up included). Settling is order-independent and
# restarted shards skip saved states, so the result does not depend on restarts.
# Run inside `conda activate cable`.
set -u
manifest=$1
n=$2
stall=${3:-180}
max_restarts=20
logdir=runs/logs/states
mkdir -p "$logdir"
stem=$(basename "$manifest" .json)

run_shard() {
  local k=$1 log="$logdir/${stem}_$k.log" restarts=0
  while true; do
    CUDA_VISIBLE_DEVICES=$((k % 4)) python -m cablerouting make-states --manifest "$manifest" --shard "$k/$n" >>"$log" 2>&1 &
    local pid=$! last count now c
    last=$(date +%s)
    count=$(grep -c "settled" "$log")
    while kill -0 "$pid" 2>/dev/null; do
      sleep 5
      c=$(grep -c "settled" "$log")
      now=$(date +%s)
      if [ "$c" != "$count" ]; then count=$c; last=$now; fi
      if ((now - last > stall)); then
        echo "[watchdog] shard $k: no new state for ${stall}s; restarting" | tee -a "$log"
        kill "$pid"; sleep 5; kill -9 "$pid" 2>/dev/null
      fi
    done
    if wait "$pid"; then echo "[watchdog] shard $k done"; return 0; fi
    if grep -q "Traceback" "$log"; then echo "[watchdog] shard $k failed; see $log"; return 1; fi
    restarts=$((restarts + 1))
    if ((restarts > max_restarts)); then echo "[watchdog] shard $k: too many restarts"; return 1; fi
  done
}

for ((k = 0; k < n; k++)); do run_shard "$k" & done
wait
python -m cablerouting make-states --manifest "$manifest"
