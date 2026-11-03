#!/bin/bash
# Queue validation evaluations (deviation R10) for candidate checkpoints as training writes them.
#
#   scripts/study/watch_checkpoints.sh RUN_DIR [RUN_DIR ...]
#
# Candidates per run: the first checkpoint at or after each target in STEPS (default 5880, about the
# paper's 6,200 steps, then 9800, 14700, 19600) plus sac_final. Each gets 100
# single-clip validation starts (seeds 8,000,000+) and 100 route-stage validation starts (seeds
# 8,100,000+, train-split stage geometry). Exits once every candidate of every run is queued.
set -u
ROOT=/home/guy.azran/workspace/cable-routing-vlm-rl
cd "$ROOT"
set +u; source ~/miniforge3/etc/profile.d/conda.sh && conda activate cable; set -u
Q=runs/study/queue/insval
STEPS="${STEPS:-5880 9800 14700 19600 final}"
queue_ckpt() {  # run_dir tag zip
  local name; name=$(basename "$1")_$2
  python scripts/study/enqueue.py insertion-eval --queue $Q --policy "$3" --seed-start 8000000 --n 100 --chunk 10 \
    --route-stage-prob 0 --out "runs/study/insval/${name}_single" --priority "p20"
  python scripts/study/enqueue.py insertion-eval --queue $Q --policy "$3" --seed-start 8100000 --n 100 --chunk 10 \
    --route-stage-prob 1 --stages manifests/shapes_train.json --out "runs/study/insval/${name}_stage" --priority "p20"
}
while true; do
  pending=0
  for run in "$@"; do
    for s in $STEPS; do
      mark="$run/.queued_$s"
      [ -e "$mark" ] && continue
      if [ "$s" = final ]; then zip="$run/sac_final.zip"; else
        # the first checkpoint at or after the target step (checkpoint spacing depends on n_envs and warm-up)
        zip=$(ls "$run"/sac_*_steps.zip 2>/dev/null | sed -E 's/.*sac_([0-9]+)_steps.zip/\1/' | sort -n |
              awk -v t="$s" '$1 >= t {print; exit}')
        zip=${zip:+$run/sac_${zip}_steps.zip}
      fi
      if [ -n "$zip" ] && [ -e "$zip" ]; then
        sleep 5  # let the writer finish the file
        queue_ckpt "$run" "$s" "$zip" && touch "$mark" && echo "$(date +%H:%M) queued $zip"
      else
        pending=$((pending + 1))
      fi
    done
  done
  [ $pending -eq 0 ] && { echo "all candidates queued"; exit 0; }
  sleep 30
done
