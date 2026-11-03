#!/bin/bash
# One SAC insertion-training run as a Slurm job; resumes from its newest checkpoint after a requeue.
#
#   sbatch -p public -c 100 --gres=gpu:2 --mem=360G --requeue -o runs/study/logs/train_%j.out \
#          scripts/study/train.sh OUT N_ENVS STEPS SEED [extra train-insertion args]
#
# Watchdog: a simulator child that dies (e.g. "omni.rtx: Out of resource descriptors") leaves the
# vectorised environment waiting forever. If no episode is logged for STALL_S seconds (start-up
# included), the whole process group is killed and training resumes from its newest checkpoint.
set -u
OUT=$1; N_ENVS=$2; STEPS=$3; SEED=$4; shift 4
ROOT=/home/guy.azran/workspace/cable-routing-vlm-rl
STALL_S=${STALL_S:-720}
cd "$ROOT"
set +u; source ~/miniforge3/etc/profile.d/conda.sh && conda activate cable; set -u
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export CABLE_SIM_THREADS=${CABLE_SIM_THREADS:-4} CABLE_PHYSX_THREADS=${CABLE_PHYSX_THREADS:-1}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "train ${SLURM_JOB_ID:-local} on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} cpus=$(nproc): $OUT n_envs=$N_ENVS steps=$STEPS seed=$SEED $*"
mkdir -p "$OUT"
for attempt in $(seq 1 30); do
  [ -e "$OUT/sac_final.zip" ] && { echo "training complete"; exit 0; }
  setsid python -m cablerouting train-insertion --out "$OUT" --n-envs "$N_ENVS" --steps "$STEPS" --seed "$SEED" \
    --ckpt-every 1000 "$@" &
  pid=$!
  start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    last=$(stat -c %Y "$OUT"/monitor*.csv 2>/dev/null | sort -n | tail -1)
    last=$(( ${last:-0} > start ? ${last:-0} : start ))
    if (( $(date +%s) - last > STALL_S )); then
      echo "[watchdog $(date +%H:%M:%S)] no episode for ${STALL_S}s; killing process group $pid (attempt $attempt)"
      kill -9 -- -"$pid" 2>/dev/null
      sleep 10
      break
    fi
  done
  wait "$pid"
  echo "[train] attempt $attempt ended with status $? at $(date +%H:%M:%S)"
done
[ -e "$OUT/sac_final.zip" ] && exit 0 || exit 1
