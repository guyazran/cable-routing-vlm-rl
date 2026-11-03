#!/bin/bash
# One Slurm job ("pod") of queue workers, optionally with its own planner server.
#
#   sbatch -p public -c 64 --gres=gpu:1 --mem=300G --requeue -o runs/study/logs/pod_%j.out \
#          scripts/study/pod.sh WORKERS KIND VLM QUEUE [QUEUE ...]
#
#   WORKERS  simulator worker processes (one Isaac process each)
#   KIND     episode | insertion_eval
#   VLM      none | local (first two GPUs of the job serve Qwen3.8-27B with TP=2; workers use it)
#            | http://host:port/v1 (an existing server)
#   QUEUE    queue directories, highest priority first
#   env CONTROLLERS: take only episodes of these controllers (e.g. vlm,vlm+progress for planner pods)
#   env IDLE_EXIT: seconds a worker waits on empty queues before exiting (default 600)
#   env WORKER_BASE: first worker index (default 0); give pods sharing one Slurm job disjoint ranges,
#   since a worker id is <job id>-<index> and a worker takes back every unfinished claim under its id.
#
# Workers exit when the queues stay empty; a worker that exits non-zero (crash, transport error)
# is restarted under the same worker id and takes back its own claim. A requeued job (preemption)
# keeps its job id, so its workers take back their claims too (eval/workqueue.py).
set -u
WORKERS=$1; KIND=$2; VLM=$3; shift 3; QUEUES=("$@")
ROOT=/home/guy.azran/workspace/cable-routing-vlm-rl
cd "$ROOT"
set +u; source ~/miniforge3/etc/profile.d/conda.sh && conda activate cable; set -u
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6       # Kit's libicu needs the env's libstdc++ (clair1-node-quirks)
export CABLE_SIM_THREADS=${CABLE_SIM_THREADS:-4} CABLE_PHYSX_THREADS=${CABLE_PHYSX_THREADS:-1}
export PYTHONUNBUFFERED=1
LOGDIR=$ROOT/runs/study/logs/pod_${SLURM_JOB_ID:-local}
mkdir -p "$LOGDIR"
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "pod ${SLURM_JOB_ID:-local} on $(hostname): ${#GPUS[@]} GPUs (${GPUS[*]}), $(nproc) cpus, $WORKERS $KIND workers, vlm=$VLM, queues=${QUEUES[*]}"

URL=""
SIM_GPUS=("${GPUS[@]}")
if [ "$VLM" = local ]; then
  PORT=$((8100 + ${SLURM_JOB_ID:-0} % 800))
  export PATH=$HOME/.local/bin:$PATH
  CUDA_VISIBLE_DEVICES=${GPUS[0]},${GPUS[1]} PORT=$PORT HOST=127.0.0.1 MAX_NUM_SEQS=${MAX_NUM_SEQS:-16} \
      LOG_DIR=$LOGDIR setsid scripts/serve_vlm.sh main > "$LOGDIR/vllm.log" 2>&1 &
  URL=http://127.0.0.1:$PORT/v1
  SIM_GPUS=("${GPUS[@]:2}")
  [ ${#SIM_GPUS[@]} -eq 0 ] && SIM_GPUS=("${GPUS[1]}")
  until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
    sleep 10
    if ! pgrep -u "$USER" -f "[v]llm serve.*--port $PORT" > /dev/null; then echo "vLLM died; see $LOGDIR/vllm.log"; exit 1; fi
  done
  echo "vLLM ready at $URL"
elif [ "$VLM" != none ]; then
  URL=$VLM
fi

run_worker() {
  local i=$1 gpu=${SIM_GPUS[$(( $1 % ${#SIM_GPUS[@]} ))]} rc idx=$(( ${WORKER_BASE:-0} + $1 ))
  while true; do
    CUDA_VISIBLE_DEVICES=$gpu python -m cablerouting worker --queue "${QUEUES[@]}" --index "$idx" --kinds "$KIND" \
      ${URL:+--vlm-url "$URL"} ${CONTROLLERS:+--controllers "$CONTROLLERS"} --idle-exit "${IDLE_EXIT:-600}" \
      >> "$LOGDIR/worker_$idx.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && return 0
    echo "[pod] worker $idx exited with $rc; restarting" >> "$LOGDIR/worker_$idx.log"
    sleep 10
  done
}
PIDS=()
for ((i = 0; i < WORKERS; i++)); do run_worker "$i" & PIDS+=($!); sleep 2; done
wait "${PIDS[@]}"
[ "$VLM" = local ] && pkill -u "$USER" -f "[v]llm serve.*--port $PORT"
echo "pod ${SLURM_JOB_ID:-local} done"
