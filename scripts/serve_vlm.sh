#!/bin/bash
# Serve the planner with vLLM (user-global uv tool; see scripts/install/install_vllm.sh).
#
#   scripts/serve_vlm.sh main   # Qwen3.8-27B, BF16, TP=2 on GPUs 0,1 (primary study)
#   scripts/serve_vlm.sh dev    # Qwen3.5-9B, BF16, 1 GPU (debugging only)
#
# Protocol Section 4 candidate configuration with the pinned revision. Multimodal
# serving on 2x L40S is NOT covered by the upstream recipe and must pass
# scripts/smoke/vlm_smoke.py before use. The simulator gets the remaining GPU(s).
set -euo pipefail
MODE=${1:-main}
PORT=${PORT:-8000}
LOG_DIR=${LOG_DIR:-$(dirname "$0")/../runs/serving}
mkdir -p "$LOG_DIR"
# The Rust frontend (recipe default) ignores structured-output settings and rejects
# --tokenizer-revision; the Python frontend supports both.
export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # serve only the pinned, pre-downloaded revision
# JIT toolchain for FlashInfer kernels: ninja and the CUDA 13 toolkit both ship inside the
# vLLM tool venv (the node's system CUDA is 12.4). The top-k/top-p sampler is the native
# PyTorch one, so no kernel needs compiling on the sampling path.
VLLM_VENV=$(dirname "$(dirname "$(readlink -f "$(command -v vllm)")")")
export PATH="$VLLM_VENV/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-$VLLM_VENV/lib/python3.12/site-packages/nvidia/cu13}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

case "$MODE" in
  main)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    MODEL=Qwen/Qwen3.8-27B; REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; TP=2 ;;
  dev)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
    MODEL=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a; TP=1 ;;
  *) echo "usage: $0 main|dev" >&2; exit 2 ;;
esac

vllm serve "$MODEL" \
  --revision "$REV" \
  --tokenizer-revision "$REV" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --max-model-len 32768 \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --limit-mm-per-prompt '{"image": 4}' \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --host "${HOST:-127.0.0.1}" \
  --port "$PORT" 2>&1 | tee "$LOG_DIR/vllm_${MODE}_$(date +%Y%m%d_%H%M%S).log"
