#!/bin/bash
# User-global VLM serving environment: vLLM as a uv tool (its own Python 3.12 venv
# under ~/.local/share/uv/tools/vllm, executables in ~/.local/bin). Kept separate
# from the `cable` simulator environment (protocol Section 4); the planner client
# in `cable` talks to it over the OpenAI-compatible HTTP API.
set -euo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.32.0}
# Keep uv's wheel cache on node-local disk (NFS makes unpacking very slow) and limit
# concurrent downloads (uv stalled on the large CUDA wheels with the default).
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-$USER} UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-2} UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
uv tool install --python 3.12 "vllm==${VLLM_VERSION}" --with openai --with "huggingface_hub[hf_xet]"
vllm --version
