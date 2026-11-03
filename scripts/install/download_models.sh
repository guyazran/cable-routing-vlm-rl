#!/bin/bash
# Download the planner weights at pinned revisions into the HF cache.
# Revisions are recorded in configs/versions.yaml; change them there and here together.
set -euo pipefail
MAIN_MODEL=Qwen/Qwen3.8-27B
MAIN_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
DEV_MODEL=Qwen/Qwen3.5-9B
DEV_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a

dl() { uv run --no-project --with "huggingface_hub[hf_xet]" hf download "$1" --revision "$2"; }
dl "$MAIN_MODEL" "$MAIN_REV"
dl "$DEV_MODEL" "$DEV_REV"
