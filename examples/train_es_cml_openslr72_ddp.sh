#!/usr/bin/env bash
# Train the combined Spanish multi-speaker base model with automatic resume.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec python -m optispeech.train \
  experiment=es_cml_openslr72_ddp \
  run_name=es_cml_openslr72_ddp \
  "$@"
