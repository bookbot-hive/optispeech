#!/usr/bin/env bash
# Continue Spanish es_speaker DDP training with a fresh LR schedule.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec python -m optispeech.train \
  experiment=es_speaker_ddp_finetune \
  run_name=es_speaker_ddp_v2 \
  "$@"
