#!/usr/bin/env bash
# Launch Spanish es_speaker DDP training with automatic resume on failure.
# Effective batch size: 6 per GPU x 2 GPUs = 12.
#
# Run directly:
#   bash examples/train_es_speaker_ddp.sh
#
# Run in tmux:
#   tmux new -s es_train "bash examples/train_es_speaker_ddp.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/examples/train_es_speaker_ddp_autoresume.sh"
