#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv/bin/activate

python scripts/prepare_es_dataset_cml_openslr72.py \
  --output-dir data/es_cml_openslr72_raw \
  --min-levenshtein 0.9 \
  --sample-rate 24000 \
  --opus-bitrate 48k \
  --val-ratio 0.02 \
  --seed 1234 \
  --workers 16

python -m optispeech.tools.preprocess_dataset \
  es_cml_openslr72 \
  data/es_cml_openslr72_raw \
  data/es_cml_openslr72 \
  --text-normalization-recipe spanish_cml_openslr72 \
  --resume \
  -w 16

python -m optispeech.tools.generate_data_statistics es_cml_openslr72 -b 32 -w 16

echo "Copy the values from data/es_cml_openslr72/stats.json into configs/data/es_cml_openslr72.yaml before training."
