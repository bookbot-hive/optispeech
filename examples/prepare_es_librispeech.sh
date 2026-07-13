source .venv/bin/activate

python scripts/prepare_mls_es_dataset.py \
  --input-dir "/mnt/Store04/Datasets/Colombian Spanish/Spanish Librispeech/mls_spanish_opus" \
  --output-dir data/es_mls_raw \
  --val-ratio 0.02 \
  --seed 1234 \
  --audio-mode symlink \
  --workers 16

python -m optispeech.tools.preprocess_dataset \
  es_mls \
  data/es_mls_raw \
  data/es_mls \
  --text-normalization-recipe spanish_librispeech \
  -w 16

python -m optispeech.tools.generate_data_statistics es_mls -b 32 -w 16
