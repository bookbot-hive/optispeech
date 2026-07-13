python scripts/prepare_es_dataset.py

uv run python -m optispeech.tools.preprocess_dataset es_speaker ./data/es_speaker_raw ./data/es_speaker --text-normalization-recipe spanish_vox_cpm

uv run python -m optispeech.tools.generate_data_statistics es_speaker
