#!/bin/bash
source .venv/bin/activate
python3 -m optispeech.tools.preprocess_dataset en_multi_althaf /home/s44504/en-Multi-Althaf-expressive-2 ./data/en_multi_althaf
python3 -m optispeech.tools.generate_data_statistics en_multi_althaf

export CUDA_VISIBLE_DEVICES=0
python3 -m optispeech.train experiment=en_multi_althaf
