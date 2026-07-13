#!/usr/bin/env bash
# Run ONNX inference from the project venv (required for correct Spanish phoneme IDs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv/bin/activate

# EN inference
# python scripts/onnx_inference.py \
#     --input "Superb!" \
#     --language en \
#     --voice gb \
#     --output_path "./outputs" \
#     --output_name "superb!.wav" \
#     --model_path "logs/OptiSpeech-ConvNext-TTS-en-Multi-Althaf-v3/convnext-tts-en-Multi-Althaf.onnx"

# ID inference
# python scripts/onnx_inference.py \
#     --input "hebat." \
#     --language id \
#     --output_path "./outputs" \
#     --output_name "hebat.wav" \
#     --model_path "logs/OptiSpeech-ConvNext-TTS-id-ID-Althaf-v2/convnext-tts-id-ID-Althaf.onnx"

# ES Inference
# uv run python -m scripts.onnx_inference_es \
#     --input "ola como estas." \
#     --output_path "./outputs" \
#     --output_name "hola_como_estas.wav" \
#     --model_path "logs/train/es_speaker_ddp_v2/convnext-tts-ES-Marcia.onnx"



uv run python -m scripts.onnx_inference_es \
    --input "—¡Hola Soy un Bookbot. Uso el reconocimiento de voz para escuchar a un niño leer y ayudarlo a mejorar. Pero todavía no puedo escucharte." \
    --output_path "./outputs" \
    --output_name "test_es_v2.wav" \
    --model_path "logs/train/es_speaker_ddp_v2/convnext-tts-ES-Marcia.onnx"

uv run python -m scripts.onnx_inference_es \
    --input "—¡Hola! Soy un Bookbot. Uso el reconocimiento de voz para escuchar a un niño leer y ayudarlo a mejorar. Pero todavía no puedo escucharte." \
    --output_path "./outputs" \
    --output_name "test_es_v1.wav" \
    --model_path "logs/train/es_speaker_ddp_v2/convnext-tts-ES-Marcia.onnx"
