#!/usr/bin/env python3
"""ONNX inference for English multi-speaker OptiSpeech models."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf

try:
    from text_processor import TTSTextProcessors
except ImportError:
    from text_processor import TextProcessor as TTSTextProcessors

LANGUAGE = "en"

VOICE2SPEAKER = {
    "au": 0,
    "gb": 1,
    "us": 2,
}

SAMPLE_RATE = 44_100
HOP_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="OptiSpeech English ONNX inference")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text or phonemes to synthesize",
    )
    parser.add_argument(
        "--is_phoneme",
        action="store_true",
        help="Whether the input is phonemes (default: False)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        choices=["au", "gb", "us"],
        default="gb",
        help="English accent/voice (default: gb)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Output directory path (default: current directory)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output.wav",
        help="Output filename (default: output.wav)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the English ONNX model",
    )
    return parser.parse_args()


def synthesize_speech(args):
    session = onnxruntime.InferenceSession(args.model_path)

    text_processor = TTSTextProcessors(
        None,
        None,
        language=LANGUAGE,
        use_cosmos=False,
        cosmos_config=None,
        online_g2p=True,
    )

    result = text_processor.get_input_ids(
        args.input,
        phonemes=args.is_phoneme,
        return_phonemes=True,
        push_oov_to_cosmos=False,
        add_blank_token=True,
        normalize=True,
    )

    input_ids = np.array([result["input_ids"]], dtype=np.int64)
    input_lengths = np.array([len(input_ids[0])], dtype=np.int64)
    inputs = {
        "x": input_ids,
        "x_lengths": input_lengths,
        "scales": np.array([0.6, 1.0, 1.0], dtype=np.float32),
        "sids": np.array([VOICE2SPEAKER[args.voice]], dtype=np.int64),
        "lids": np.array([0], dtype=np.int64),
    }

    audio, _, durations = session.run(None, inputs)

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    sf.write(str(output_file), audio[0], SAMPLE_RATE)

    duration = durations.sum() * HOP_SIZE / SAMPLE_RATE
    return duration, output_file


def main():
    args = parse_args()
    duration, output_file = synthesize_speech(args)
    print(f"Audio generated successfully! Duration: {duration:.2f} seconds")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
