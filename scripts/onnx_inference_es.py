#!/usr/bin/env python3
"""Inference for Spanish single-speaker es_speaker OptiSpeech models."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
import torch

from optispeech.model import OptiSpeech
from optispeech.text import TextProcessor
from optispeech.values import InferenceInputs

LANGUAGE = "es"
DEFAULT_MODEL = "logs/train/es_speaker_ddp/convnext-tts-ES-Marcia.onnx"
SAMPLE_RATE = 44_100
HOP_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="OptiSpeech Spanish inference")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Spanish input text or phonemes to synthesize",
    )
    parser.add_argument(
        "--is_phoneme",
        action="store_true",
        help="Whether the input is phonemes (default: False)",
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
        default=DEFAULT_MODEL,
        help=f"Path to a Spanish ONNX model or checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for checkpoint inference")
    return parser.parse_args()


def load_inference_args(model_path: str) -> tuple[int, dict[str, float]]:
    meta = onnxruntime.InferenceSession(model_path).get_modelmeta()
    infer_params = json.loads(meta.custom_metadata_map["inference"])
    sample_rate = infer_params["sample_rate"]
    inference_args = infer_params["inference_args"]
    return sample_rate, inference_args


def text_to_ids(text_processor: TextProcessor, text: str, is_phoneme: bool) -> dict:
    """Match configs/data/text_processor/es.yaml + training preprocessing."""
    if is_phoneme:
        return text_processor.tokenizers[LANGUAGE].get_input_ids(
            text,
            phonemes=True,
            return_phonemes=True,
            add_blank_token=text_processor.add_blank,
            normalize=text_processor.normalize_text,
        )
    return text_processor(text, lang=LANGUAGE)


def _inference_arg_value(inference_args, name: str) -> float:
    if isinstance(inference_args, dict):
        return inference_args[name]
    return getattr(inference_args, name)


def _build_inference_inputs(args, result: dict, inference_args) -> InferenceInputs:
    input_ids = result["input_ids"]
    return InferenceInputs.from_ids_and_lengths(
        ids=[input_ids],
        lengths=[len(input_ids)],
        clean_text=result.get("phonemes", result.get("normalized_text", args.input)),
        d_factor=_inference_arg_value(inference_args, "d_factor"),
        p_factor=_inference_arg_value(inference_args, "p_factor"),
        e_factor=_inference_arg_value(inference_args, "e_factor"),
    )


def _synthesize_onnx(args, result: dict):
    sample_rate, inference_args = load_inference_args(args.model_path)
    session = onnxruntime.InferenceSession(args.model_path)

    input_ids = np.array([result["input_ids"]], dtype=np.int64)
    input_lengths = np.array([len(input_ids[0])], dtype=np.int64)
    scales = np.array(
        [
            inference_args["d_factor"],
            inference_args["p_factor"],
            inference_args["e_factor"],
        ],
        dtype=np.float32,
    )

    scales = np.array([1.1, 1.0, 1.0], dtype=np.float32)
    inputs = {
        "x": input_ids,
        "x_lengths": input_lengths,
        "scales": scales,
    }

    audio, _, durations = session.run(None, inputs)
    return audio.squeeze(), sample_rate, durations.sum() * HOP_SIZE / sample_rate


def _synthesize_checkpoint(args, result: dict):
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    model = OptiSpeech.load_from_checkpoint(args.model_path, map_location="cpu")
    model.to(device)
    model.eval()

    inference_inputs = _build_inference_inputs(args, result, model.inference_args)
    inference_inputs = inference_inputs.as_torch().to(device)
    synth_outs = model.synthesise(inference_inputs)

    wav = synth_outs.wav.squeeze()
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()

    durations = synth_outs.durations
    if isinstance(durations, torch.Tensor):
        durations = durations.detach().cpu().numpy()
    duration = durations.sum() * model.hop_length / model.sample_rate
    return wav, model.sample_rate, duration


def synthesize_speech(args):
    text_processor = TextProcessor(
        tokenizer=None,
        add_blank=False,
        add_bos_eos=False,
        normalize_text=True,
        languages=[LANGUAGE],
    )

    result = text_to_ids(text_processor, args.input, args.is_phoneme)

    if not result["input_ids"]:
        raise ValueError(
            "No phoneme IDs produced for the input text. "
            "Run inference from the project venv: source .venv/bin/activate && uv sync --extra es"
        )

    model_path = Path(args.model_path)
    if model_path.suffix == ".ckpt":
        audio, sample_rate, duration = _synthesize_checkpoint(args, result)
    elif model_path.suffix == ".onnx":
        audio, sample_rate, duration = _synthesize_onnx(args, result)
    else:
        raise ValueError(f"Unsupported model type for {model_path}. Expected .onnx or .ckpt")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    sf.write(str(output_file), audio, sample_rate)

    return duration, output_file, result


def main():
    args = parse_args()
    duration, output_file, result = synthesize_speech(args)
    print(f"Phonemes: {result.get('phonemes', result.get('normalized_text', ''))}")
    print(f"Input IDs: {result['input_ids']}")
    print(f"Audio generated successfully! Duration: {duration:.2f} seconds")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
