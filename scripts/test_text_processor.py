#!/usr/bin/env python3
"""Interactive CLI to inspect TTSTextProcessors output for OptiSpeech training/inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from optispeech.text import TEXT_PROCESSOR_CONFIG

try:
    from text_processor import TTSTextProcessors
except ImportError as exc:
    raise SystemExit(
        "text_processor is not installed. Run: source .venv/bin/activate && uv sync --extra es"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test TTSTextProcessors phonemization and token IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("text", nargs="?", help="Input text or phoneme string")
    parser.add_argument("--input", "-i", dest="text_opt", help="Input text (alternative to positional arg)")

    parser.add_argument("--language", "-l", default="es", choices=["en", "es", "id", "sw"], help="Language code")
    parser.add_argument("--phonemes", action="store_true", help="Treat input as phonemes instead of raw text")
    parser.add_argument(
        "--show-words",
        action="store_true",
        help="Print per-token word index mapping (word_idx)",
    )
    parser.add_argument("--json", action="store_true", help="Print full result as JSON")

    get_input_ids = parser.add_argument_group("get_input_ids options")
    get_input_ids.add_argument("--return-phonemes", action="store_true", default=True, help="Include phoneme string")
    get_input_ids.add_argument("--no-return-phonemes", action="store_false", dest="return_phonemes")
    get_input_ids.add_argument("--normalize", action="store_true", default=True, help="Normalize input text")
    get_input_ids.add_argument("--no-normalize", action="store_false", dest="normalize")
    get_input_ids.add_argument("--add-blank-token", action="store_true", default=False, help="Append blank token")
    get_input_ids.add_argument("--push-oov-to-cosmos", action="store_true", default=False)
    get_input_ids.add_argument("--accent", default=None, help="English accent, e.g. en-us (en only)")
    get_input_ids.add_argument("--emphasize-model", default=None, help="LLM model name for emphasis tagging")

    processor_init = parser.add_argument_group("TTSTextProcessors options")
    processor_init.add_argument(
        "--emphasis-model-path",
        default=None,
        help="Override emphasis ONNX model path (defaults from optispeech config)",
    )
    processor_init.add_argument(
        "--emphasis-db-path",
        default=None,
        help="Override emphasis lookup JSON path (defaults from optispeech config)",
    )
    processor_init.add_argument("--use-cosmos", action="store_true", default=False)
    processor_init.add_argument("--animation-tags-path", default=None)
    processor_init.add_argument("--online-g2p", action="store_true", default=False)
    processor_init.add_argument("--cache", action="store_true", default=False)
    processor_init.add_argument("--llm-call-mode", default="api", choices=["api", "cli"])
    processor_init.add_argument("--dev", action="store_true", default=False, help="Write dev trajectory logs")
    processor_init.add_argument("--dev-output-dir", default=None)

    return parser.parse_args()


def resolve_paths(language: str, emphasis_model_path: str | None, emphasis_db_path: str | None) -> tuple[str | None, str | None]:
    lang = language.strip().lower()
    model_path = emphasis_model_path if emphasis_model_path is not None else TEXT_PROCESSOR_CONFIG["MODEL_DIRS"].get(lang)
    db_path = emphasis_db_path if emphasis_db_path is not None else TEXT_PROCESSOR_CONFIG["DB_PATHS"].get(lang)
    return model_path, db_path


def build_processor(args: argparse.Namespace) -> TTSTextProcessors:
    emphasis_model_path, emphasis_db_path = resolve_paths(args.language, args.emphasis_model_path, args.emphasis_db_path)
    cosmos_config = TEXT_PROCESSOR_CONFIG["COSMOS_CONFIG"] if args.use_cosmos else None
    return TTSTextProcessors(
        emphasis_model_path,
        emphasis_db_path,
        language=args.language,
        use_cosmos=args.use_cosmos,
        cosmos_config=cosmos_config,
        animation_tags_path=args.animation_tags_path,
        online_g2p=args.online_g2p,
        cache=args.cache,
        llm_call_mode=args.llm_call_mode,
        dev=args.dev,
        dev_output_dir=args.dev_output_dir,
    )


def format_word_mapping(phonemes: str, phoneme_tokens: list[str], word_idx: list[int]) -> str:
    lines = ["Token -> word_idx"]
    for token, idx in zip(phoneme_tokens, word_idx):
        lines.append(f"  {token!r:12} -> {idx}")
    return "\n".join(lines)


def split_phoneme_tokens(processor: TTSTextProcessors, phonemes: str) -> list[str]:
    return processor.tokenizer.split_phonemes(phonemes)


def main() -> int:
    args = parse_args()
    text = args.text_opt or args.text
    if not text:
        print("Error: provide input text as a positional argument or --input", file=sys.stderr)
        return 1

    processor = build_processor(args)
    result = processor.get_input_ids(
        text,
        phonemes=args.phonemes,
        return_phonemes=args.return_phonemes,
        push_oov_to_cosmos=args.push_oov_to_cosmos,
        add_blank_token=args.add_blank_token,
        normalize=args.normalize,
        accent=args.accent,
        emphasize_model=args.emphasize_model,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if "normalized_text" in result:
        print(f"Normalized text: {result['normalized_text']}")
    if "phonemes" in result:
        print(f"Phonemes:        {result['phonemes']}")
    print(f"Input IDs ({len(result['input_ids'])}): {result['input_ids']}")

    if args.show_words and "word_idx" in result and "phonemes" in result:
        tokens = split_phoneme_tokens(processor, result["phonemes"])
        print(format_word_mapping(result["phonemes"], tokens, result["word_idx"]))

    if "trajectory_path" in result:
        print(f"Trajectory:      {result['trajectory_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
