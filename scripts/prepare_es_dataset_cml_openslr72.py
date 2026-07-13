"""Prepare a combined CML-TTS + OpenSLR72 Spanish corpus for OptiSpeech.

The output is LJSpeech-like with speaker IDs::

    <output-dir>/
        train/metadata.csv  # filestem|speaker|text
        train/wav/*.opus
        val/metadata.csv
        val/wav/*.opus

CML train/dev/test are pooled with OpenSLR72 and split per speaker. Audio is
resampled to 24 kHz before encoding as mono Opus. Opus itself uses a 48 kHz
internal decode clock; OptiSpeech subsequently loads it at the configured
24 kHz feature-extractor rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm


DEFAULT_CML_DIR = Path(
    "/mnt/Store04/Datasets/Colombian Spanish/CML-TTS/cml_tts_dataset_spanish_v0.1"
)
DEFAULT_OPENSLR72_DIR = Path("/mnt/Store04/Datasets/Colombian Spanish/OpenSLR72")


@dataclass(frozen=True)
class Entry:
    stem: str
    speaker: str
    text: str
    src: Path
    duration: float
    source: str


@dataclass
class LoadStats:
    accepted: int = 0
    below_levenshtein: int = 0
    contains_digits: int = 0
    missing_audio: int = 0


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def load_cml(cml_dir: Path, min_levenshtein: float) -> tuple[list[Entry], LoadStats, list[str]]:
    entries: list[Entry] = []
    stats = LoadStats()
    digit_examples: list[str] = []

    for split in ("train", "dev", "test"):
        metadata_path = cml_dir / f"{split}.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing CML metadata: {metadata_path}")
        with metadata_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="|")
            required = {"wav_filename", "transcript", "levenshtein", "duration", "client_id"}
            missing_columns = required - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(f"{metadata_path} is missing columns: {sorted(missing_columns)}")

            for row in reader:
                score = float(row["levenshtein"])
                if score < min_levenshtein:
                    stats.below_levenshtein += 1
                    continue
                text = row["transcript"].strip()
                if any(character.isdigit() for character in text):
                    stats.contains_digits += 1
                    if len(digit_examples) < 10:
                        digit_examples.append(text)
                    continue
                src = cml_dir / row["wav_filename"]
                if not src.is_file():
                    stats.missing_audio += 1
                    continue
                source_stem = Path(row["wav_filename"]).stem
                entries.append(
                    Entry(
                        stem=f"cml_{split}_{source_stem}",
                        speaker=f"cml_{row['client_id'].strip()}",
                        text=text,
                        src=src,
                        duration=float(row["duration"]),
                        source="cml",
                    )
                )
                stats.accepted += 1
    return entries, stats, digit_examples


def load_openslr72(openslr72_dir: Path) -> tuple[list[Entry], LoadStats]:
    entries: list[Entry] = []
    stats = LoadStats()
    sources = (
        ("female", "line_index_female.tsv", "es_co_female"),
        ("male", "line_index_male.tsv", "es_co_male"),
    )
    for gender, index_name, audio_subdir in sources:
        index_path = openslr72_dir / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing OpenSLR72 index: {index_path}")
        with index_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    source_stem, text = line.split("\t", 1)
                except ValueError as error:
                    raise ValueError(f"Malformed {index_path}:{line_number}") from error
                stem_parts = source_stem.split("_")
                if len(stem_parts) < 3:
                    raise ValueError(f"Unexpected OpenSLR72 stem at {index_path}:{line_number}: {source_stem}")
                src = openslr72_dir / audio_subdir / f"{source_stem}.wav"
                if not src.is_file():
                    stats.missing_audio += 1
                    continue
                entries.append(
                    Entry(
                        stem=f"openslr72_{gender}_{source_stem}",
                        speaker=f"openslr72_{gender}_{stem_parts[1]}",
                        text=text.strip(),
                        src=src,
                        duration=wav_duration(src),
                        source="openslr72",
                    )
                )
                stats.accepted += 1
    return entries, stats


def validate_unique_entries(entries: list[Entry]) -> None:
    stems: set[str] = set()
    for entry in entries:
        if entry.stem in stems:
            raise ValueError(f"Duplicate output stem: {entry.stem}")
        stems.add(entry.stem)


def split_by_speaker(entries: list[Entry], val_ratio: float, seed: int) -> tuple[list[Entry], list[Entry]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    by_speaker: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_speaker[entry.speaker].append(entry)

    rng = random.Random(seed)
    train_entries: list[Entry] = []
    val_entries: list[Entry] = []
    for speaker in sorted(by_speaker):
        speaker_entries = sorted(by_speaker[speaker], key=lambda entry: entry.stem)
        rng.shuffle(speaker_entries)
        if len(speaker_entries) == 1:
            train_entries.extend(speaker_entries)
            continue
        n_val = max(1, int(round(len(speaker_entries) * val_ratio)))
        n_val = min(n_val, len(speaker_entries) - 1)
        val_entries.extend(speaker_entries[:n_val])
        train_entries.extend(speaker_entries[n_val:])

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def probe_opus(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        stream = json.loads(result.stdout)["streams"][0]
        return stream["codec_name"] == "opus" and int(stream["channels"]) == 1
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False


def encode_opus(src: Path, dst: Path, sample_rate: int, bitrate: str, overwrite: bool) -> tuple[str, str | None]:
    if not overwrite and probe_opus(dst):
        return "skipped", None
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.stem}.tmp.opus")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(src),
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-compression_level",
        "10",
        "-application",
        "audio",
        "-cutoff",
        str(sample_rate // 2),
        "-y",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not probe_opus(temporary):
        temporary.unlink(missing_ok=True)
        error = result.stderr.strip() or "ffprobe rejected encoded output"
        return "failed", error
    temporary.replace(dst)
    return "encoded", None


def write_metadata(entries: list[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="|", lineterminator="\n")
        for entry in entries:
            writer.writerow((entry.stem, entry.speaker, entry.text))
    temporary.replace(path)


def materialize_split(
    entries: list[Entry],
    split_dir: Path,
    sample_rate: int,
    bitrate: str,
    workers: int,
    overwrite: bool,
) -> tuple[list[Entry], dict[str, int], list[str]]:
    audio_dir = split_dir / "wav"
    results: dict[str, str] = {}
    errors: list[str] = []

    def convert(entry: Entry) -> tuple[Entry, str, str | None]:
        status, error = encode_opus(entry.src, audio_dir / f"{entry.stem}.opus", sample_rate, bitrate, overwrite)
        return entry, status, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert, entry): entry for entry in entries}
        for future in tqdm(as_completed(futures), total=len(futures), desc=split_dir.name, unit="file"):
            entry, status, error = future.result()
            results[entry.stem] = status
            if error:
                errors.append(f"{entry.stem}: {error}")

    successful = [entry for entry in entries if results.get(entry.stem) in {"encoded", "skipped"}]
    write_metadata(successful, split_dir / "metadata.csv")
    counts = {status: sum(result == status for result in results.values()) for status in ("encoded", "skipped", "failed")}
    return successful, counts, errors


def validate_tools() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable not found: {executable}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cml-dir", type=Path, default=DEFAULT_CML_DIR)
    parser.add_argument("--openslr72-dir", type=Path, default=DEFAULT_OPENSLR72_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("data/es_cml_openslr72_raw"))
    parser.add_argument("--min-levenshtein", type=float, default=0.9)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--opus-bitrate", default="48k")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="Re-encode valid existing Opus outputs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_levenshtein <= 1.0:
        raise ValueError("--min-levenshtein must be between 0 and 1")
    if args.sample_rate != 24000:
        raise ValueError("This dataset configuration requires --sample-rate 24000")
    if not re.fullmatch(r"[1-9]\d*[kK]", args.opus_bitrate):
        raise ValueError("--opus-bitrate must look like 48k")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    validate_tools()

    cml_entries, cml_stats, digit_examples = load_cml(args.cml_dir, args.min_levenshtein)
    openslr_entries, openslr_stats = load_openslr72(args.openslr72_dir)
    entries = cml_entries + openslr_entries
    validate_unique_entries(entries)
    train_entries, val_entries = split_by_speaker(entries, args.val_ratio, args.seed)

    train_speakers = {entry.speaker for entry in train_entries}
    val_speakers = {entry.speaker for entry in val_entries}
    if val_speakers - train_speakers:
        raise RuntimeError(f"Validation-only speakers: {sorted(val_speakers - train_speakers)}")

    print(f"CML accepted: {cml_stats.accepted:,}")
    print(f"CML below Levenshtein threshold: {cml_stats.below_levenshtein:,}")
    print(f"CML rejected for digits: {cml_stats.contains_digits:,}")
    for example in digit_examples:
        print(f"  digit example: {example}")
    print(f"OpenSLR72 accepted: {openslr_stats.accepted:,}")
    print(f"Combined duration: {sum(entry.duration for entry in entries) / 3600:.2f} hours")
    print(f"Speakers: {len(train_speakers | val_speakers):,}")
    print(f"Train: {len(train_entries):,}; validation: {len(val_entries):,}")

    train_ok, train_counts, train_errors = materialize_split(
        train_entries, args.output_dir / "train", args.sample_rate, args.opus_bitrate, args.workers, args.overwrite
    )
    val_ok, val_counts, val_errors = materialize_split(
        val_entries, args.output_dir / "val", args.sample_rate, args.opus_bitrate, args.workers, args.overwrite
    )
    errors = train_errors + val_errors
    summary = {
        "cml": asdict(cml_stats),
        "openslr72": asdict(openslr_stats),
        "input_utterances": len(entries),
        "input_hours": round(sum(entry.duration for entry in entries) / 3600, 6),
        "speakers": len(train_speakers | val_speakers),
        "train_utterances": len(train_ok),
        "val_utterances": len(val_ok),
        "train_conversion": train_counts,
        "val_conversion": val_counts,
        "conversion_failures": len(errors),
        "sample_rate_before_opus_encoding": args.sample_rate,
        "opus_bitrate": args.opus_bitrate,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if errors:
        print(f"ERROR: {len(errors)} conversions failed", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
