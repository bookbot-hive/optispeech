"""
Prepare MLS Spanish Opus data for OptiSpeech preprocessing.

The output is LJSpeech-like, but with a speaker column:

    <output-dir>/
        train/
            metadata.csv  # filestem|speaker|text
            wav/
        val/
            metadata.csv
            wav/

By default, MLS train/dev/test are pooled and split per speaker so validation
does not contain speaker IDs absent from training.
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

def read_segments(path: Path) -> dict[str, float]:
    durations = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            key, _source, start, end = parts[:4]
            durations[key] = float(end) - float(start)
    return durations


def load_split(root: Path, split: str, max_duration: float | None) -> list[dict]:
    split_dir = root / split
    transcript_path = split_dir / "transcripts.txt"
    segment_path = split_dir / "segments.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcripts file: {transcript_path}")
    if not segment_path.exists():
        raise FileNotFoundError(f"Missing segments file: {segment_path}")

    durations = read_segments(segment_path)
    entries = []
    skipped_duration = 0
    missing_audio = 0
    with transcript_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            key, text = line.split("\t", 1)
            speaker, book, _utt = key.split("_", 2)
            duration = durations.get(key)
            if max_duration is not None and duration is not None and duration > max_duration:
                skipped_duration += 1
                continue
            src = split_dir / "audio" / speaker / book / f"{key}.opus"
            if not src.exists():
                missing_audio += 1
                continue
            entries.append(
                {
                    "stem": f"{split}_{key}",
                    "speaker": speaker,
                    "text": text.strip(),
                    "src": src,
                }
            )

    if skipped_duration:
        print(f"  {split}: skipped {skipped_duration} entries over {max_duration}s")
    if missing_audio:
        print(f"  {split}: WARNING missing {missing_audio} audio files", file=sys.stderr)
    print(f"  {split}: loaded {len(entries)} entries")
    return entries


def split_by_speaker(entries: list[dict], val_ratio: float, seed: int, min_train_per_speaker: int) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    by_speaker = defaultdict(list)
    for entry in entries:
        by_speaker[entry["speaker"]].append(entry)

    rng = random.Random(seed)
    train_entries = []
    val_entries = []
    speakers_without_val = 0
    for speaker_entries in by_speaker.values():
        speaker_entries = list(speaker_entries)
        rng.shuffle(speaker_entries)
        max_val = max(0, len(speaker_entries) - min_train_per_speaker)
        n_val = min(max(1, int(round(len(speaker_entries) * val_ratio))), max_val)
        if n_val == 0:
            speakers_without_val += 1
            train_entries.extend(speaker_entries)
            continue
        val_entries.extend(speaker_entries[:n_val])
        train_entries.extend(speaker_entries[n_val:])

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    if speakers_without_val:
        print(
            f"  {speakers_without_val} speakers had <= {min_train_per_speaker} utterances "
            "and were kept train-only"
        )
    return train_entries, val_entries


def convert_opus_to_wav(src: Path, dst: Path, sample_rate: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(src),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(dst),
            "-y",
            "-loglevel",
            "error",
        ],
        capture_output=True,
    )
    return result.returncode == 0


def materialize_audio(src: Path, dst: Path, sample_rate: int, audio_mode: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    if audio_mode == "convert":
        return convert_opus_to_wav(src, dst, sample_rate)
    if audio_mode == "symlink":
        dst.symlink_to(src.resolve())
        return True
    if audio_mode == "copy":
        shutil.copy2(src, dst)
        return True
    raise ValueError(f"Unsupported audio mode: {audio_mode}")


def write_split(entries: list[dict], split_dir: Path, sample_rate: int, n_workers: int, audio_mode: str) -> None:
    wav_dir = split_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.csv"
    audio_suffix = ".wav" if audio_mode == "convert" else ".opus"

    def convert(entry: dict) -> tuple[str, bool]:
        dst = wav_dir / f"{entry['stem']}{audio_suffix}"
        return entry["stem"], materialize_audio(entry["src"], dst, sample_rate, audio_mode)

    action = {"convert": "Converting", "symlink": "Linking", "copy": "Copying"}[audio_mode]
    print(f"  {action} {len(entries)} files -> {split_dir.name}/wav/ ...")
    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(convert, entry): entry for entry in entries}
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            stem, ok = future.result()
            results[stem] = ok

    failed = [stem for stem, ok in results.items() if not ok]
    if failed:
        print(f"  WARNING: {len(failed)} conversions failed; first failures: {failed[:5]}", file=sys.stderr)

    written = 0
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        for entry in entries:
            if results.get(entry["stem"]):
                writer.writerow([entry["stem"], entry["speaker"], entry["text"]])
                written += 1
    print(f"  Wrote {written} entries -> {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/mnt/Store04/Datasets/Colombian Spanish/Spanish Librispeech/mls_spanish_opus"),
        help="MLS Spanish Opus root containing train/dev/test directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/es_mls_raw"))
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-train-per-speaker", type=int, default=1)
    parser.add_argument(
        "--audio-mode",
        choices=["symlink", "copy", "convert"],
        default="symlink",
        help="How to materialize split audio. symlink/copy keep Opus; convert writes WAV.",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading MLS Spanish from {args.input_dir}")
    entries = []
    for split in ("train", "dev", "test"):
        entries.extend(load_split(args.input_dir, split, args.max_duration))

    train_entries, val_entries = split_by_speaker(
        entries=entries,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_train_per_speaker=args.min_train_per_speaker,
    )
    train_speakers = {entry["speaker"] for entry in train_entries}
    val_speakers = {entry["speaker"] for entry in val_entries}
    missing_train_speakers = val_speakers - train_speakers
    print(
        f"  Train: {len(train_entries)}  Val: {len(val_entries)}  "
        f"Speakers: {len(train_speakers | val_speakers)}"
    )
    if missing_train_speakers:
        raise RuntimeError(f"Validation speakers missing from train: {sorted(missing_train_speakers)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Processing train split ...")
    write_split(train_entries, args.output_dir / "train", args.sample_rate, args.workers, args.audio_mode)
    print("Processing val split ...")
    write_split(val_entries, args.output_dir / "val", args.sample_rate, args.workers, args.audio_mode)
    print("Done.")


if __name__ == "__main__":
    main()
