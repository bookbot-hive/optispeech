"""
Converts the Spanish TTS pipeline passed_manifest.jsonl into LJSpeech format
for optispeech preprocessing.

Usage:
    python scripts/prepare_es_dataset.py \
        --manifest /path/to/passed_manifest.jsonl \
        --output-dir data/es_speaker_raw \
        [--max-duration 20.0] \
        [--val-ratio 0.05] \
        [--seed 1234] \
        [--workers 8]

Output structure:
    <output-dir>/
        train/
            metadata.csv   # filestem|text
            wav/
        val/
            metadata.csv
            wav/
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

def convert_opus_to_wav(src: Path, dst: Path, sample_rate: int = 44100) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-i", str(src), "-ar", str(sample_rate), "-ac", "1", str(dst), "-y", "-loglevel", "error"],
        capture_output=True,
    )
    return result.returncode == 0


def load_manifest(path: Path, max_duration: float) -> list[dict]:
    base_dir = path.parent
    entries = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["duration"] > max_duration:
                skipped += 1
                continue
            rel_audio = item.get("path") or item.get("audio")
            text = item.get("transcript") or item.get("text")
            if not rel_audio or text is None:
                raise ValueError(f"Unsupported manifest row in {path}: {item}")
            audio_path = base_dir / rel_audio
            stem = rel_audio.replace("/", "_").removesuffix(".opus")
            entries.append({
                "stem": stem,
                "text": text.strip(),
                "src": audio_path,
            })
    if skipped:
        print(f"  Skipped {skipped} entries exceeding max_duration={max_duration}s")
    return entries


def convert_or_link_audio(src: Path, dst: Path, sample_rate: int, audio_mode: str) -> bool:
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
        ok = convert_or_link_audio(entry["src"], dst, sample_rate, audio_mode)
        return entry["stem"], ok

    action = {"convert": "Converting", "symlink": "Linking", "copy": "Copying"}[audio_mode]
    print(f"  {action} {len(entries)} files -> {split_dir.name}/wav/ ...")
    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(convert, e): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            stem, ok = future.result()
            results[stem] = ok

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"  WARNING: {len(failed)} conversions failed: {failed[:5]}", file=sys.stderr)

    written = 0
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for entry in entries:
            if results.get(entry["stem"]):
                writer.writerow([entry["stem"], entry["text"]])
                written += 1

    print(f"  Wrote {written} entries → {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", dest="manifests")
    parser.add_argument("--narration-manifest", type=Path)
    parser.add_argument("--words-manifest", type=Path)
    parser.add_argument("--manifests", type=Path, nargs="+", dest="manifest_list")
    parser.add_argument("--output-dir", type=Path, default=Path("data/es_speaker_raw"))
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--audio-mode",
        choices=["convert", "symlink", "copy"],
        default="convert",
        help="How to materialize split audio. symlink/copy keep Opus; convert writes WAV.",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifests = []
    if args.manifests:
        manifests.extend(args.manifests)
    if args.narration_manifest:
        manifests.append(args.narration_manifest)
    if args.words_manifest:
        manifests.append(args.words_manifest)
    if args.manifest_list:
        manifests.extend(args.manifest_list)
    if not manifests:
        manifests = [
            Path("/mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/narration/_manifest.jsonl"),
            Path("/mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/words/_manifest.jsonl"),
        ]
    for manifest in manifests:
        if not manifest.exists():
            print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading manifests (max_duration={args.max_duration}s):")
    entries = []
    for manifest in manifests:
        print(f"  {manifest}")
        entries.extend(load_manifest(manifest, args.max_duration))
    print(f"  Total entries: {len(entries)}")

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_ratio))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]
    print(f"  Train: {len(train_entries)}  Val: {len(val_entries)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing train split ...")
    write_split(train_entries, args.output_dir / "train", args.sample_rate, args.workers, args.audio_mode)

    print("Processing val split ...")
    write_split(val_entries, args.output_dir / "val", args.sample_rate, args.workers, args.audio_mode)

    print("Done.")


if __name__ == "__main__":
    main()
