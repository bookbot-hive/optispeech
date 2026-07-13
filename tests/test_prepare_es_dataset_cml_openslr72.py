import csv
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from scripts.prepare_es_dataset_cml_openslr72 import (
    Entry,
    encode_opus,
    load_cml,
    probe_opus,
    split_by_speaker,
)


def write_wav(path: Path, sample_rate: int = 48000, seconds: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(b"\0\0" * int(sample_rate * seconds))


def write_cml_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["wav_filename", "transcript", "levenshtein", "duration", "client_id"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)


def test_cml_threshold_and_digit_rejection(tmp_path: Path):
    audio = tmp_path / "train/audio/1/2/good.wav"
    write_wav(audio, sample_rate=24000)
    accepted = {
        "wav_filename": "train/audio/1/2/good.wav",
        "transcript": "Texto aceptado",
        "levenshtein": "0.9",
        "duration": "0.05",
        "client_id": "1",
    }
    digit = dict(accepted, transcript="Texto con 2 notas")
    below = dict(accepted, transcript="Texto inferior", levenshtein="0.899")
    write_cml_csv(tmp_path / "train.csv", [accepted, digit, below])
    write_cml_csv(tmp_path / "dev.csv", [])
    write_cml_csv(tmp_path / "test.csv", [])

    entries, stats, examples = load_cml(tmp_path, 0.9)

    assert [entry.stem for entry in entries] == ["cml_train_good"]
    assert entries[0].speaker == "cml_1"
    assert stats.accepted == 1
    assert stats.contains_digits == 1
    assert stats.below_levenshtein == 1
    assert examples == ["Texto con 2 notas"]


def test_split_is_deterministic_and_has_no_validation_only_speakers(tmp_path: Path):
    entries = [
        Entry(f"stem_{speaker}_{index}", speaker, "texto", tmp_path / "x.wav", 1.0, "fixture")
        for speaker in ("a", "b")
        for index in range(5)
    ]
    first = split_by_speaker(entries, 0.2, 1234)
    second = split_by_speaker(entries, 0.2, 1234)

    assert [[entry.stem for entry in split] for split in first] == [
        [entry.stem for entry in split] for split in second
    ]
    assert {entry.speaker for entry in first[1]} <= {entry.speaker for entry in first[0]}
    assert len(first[1]) == 2


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
def test_encodes_valid_mono_opus_and_resumes(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.opus"
    write_wav(source)

    status, error = encode_opus(source, output, 24000, "48k", overwrite=False)
    assert (status, error) == ("encoded", None)
    assert probe_opus(output)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels", "-of", "csv=p=0", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "opus,1"

    status, error = encode_opus(source, output, 24000, "48k", overwrite=False)
    assert (status, error) == ("skipped", None)
