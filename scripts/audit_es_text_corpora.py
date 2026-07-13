"""Audit Spanish TTS corpora for text patterns that may need normalization."""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from recipes.text_normalization.spanish_common_only import preprocess_text as preprocess_common_text
from recipes.text_normalization.spanish_librispeech import preprocess_text as preprocess_librispeech_text
from recipes.text_normalization.spanish_vox_cpm import preprocess_text as preprocess_vox_cpm_text


SPANISH_LETTERS = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
TOKEN_RE = re.compile(
    rf"[{SPANISH_LETTERS}]+(?:[.'’][{SPANISH_LETTERS}]+)*|\d+(?:[.,]\d+)*|[^\s]",
    re.UNICODE,
)
WORD_RE = re.compile(rf"^[{SPANISH_LETTERS}]+(?:[.'’][{SPANISH_LETTERS}]+)*$", re.UNICODE)
ROMAN_RE = re.compile(r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)
HAS_DIGIT_RE = re.compile(r"\d")
HAS_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
HAS_SPANISH_LETTER_RE = re.compile(rf"[{SPANISH_LETTERS}]")

KNOWN_ABBREVIATIONS = {
    "adm",
    "cap",
    "cf",
    "col",
    "dra",
    "dr",
    "etc",
    "excmo",
    "fr",
    "gral",
    "ing",
    "lic",
    "mr",
    "mrs",
    "ms",
    "núm",
    "pag",
    "pág",
    "prof",
    "sra",
    "srta",
    "sr",
    "sto",
    "sta",
    "ud",
    "uds",
    "vd",
    "vds",
}

ARCHAIC_ACCENTED_FORMS = {
    "á",
    "é",
    "ó",
    "fué",
    "fuí",
    "dió",
    "vió",
    "vióse",
    "vióle",
    "viólo",
    "vióme",
    "vióte",
    "fé",
    "pié",
    "rió",
}

ROMAN_CONTEXT_WORDS = {
    "acto",
    "cap",
    "capitulo",
    "capítulo",
    "canto",
    "libro",
    "parte",
    "siglo",
}

COMMON_ROMAN_AMBIGUOUS_WORDS = {"mi", "di", "vi"}

ALLOWED_CHARS = set(
    " \t\r\n"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "ÁÉÍÓÚÜÑáéíóúüñ"
    "0123456789"
    ".,;:!?¡¿"
    "\"'’"
    "()[]{}"
    "-—–…"
    "«»"
    "/"
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def normalize_with_profile(text: str, profile: str) -> str:
    if profile == "none":
        return text
    if profile == "common":
        return preprocess_common_text(text)
    if profile in {"mls_pretrain", "spanish_librispeech"}:
        return preprocess_librispeech_text(text)
    if profile in {"finetune", "spanish_vox_cpm"}:
        return preprocess_vox_cpm_text(text)
    raise ValueError(f"Unsupported normalization profile: {profile}")


def normalize_token(token: str) -> str:
    return token.strip(" \t\r\n.,;:!?¡¿\"'’“”«»()[]{}").lower()


def is_probable_roman(token: str, previous_clean: str | None) -> bool:
    stripped = token.strip(" \t\r\n.,;:!?¡¿\"'’“”«»()[]{}")
    clean = stripped.lower()
    if len(clean) < 2 or not ROMAN_RE.fullmatch(clean):
        return False
    if clean in COMMON_ROMAN_AMBIGUOUS_WORDS:
        return False
    if previous_clean in ROMAN_CONTEXT_WORDS:
        return True
    # Avoid counting common Spanish words like mi/di/vi as roman numerals.
    return stripped.isupper() and len(stripped) >= 4


def add_example(examples: dict[str, list[dict]], key: str, item: dict, token: str | None = None, limit: int = 8):
    bucket = examples[key]
    if len(bucket) >= limit:
        return
    entry = {"id": item["id"], "text": item["text"][:240]}
    if token is not None:
        entry["token"] = token
    bucket.append(entry)


def read_metadata(path: Path, corpus: str, normalize_profile: str = "none"):
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) == 2:
                stem, text = row
                speaker = None
            elif len(row) >= 3:
                stem, speaker, text = row[:3]
            else:
                continue
            yield {
                "corpus": corpus,
                "id": stem,
                "speaker": speaker,
                "text": normalize_with_profile(text, normalize_profile),
                "duration": None,
            }


def read_manifest(path: Path, corpus: str, normalize_profile: str = "none"):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text") or item.get("transcript") or ""
            yield {
                "corpus": corpus,
                "id": item.get("audio") or item.get("path") or item.get("id") or "",
                "speaker": None,
                "text": normalize_with_profile(text, normalize_profile),
                "duration": item.get("duration"),
            }


def audit_items(items: list[dict]) -> dict:
    counts = Counter()
    token_counts = Counter()
    digits = Counter()
    abbreviations = Counter()
    period_abbreviations = Counter()
    romans = Counter()
    archaic = Counter()
    unusual_tokens = Counter()
    unusual_chars = Counter()
    punctuation = Counter()
    examples = defaultdict(list)
    speakers = set()
    durations = []

    for item in items:
        text = item["text"]
        counts["utterances"] += 1
        counts["characters"] += len(text)
        if item.get("speaker"):
            speakers.add(item["speaker"])
        if isinstance(item.get("duration"), (int, float)) and math.isfinite(item["duration"]):
            durations.append(float(item["duration"]))

        has_digit = False
        has_abbr = False
        has_period_abbr = False
        has_roman = False
        has_archaic = False
        has_unusual = False

        for char in text:
            if not char.isalnum() and not char.isspace():
                punctuation[char] += 1
            if char not in ALLOWED_CHARS:
                unusual_chars[char] += 1
                has_unusual = True

        tokens = tokenize(text)
        for idx, token in enumerate(tokens):
            clean = normalize_token(token)
            if not clean:
                continue
            previous_clean = normalize_token(tokens[idx - 1]) if idx > 0 else None
            token_counts[clean] += 1

            if HAS_DIGIT_RE.search(token):
                digits[token] += 1
                has_digit = True
                add_example(examples, "digits", item, token)

            if clean in KNOWN_ABBREVIATIONS:
                abbreviations[clean] += 1
                has_abbr = True
                add_example(examples, "abbreviations", item, token)

            if re.fullmatch(rf"(?:[{SPANISH_LETTERS}]\.){{2,}}|[{SPANISH_LETTERS}]{{1,6}}\.", token, re.I):
                period_abbreviations[token.lower()] += 1
                has_period_abbr = True
                add_example(examples, "period_abbreviations", item, token)

            if is_probable_roman(token, previous_clean):
                romans[clean.upper()] += 1
                has_roman = True
                add_example(examples, "roman_numerals", item, token)

            if clean in ARCHAIC_ACCENTED_FORMS:
                archaic[clean] += 1
                has_archaic = True
                add_example(examples, "archaic_accents", item, token)

            token_has_unusual_char = any(char not in ALLOWED_CHARS for char in token)
            token_has_letters = HAS_SPANISH_LETTER_RE.search(token) or HAS_ASCII_LETTER_RE.search(token)
            token_is_plain_word = WORD_RE.fullmatch(token) is not None
            if token_has_unusual_char or (token_has_letters and not token_is_plain_word and not HAS_DIGIT_RE.search(token)):
                unusual_tokens[token] += 1
                has_unusual = True
                add_example(examples, "unusual_tokens", item, token)

        counts["utterances_with_digits"] += int(has_digit)
        counts["utterances_with_known_abbreviations"] += int(has_abbr)
        counts["utterances_with_period_abbreviations"] += int(has_period_abbr)
        counts["utterances_with_roman_numerals"] += int(has_roman)
        counts["utterances_with_archaic_accents"] += int(has_archaic)
        counts["utterances_with_unusual_tokens_or_chars"] += int(has_unusual)

    duration_stats = {}
    if durations:
        durations_sorted = sorted(durations)
        duration_stats = {
            "total_hours": round(sum(durations) / 3600.0, 3),
            "mean_seconds": round(sum(durations) / len(durations), 3),
            "max_seconds": round(max(durations), 3),
            "p95_seconds": round(durations_sorted[int(0.95 * (len(durations_sorted) - 1))], 3),
        }

    return {
        "counts": dict(counts),
        "speakers": len(speakers) if speakers else None,
        "duration_stats": duration_stats,
        "top_tokens": token_counts.most_common(30),
        "digits": digits.most_common(40),
        "known_abbreviations": abbreviations.most_common(40),
        "period_abbreviations": period_abbreviations.most_common(40),
        "roman_numerals": romans.most_common(40),
        "archaic_accents": archaic.most_common(40),
        "unusual_tokens": unusual_tokens.most_common(80),
        "unusual_chars": unusual_chars.most_common(80),
        "punctuation": punctuation.most_common(80),
        "examples": dict(examples),
    }


def pct(part: int, total: int) -> str:
    return f"{(100.0 * part / total):.2f}%" if total else "0.00%"


def render_section(name: str, result: dict) -> str:
    counts = result["counts"]
    total = counts.get("utterances", 0)
    lines = [f"## {name}", ""]
    lines.append(f"- Utterances: {total:,}")
    if result["speakers"] is not None:
        lines.append(f"- Speakers: {result['speakers']:,}")
    if result["duration_stats"]:
        stats = result["duration_stats"]
        lines.append(
            f"- Audio durations: {stats['total_hours']:,}h total, "
            f"mean {stats['mean_seconds']}s, p95 {stats['p95_seconds']}s, max {stats['max_seconds']}s"
        )
    lines.extend(
        [
            f"- Utterances with digits: {counts.get('utterances_with_digits', 0):,} ({pct(counts.get('utterances_with_digits', 0), total)})",
            f"- Utterances with known abbreviations: {counts.get('utterances_with_known_abbreviations', 0):,} ({pct(counts.get('utterances_with_known_abbreviations', 0), total)})",
            f"- Utterances with period abbreviations: {counts.get('utterances_with_period_abbreviations', 0):,} ({pct(counts.get('utterances_with_period_abbreviations', 0), total)})",
            f"- Utterances with roman numerals: {counts.get('utterances_with_roman_numerals', 0):,} ({pct(counts.get('utterances_with_roman_numerals', 0), total)})",
            f"- Utterances with archaic accented forms: {counts.get('utterances_with_archaic_accents', 0):,} ({pct(counts.get('utterances_with_archaic_accents', 0), total)})",
            f"- Utterances with unusual tokens/chars: {counts.get('utterances_with_unusual_tokens_or_chars', 0):,} ({pct(counts.get('utterances_with_unusual_tokens_or_chars', 0), total)})",
            "",
        ]
    )

    for title, key in [
        ("Top digit tokens", "digits"),
        ("Known abbreviations", "known_abbreviations"),
        ("Period abbreviations", "period_abbreviations"),
        ("Roman numerals", "roman_numerals"),
        ("Archaic accented forms", "archaic_accents"),
        ("Unusual tokens", "unusual_tokens"),
        ("Unusual chars", "unusual_chars"),
    ]:
        lines.append(f"### {title}")
        values = result[key][:20]
        lines.append(", ".join(f"`{token}`={count:,}" for token, count in values) if values else "_None found._")
        lines.append("")

    if result["examples"]:
        lines.append("### Examples")
        for key, rows in result["examples"].items():
            lines.append(f"**{key}**")
            for row in rows[:5]:
                token = f" `{row['token']}`:" if "token" in row else ":"
                lines.append(f"- {row['id']}{token} {row['text']}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mls-train", type=Path, default=Path("data/es_mls_raw/train/metadata.csv"))
    parser.add_argument("--mls-val", type=Path, default=Path("data/es_mls_raw/val/metadata.csv"))
    parser.add_argument(
        "--narration-manifest",
        type=Path,
        default=Path("/mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/narration/_manifest.jsonl"),
    )
    parser.add_argument(
        "--words-manifest",
        type=Path,
        default=Path("/mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/words/_manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/es_text_audit"))
    parser.add_argument(
        "--mls-normalize-profile",
        choices=["none", "common", "mls_pretrain", "finetune", "spanish_librispeech", "spanish_vox_cpm"],
        default="none",
    )
    parser.add_argument(
        "--finetune-normalize-profile",
        choices=["none", "common", "mls_pretrain", "finetune", "spanish_librispeech", "spanish_vox_cpm"],
        default="none",
    )
    args = parser.parse_args()

    corpora = {
        "MLS pretrain": list(read_metadata(args.mls_train, "mls_train", args.mls_normalize_profile))
        + list(read_metadata(args.mls_val, "mls_val", args.mls_normalize_profile)),
        "Fine-tune narration": list(read_manifest(args.narration_manifest, "narration", args.finetune_normalize_profile)),
        "Fine-tune words": list(read_manifest(args.words_manifest, "words", args.finetune_normalize_profile)),
    }
    corpora["Fine-tune combined"] = corpora["Fine-tune narration"] + corpora["Fine-tune words"]

    results = {name: audit_items(items) for name, items in corpora.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit.json"
    md_path = args.output_dir / "audit.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Spanish Text Corpus Audit\n\n" + "\n".join(render_section(name, result) for name, result in results.items()),
        encoding="utf-8",
    )
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
