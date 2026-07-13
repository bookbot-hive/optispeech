"""Shared Spanish text normalization helpers for recipe modules."""

from __future__ import annotations

import re
import unicodedata


_WORD_BOUNDARY_TEMPLATE = r"(?<![\wÁÉÍÓÚÜÑáéíóúüñ]){}(?![\wÁÉÍÓÚÜÑáéíóúüñ])"

COMMON_REPLACEMENTS = (
    ("…", "..."),
    ("–", "-"),
    ("«", '"'),
    ("»", '"'),
    ("“", '"'),
    ("”", '"'),
)

MLS_WORD_REPLACEMENTS = {
    "á": "a",
    "é": "e",
    "ó": "o",
    "fué": "fue",
    "fuí": "fui",
    "dió": "dio",
    "vió": "vio",
    "vióse": "viose",
    "vióle": "viole",
    "viólo": "violo",
    "vióme": "viome",
    "vióte": "viote",
    "fé": "fe",
    "pié": "pie",
    "rió": "rio",
}

FINETUNE_WORD_REPLACEMENTS = {
    # Gruut emits the same phonemes for rió/rio, while río remains distinct.
    "rió": "rio",
}

FINETUNE_PHRASE_REPLACEMENTS = {
    "Dra. Sí Puede": "Doctor Sí Puede",
    "Dra Sí Puede": "Doctor Sí Puede",
}

ABBREVIATION_REPLACEMENTS = {
    "vd": "usted",
    "vds": "ustedes",
    "ud": "usted",
    "uds": "ustedes",
    "sr": "señor",
    "sra": "señora",
    "srta": "señorita",
    "dr": "doctor",
    "dra": "doctora",
    "mr": "señor",
    "mrs": "señora",
    "sto": "santo",
    "sta": "santa",
    "núm": "número",
    "pág": "página",
    "pag": "página",
    "etc": "etcétera",
}

FINETUNE_ABBREVIATION_REPLACEMENTS = {
    "dr": "doctor",
    "dra": "doctora",
    "sto": "santo",
}

DIGIT_REPLACEMENTS = {
    "2B": "dos B",
    "10": "diez",
    "15": "quince",
    "2": "dos",
    "30": "treinta",
    "115": "ciento quince",
    "5.974.000.000.000.000.000.000": "cinco mil novecientos setenta y cuatro trillones",
}


def match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def replace_whole_words(text: str, replacements: dict[str, str]) -> str:
    for source, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(_WORD_BOUNDARY_TEMPLATE.format(re.escape(source)), re.IGNORECASE)
        text = pattern.sub(lambda match: match_case(match.group(0), replacement), text)
    return text


def replace_abbreviations(text: str, replacements: dict[str, str]) -> str:
    for source, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(_WORD_BOUNDARY_TEMPLATE.format(re.escape(source)) + r"\.?", re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            matched_word = match.group(0).rstrip(".")
            return match_case(matched_word, replacement)

        text = pattern.sub(replace, text)
    return text


def replace_digit_tokens(text: str) -> str:
    for source, replacement in sorted(DIGIT_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<![\w.]){re.escape(source)}(?!\w)", re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text


def replace_phrases(text: str, replacements: dict[str, str]) -> str:
    for source, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(re.escape(source), replacement, text, flags=re.IGNORECASE)
    return text


def normalize_common_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for source, replacement in COMMON_REPLACEMENTS:
        text = text.replace(source, replacement)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", text)
    text = re.sub(r"([,;:])(?!\s|$)", r"\1 ", text)
    return text.strip()
