"""Spanish common punctuation/Unicode cleanup recipe."""

from recipes.text_normalization.spanish_common import normalize_common_text


def preprocess_text(text: str) -> str:
    return normalize_common_text(text)
