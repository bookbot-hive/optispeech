"""Spanish MLS pretraining normalization recipe.

Every text normalization recipe must define:

    preprocess_text(text: str) -> str
"""

from recipes.text_normalization.spanish_common import (
    ABBREVIATION_REPLACEMENTS,
    MLS_WORD_REPLACEMENTS,
    normalize_common_text,
    replace_abbreviations,
    replace_whole_words,
)


def preprocess_text(text: str) -> str:
    text = normalize_common_text(text)
    text = replace_whole_words(text, MLS_WORD_REPLACEMENTS)
    text = replace_abbreviations(text, ABBREVIATION_REPLACEMENTS)
    return text
