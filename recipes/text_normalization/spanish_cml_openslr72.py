"""Spanish normalization for the combined CML-TTS and OpenSLR72 corpus.

The CML replacements canonicalize audited historical spellings and common
abbreviations. They are deliberately word-specific, so applying this recipe to
OpenSLR72 is equivalent to common typography cleanup for its current texts.
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
