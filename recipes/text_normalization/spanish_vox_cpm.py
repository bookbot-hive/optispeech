"""Spanish single-speaker fine-tuning normalization recipe.

Every text normalization recipe must define:

    preprocess_text(text: str) -> str
"""

from recipes.text_normalization.spanish_common import (
    FINETUNE_ABBREVIATION_REPLACEMENTS,
    FINETUNE_PHRASE_REPLACEMENTS,
    FINETUNE_WORD_REPLACEMENTS,
    normalize_common_text,
    replace_abbreviations,
    replace_digit_tokens,
    replace_phrases,
    replace_whole_words,
)


def preprocess_text(text: str) -> str:
    text = normalize_common_text(text)
    text = replace_digit_tokens(text)
    text = replace_phrases(text, FINETUNE_PHRASE_REPLACEMENTS)
    text = replace_whole_words(text, FINETUNE_WORD_REPLACEMENTS)
    text = replace_abbreviations(text, FINETUNE_ABBREVIATION_REPLACEMENTS)
    return text
