from typing import Any

from .normalization import UNICODE_NORM_FORM
from pkg_resources import resource_filename
import os

TEXT_PROCESSOR_CONFIG = {
    "MODEL_DIRS": {
        "en": "bookbot/roberta-base-emphasis-onnx-quantized",
    },
    "DB_PATHS": {
        "en": resource_filename('text_processor', 'data/en_word_emphasis_lookup_mix_homographs.json'),
    },
    "COSMOS_CONFIG": {
        "url": os.getenv("COSMOS_DB_URL"),
        "key": os.getenv("COSMOS_DB_KEY"),
        "database_name": "Bookbot"
    }
}

class TextProcessor:
    def __init__(
        self, tokenizer: str|Any, add_blank: str, add_bos_eos: str, normalize_text: bool, languages: list[str]
    ):
        self.tokenizer_ref = tokenizer
        self.add_blank = add_blank
        self.add_bos_eos = add_bos_eos
        self.normalize_text = normalize_text
        self.languages = languages
        self.tokenizers = {}
        from text_processor import TextProcessor
        for language in languages:
            if "en" in language:
                language = "en"
            if language not in self.tokenizers:
                self.tokenizers[language] = TextProcessor(TEXT_PROCESSOR_CONFIG["MODEL_DIRS"][language], 
                                                        TEXT_PROCESSOR_CONFIG["DB_PATHS"][language], 
                                                        language=language) 
        self.num_languages = len(languages)
        self.is_multi_language = self.num_languages > 1
        self.default_language = languages[0].strip().lower()

    def __call__(self, text, lang, split_sentences: bool = False):
        if lang is None:
            lang = self.default_language
        lang = lang.strip().lower()
        if lang not in self.languages:
            raise ValueError(f"Language {lang} does not exist in the supported language list.")
        if "en" in lang:
            lang = "en"
        return self.tokenizers[lang].get_input_ids(text, 
                                                   phonemes=False, 
                                                   return_phonemes=True, 
                                                   add_blank_token=self.add_blank,
                                                   normalize=self.normalize_text)

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)

    def asdict(self):
        return dict(
            tokenizer=self.tokenizer.name,
            add_blank=self.add_blank,
            add_bos_eos=self.add_bos_eos,
            normalize_text=self.normalize_text,
            languages=self.languages,
        )
