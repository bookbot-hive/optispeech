from abc import ABC, abstractmethod

from piper_phonemize import phonemize_espeak
from gruut import sentences

from ..utils import intersperse
from . import symbols
from . import gruut_symbols
from .normalization import UNICODE_NORM_FORM, collapse_whitespace, preprocess_text

# tokenizer registry
_TOKENIZERS = {}


class BaseTokenizer(ABC):
    name: str
    input_symbols: dict[str, int]
    special_symbols: dict[str, int]

    def __init_subclass__(cls, /, **kwargs):
        _TOKENIZERS.setdefault(cls.name, cls)

    @classmethod
    def get_tokenizer_by_name(cls, name):
        try:
            return _TOKENIZERS[name]
        except KeyError:
            raise ValueError(f"Tokenizer `{name}` does not exist.")

    def __init__(
        self,
        add_blank: bool,
        add_bos_eos: bool,
        normalize_text: bool,
    ):
        self.add_blank = add_blank
        self.add_bos_eos = add_bos_eos
        self.normalize_text = normalize_text

    @abstractmethod
    def __call__(
        self, text: str, language: str, *, split_sentences: bool = True
    ) -> tuple[list[int] | list[list[int]], str]:
        """Return input IDs."""

    def preprocess_text(self, text: str, language: str = None) -> str:
        return preprocess_text(text, language, normalize=self.normalize_text)


class GruutTokenizer(BaseTokenizer):
    name = "gruut"
    input_symbols = gruut_symbols.SYMBOL_TO_ID
    special_symbols = dict(
        pad=gruut_symbols.PAD,
        bos=gruut_symbols.BOS,
        eos=gruut_symbols.EOS,
    )

    def __call__(
        self, text: str, language: str, *, split_sentences: bool = False
    ) -> tuple[list[int] | list[list[int]], str]:
        phonemes, normalized_text = self.phonemize_text(text, language)
        phoneme_ids = gruut_symbols.phonemes_to_ids(phonemes)
        return phoneme_ids, normalized_text

    def phonemize_text(self, text: str, language: str) -> str:
        text = self.preprocess_text(text, language)
        phonemes = []
        for sentence in sentences(text, lang=language):
            sent_ph = []
            for idx, word in enumerate(sentence):
                if word.is_major_break or word.is_minor_break:
                    sent_ph.append(word.text)
                elif word.text == '"':
                    sent_ph.append('"')
                elif word.phonemes:
                    sent_ph += word.phonemes

                if word.trailing_ws and idx < len(sentence) - 1:
                    sent_ph.append(" ")
            phonemes += sent_ph
        return phonemes, text


class IPATokenizer(BaseTokenizer):
    name = "ipa"
    input_symbols = symbols.SYMBOL_TO_ID
    special_symbols = dict(
        pad=symbols.PAD,
        bos=symbols.BOS,
        eos=symbols.EOS,
    )

    def __call__(
        self, text: str, language: str, *, split_sentences: bool = True
    ) -> tuple[list[int] | list[list[int]], str]:
        phonemes, normalized_text = self.phonemize_text(text, language)
        if not split_sentences:
            phonemes = [phoneme for sentence_phonemes in phonemes for phoneme in sentence_phonemes]
            phonemes = list(collapse_whitespace("".join(phonemes)))
            phoneme_ids = symbols.phonemes_to_ids(phonemes)
            if self.add_blank:
                phoneme_ids = intersperse(phoneme_ids, 0)
            if self.add_bos_eos:
                phoneme_ids = [
                    symbols.BOS_ID,
                    *phoneme_ids,
                    symbols.EOS_ID,
                ]
        else:
            phoneme_ids = []
            for sent_ph in phonemes:
                sent_phonemes = list(collapse_whitespace("".join(sent_ph)))
                phids = symbols.phonemes_to_ids(sent_phonemes)
                if self.add_blank:
                    phids = intersperse(phids, 0)
                if self.add_bos_eos:
                    phids = [symbols.BOS_ID, *phids, symbols.EOS_ID]
                phoneme_ids.append(phids)
        return phoneme_ids, normalized_text

    def phonemize_text(self, text: str, language: str) -> str:
        # Preprocess
        text = self.preprocess_text(text, language)
        # Phonemize
        phonemes = phonemize_espeak(text, language)
        return phonemes, text
