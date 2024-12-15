SYMBOLS = [
    "_",
    "^",
    "$",
    " ",
    "!",
    '"',
    "#",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    "?",
    "a",
    "aɪ",
    "aɪə",
    "aʊ",
    "b",
    "d",
    "d͡ʒ",
    "eə",
    "eɪ",
    "f",
    "h",
    "i",
    "iə",
    "iː",
    "j",
    "k",
    "l",
    "m",
    "n",
    "nʲ",
    "n̩",
    "oʊ",
    "p",
    "r",
    "s",
    "t",
    "t͡ʃ",
    "u",
    "uː",
    "v",
    "w",
    "x",
    "z",
    "æ",
    "ð",
    "ŋ",
    "ɐ",
    "ɑ",
    "ɑː",
    "ɑ̃",
    "ɒ",
    "ɔ",
    "ɔɪ",
    "ɔː",
    "ɔ̃",
    "ə",
    "əl",
    "əʊ",
    "ɚ",
    "ɛ",
    "ɜː",
    "ɡ",
    "ɡʲ",
    "ɪ",
    "ɬ",
    "ɹ",
    "ʃ",
    "ʊ",
    "ʊə",
    "ʌ",
    "ʒ",
    "ˈa",
    "ˈaɪ",
    "ˈaɪə",
    "ˈaʊ",
    "ˈeə",
    "ˈeɪ",
    "ˈi",
    "ˈiə",
    "ˈiː",
    "ˈiːː",
    "ˈoʊ",
    "ˈu",
    "ˈuː",
    "ˈæ",
    "ˈɐ",
    "ˈɑ",
    "ˈɑː",
    "ˈɑ̃",
    "ˈɒ",
    "ˈɔ",
    "ˈɔɪ",
    "ˈɔː",
    "ˈə",
    "ˈəl",
    "ˈəʊ",
    "ˈɚ",
    "ˈɛ",
    "ˈɜː",
    "ˈɪ",
    "ˈʊ",
    "ˈʊə",
    "ˈʌ",
    "ˌa",
    "ˌaɪ",
    "ˌaɪə",
    "ˌaʊ",
    "ˌeə",
    "ˌeɪ",
    "ˌi",
    "ˌiə",
    "ˌiː",
    "ˌoʊ",
    "ˌu",
    "ˌuː",
    "ˌæ",
    "ˌɑ",
    "ˌɑː",
    "ˌɒ",
    "ˌɔ",
    "ˌɔɪ",
    "ˌɔː",
    "ˌə",
    "ˌəʊ",
    "ˌɚ",
    "ˌɛ",
    "ˌɜː",
    "ˌɪ",
    "ˌʊ",
    "ˌʊə",
    "ˌʌ",
    "θ",
]


# Special symbols
PAD = "_"
BOS = "^"
EOS = "$"

# Special symbol ids
PAD_ID = SYMBOLS.index(PAD)
BOS_ID = SYMBOLS.index(BOS)
EOS_ID = SYMBOLS.index(EOS)
SPACE_ID = SYMBOLS.index(" ")

# Mappings from symbol to numeric ID and vice versa:
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}  # pylint: disable=unnecessary-comprehension


def phonemes_to_ids(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    for symbol in text:
        symbol_id = SYMBOL_TO_ID[symbol]
        sequence.append(symbol_id)
    return sequence


def ids_to_phonemes(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = ID_TO_SYMBOL[symbol_id]
        result += s
    return result
