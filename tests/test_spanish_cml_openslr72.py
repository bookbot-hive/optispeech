from recipes.text_normalization.spanish_cml_openslr72 import preprocess_text


def test_normalizes_historical_cml_spellings_and_abbreviations():
    assert preprocess_text("El Sr. vió el pié.") == "El Señor vio el pie."


def test_preserves_modern_diacritics():
    text = "María vio el río del país y volvió a oírlo."
    assert preprocess_text(text) == text


def test_normalizes_typography_without_removing_punctuation():
    assert preprocess_text("— ¿Hola…?  ¡Sí;claro!") == '— ¿Hola...? ¡Sí; claro!'


def test_is_effectively_common_cleanup_for_openslr72():
    text = "¿Puedes revisar si hay alguna tienda cerca de la casa de mis papás?"
    assert preprocess_text(text) == text
