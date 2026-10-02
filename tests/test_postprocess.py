"""Detokenisation of WordPiece decoder output."""

from __future__ import annotations

import pytest

from cybert.data.readability import fkgl, tokenize_words
from cybert.eval.metrics import normalize, sari_sentence
from cybert.inference.postprocess import detokenize, detokenize_batch


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("islam ' s holiest city", "islam's holiest city"),
        ("able - bodied muslims", "able-bodied muslims"),
        ("the cat sat , quietly .", "the cat sat, quietly."),
        ("it did n ' t work", "it didn't work"),
        ("they ' ll arrive", "they'll arrive"),
        ("about 3 . 5 million", "about 3.5 million"),
        ("( see below )", "(see below)"),
        ("the input / output pair", "the input/output pair"),
        ("spaced   out    words", "spaced out words"),
    ],
)
def test_detokenize_cases(raw, expected):
    assert detokenize(raw) == expected


def test_detokenize_handles_empty_and_batch():
    assert detokenize("") == ""
    assert detokenize_batch(["a ,  b", ""]) == ["a, b", ""]


def test_detokenize_fixes_the_fkgl_word_count():
    """The reason this exists: FKGL counts words by regex.

    ``islam ' s`` is counted as the two words ``islam`` and ``s``, and
    ``able - bodied`` as two more, so the raw form describes the tokenisation
    rather than the text.

    Note the direction: correcting the count *raises* FKGL here. Merging
    ``islam`` + ``s`` into the three-syllable ``islam's`` removes a word
    without removing its syllables, so syllables-per-word rises and the
    11.8 coefficient dominates the 0.39 words-per-sentence term. The
    detokenised value is the honest one -- these really are single words.
    """
    raw = "jeddah is islam ' s holiest city , which able - bodied muslims visit ."
    clean = detokenize(raw)

    assert tokenize_words(clean) == [
        "jeddah", "is", "islam's", "holiest", "city", "which",
        "able-bodied", "muslims", "visit",
    ]
    assert len(tokenize_words(raw)) == len(tokenize_words(clean)) + 2
    assert fkgl(clean) != fkgl(raw)


def test_detokenize_leaves_sari_unchanged():
    """SARI's tokeniser splits punctuation off, so both forms normalise alike."""
    raw = "islam ' s holiest city"
    clean = detokenize(raw)
    assert normalize(raw) == normalize(clean)

    source = "Jeddah is the principal gateway to Islam's holiest city."
    refs = ["Jeddah is the main gateway to Islam's holiest city."]
    assert sari_sentence(source, raw, refs)["sari"] == pytest.approx(
        sari_sentence(source, clean, refs)["sari"]
    )
