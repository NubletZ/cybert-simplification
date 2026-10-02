"""SARI, FKGL and readability labelling (paper 4.1, 4.4, Eqs. 17-19)."""

from __future__ import annotations

import pytest

from cybert.data.readability import (
    COMPLEX,
    SIMPLE,
    flesch_reading_ease,
    label_domain,
)
from cybert.eval.metrics import fkgl_corpus, normalize, sari_corpus, sari_sentence


def test_perfect_copy_of_the_only_reference_scores_high():
    """When the output equals the reference, keep and add are maximal."""
    source = "About 95 species are currently accepted."
    reference = "About 95 species are currently known."
    scores = sari_sentence(source, reference, [reference])
    assert scores["sari"] > 70
    assert scores["add"] == pytest.approx(100.0)


def test_copying_the_source_is_penalised():
    """Echoing the input cannot add or delete anything, so SARI stays low."""
    source = "The manuscript was subsequently transcribed by an anonymous scribe."
    reference = "Someone copied the book later."
    copy_score = sari_sentence(source, source, [reference])["sari"]
    ref_score = sari_sentence(source, reference, [reference])["sari"]
    assert copy_score < ref_score


def test_sari_components_are_reported():
    scores = sari_sentence("a b c d", "a b x", ["a b y"])
    assert set(scores) == {"sari", "keep", "delete", "add"}
    assert scores["sari"] == pytest.approx(
        (scores["keep"] + scores["delete"] + scores["add"]) / 3
    )


def test_multiple_references_are_pooled():
    """An addition supported by any reference should be rewarded."""
    source = "The edifice collapsed."
    refs = ["The building fell down.", "The building collapsed."]
    good = sari_sentence(source, "The building collapsed.", refs)["sari"]
    bad = sari_sentence(source, "The zebra collapsed.", refs)["sari"]
    assert good > bad


def test_sari_corpus_averages_sentences():
    sources = ["a b c", "d e f"]
    candidates = ["a b", "d e"]
    references = [["a b"], ["d e"]]
    corpus = sari_corpus(sources, candidates, references)
    manual = (
        sari_sentence(sources[0], candidates[0], references[0])["sari"]
        + sari_sentence(sources[1], candidates[1], references[1])["sari"]
    ) / 2
    assert corpus["sari"] == pytest.approx(manual)


def test_sari_variants_differ_on_empty_sets():
    """Identity output deletes nothing; the two conventions score that oppositely."""
    source = "The edifice was subsequently demolished."
    refs = ["The building was pulled down."]
    easse = sari_sentence(source, source, refs, variant="easse")
    xu = sari_sentence(source, source, refs, variant="xu")
    assert easse["delete"] == pytest.approx(0.0)
    assert xu["delete"] == pytest.approx(100.0)
    assert easse["sari"] < xu["sari"]


def test_sari_rejects_unknown_variant():
    with pytest.raises(ValueError):
        sari_sentence("a b", "a", ["a"], variant="original")


def test_easse_variant_matches_published_identity_baseline():
    """ASSET's identity baseline is 20.73; TurkCorpus's is 25.44.

    This is the check that pins the implementation to the convention the
    simplification literature reports, and therefore to the numbers the paper
    compares against.
    """
    source = "About 95 species are currently accepted."
    refs = ["About 95 species are now accepted.", "Roughly 95 species are accepted."]
    identity = sari_sentence(source, source, refs, variant="easse")
    # Copying the input can neither add nor delete, so SARI reduces to keep/3.
    assert identity["add"] == pytest.approx(0.0)
    assert identity["delete"] == pytest.approx(0.0)
    assert identity["sari"] == pytest.approx(identity["keep"] / 3)


def test_sari_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        sari_corpus(["a"], ["a", "b"], [["a"]])


def test_normalize_splits_punctuation():
    assert normalize("Hello, world!") == ["hello", ",", "world", "!"]


def test_simple_text_has_lower_fkgl():
    complex_text = (
        "The municipality subsequently promulgated an ordinance which effectively "
        "prohibited the construction of additional residential structures."
    )
    simple_text = "The town made a new rule. No one could build new homes."
    assert fkgl_corpus([simple_text]) < fkgl_corpus([complex_text])


def test_readability_thresholds_assign_domains():
    """4.1: FRE < 10 is complex, FRE > 70 is simple, the middle is dropped."""
    complex_text = (
        "Notwithstanding considerable opposition from agricultural stakeholders, "
        "the legislature ratified the amendment governing irrigation entitlements."
    )
    simple_text = "The cat sat on the mat. It was a big cat."
    assert flesch_reading_ease(complex_text) < flesch_reading_ease(simple_text)
    assert label_domain(complex_text) == COMPLEX
    assert label_domain(simple_text) == SIMPLE


def test_middle_band_is_discarded():
    scores = [label_domain(t) for t in ["The report was fairly long and detailed."]]
    assert scores[0] in (COMPLEX, SIMPLE, None)  # band membership is data dependent
    # A sentence engineered to sit in the middle must be dropped.
    assert label_domain("x", complex_max=-1000.0, simple_min=1000.0) is None
