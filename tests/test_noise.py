"""POS-guided span-infilling corruption (paper 3.2.1, Eqs. 1-3)."""

from __future__ import annotations

import statistics

import pytest

from cybert.data.noise import MASK_TOKEN, SpanInfillingNoise
from cybert.data.pos_tagger import whitespace_tokenize

K_POS = ("VERB", "NOUN", "AUX", "ADJ", "ADV")


def make_noise(**kwargs) -> SpanInfillingNoise:
    defaults = dict(poisson_lambda=3.0, mask_ratio=0.3, max_spans=8, seed=0)
    defaults.update(kwargs)
    return SpanInfillingNoise(**defaults)


def test_poisson_mean_matches_lambda():
    noise = make_noise()
    samples = [noise._poisson() for _ in range(20000)]
    assert statistics.mean(samples) == pytest.approx(3.0, abs=0.1)
    # Eq. 3.2.1 relies on L = 0 being reachable; that is the insertion signal.
    assert 0 in samples


def test_disabled_noise_is_identity():
    noise = make_noise(enabled=False)
    tokens = "the quick brown fox jumps over the lazy dog".split()
    assert noise.corrupt_tokens(tokens) == tokens


def test_mask_token_is_inserted():
    noise = make_noise()
    tokens = "the committee subsequently approved the revised proposal today".split()
    out = noise.corrupt_tokens(tokens)
    assert MASK_TOKEN in out


def test_zero_length_span_inserts_without_deleting():
    """L = 0 must lengthen the sequence: a [MASK] appears, nothing is removed."""
    noise = make_noise(mask_ratio=1.0, max_spans=1)
    noise._poisson = lambda: 0  # force a zero-length span
    tokens = "alpha beta gamma delta".split()
    out = noise.corrupt_tokens(tokens)
    assert len(out) == len(tokens) + 1
    assert out.count(MASK_TOKEN) == 1
    assert [t for t in out if t != MASK_TOKEN] == tokens


def test_pos_guided_starts_land_on_target_categories():
    """With a POS-tagged sentence, span starts should sit on K_POS tokens."""
    tokens = "the cat quickly ate a very large fish".split()
    tags = ["DET", "NOUN", "ADV", "VERB", "DET", "ADV", "ADJ", "NOUN"]
    pos_words = {tokens[i] for i, tag in enumerate(tags) if tag in K_POS}

    hits = 0
    trials = 200
    for seed in range(trials):
        noise = make_noise(seed=seed, mask_ratio=0.15, max_spans=1)
        out = noise.corrupt_tokens(tokens, tags)
        # The token immediately following a [MASK] is either the anchor (for a
        # zero-length span) or the token after the deleted span. Instead of
        # reasoning about that, recompute which token index was removed.
        removed = [t for t in tokens if t not in out]
        if removed and removed[0] in pos_words:
            hits += 1
        elif not removed:
            hits += 1  # zero-length insertion, anchored on a candidate
    assert hits / trials >= 0.9


def test_without_tags_falls_back_to_uniform_starts():
    """tags=None is the random-infilling ablation and must still corrupt."""
    noise = make_noise(seed=3)
    tokens = "one two three four five six seven eight".split()
    out = noise.corrupt_tokens(tokens, None)
    assert MASK_TOKEN in out


def test_spans_do_not_overlap():
    """Every surviving token must appear exactly once, in its original order."""
    tokens = [f"w{i}" for i in range(40)]
    for seed in range(50):
        noise = make_noise(seed=seed, mask_ratio=0.4, max_spans=6)
        out = noise.corrupt_tokens(tokens)
        survivors = [t for t in out if t != MASK_TOKEN]
        assert survivors == [t for t in tokens if t in survivors]
        assert len(survivors) == len(set(survivors))


def test_corrupt_roundtrips_through_strings():
    noise = make_noise(seed=1)
    text = "the municipality promulgated an ordinance restricting new construction"
    corrupted = noise.corrupt(text)
    assert isinstance(corrupted, str)
    assert len(whitespace_tokenize(corrupted)) > 0


def test_empty_input_is_safe():
    noise = make_noise()
    assert noise.corrupt_tokens([]) == []
    assert noise.corrupt("") == ""
