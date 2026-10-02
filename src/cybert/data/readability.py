"""Readability scoring and domain labelling.

Paper 4.1 assigns unpaired Wikipedia sentences to the complex / simple domains
by Flesch Reading Ease, following Surya et al. (2019): ``FRE < 10`` is complex,
``FRE > 70`` is simple. Everything in between is discarded.

FKGL (Eq. 19) is also defined here because it is one of the reported metrics.
"""

from __future__ import annotations

import re

COMPLEX = 0
SIMPLE = 1

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
_VOWELS = "aeiouy"


def count_syllables(word: str) -> int:
    """Heuristic English syllable count.

    Standard vowel-group counter with the usual silent-``e`` correction. Exact
    syllabification needs a pronunciation dictionary; both FRE and FKGL are
    corpus-level averages, so the heuristic is what these metrics are normally
    computed with.
    """
    word = word.lower().strip("'-")
    if not word:
        return 0
    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in _VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(1, count)


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def count_sentences(text: str) -> int:
    parts = [p for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return max(1, len(parts))


def flesch_reading_ease(text: str) -> float:
    """FRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)."""
    words = tokenize_words(text)
    if not words:
        return 0.0
    sentences = count_sentences(text)
    syllables = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def fkgl(text: str) -> float:
    """Flesch-Kincaid Grade Level, Eq. 19."""
    words = tokenize_words(text)
    if not words:
        return 0.0
    sentences = count_sentences(text)
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59


def corpus_fkgl(texts: list[str]) -> float:
    """Corpus-level FKGL: ratios are pooled before the formula is applied.

    This is how FKGL is reported for simplification systems -- averaging the
    per-sentence scores instead would over-weight very short outputs.
    """
    total_words = 0
    total_sentences = 0
    total_syllables = 0
    for text in texts:
        words = tokenize_words(text)
        if not words:
            continue
        total_words += len(words)
        total_sentences += count_sentences(text)
        total_syllables += sum(count_syllables(w) for w in words)
    if total_words == 0 or total_sentences == 0:
        return 0.0
    return 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59


def label_domain(text: str, complex_max: float = 10.0, simple_min: float = 70.0) -> int | None:
    """Return ``COMPLEX``, ``SIMPLE``, or ``None`` for the discarded middle band."""
    score = flesch_reading_ease(text)
    if score < complex_max:
        return COMPLEX
    if score > simple_min:
        return SIMPLE
    return None
