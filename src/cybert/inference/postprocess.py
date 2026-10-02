"""Detokenisation of decoder output.

The decoder emits WordPiece ids, and ``batch_decode`` puts a space around every
piece: ``Islam's`` comes back as ``islam ' s`` and ``able-bodied`` as
``able - bodied``.

SARI is unaffected -- its tokeniser splits punctuation off both the candidate
and the references, so the two forms normalise identically. FKGL is *not*:
``[A-Za-z][A-Za-z'\\-]*`` counts ``islam ' s`` as two words rather than one,
which raises words-per-sentence and inflates the reported grade level. Reported
readability should describe the text a reader would see, so output is
detokenised before scoring.

Casing is not restored. ``bert-base-uncased`` discards it in the tokeniser, so
the information is genuinely gone; SARI and FKGL are both case-insensitive, so
nothing measured here depends on it.
"""

from __future__ import annotations

import re

# " ' s" -> "'s"; also 'll, 've, 're, 'd, 'm, n't
_CLITIC = re.compile(r"\s+'\s*(s|ll|ve|re|d|m|t)\b")
# "did n ' t" -> "didn't"; the preceding space is absorbed too. ("didn ' t",
# which is what bert-base-uncased actually produces, is handled by _CLITIC.)
_NT = re.compile(r"(?<=\w)\s+n\s*'\s*t\b")
# Space before closing punctuation.
_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}])")
# Space after opening punctuation.
_AFTER_OPEN = re.compile(r"([(\[{$])\s+")
# Hyphen and slash between word characters: "able - bodied" -> "able-bodied".
_INFIX = re.compile(r"(?<=\w)\s+([-/])\s+(?=\w)")
# Space inside a decimal or thousands separator: "3 . 5" -> "3.5".
_NUMERIC = re.compile(r"(?<=\d)\s+([.,])\s+(?=\d)")
_QUOTES = re.compile(r'\s+"\s*')
_MULTISPACE = re.compile(r"\s{2,}")


def detokenize(text: str) -> str:
    """Undo WordPiece spacing so the string reads as ordinary text."""
    if not text:
        return text
    out = _NT.sub("n't", text)
    out = _CLITIC.sub(r"'\1", out)
    out = _NUMERIC.sub(r"\1", out)
    out = _INFIX.sub(r"\1", out)
    out = _BEFORE_PUNCT.sub(r"\1", out)
    out = _AFTER_OPEN.sub(r"\1", out)
    # Pair up double quotes: alternate opening and closing.
    parts = _QUOTES.split(out)
    if len(parts) > 1:
        rebuilt = parts[0]
        for i, part in enumerate(parts[1:]):
            rebuilt += '"' + part if i % 2 == 0 else '" ' + part
        out = rebuilt
    out = _MULTISPACE.sub(" ", out)
    return out.strip()


def detokenize_batch(texts: list[str]) -> list[str]:
    return [detokenize(t) for t in texts]
