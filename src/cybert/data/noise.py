"""POS-guided span-infilling corruption (paper 3.2.1, Eqs. 1-3).

The corruption operates on whitespace tokens, before WordPiece, for two
reasons: POS tags align one-to-one with the units being deleted, and the
corrupted string re-tokenizes cleanly with ``[MASK]`` left intact.

Procedure, per the paper:

1. Candidate span starts come from ``S_POS(x) = {i : POS(x_i) in K_POS}``
   (Eq. 2), restricted to positions that keep spans non-overlapping.
2. A start is drawn ``Uniform(V_POS)`` when that set is non-empty and
   ``Uniform(V_all)`` otherwise (Eq. 3).
3. Span length ``L ~ Poisson(3)``; ``x[i:i+L]`` is replaced by a *single*
   ``[MASK]``. ``L = 0`` inserts a ``[MASK]`` without deleting anything, which
   is the insertion signal the paper ties to SARI's Add score.
4. Spans that would run past the end are truncated.

Sampling stops once ``mask_ratio`` of the tokens have been covered or
``max_spans`` spans have been placed.
"""

from __future__ import annotations

import random

from .pos_tagger import whitespace_tokenize

MASK_TOKEN = "[MASK]"


class SpanInfillingNoise:
    """Callable corruption function.

    Parameters mirror :class:`cybert.config.NoiseConfig`. ``enabled=False``
    returns the input unchanged (the "w/o noise" rows of Table 8); the
    ``use_pos`` flag is applied by the caller, which simply passes ``tags=None``
    to get uniform span starts (the "w/ IN" rows).
    """

    def __init__(
        self,
        poisson_lambda: float = 3.0,
        mask_ratio: float = 0.30,
        max_spans: int = 8,
        pos_categories: tuple[str, ...] = ("VERB", "NOUN", "AUX", "ADJ", "ADV"),
        enabled: bool = True,
        mask_token: str = MASK_TOKEN,
        seed: int | None = None,
    ) -> None:
        self.poisson_lambda = poisson_lambda
        self.mask_ratio = mask_ratio
        self.max_spans = max_spans
        self.pos_categories = set(pos_categories)
        self.enabled = enabled
        self.mask_token = mask_token
        self._rng = random.Random(seed)

    # -- sampling helpers -------------------------------------------------- #

    def _poisson(self) -> int:
        """Knuth's method; the sample is a span *length*, so 0 is meaningful."""
        import math

        limit = math.exp(-self.poisson_lambda)
        k = 0
        p = 1.0
        while True:
            p *= self._rng.random()
            if p <= limit:
                return k
            k += 1
            if k > 50:  # guard; P(k>50 | lambda=3) is vanishing
                return k

    def _candidate_starts(
        self, n_tokens: int, tags: list[str] | None, taken: list[bool]
    ) -> tuple[list[int], list[int]]:
        """Return ``(V_POS, V_all)``: free positions, POS-restricted and not."""
        v_all = [i for i in range(n_tokens) if not taken[i]]
        if tags is None:
            return [], v_all
        v_pos = [i for i in v_all if i < len(tags) and tags[i] in self.pos_categories]
        return v_pos, v_all

    # -- main entry point -------------------------------------------------- #

    def corrupt_tokens(self, tokens: list[str], tags: list[str] | None = None) -> list[str]:
        """Corrupt a list of whitespace tokens, returning the corrupted list."""
        if not self.enabled or not tokens:
            return list(tokens)

        n = len(tokens)
        budget = max(1, int(round(self.mask_ratio * n)))
        # Each span replaces its tokens with one [MASK], so a span of length L
        # "covers" L tokens. Zero-length spans cost nothing against the budget
        # but still count as a span.
        taken = [False] * n
        spans: list[tuple[int, int]] = []  # (start, length)
        covered = 0

        for _ in range(self.max_spans):
            if covered >= budget:
                break
            v_pos, v_all = self._candidate_starts(n, tags, taken)
            pool = v_pos if v_pos else v_all
            if not pool:
                break
            start = self._rng.choice(pool)

            length = self._poisson()
            # Truncate at the sentence boundary and at the next occupied token,
            # which is what keeps spans non-overlapping.
            max_len = 0
            while start + max_len < n and not taken[start + max_len]:
                max_len += 1
            length = min(length, max_len)

            spans.append((start, length))
            if length == 0:
                # An insertion still consumes the anchor position, otherwise the
                # same slot could be chosen repeatedly.
                taken[start] = True
            else:
                for j in range(start, start + length):
                    taken[j] = True
                covered += length

        if not spans:
            return list(tokens)

        return self._apply(tokens, spans)

    def _apply(self, tokens: list[str], spans: list[tuple[int, int]]) -> list[str]:
        """Rebuild the sequence with each span replaced by one ``[MASK]``."""
        spans = sorted(spans)
        out: list[str] = []
        i = 0
        span_idx = 0
        while i < len(tokens):
            if span_idx < len(spans) and spans[span_idx][0] == i:
                start, length = spans[span_idx]
                out.append(self.mask_token)
                span_idx += 1
                if length == 0:
                    # Insertion: the anchor token itself survives.
                    out.append(tokens[i])
                    i += 1
                else:
                    i += length
            else:
                out.append(tokens[i])
                i += 1
        # A span anchored past the last token (only possible for an empty
        # sequence) would be dropped above; nothing to do here.
        return out

    def corrupt(self, text: str, tags: list[str] | None = None) -> str:
        """Corrupt a raw sentence string."""
        tokens = whitespace_tokenize(text)
        return " ".join(self.corrupt_tokens(tokens, tags))

    def __call__(self, text: str, tags: list[str] | None = None) -> str:
        return self.corrupt(text, tags)


def build_noise(noise_cfg, seed: int | None = None) -> SpanInfillingNoise:
    """Construct the corruption function from a :class:`NoiseConfig`."""
    return SpanInfillingNoise(
        poisson_lambda=noise_cfg.poisson_lambda,
        mask_ratio=noise_cfg.mask_ratio,
        max_spans=noise_cfg.max_spans,
        pos_categories=tuple(noise_cfg.pos_categories),
        enabled=noise_cfg.enabled,
        seed=seed,
    )
