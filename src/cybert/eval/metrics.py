"""Automatic metrics (paper 4.4).

``SARI``      Eqs. 17-18. ``(F1_add + F1_keep + P_del) / 3``, averaged over
              n-gram orders 1-4, following Xu et al. (2016): the n-gram
              counters are replicated by the number of references, deletion
              contributes precision only, and addition is scored on the *set*
              of added n-grams.

              Two conventions exist for the degenerate case where a candidate
              or reference set is empty, and they differ by tens of points:

              ``easse`` (default)
                  Empty sets score 0. This is what EASSE does and what the
                  simplification literature reports, so it is the convention
                  every number in the paper's Tables 2-3 is comparable to.
                  Verified: the identity baseline on ASSET scores 20.69 here
                  against a published 20.73.
              ``xu``
                  Empty sets score 1, as in the original release of the SARI
                  script (and in HuggingFace ``evaluate``'s port of it). This
                  inflates copy-like outputs enormously -- the identity
                  baseline on ASSET scores 54.03, because deleting nothing
                  earns a perfect deletion precision. Provided for
                  cross-checking against that implementation, not for
                  reporting.
``FKGL``      Eq. 19, pooled at corpus level (see ``data.readability``).
``BERTScore`` Semantic preservation, delegated to the ``bert-score`` package.

Tokenisation is lowercased and punctuation-separated, matching the standard
SARI evaluation setup so scores are comparable with published numbers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from ..data.readability import corpus_fkgl, fkgl

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def normalize(text: str) -> list[str]:
    """Lowercase and split off punctuation."""
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(
    source: list[tuple[str, ...]],
    candidate: list[tuple[str, ...]],
    references: list[list[tuple[str, ...]]],
    empty_score: float = 0.0,
) -> tuple[float, float, float]:
    """Keep-F1, delete-precision and add-F1 at a single n-gram order.

    ``empty_score`` is the value assigned when a set is empty: 0.0 for the
    EASSE convention, 1.0 for the original Xu script.
    """
    num_ref = len(references)
    ref_counter: Counter = Counter(g for ref in references for g in ref)

    src_counter = Counter(source)
    src_rep = Counter({g: c * num_ref for g, c in src_counter.items()})
    cand_counter = Counter(candidate)
    cand_rep = Counter({g: c * num_ref for g, c in cand_counter.items()})

    # -- KEEP: n-grams retained from the source that the references also keep.
    keep_cand = src_rep & cand_rep
    keep_good = keep_cand & ref_counter
    keep_all = src_rep & ref_counter

    keep_p_num = sum(keep_good[g] / keep_cand[g] for g in keep_good)
    keep_r_num = sum(keep_good.values())
    keep_precision = keep_p_num / len(keep_cand) if keep_cand else empty_score
    keep_recall = keep_r_num / sum(keep_all.values()) if keep_all else empty_score
    keep_f1 = _f1(keep_precision, keep_recall)

    # -- DELETE: precision only, per Eq. 18. An n-gram counts as correctly
    # deleted when the candidate dropped it and no reference kept it.
    del_cand = src_rep - cand_rep
    del_good = del_cand - ref_counter
    del_p_num = sum(del_good[g] / del_cand[g] for g in del_good)
    del_precision = del_p_num / len(del_cand) if del_cand else empty_score

    # -- ADD: scored over the set of n-grams absent from the source.
    add_cand = set(cand_counter) - set(src_counter)
    add_all = set(ref_counter) - set(src_counter)
    add_good = add_cand & add_all
    add_precision = len(add_good) / len(add_cand) if add_cand else empty_score
    add_recall = len(add_good) / len(add_all) if add_all else empty_score
    add_f1 = _f1(add_precision, add_recall)

    return keep_f1, del_precision, add_f1


def _f1(precision: float, recall: float) -> float:
    """Eq. 17."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


SARI_VARIANTS = {"easse": 0.0, "xu": 1.0}


def sari_sentence(
    source: str,
    candidate: str,
    references: Sequence[str],
    max_n: int = 4,
    variant: str = "easse",
) -> dict[str, float]:
    """SARI for one sentence, returning the total and its three components."""
    if variant not in SARI_VARIANTS:
        raise ValueError(f"unknown SARI variant {variant!r}; expected one of {sorted(SARI_VARIANTS)}")
    empty_score = SARI_VARIANTS[variant]

    src_tokens = normalize(source)
    cand_tokens = normalize(candidate)
    ref_tokens = [normalize(r) for r in references]

    keeps, dels, adds = [], [], []
    for n in range(1, max_n + 1):
        keep, delete, add = _sari_ngram(
            _ngrams(src_tokens, n),
            _ngrams(cand_tokens, n),
            [_ngrams(r, n) for r in ref_tokens],
            empty_score,
        )
        keeps.append(keep)
        dels.append(delete)
        adds.append(add)

    keep_avg = 100 * sum(keeps) / max_n
    del_avg = 100 * sum(dels) / max_n
    add_avg = 100 * sum(adds) / max_n
    return {
        "sari": (keep_avg + del_avg + add_avg) / 3,
        "keep": keep_avg,
        "delete": del_avg,
        "add": add_avg,
    }


def sari_corpus(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
    variant: str = "easse",
) -> dict[str, float]:
    """Corpus SARI: the mean of the per-sentence scores, as in the paper."""
    if not (len(sources) == len(candidates) == len(references)):
        raise ValueError("sources, candidates and references must align")
    if not sources:
        return {"sari": 0.0, "keep": 0.0, "delete": 0.0, "add": 0.0}

    totals = {"sari": 0.0, "keep": 0.0, "delete": 0.0, "add": 0.0}
    for src, cand, refs in zip(sources, candidates, references):
        scores = sari_sentence(src, cand, refs, max_n, variant)
        for key in totals:
            totals[key] += scores[key]
    return {key: value / len(sources) for key, value in totals.items()}


def fkgl_corpus(texts: Sequence[str]) -> float:
    """Corpus-level FKGL (Eq. 19)."""
    return corpus_fkgl(list(texts))


def fkgl_reduction(sources: Sequence[str], candidates: Sequence[str]) -> float:
    """Mean per-sentence FKGL change, as reported in Table 11.

    Negative means the output reads at a lower grade level than the input.
    """
    if not sources:
        return 0.0
    deltas = [fkgl(c) - fkgl(s) for s, c in zip(sources, candidates)]
    return sum(deltas) / len(deltas)


def bertscore(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    model_type: str = "bert-base-uncased",
    batch_size: int = 32,
    device: str | None = None,
) -> float:
    """Mean BERTScore F1 against the best-matching reference.

    Returns ``float('nan')`` when ``bert-score`` is not installed, so a run
    without the optional dependency still reports SARI and FKGL.
    """
    try:
        from bert_score import score as _score
    except ImportError:  # pragma: no cover - optional dependency
        return float("nan")

    flat_candidates: list[str] = []
    flat_references: list[str] = []
    groups: list[int] = []
    for cand, refs in zip(candidates, references):
        for ref in refs:
            flat_candidates.append(cand)
            flat_references.append(ref)
        groups.append(len(refs))

    if not flat_candidates:
        return float("nan")

    _, _, f1 = _score(
        flat_candidates,
        flat_references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )

    best: list[float] = []
    offset = 0
    for size in groups:
        best.append(float(f1[offset : offset + size].max()))
        offset += size
    return sum(best) / len(best)


def length_bucket(source: str, short_max: int = 15, medium_max: int = 30) -> str:
    """Short / medium / long buckets for the Table 11 breakdown."""
    n = len(normalize(source))
    if n <= short_max:
        return "short"
    if n <= medium_max:
        return "medium"
    return "long"


def evaluate_all(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    compute_bertscore: bool = True,
    device: str | None = None,
    sari_variant: str = "easse",
) -> dict[str, float]:
    """The metric block reported in Tables 2, 3 and 11."""
    results = sari_corpus(sources, candidates, references, variant=sari_variant)
    results["fkgl"] = fkgl_corpus(candidates)
    results["fkgl_reduction"] = fkgl_reduction(sources, candidates)
    if compute_bertscore:
        results["bertscore"] = bertscore(candidates, references, device=device)
    return results
