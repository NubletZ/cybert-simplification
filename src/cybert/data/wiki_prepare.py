"""Build the unpaired complex / simple corpora from an English Wikipedia dump.

Paper 4.1 trains on 4M unpaired Wikipedia sentences split by Flesch Reading
Ease: ``FRE < 10`` is complex, ``FRE > 70`` is simple. Sentences between the
thresholds are discarded, which is why the yield is far below the input size.

The input may be a raw text file (one sentence or one paragraph per line), a
directory of such files, or a ``.bz2``/``.gz`` archive of the same. Extracting
plain text from a MediaWiki XML dump is out of scope -- run ``wikiextractor``
first and point this script at its output.

``--max-sentences`` caps the per-domain output, which is what the unpaired-data
scaling sweep of Figure 7 varies.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import logging
import random
import re
from pathlib import Path
from typing import Iterator

from .readability import COMPLEX, SIMPLE, label_domain

logger = logging.getLogger(__name__)

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_WS = re.compile(r"\s+")
_DOC_TAG = re.compile(r"^<[/]?doc\b")


def _open(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def iter_lines(source: Path) -> Iterator[str]:
    paths = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
    for path in paths:
        with _open(path) as fh:
            for line in fh:
                yield line


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_BOUNDARY.split(paragraph) if s.strip()]


def clean(sentence: str) -> str:
    return _WS.sub(" ", sentence).strip()


def is_acceptable(sentence: str, min_words: int, max_words: int) -> bool:
    """Drop fragments, list items, and sentences outside the length window."""
    if _DOC_TAG.match(sentence):
        return False
    words = sentence.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not sentence[0].isupper():
        return False
    if sentence[-1] not in ".!?":
        return False
    # Wikitable / markup leftovers.
    if any(marker in sentence for marker in ("|", "{{", "}}", "==", "[[")):
        return False
    letters = sum(c.isalpha() for c in sentence)
    return letters / max(1, len(sentence)) >= 0.6


def prepare(
    source: Path,
    out_dir: Path,
    max_sentences: int | None,
    complex_max: float,
    simple_min: float,
    min_words: int,
    max_words: int,
    valid_size: int,
    seed: int,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    complex_side: list[str] = []
    simple_side: list[str] = []
    seen: set[str] = set()
    scanned = 0

    for line in iter_lines(source):
        line = line.strip()
        if not line:
            continue
        for sentence in split_sentences(line):
            sentence = clean(sentence)
            scanned += 1
            if not is_acceptable(sentence, min_words, max_words):
                continue
            if sentence in seen:
                continue
            label = label_domain(sentence, complex_max, simple_min)
            if label is None:
                continue
            seen.add(sentence)
            if label == COMPLEX and (max_sentences is None or len(complex_side) < max_sentences):
                complex_side.append(sentence)
            elif label == SIMPLE and (max_sentences is None or len(simple_side) < max_sentences):
                simple_side.append(sentence)
            if (
                max_sentences is not None
                and len(complex_side) >= max_sentences
                and len(simple_side) >= max_sentences
            ):
                break
        else:
            continue
        break

    rng = random.Random(seed)
    rng.shuffle(complex_side)
    rng.shuffle(simple_side)

    def dump(name: str, rows: list[str]) -> None:
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + ("\n" if rows else ""))

    n_valid_c = min(valid_size, len(complex_side) // 10)
    n_valid_s = min(valid_size, len(simple_side) // 10)
    dump("valid.complex.txt", complex_side[:n_valid_c])
    dump("valid.simple.txt", simple_side[:n_valid_s])
    dump("train.complex.txt", complex_side[n_valid_c:])
    dump("train.simple.txt", simple_side[n_valid_s:])

    stats = {
        "scanned": scanned,
        "train_complex": len(complex_side) - n_valid_c,
        "train_simple": len(simple_side) - n_valid_s,
        "valid_complex": n_valid_c,
        "valid_simple": n_valid_s,
    }
    logger.info("prepared corpus: %s", stats)
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="text file or directory")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="cap per domain; the paper uses 2M per side (4M total)",
    )
    parser.add_argument("--fre-complex-max", type=float, default=10.0)
    parser.add_argument("--fre-simple-min", type=float, default=70.0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--valid-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_argparser().parse_args(argv)
    stats = prepare(
        source=args.source,
        out_dir=args.out_dir,
        max_sentences=args.max_sentences,
        complex_max=args.fre_complex_max,
        simple_min=args.fre_simple_min,
        min_words=args.min_words,
        max_words=args.max_words,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
