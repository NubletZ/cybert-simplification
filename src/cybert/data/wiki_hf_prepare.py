"""Stream an unpaired Wikipedia corpus from the Hugging Face Hub.

    python -m cybert.data.wiki_hf_prepare --out-dir data/wiki --target 200000

Paper 4.1 builds its unpaired corpus from an English Wikipedia dump, split by
Flesch Reading Ease: ``FRE < 10`` is complex, ``FRE > 70`` is simple, and the
band between is discarded. This module does the same over
``wikimedia/wikipedia`` in *streaming* mode, so nothing is downloaded to disk
beyond the sentences that survive filtering.

The paper uses 4M sentences. ``--target`` caps each domain so a smaller,
tractable corpus can be built; ``cybert.eval`` has no dependency on the size,
and Figure 7 is precisely the experiment that varies it.

Sentence cleaning is shared with :mod:`cybert.data.wiki_prepare`, so a corpus
built from a local dump and one built from the Hub go through identical
filters.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

from .readability import COMPLEX, SIMPLE, label_domain
from .wiki_prepare import clean, is_acceptable, split_sentences

logger = logging.getLogger(__name__)

DEFAULT_REPO = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.en"


def stream_sentences(
    repo: str,
    config: str,
    target: int,
    complex_max: float,
    simple_min: float,
    min_words: int,
    max_words: int,
    max_articles: int | None,
    log_every: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Stream articles until both domains hit ``target`` (or the stream ends).

    ``FRE < 10`` is a demanding threshold, so the complex side fills far more
    slowly than the simple side; the loop keeps going until *both* are full and
    simply stops adding to whichever side is already done.
    """
    from datasets import load_dataset

    dataset = load_dataset(repo, config, split="train", streaming=True)

    complex_side: list[str] = []
    simple_side: list[str] = []
    seen: set[str] = set()
    articles = 0
    kept = 0
    started = time.time()

    for record in dataset:
        articles += 1
        if max_articles is not None and articles > max_articles:
            break

        for paragraph in record["text"].split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for sentence in split_sentences(paragraph):
                sentence = clean(sentence)
                if not is_acceptable(sentence, min_words, max_words):
                    continue
                if sentence in seen:
                    continue
                label = label_domain(sentence, complex_max, simple_min)
                if label is None:
                    continue
                if label == COMPLEX and len(complex_side) < target:
                    complex_side.append(sentence)
                elif label == SIMPLE and len(simple_side) < target:
                    simple_side.append(sentence)
                else:
                    continue
                seen.add(sentence)
                kept += 1

        if articles % log_every == 0:
            elapsed = time.time() - started
            logger.info(
                "%d articles | complex %d/%d | simple %d/%d | %.0f sent/s",
                articles,
                len(complex_side),
                target,
                len(simple_side),
                target,
                kept / max(1e-6, elapsed),
            )

        if len(complex_side) >= target and len(simple_side) >= target:
            break

    stats = {
        "articles_scanned": articles,
        "complex": len(complex_side),
        "simple": len(simple_side),
        "seconds": int(time.time() - started),
    }
    return complex_side, simple_side, stats


def prepare(
    out_dir: Path,
    repo: str = DEFAULT_REPO,
    config: str = DEFAULT_CONFIG,
    target: int = 200_000,
    complex_max: float = 10.0,
    simple_min: float = 70.0,
    min_words: int = 5,
    max_words: int = 64,
    max_articles: int | None = None,
    valid_size: int = 1000,
    seed: int = 42,
    log_every: int = 20_000,
) -> dict[str, int]:
    complex_side, simple_side, stats = stream_sentences(
        repo, config, target, complex_max, simple_min, min_words, max_words, max_articles, log_every
    )

    rng = random.Random(seed)
    rng.shuffle(complex_side)
    rng.shuffle(simple_side)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_valid_c = min(valid_size, len(complex_side) // 10)
    n_valid_s = min(valid_size, len(simple_side) // 10)
    for name, rows in (
        ("valid.complex.txt", complex_side[:n_valid_c]),
        ("valid.simple.txt", simple_side[:n_valid_s]),
        ("train.complex.txt", complex_side[n_valid_c:]),
        ("train.simple.txt", simple_side[n_valid_s:]),
    ):
        (out_dir / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        logger.info("wrote %s (%d sentences)", out_dir / name, len(rows))

    stats.update({"train_complex": len(complex_side) - n_valid_c, "train_simple": len(simple_side) - n_valid_s})
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/wiki"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--target", type=int, default=200_000, help="sentences per domain")
    parser.add_argument("--fre-complex-max", type=float, default=10.0)
    parser.add_argument("--fre-simple-min", type=float, default=70.0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20_000, help="articles between progress logs")
    args = parser.parse_args(argv)

    stats = prepare(
        log_every=args.log_every,
        out_dir=args.out_dir,
        repo=args.repo,
        config=args.config,
        target=args.target,
        complex_max=args.fre_complex_max,
        simple_min=args.fre_simple_min,
        min_words=args.min_words,
        max_words=args.max_words,
        max_articles=args.max_articles,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
