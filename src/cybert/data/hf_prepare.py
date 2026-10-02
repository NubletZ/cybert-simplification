"""Build ASSET and TurkCorpus corpora from the Hugging Face Hub.

    python -m cybert.data.hf_prepare --out-dir data

Both evaluation corpora of paper 4.1 are pulled directly from the Hub:

``facebook/asset``
    Parquet, fields ``original`` / ``simplifications``. 2000 validation and
    359 test sentences, 10 references each.

``waboucay/turk_corpus``
    Raw ``*.8turkers.organized.tsv``: column 0 is an id, column 1 the original,
    columns 2-9 the eight turker references. Its loading script no longer works
    with ``datasets`` v3+, so the TSVs are fetched with ``hf_hub_download`` and
    parsed here. 2000 tune and 359 test sentences.

Written per dataset:

``train.paired.tsv`` / ``test.paired.tsv``
    ``complex <TAB> ref1 <TAB> ref2 ...``. The official test splits are used
    only for ``test.paired.tsv``; training draws on validation/tune, matching
    the paper's statement that test splits are reserved strictly for testing.

``train.complex.txt`` / ``train.simple.txt``
    An unpaired pool for Phase 2, derived by *unaligning* the paired data: the
    originals form the complex domain and the references the simple domain,
    then each side is shuffled independently. This is a stand-in for the
    paper's 4M unpaired Wikipedia sentences. It is far smaller and drawn from
    the same content, so it exercises the cycle objective but cannot reproduce
    the data scale that Figure 7 shows to drive performance -- use
    ``cybert.cli prepare-data`` on a Wikipedia dump for that.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

ASSET_REPO = "facebook/asset"
TURK_REPO = "waboucay/turk_corpus"
TURK_FILES = {"train": "tune.8turkers.organized.tsv", "test": "test.8turkers.organized.tsv"}


def load_asset() -> dict[str, list[tuple[str, list[str]]]]:
    """Return ``{"train": rows, "test": rows}`` of ``(original, references)``."""
    from datasets import load_dataset

    dataset = load_dataset(ASSET_REPO, name="simplification")
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for split, key in (("validation", "train"), ("test", "test")):
        rows = []
        for record in dataset[split]:
            refs = [r.strip() for r in record["simplifications"] if r and r.strip()]
            if record["original"].strip() and refs:
                rows.append((record["original"].strip(), refs))
        out[key] = rows
    return out


def load_turkcorpus() -> dict[str, list[tuple[str, list[str]]]]:
    """Parse the organized TSVs: id, original, then eight references."""
    from huggingface_hub import hf_hub_download

    out: dict[str, list[tuple[str, list[str]]]] = {}
    for key, filename in TURK_FILES.items():
        path = hf_hub_download(TURK_REPO, filename, repo_type="dataset")
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 3:
                    continue
                original = columns[1].strip()
                refs = [c.strip() for c in columns[2:] if c.strip()]
                if original and refs:
                    rows.append((original, refs))
        out[key] = rows
    return out


def write_paired(path: Path, rows: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join([source, *refs]) for source, refs in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s (%d rows)", path, len(rows))


def write_unpaired(
    out_dir: Path, rows: list[tuple[str, list[str]]], seed: int = 42, valid_size: int = 200
) -> None:
    """Split the paired rows into two independently shuffled domain corpora."""
    complex_side = [source for source, _ in rows]
    simple_side = [ref for _, refs in rows for ref in refs]

    rng = random.Random(seed)
    rng.shuffle(complex_side)
    rng.shuffle(simple_side)

    n_valid_c = min(valid_size, len(complex_side) // 10)
    n_valid_s = min(valid_size, len(simple_side) // 10)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows_out in (
        ("valid.complex.txt", complex_side[:n_valid_c]),
        ("valid.simple.txt", simple_side[:n_valid_s]),
        ("train.complex.txt", complex_side[n_valid_c:]),
        ("train.simple.txt", simple_side[n_valid_s:]),
    ):
        (out_dir / name).write_text("\n".join(rows_out) + "\n", encoding="utf-8")
        logger.info("wrote %s (%d sentences)", out_dir / name, len(rows_out))


def prepare(out_dir: Path, seed: int = 42) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for name, loader in (("asset", load_asset), ("turkcorpus", load_turkcorpus)):
        splits = loader()
        target = out_dir / name
        write_paired(target / "train.paired.tsv", splits["train"])
        write_paired(target / "test.paired.tsv", splits["test"])
        # A held-out slice of the training rows drives early stopping, so the
        # official test split is never touched during training.
        holdout = max(1, len(splits["train"]) // 10)
        write_paired(target / "valid.paired.tsv", splits["train"][:holdout])
        write_unpaired(target, splits["train"][holdout:], seed=seed)
        stats[name] = {
            "train_pairs": len(splits["train"]) - holdout,
            "valid_pairs": holdout,
            "test_pairs": len(splits["test"]),
            "refs_per_source": len(splits["test"][0][1]) if splits["test"] else 0,
        }
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    stats = prepare(args.out_dir, args.seed)
    for name, values in stats.items():
        print(name, values)


if __name__ == "__main__":  # pragma: no cover
    main()
