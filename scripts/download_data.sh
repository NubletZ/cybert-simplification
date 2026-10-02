#!/usr/bin/env bash
# Fetch the evaluation corpora and the spaCy tagger used by POS-guided
# denoising, then convert them to the TSV layout the loaders expect:
#
#   complex <TAB> ref1 <TAB> ref2 ...
#
# ASSET has 10 references per sentence, TurkCorpus has 8 (paper 4.1); both
# provide 359 test sentences. The English Wikipedia dump used for the unpaired
# corpus is far too large to pull here -- see the note at the end.

set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "==> spaCy tagger"
python -m spacy download en_core_web_sm

echo "==> ASSET"
ASSET_DIR="$DATA_DIR/asset"
mkdir -p "$ASSET_DIR/raw"
ASSET_BASE="https://raw.githubusercontent.com/facebookresearch/asset/main/dataset"
for split in valid test; do
  curl -fsSL "$ASSET_BASE/asset.$split.orig" -o "$ASSET_DIR/raw/asset.$split.orig"
  for i in $(seq 0 9); do
    curl -fsSL "$ASSET_BASE/asset.$split.simp.$i" -o "$ASSET_DIR/raw/asset.$split.simp.$i"
  done
done

echo "==> TurkCorpus"
TURK_DIR="$DATA_DIR/turkcorpus"
mkdir -p "$TURK_DIR/raw"
TURK_BASE="https://raw.githubusercontent.com/cocoxu/simplification/master/data"
for split in tune test; do
  curl -fsSL "$TURK_BASE/turkcorpus/truecase/$split.8turkers.tok.norm" \
    -o "$TURK_DIR/raw/$split.orig" || true
  for i in $(seq 0 7); do
    curl -fsSL "$TURK_BASE/turkcorpus/truecase/$split.8turkers.tok.turk.$i" \
      -o "$TURK_DIR/raw/$split.simp.$i" || true
  done
done

echo "==> building paired TSVs"
python - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])


def build(out_path: Path, orig: Path, refs: list[Path]) -> None:
    if not orig.exists() or not all(r.exists() for r in refs):
        print(f"skipping {out_path}: missing source files")
        return
    sources = orig.read_text(encoding="utf-8").splitlines()
    columns = [r.read_text(encoding="utf-8").splitlines() for r in refs]
    rows = []
    for i, source in enumerate(sources):
        simplifications = [c[i].strip() for c in columns if i < len(c) and c[i].strip()]
        if source.strip() and simplifications:
            rows.append("\t".join([source.strip(), *simplifications]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} rows)")


asset = data_dir / "asset" / "raw"
build(
    data_dir / "asset" / "valid.paired.tsv",
    asset / "asset.valid.orig",
    [asset / f"asset.valid.simp.{i}" for i in range(10)],
)
build(
    data_dir / "asset" / "test.paired.tsv",
    asset / "asset.test.orig",
    [asset / f"asset.test.simp.{i}" for i in range(10)],
)
# ASSET has no dedicated training split; its validation set doubles as the
# interleaved supervised data (paper 4.2), with the test split untouched.
valid = data_dir / "asset" / "valid.paired.tsv"
if valid.exists():
    (data_dir / "asset" / "train.paired.tsv").write_text(
        valid.read_text(encoding="utf-8"), encoding="utf-8"
    )

turk = data_dir / "turkcorpus" / "raw"
build(
    data_dir / "turkcorpus" / "valid.paired.tsv",
    turk / "tune.orig",
    [turk / f"tune.simp.{i}" for i in range(8)],
)
build(
    data_dir / "turkcorpus" / "test.paired.tsv",
    turk / "test.orig",
    [turk / f"test.simp.{i}" for i in range(8)],
)
PY

cat <<'EOF'

Unpaired Wikipedia corpus
-------------------------
The paper trains on 4M unpaired sentences from an English Wikipedia dump
(4.1). Fetch a dump, extract plain text with wikiextractor, then run:

  python -m cybert.cli prepare-data \
      --source path/to/extracted_text \
      --out-dir data/wiki \
      --max-sentences 2000000

That applies the Flesch Reading Ease split (FRE < 10 complex, FRE > 70 simple)
and writes train/valid files for both domains.
EOF
