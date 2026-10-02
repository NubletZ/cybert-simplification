# CyBERT

Implementation of **CyBERT: Knowledge-Guided Sentence Simplification via POS-Guided
Denoising and Content–Style Disentanglement**.

Sentence simplification is treated as readability-oriented style transfer between a complex domain
and a simple domain. The model is a BERT encoder, a content–style latent interface, and an
attention-based LSTM decoder, trained in two phases:

1. **Phase 1** — POS-guided span-infilling denoising reconstruction, plus the latent disentanglement
   objectives (Eq. 8).
2. **Phase 2** — cycle-consistent adversarial style transfer over unpaired complex/simple corpora,
   with optional interleaved paired supervision (Eq. 16).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export PYTHONPATH=src          # or: pip install -e .
```

## Test setup

A tiny bundled corpus (`data/toy/`) exercises the whole pipeline in minutes, on CPU if necessary.
It is a correctness gate — the scores it produces mean nothing.

```bash
pytest tests/ -v

python -m cybert.cli train-phase1 --config configs/toy.yaml
python -m cybert.cli train-phase2 --config configs/toy.yaml \
       --init-from runs/toy/best.pt
python -m cybert.cli generate --ckpt runs/toy/best.pt \
       --input data/toy/valid.complex.txt --beam 2
```

## Real runs

Everything comes from the Hugging Face Hub — no manual downloads:

```bash
python -m cybert.data.hf_prepare      --out-dir data                       # ASSET + TurkCorpus
python -m cybert.data.wiki_hf_prepare --out-dir data/wiki --target 200000  # unpaired Wikipedia
mkdir -p data/combined
cat data/asset/train.paired.tsv data/turkcorpus/train.paired.tsv > data/combined/train.paired.tsv
cat data/asset/valid.paired.tsv data/turkcorpus/valid.paired.tsv > data/combined/valid.paired.tsv
python -m spacy download en_core_web_sm
```

| corpus | source | rows |
|---|---|---|
| ASSET | `facebook/asset` (parquet) | 1,800 train / 200 valid / 359 test, 10 refs |
| TurkCorpus | `waboucay/turk_corpus` (raw TSV) | 1,800 train / 200 valid / 359 test, 8 refs |
| Wikipedia | `wikimedia/wikipedia`, streamed | 199k complex + 199k simple |

TurkCorpus needs the raw-TSV path (column 0 id, column 1 original, columns 2–9 references) because
its loading script no longer runs under `datasets` v3+. The Wikipedia FRE split of §4.1 separates
the domains cleanly — complex side FKGL 18.8 / 24.4 words, simple side FKGL 5.1 / 13.7 words — from
74k streamed articles in 43 s.

```bash
python -m cybert.cli train-phase1 --config configs/paper.yaml
python -m cybert.cli train-phase2 --config configs/paper.yaml \
       --init-from runs/paper_phase1/best.pt --set run.output_dir=runs/paper_phase2

python -m cybert.eval.report_benchmark --ckpt runs/paper_phase2/best.pt \
       --out runs/benchmark.json --samples-dir runs/samples
python -m cybert.cli analyze --ckpt runs/paper_phase2/best.pt
python -m cybert.cli profile --config configs/base.yaml
```

`scripts/download_data.sh` still fetches the corpora from their original repositories, and
`cybert.cli prepare-data` handles a local Wikipedia dump instead of the streamed one.

Any config value can be overridden inline:

```bash
python -m cybert.cli train-phase2 --config configs/phase2.yaml \
       --set phase2.lambda_paired=0 phase2.cycle_mode=gumbel model.fusion=concat
```
