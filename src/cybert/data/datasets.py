"""Datasets and collators.

Three shapes are needed:

``Phase1Dataset``
    Single sentences with their domain label, corrupted on the fly. Phase 1
    reconstructs the original from the corruption (paper 3.2.2).

``UnpairedDataset``
    Independent draws of ``x ~ p_complex`` and ``y ~ p_simple``, the CycleGAN
    setting of 3.2.3. There is no alignment between the two sides.

``PairedDataset``
    ``complex <TAB> simple [<TAB> simple2 ...]`` rows from ASSET / TurkCorpus,
    used for the interleaved supervised term of 4.2 and for evaluation, where
    the extra columns are the multiple references SARI needs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .noise import SpanInfillingNoise
from .pos_tagger import POSTagger, whitespace_tokenize
from .readability import COMPLEX, SIMPLE, label_domain


def read_lines(path: str | Path) -> list[str]:
    """Read a corpus file, dropping blank lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def read_paired_tsv(path: str | Path) -> list[tuple[str, list[str]]]:
    """Read ``complex <TAB> ref1 [<TAB> ref2 ...]`` rows."""
    rows: list[tuple[str, list[str]]] = []
    for line in read_lines(path):
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2:
            continue
        rows.append((parts[0], [p for p in parts[1:] if p]))
    return rows


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class Phase1Dataset(Dataset):
    """Labelled sentences for denoising reconstruction + disentanglement."""

    def __init__(
        self,
        complex_path: str | Path,
        simple_path: str | Path,
        max_items: int | None = None,
    ) -> None:
        items: list[tuple[str, int]] = []
        items += [(t, COMPLEX) for t in read_lines(complex_path)]
        items += [(t, SIMPLE) for t in read_lines(simple_path)]
        if max_items is not None:
            items = items[:max_items]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text, label = self.items[index]
        return {"text": text, "label": label}


class UnpairedDataset(Dataset):
    """Independent complex / simple draws for cycle-consistent training.

    One epoch is defined as a pass over the larger side; the smaller side is
    cycled with a fresh shuffle each time it wraps, so the pairing between the
    domains is never fixed.
    """

    def __init__(
        self,
        complex_path: str | Path,
        simple_path: str | Path,
        seed: int = 42,
        max_items: int | None = None,
    ) -> None:
        self.complex = read_lines(complex_path)
        self.simple = read_lines(simple_path)
        if not self.complex or not self.simple:
            raise ValueError("both domains need at least one sentence")
        if max_items is not None:
            self.complex = self.complex[:max_items]
            self.simple = self.simple[:max_items]
        self.length = max(len(self.complex), len(self.simple))
        rng = random.Random(seed)
        self._perm_complex = list(range(len(self.complex)))
        self._perm_simple = list(range(len(self.simple)))
        rng.shuffle(self._perm_complex)
        rng.shuffle(self._perm_simple)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = self.complex[self._perm_complex[index % len(self.complex)]]
        y = self.simple[self._perm_simple[index % len(self.simple)]]
        return {"complex": x, "simple": y}


class PairedDataset(Dataset):
    """Aligned complex -> simple rows, optionally multi-reference."""

    def __init__(self, path: str | Path, max_items: int | None = None) -> None:
        rows = read_paired_tsv(path)
        if max_items is not None:
            rows = rows[:max_items]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, refs = self.rows[index]
        return {"source": source, "target": refs[0], "references": refs}


# --------------------------------------------------------------------------- #
# Collators
# --------------------------------------------------------------------------- #


class TextEncoder:
    """Tokenization shared by every collator.

    Encoder inputs and decoder targets both live in BERT's WordPiece vocabulary,
    so the decoder's output projection is over the same ~30k types (4.3).
    """

    def __init__(self, tokenizer, max_length: int = 64) -> None:
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

    def encode_source(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        batch = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

    def encode_target(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Build decoder inputs and labels.

        ``decoder_input = [CLS] w1 ... wn`` and ``labels = w1 ... wn [SEP]``,
        with pads masked out of the loss by ``-100``.
        """
        batch = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids = batch["input_ids"]  # [CLS] w1 ... wn [SEP] [PAD]...
        mask = batch["attention_mask"]
        decoder_input = ids[:, :-1].contiguous()
        labels = ids[:, 1:].contiguous()
        label_mask = mask[:, 1:].contiguous()
        labels = labels.masked_fill(label_mask == 0, -100)
        return {
            "decoder_input_ids": decoder_input,
            "labels": labels,
            "decoder_attention_mask": mask[:, :-1].contiguous(),
        }

    def decode(self, ids: torch.Tensor) -> list[str]:
        return self.tok.batch_decode(ids, skip_special_tokens=True)


class Phase1Collator:
    """Corrupt, tokenize and batch for Phase 1.

    POS tagging happens here (batched) so the tagger sees a whole batch at once
    and the disk cache absorbs the cost after the first epoch.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        noise: SpanInfillingNoise,
        tagger: POSTagger | None,
        use_pos: bool = True,
    ) -> None:
        self.encoder = encoder
        self.noise = noise
        self.tagger = tagger
        self.use_pos = use_pos

    def _corrupt_batch(self, texts: list[str]) -> list[str]:
        if not self.noise.enabled:
            return list(texts)
        tags_batch: list[list[str] | None]
        if self.use_pos and self.tagger is not None:
            tags_batch = list(self.tagger.tag_batch(texts))
        else:
            tags_batch = [None] * len(texts)
        out = []
        for text, tags in zip(texts, tags_batch):
            tokens = whitespace_tokenize(text)
            out.append(" ".join(self.noise.corrupt_tokens(tokens, tags)))
        return out

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        corrupted = self._corrupt_batch(texts)

        source = self.encoder.encode_source(corrupted)
        clean = self.encoder.encode_source(texts)
        target = self.encoder.encode_target(texts)
        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "clean_input_ids": clean["input_ids"],
            "clean_attention_mask": clean["attention_mask"],
            "style_labels": labels,
            "texts": texts,
            "corrupted_texts": corrupted,
            **target,
        }


class UnpairedCollator:
    """Batch independent complex / simple sentences for Phase 2."""

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        complex_texts = [b["complex"] for b in batch]
        simple_texts = [b["simple"] for b in batch]
        cx = self.encoder.encode_source(complex_texts)
        sx = self.encoder.encode_source(simple_texts)
        return {
            "complex_input_ids": cx["input_ids"],
            "complex_attention_mask": cx["attention_mask"],
            "simple_input_ids": sx["input_ids"],
            "simple_attention_mask": sx["attention_mask"],
            "complex_texts": complex_texts,
            "simple_texts": simple_texts,
        }


class PairedCollator:
    """Batch aligned rows for the supervised term and for evaluation."""

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        sources = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]
        src = self.encoder.encode_source(sources)
        tgt = self.encoder.encode_target(targets)
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "sources": sources,
            "targets": targets,
            "references": [b["references"] for b in batch],
            **tgt,
        }


def split_by_readability(
    texts: Sequence[str], complex_max: float = 10.0, simple_min: float = 70.0
) -> tuple[list[str], list[str]]:
    """Partition raw sentences into the two readability domains (paper 4.1)."""
    complex_side: list[str] = []
    simple_side: list[str] = []
    for text in texts:
        label = label_domain(text, complex_max, simple_min)
        if label == COMPLEX:
            complex_side.append(text)
        elif label == SIMPLE:
            simple_side.append(text)
    return complex_side, simple_side
