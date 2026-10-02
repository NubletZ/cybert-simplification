"""Generation + scoring, shared by validation and final test evaluation.

Both phases early-stop on validation SARI (4.3), so the same code path is used
during training and for the final numbers. Decoding follows Appendix G.2:
beam ``k = 4``, temperature ``tau = 1.0``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from ..config import CyBERTConfig
from ..data.datasets import PairedDataset, TextEncoder, read_paired_tsv
from ..data.readability import COMPLEX, SIMPLE
from ..inference.generate import beam_search, greedy_decode
from ..inference.postprocess import detokenize_batch
from ..models.generator import CyBERTGenerator
from .metrics import evaluate_all, length_bucket, sari_corpus

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_texts(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    sources: Sequence[str],
    cfg: CyBERTConfig,
    device: torch.device,
    target_label: int = SIMPLE,
    batch_size: int | None = None,
) -> list[str]:
    """Transfer ``sources`` into the target domain and decode to strings."""
    generator.eval()
    batch_size = batch_size or cfg.optim.batch_size
    outputs: list[str] = []

    for start in range(0, len(sources), batch_size):
        chunk = list(sources[start : start + batch_size])
        batch = text_encoder.encode_source(chunk)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        encoded = generator.encode(input_ids=input_ids, attention_mask=attention_mask)
        style = generator.target_style(target_label, input_ids.size(0), encoded.style)

        if cfg.decode.beam_size > 1:
            ids = beam_search(
                generator,
                encoded,
                style=style,
                beam_size=cfg.decode.beam_size,
                max_length=cfg.decode.max_length,
                temperature=cfg.decode.temperature,
                length_penalty=cfg.decode.length_penalty,
                no_repeat_ngram_size=cfg.decode.no_repeat_ngram_size,
            )
        else:
            ids = greedy_decode(
                generator,
                encoded,
                style=style,
                max_length=cfg.decode.max_length,
                temperature=cfg.decode.temperature,
            )
        outputs.extend(text_encoder.decode(ids))

    # WordPiece decoding spaces out punctuation and clitics; FKGL counts words
    # by regex, so the raw form would inflate the reported grade level.
    return detokenize_batch(outputs)


def evaluate_paired(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    cfg: CyBERTConfig,
    device: torch.device,
    paired_path: str,
    max_samples: int | None = None,
    compute_bertscore: bool = False,
    return_outputs: bool = False,
) -> dict:
    """Score the model against an aligned test/validation file.

    ``compute_bertscore`` is off during training because loading a second BERT
    for scoring on every epoch dominates the epoch time.
    """
    dataset = PairedDataset(paired_path, max_items=max_samples)
    sources = [row["source"] for row in dataset]
    references = [row["references"] for row in dataset]

    candidates = generate_texts(
        generator, text_encoder, sources, cfg, device, target_label=SIMPLE
    )

    results = evaluate_all(
        sources, candidates, references, compute_bertscore=compute_bertscore, device=str(device)
    )
    if return_outputs:
        results["outputs"] = candidates
        results["sources"] = sources
    return results


def evaluate_by_length(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Short / medium / long breakdown, reproducing Table 11."""
    buckets: dict[str, list[int]] = {"short": [], "medium": [], "long": []}
    for i, source in enumerate(sources):
        buckets[length_bucket(source)].append(i)

    results: dict[str, dict[str, float]] = {}
    for name, indices in buckets.items():
        if not indices:
            continue
        results[name] = sari_corpus(
            [sources[i] for i in indices],
            [candidates[i] for i in indices],
            [references[i] for i in indices],
        )
        results[name]["count"] = float(len(indices))
    return results


def quick_validation_sari(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    cfg: CyBERTConfig,
    device: torch.device,
    paired_path: str | None,
    max_samples: int,
) -> dict[str, float]:
    """Validation SARI used for early stopping in both phases.

    Falls back to the unpaired validation complex file with the *source* as its
    own single reference when no paired validation set is configured. That
    fallback score is not comparable to published SARI -- it only provides a
    monotone signal for model selection when no aligned development data
    exists, which is the unpaired-only setting of Table 2.
    """
    if paired_path:
        return evaluate_paired(
            generator,
            text_encoder,
            cfg,
            device,
            paired_path,
            max_samples=max_samples,
            compute_bertscore=False,
        )

    from ..data.datasets import read_lines

    sources = read_lines(cfg.data.valid_complex)[:max_samples]
    candidates = generate_texts(generator, text_encoder, sources, cfg, device, SIMPLE)
    references = [[s] for s in sources]
    logger.warning(
        "no paired validation file configured; using self-reference SARI for "
        "model selection only"
    )
    return evaluate_all(sources, candidates, references, compute_bertscore=False)


def transfer_both_directions(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    cfg: CyBERTConfig,
    device: torch.device,
    complex_sentences: Sequence[str],
    simple_sentences: Sequence[str],
) -> dict[str, list[str]]:
    """Run G (complex -> simple) and F (simple -> complex) for inspection."""
    return {
        "simplified": generate_texts(
            generator, text_encoder, complex_sentences, cfg, device, SIMPLE
        ),
        "complexified": generate_texts(
            generator, text_encoder, simple_sentences, cfg, device, COMPLEX
        ),
    }


def read_sources(path: str) -> list[str]:
    """Read either a plain sentence file or the source column of a paired TSV."""
    if str(path).endswith(".tsv"):
        return [source for source, _ in read_paired_tsv(path)]
    from ..data.datasets import read_lines

    return read_lines(path)
