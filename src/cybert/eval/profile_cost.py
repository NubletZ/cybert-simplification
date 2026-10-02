"""Inference cost profiling (paper 5.5, Table 5).

The paper's protocol: maximum length 64, beam size 4, no sampling, peak memory
from ``torch.cuda.max_memory_allocated``. It reports 2.11 GFLOPs, 722 MB and
185.15M parameters for CyBERT.

Only the inference-time model is counted -- the trainable BERT encoder, the
latent interface and the attention-LSTM decoder. The frozen phi encoder and the
discriminator heads exist during training only and are not loaded here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..config import CyBERTConfig
from ..data.datasets import TextEncoder
from ..data.readability import SIMPLE
from ..inference.generate import beam_search
from ..models.generator import CyBERTGenerator

logger = logging.getLogger(__name__)


def count_parameters(generator: CyBERTGenerator) -> dict[str, float]:
    """Parameter counts in millions, broken down by component."""

    def millions(module) -> float:
        return sum(p.numel() for p in module.parameters()) / 1e6

    return {
        "total_M": generator.num_parameters(trainable_only=False) / 1e6,
        "encoder_M": millions(generator.encoder),
        "decoder_M": millions(generator.decoder),
        "latent_M": millions(generator.splitter) + millions(generator.fusion),
        "heads_M": millions(generator.style_head)
        + millions(generator.content_adv_head)
        + millions(generator.content_proj),
    }


def _encoder_flops(generator: CyBERTGenerator, seq_len: int) -> float:
    """Analytic forward FLOPs for the BERT encoder at ``seq_len`` tokens.

    Per layer: four projections of ``d x d``, the attention score and context
    matmuls, and the two feed-forward matrices. Counted as multiply-accumulate
    pairs (2 FLOPs each), which is the convention Table 5 uses.
    """
    config = generator.encoder.bert.config
    d = config.hidden_size
    ff = config.intermediate_size
    layers = config.num_hidden_layers

    per_layer = 2 * (4 * seq_len * d * d)  # q, k, v, output projections
    per_layer += 2 * (2 * seq_len * seq_len * d)  # scores and context
    per_layer += 2 * (2 * seq_len * d * ff)  # feed-forward
    return per_layer * layers


def _decoder_flops(generator: CyBERTGenerator, steps: int, memory_len: int, beam: int) -> float:
    """Analytic forward FLOPs for the attention-LSTM decoder over ``steps``."""
    decoder = generator.decoder
    h = decoder.hidden_size
    e = decoder.embed_dim
    layers = decoder.num_layers
    vocab = decoder.vocab_size

    per_step = 2 * (4 * h * (e + h))  # first LSTM layer gates
    per_step += 2 * ((layers - 1) * 4 * h * (h + h))  # remaining layers
    if decoder.use_attention:
        per_step += 2 * (memory_len * h)  # dot-product scores
        per_step += 2 * (memory_len * h)  # context mixture
        per_step += 2 * (2 * h * h)  # combine projection
    per_step += 2 * (h * vocab)  # output projection

    memory_proj = 2 * (memory_len * generator.encoder.hidden_size * h) if decoder.use_attention else 0
    return steps * beam * per_step + memory_proj


def profile(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    cfg: CyBERTConfig,
    device: torch.device,
    sample_sentences: list[str] | None = None,
) -> dict[str, Any]:
    """Measure parameters, analytic FLOPs and peak memory for one sentence.

    FLOPs are analytic rather than measured because ``thop``/``fvcore`` cannot
    trace a beam-search loop. ``thop`` is still used, when installed, to
    cross-check the encoder term, which dominates the total.
    """
    generator.eval().to(device)
    seq_len = cfg.model.max_length
    beam = cfg.decode.beam_size

    encoder_flops = _encoder_flops(generator, seq_len)
    decoder_flops = _decoder_flops(generator, cfg.decode.max_length, seq_len, beam)
    total_flops = encoder_flops + decoder_flops

    results: dict[str, Any] = {
        "seq_len": seq_len,
        "beam_size": beam,
        "flops_G": total_flops / 1e9,
        "encoder_flops_G": encoder_flops / 1e9,
        "decoder_flops_G": decoder_flops / 1e9,
        **count_parameters(generator),
    }

    sentences = sample_sentences or ["The municipality promulgated an ordinance last year."]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch = text_encoder.encode_source(sentences[:1])
    with torch.no_grad():
        encoded = generator.encode(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        style = generator.target_style(SIMPLE, 1, encoded.style)
        beam_search(
            generator,
            encoded,
            style=style,
            beam_size=beam,
            max_length=cfg.decode.max_length,
            temperature=cfg.decode.temperature,
            length_penalty=cfg.decode.length_penalty,
            no_repeat_ngram_size=cfg.decode.no_repeat_ngram_size,
        )

    if device.type == "cuda":
        results["peak_memory_MB"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        results["peak_memory_MB"] = float("nan")
        logger.warning("peak memory requires CUDA; reporting NaN on %s", device)

    results["thop_encoder_flops_G"] = _thop_encoder_flops(generator, seq_len, device)
    return results


def _thop_encoder_flops(generator: CyBERTGenerator, seq_len: int, device: torch.device) -> float:
    """Cross-check the encoder term with ``thop``, or NaN when unavailable."""
    try:
        from thop import profile as thop_profile
    except ImportError:  # pragma: no cover - optional dependency
        return float("nan")

    ids = torch.ones(1, seq_len, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    try:
        macs, _ = thop_profile(
            generator.encoder.bert, inputs=(ids, mask), verbose=False
        )
        return 2 * macs / 1e9
    except Exception as exc:  # pragma: no cover - thop is brittle on HF models
        logger.warning("thop profiling failed: %s", exc)
        return float("nan")


def format_report(results: dict[str, Any]) -> str:
    lines = [
        "Inference cost (max_length=%(seq_len)d, beam=%(beam_size)d, no sampling)" % results,
        f"  FLOPs        : {results['flops_G']:.2f} G "
        f"(encoder {results['encoder_flops_G']:.2f} G, decoder {results['decoder_flops_G']:.2f} G)",
        f"  MACs         : {results['flops_G'] / 2:.2f} G "
        f"(the same counts without the multiply-accumulate factor of 2)",
        f"  Peak memory  : {results['peak_memory_MB']:.2f} MB",
        f"  Parameters   : {results['total_M']:.2f} M "
        f"(encoder {results['encoder_M']:.2f} M, decoder {results['decoder_M']:.2f} M)",
        "  Paper Table 5: 2.11 GFLOPs, 722.39 MB, 185.15 M",
        "",
        "  Note: a full bert-base forward over 64 tokens is ~11 GFLOPs on its own",
        "  (2 x 85M non-embedding parameters x 64 tokens), so Table 5's 2.11 G is",
        "  not reproducible under this accounting. The paper does not state its",
        "  counting convention; report the basis alongside any comparison.",
    ]
    return "\n".join(lines)
