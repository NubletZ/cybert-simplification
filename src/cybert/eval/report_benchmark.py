"""Benchmark a checkpoint on ASSET and TurkCorpus and tabulate the result.

    python -m cybert.eval.report_benchmark --ckpt runs/paper_phase2/best.pt

Produces the row CyBERT occupies in Tables 2 and 3 -- SARI with its add / keep
/ delete components, and FKGL -- next to two reference points that make the
number interpretable:

``identity``
    Output = input. The floor any simplification system must clear. Scores
    20.69 on ASSET and 26.05 on TurkCorpus under the EASSE convention, against
    published values of 20.73 and 25.44.
``gold reference``
    One human reference scored against the others. The practical ceiling.

Both SARI conventions are reported, because they differ by tens of points and
a bare "SARI = x" is not interpretable without knowing which was used. See
``cybert.eval.metrics`` for the distinction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from ..config import apply_overrides, config_from_dict
from ..data.datasets import PairedDataset, TextEncoder, read_paired_tsv
from ..eval.evaluate import evaluate_by_length, generate_texts
from ..eval.metrics import bertscore, fkgl_corpus, fkgl_reduction, sari_corpus
from ..models.generator import CyBERTGenerator
from ..training.trainer_utils import load_checkpoint, resolve_device

logger = logging.getLogger(__name__)

BENCHMARKS = {
    "asset": "data/asset/test.paired.tsv",
    "turkcorpus": "data/turkcorpus/test.paired.tsv",
}

# Tables 2 and 3, for orientation only.
PAPER = {
    "asset": {"unpaired": (43.41, 7.73), "paired": (45.58, 8.00)},
    "turkcorpus": {"unpaired": (43.59, 6.31), "paired": (48.01, 7.53)},
}


def load_model(ckpt: str, overrides: list[str] | None = None):
    from transformers import AutoTokenizer

    checkpoint = load_checkpoint(ckpt)
    cfg = config_from_dict(checkpoint["config"])
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    device = resolve_device(cfg.run.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_name)
    generator = CyBERTGenerator(cfg, tokenizer).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return cfg, generator, TextEncoder(tokenizer, cfg.model.max_length), device, checkpoint


def benchmark_one(
    generator,
    text_encoder,
    cfg,
    device,
    name: str,
    path: str,
    compute_bertscore: bool,
    limit: int | None,
) -> dict:
    dataset = PairedDataset(path, max_items=limit)
    sources = [row["source"] for row in dataset]
    references = [row["references"] for row in dataset]

    outputs = generate_texts(generator, text_encoder, sources, cfg, device)

    result: dict = {"n": len(sources), "refs_per_source": len(references[0]) if references else 0}
    for variant in ("easse", "xu"):
        scores = sari_corpus(sources, outputs, references, variant=variant)
        result[variant] = {k: round(v, 2) for k, v in scores.items()}
        # Reference points under the same convention.
        result[variant]["identity_sari"] = round(
            sari_corpus(sources, sources, references, variant=variant)["sari"], 2
        )
        result[variant]["gold_sari"] = round(
            sari_corpus(
                sources, [r[0] for r in references], [r[1:] for r in references], variant=variant
            )["sari"],
            2,
        )

    result["fkgl"] = round(fkgl_corpus(outputs), 2)
    result["fkgl_source"] = round(fkgl_corpus(sources), 2)
    result["fkgl_reduction"] = round(fkgl_reduction(sources, outputs), 2)
    result["empty_outputs"] = sum(1 for o in outputs if not o.strip())
    result["mean_output_words"] = round(
        sum(len(o.split()) for o in outputs) / max(1, len(outputs)), 1
    )
    result["by_length"] = {
        k: {kk: round(vv, 2) for kk, vv in v.items()}
        for k, v in evaluate_by_length(sources, outputs, references).items()
    }
    if compute_bertscore:
        result["bertscore"] = round(bertscore(outputs, references, device=str(device)), 4)
    result["outputs"] = outputs
    result["sources"] = sources
    return result


def format_table(results: dict[str, dict]) -> str:
    lines = []
    lines.append("SARI (EASSE convention -- comparable with the published literature)")
    lines.append(
        f"  {'benchmark':<12}{'SARI':>8}{'add':>8}{'keep':>8}{'delete':>9}"
        f"{'FKGL':>8}{'identity':>10}{'gold':>8}"
    )
    for name, r in results.items():
        e = r["easse"]
        lines.append(
            f"  {name:<12}{e['sari']:>8.2f}{e['add']:>8.2f}{e['keep']:>8.2f}{e['delete']:>9.2f}"
            f"{r['fkgl']:>8.2f}{e['identity_sari']:>10.2f}{e['gold_sari']:>8.2f}"
        )

    lines.append("")
    lines.append("SARI (Xu-script convention -- empty sets score 1; copy-inflated)")
    for name, r in results.items():
        x = r["xu"]
        lines.append(
            f"  {name:<12}{x['sari']:>8.2f}   (identity under the same convention: {x['identity_sari']:.2f})"
        )

    lines.append("")
    lines.append("Paper Tables 2/3 for orientation (convention unstated):")
    for name, entry in PAPER.items():
        if name in results:
            lines.append(
                f"  {name:<12}unpaired SARI {entry['unpaired'][0]:.2f} FKGL {entry['unpaired'][1]:.2f}"
                f"   |   paired SARI {entry['paired'][0]:.2f} FKGL {entry['paired'][1]:.2f}"
            )

    lines.append("")
    lines.append("Output sanity:")
    for name, r in results.items():
        lines.append(
            f"  {name:<12}n={r['n']:<5} empty={r['empty_outputs']:<4} "
            f"mean words={r['mean_output_words']:<6} FKGL source={r['fkgl_source']:.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--out", default=None, help="write full JSON results here")
    parser.add_argument("--samples-dir", default=None, help="write generated outputs here")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg, generator, text_encoder, device, checkpoint = load_model(args.ckpt, args.overrides)
    logger.info(
        "checkpoint phase=%s epoch=%s | beam=%d style_ratio=%.2f fusion=%s",
        checkpoint.get("phase"),
        checkpoint.get("epoch"),
        cfg.decode.beam_size,
        cfg.model.style_ratio,
        cfg.model.fusion,
    )

    results = {}
    for name in args.benchmarks:
        path = BENCHMARKS[name]
        if not Path(path).exists():
            logger.warning("skipping %s: %s not found", name, path)
            continue
        logger.info("evaluating %s", name)
        results[name] = benchmark_one(
            generator, text_encoder, cfg, device, name, path, args.bertscore, args.limit
        )

    print()
    print(format_table(results))

    if args.samples_dir:
        out_dir = Path(args.samples_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, r in results.items():
            (out_dir / f"{name}.out.txt").write_text("\n".join(r["outputs"]) + "\n", encoding="utf-8")
            logger.info("wrote %s", out_dir / f"{name}.out.txt")

    if args.out:
        payload = {
            name: {k: v for k, v in r.items() if k not in ("outputs", "sources")}
            for name, r in results.items()
        }
        payload["_meta"] = {
            "checkpoint": args.ckpt,
            "phase": checkpoint.get("phase"),
            "epoch": checkpoint.get("epoch"),
            "beam_size": cfg.decode.beam_size,
            "style_ratio": cfg.model.style_ratio,
            "fusion": cfg.model.fusion,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()


# Re-exported so ``read_paired_tsv`` stays available to callers importing this
# module directly for its BENCHMARKS map.
__all__ = ["BENCHMARKS", "PAPER", "benchmark_one", "format_table", "load_model", "read_paired_tsv"]
