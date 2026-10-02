"""Command-line entry point.

    python -m cybert.cli prepare-data  --source wiki_text/ --out-dir data/wiki
    python -m cybert.cli train-phase1  --config configs/phase1.yaml
    python -m cybert.cli train-phase2  --config configs/phase2.yaml \
                                       --init-from runs/cybert_phase1/best.pt
    python -m cybert.cli generate      --ckpt runs/cybert_phase2/best.pt \
                                       --input data/asset/test.paired.tsv
    python -m cybert.cli evaluate      --ckpt runs/cybert_phase2/best.pt
    python -m cybert.cli analyze       --ckpt runs/cybert_phase2/best.pt
    python -m cybert.cli sweep         --config configs/base.yaml
    python -m cybert.cli profile       --ckpt runs/cybert_phase2/best.pt

Any config value can be overridden inline, e.g.
``--set model.fusion=concat noise.use_pos=false phase2.lambda_paired=0``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import apply_overrides, config_from_dict, load_config

logger = logging.getLogger("cybert")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="section.key=value",
        help="inline config overrides",
    )
    parser.add_argument("--log-level", default="INFO")


def _load_for_checkpoint(args) -> tuple[CyBERTConfig, dict]:
    """Prefer the config stored in the checkpoint; allow explicit overrides."""
    from .training.trainer_utils import load_checkpoint

    checkpoint = load_checkpoint(args.ckpt)
    if args.config:
        cfg = load_config(args.config, args.overrides)
    else:
        cfg = config_from_dict(checkpoint["config"])
        if args.overrides:
            cfg = apply_overrides(cfg, args.overrides)
    return cfg, checkpoint


def _restore(args):
    """Rebuild the generator from a checkpoint and return it ready to run."""
    from transformers import AutoTokenizer

    from .data.datasets import TextEncoder
    from .models.generator import CyBERTGenerator
    from .training.trainer_utils import resolve_device

    cfg, checkpoint = _load_for_checkpoint(args)
    device = resolve_device(cfg.run.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_name)
    generator = CyBERTGenerator(cfg, tokenizer).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    text_encoder = TextEncoder(tokenizer, cfg.model.max_length)
    return cfg, generator, text_encoder, device, checkpoint


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_prepare_data(args) -> None:
    from .data.wiki_prepare import prepare

    stats = prepare(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        max_sentences=args.max_sentences,
        complex_max=args.fre_complex_max,
        simple_min=args.fre_simple_min,
        min_words=args.min_words,
        max_words=args.max_words,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


def cmd_train_phase1(args) -> None:
    from .training import phase1

    cfg = load_config(args.config, args.overrides)
    best = phase1.train(cfg, init_from=args.init_from)
    print(json.dumps(best, indent=2, default=str))


def cmd_train_phase2(args) -> None:
    from .training import phase2

    cfg = load_config(args.config, args.overrides)
    best = phase2.train(cfg, init_from=args.init_from)
    print(json.dumps(best, indent=2, default=str))


def cmd_generate(args) -> None:
    from .data.readability import COMPLEX, SIMPLE
    from .eval.evaluate import generate_texts, read_sources

    cfg, generator, text_encoder, device, _ = _restore(args)
    if args.beam is not None:
        cfg.decode.beam_size = args.beam

    sources = read_sources(args.input)
    if args.limit:
        sources = sources[: args.limit]
    target = COMPLEX if args.direction == "complexify" else SIMPLE
    outputs = generate_texts(generator, text_encoder, sources, cfg, device, target)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
        print(f"wrote {len(outputs)} lines to {path}")
    else:
        for source, output in zip(sources, outputs):
            print(f"IN : {source}\nOUT: {output}\n")


def cmd_evaluate(args) -> None:
    from .eval.evaluate import evaluate_by_length, evaluate_paired

    cfg, generator, text_encoder, device, _ = _restore(args)
    paired_path = args.paired or cfg.data.test_paired
    if not paired_path:
        raise SystemExit("no paired test file: pass --paired or set data.test_paired")

    results = evaluate_paired(
        generator,
        text_encoder,
        cfg,
        device,
        paired_path,
        max_samples=args.limit,
        compute_bertscore=not args.no_bertscore,
        return_outputs=True,
    )
    outputs = results.pop("outputs")
    sources = results.pop("sources")

    from .data.datasets import read_paired_tsv

    references = [refs for _, refs in read_paired_tsv(paired_path)][: len(sources)]
    by_length = evaluate_by_length(sources, outputs, references)

    print(json.dumps({"overall": results, "by_length": by_length}, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
        print(f"wrote outputs to {path}")


def cmd_analyze(args) -> None:
    from .eval.latent_analysis import plot_latent_spaces, plot_paired_latents

    cfg, generator, text_encoder, device, _ = _restore(args)
    out_dir = Path(args.out_dir)
    separation = plot_latent_spaces(
        generator, text_encoder, cfg, device, out_dir / "latent_spaces.png"
    )
    plot_paired_latents(
        generator, text_encoder, cfg, device, out_dir / "latent_spaces_paired.png"
    )
    print(json.dumps(separation, indent=2))


def cmd_sweep(args) -> None:
    from .eval.style_ratio_sweep import DEFAULT_RATIOS, run_sweep

    cfg = load_config(args.config, args.overrides)
    ratios = tuple(args.ratios) if args.ratios else DEFAULT_RATIOS
    results = run_sweep(cfg, ratios, args.output, run_phase2=not args.phase1_only)
    print(json.dumps(results, indent=2, default=str))


def cmd_profile(args) -> None:
    from .eval.profile_cost import format_report, profile

    if args.ckpt:
        cfg, generator, text_encoder, device, _ = _restore(args)
    else:
        # Cost depends only on the architecture, so a config alone is enough --
        # and it is the only way to profile the paper's geometry before a run
        # of that size has produced a checkpoint.
        from transformers import AutoTokenizer

        from .data.datasets import TextEncoder
        from .models.generator import CyBERTGenerator
        from .training.trainer_utils import resolve_device

        cfg = load_config(args.config, args.overrides)
        device = resolve_device(cfg.run.device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_name)
        generator = CyBERTGenerator(cfg, tokenizer).to(device).eval()
        text_encoder = TextEncoder(tokenizer, cfg.model.max_length)

    results = profile(generator, text_encoder, cfg, device)
    print(format_report(results))
    print(json.dumps(results, indent=2))


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cybert", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="build unpaired corpora from a Wikipedia dump")
    p.add_argument("--source", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-sentences", type=int, default=None)
    p.add_argument("--fre-complex-max", type=float, default=10.0)
    p.add_argument("--fre-simple-min", type=float, default=70.0)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=64)
    p.add_argument("--valid-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train-phase1", help="denoising reconstruction + disentanglement")
    _add_common(p)
    p.add_argument("--init-from", default=None)
    p.set_defaults(func=cmd_train_phase1)

    p = sub.add_parser("train-phase2", help="cycle-consistent adversarial style transfer")
    _add_common(p)
    p.add_argument("--init-from", default=None, help="Phase 1 checkpoint")
    p.set_defaults(func=cmd_train_phase2)

    p = sub.add_parser("generate", help="simplify (or complexify) sentences")
    _add_common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help="text file or paired TSV")
    p.add_argument("--output", default=None)
    p.add_argument("--direction", choices=("simplify", "complexify"), default="simplify")
    p.add_argument("--beam", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="SARI / FKGL / BERTScore on a paired test set")
    _add_common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--paired", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-bertscore", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("analyze", help="latent-space PCA and separability")
    _add_common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="runs/analysis")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("sweep", help="style-ratio sweep (Table 7)")
    _add_common(p)
    p.add_argument("--ratios", type=float, nargs="*", default=None)
    p.add_argument("--output", default="runs/style_ratio_sweep.json")
    p.add_argument("--phase1-only", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("profile", help="inference FLOPs and memory (Table 5)")
    _add_common(p)
    p.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint to profile; omit to profile the architecture from --config alone",
    )
    p.set_defaults(func=cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)


if __name__ == "__main__":
    main()
