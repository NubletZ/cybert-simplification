"""Phase 1: denoising language model + latent disentanglement (paper 3.2.2).

    L(1) = lambda_gen * L_gen
         + lambda_cls(s) * L_cls(s)
         + lambda_cls(c) * L_GRL(c)
         + lambda_emb   * L_emb(c)                                     (Eq. 8)

Each step corrupts ``x`` with POS-guided span infilling, encodes the corrupted
sentence, splits the representation, and reconstructs the *original* ``x``.
Reconstruction conditions on the sentence's own style vector -- this phase
teaches the decoder to generate and the latent to separate, not to transfer.

Early stopping is on validation SARI with patience 5 (4.3), which is measured
by decoding towards the simple domain even though the training objective is
reconstruction; that is the quantity Phase 2 will go on to optimise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import CyBERTConfig, save_config
from ..data.datasets import Phase1Collator, Phase1Dataset, TextEncoder
from ..data.noise import build_noise
from ..data.pos_tagger import POSTagger
from ..eval.evaluate import quick_validation_sari
from ..models.encoder import FrozenSemanticEncoder
from ..models.generator import CyBERTGenerator
from .losses import (
    LossRecord,
    classification_accuracy,
    content_embedding_loss,
    grl_lambda_schedule,
    reconstruction_loss,
    style_classification_loss,
)
from .trainer_utils import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    clip_gradients,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_metrics,
)

logger = logging.getLogger(__name__)


def build_components(cfg: CyBERTConfig, device: torch.device):
    """Tokenizer, generator, frozen phi and the text encoder used everywhere."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_name)
    generator = CyBERTGenerator(cfg, tokenizer).to(device)
    semantic = FrozenSemanticEncoder(cfg.model.bert_name).to(device)
    text_encoder = TextEncoder(tokenizer, cfg.model.max_length)
    return tokenizer, generator, semantic, text_encoder


def build_dataloader(cfg: CyBERTConfig, text_encoder: TextEncoder, shuffle: bool = True) -> DataLoader:
    dataset = Phase1Dataset(cfg.data.train_complex, cfg.data.train_simple)
    tagger = POSTagger(
        cfg.noise.spacy_model, cfg.noise.cache_dir, enabled=cfg.noise.use_pos
    )
    collator = Phase1Collator(
        encoder=text_encoder,
        noise=build_noise(cfg.noise, seed=cfg.data.seed),
        tagger=tagger,
        use_pos=cfg.noise.use_pos,
    )
    # POS tagging happens inside the collator and fills a shared cache. Worker
    # processes would each build their own copy and none of it would survive
    # back in the parent, so tagging pins the loader to the main process.
    num_workers = 0 if cfg.noise.use_pos and cfg.noise.enabled else cfg.data.num_workers
    if num_workers != cfg.data.num_workers:
        logger.info("POS tagging enabled: running the data loader with num_workers=0")

    loader = DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    loader.pos_tagger = tagger  # kept so the trainer can flush the cache
    return loader


def build_valid_dataloader(cfg: CyBERTConfig, text_encoder: TextEncoder) -> DataLoader | None:
    """Held-out loader for the validation reconstruction loss.

    Uses the unpaired validation files, so the measured quantity is exactly
    the Phase 1 training objective (Eq. 7) on data the model never sees.
    """
    try:
        dataset = Phase1Dataset(cfg.data.valid_complex, cfg.data.valid_simple)
    except FileNotFoundError:
        logger.warning("no unpaired validation files; reconstruction validation disabled")
        return None
    if len(dataset) == 0:
        return None

    tagger = POSTagger(cfg.noise.spacy_model, cfg.noise.cache_dir, enabled=cfg.noise.use_pos)
    collator = Phase1Collator(
        encoder=text_encoder,
        noise=build_noise(cfg.noise, seed=cfg.data.seed + 1),
        tagger=tagger,
        use_pos=cfg.noise.use_pos,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )


@torch.no_grad()
def validate_reconstruction(
    generator: CyBERTGenerator,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Mean token-level reconstruction loss (Eq. 7) on held-out sentences."""
    generator.eval()
    total = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        encoded = generator.encode(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        logits = generator.decode_teacher_forced(encoded, batch["decoder_input_ids"].to(device))
        total += float(reconstruction_loss(logits, batch["labels"].to(device)))
        count += 1
    generator.train()
    return total / max(1, count)


def compute_phase1_losses(
    generator: CyBERTGenerator,
    semantic: FrozenSemanticEncoder,
    batch: dict,
    cfg: CyBERTConfig,
    grl_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate Eq. 8 on one batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    labels = batch["labels"].to(device)
    style_labels = batch["style_labels"].to(device)
    clean_ids = batch["clean_input_ids"].to(device)
    clean_mask = batch["clean_attention_mask"].to(device)

    encoded = generator.encode(input_ids=input_ids, attention_mask=attention_mask)

    # Eq. 7 -- reconstruct the clean sentence from the corrupted encoding,
    # conditioned on the sentence's own style.
    logits = generator.decode_teacher_forced(encoded, decoder_input_ids)
    l_gen = reconstruction_loss(logits, labels)

    # Eq. 4 -- s must predict the complexity label.
    style_logits = generator.style_logits(encoded.style)
    l_cls_style = style_classification_loss(style_logits, style_labels)

    # Eq. 5 -- the same objective on c, behind gradient reversal.
    content_logits = generator.content_adv_logits(encoded.content, grl_lambda)
    l_grl_content = style_classification_loss(content_logits, style_labels)

    # Eq. 6 -- anchor c to the frozen sentence embedding of the *clean* input.
    target = semantic.encode_ids(clean_ids, clean_mask)
    l_emb = content_embedding_loss(generator.project_content(encoded.content), target)

    total = (
        cfg.phase1.lambda_gen * l_gen
        + cfg.phase1.lambda_cls_style * l_cls_style
        + cfg.phase1.lambda_cls_content * l_grl_content
        + cfg.phase1.lambda_emb * l_emb
    )

    generator.update_style_bank(encoded.style, style_labels)

    stats = {
        "loss": float(total.detach()),
        "l_gen": float(l_gen.detach()),
        "l_cls_style": float(l_cls_style.detach()),
        "l_grl_content": float(l_grl_content.detach()),
        "l_emb": float(l_emb.detach()),
        "acc_style": classification_accuracy(style_logits, style_labels),
        "acc_content_adv": classification_accuracy(content_logits, style_labels),
        "grl_lambda": grl_lambda,
    }
    return total, stats


def train(cfg: CyBERTConfig, init_from: str | None = None) -> dict:
    """Run Phase 1 and return the best validation metrics."""
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    _, generator, semantic, text_encoder = build_components(cfg, device)
    if init_from:
        checkpoint = load_checkpoint(init_from, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        logger.info("initialised generator from %s", init_from)

    loader = build_dataloader(cfg, text_encoder)
    valid_loader = build_valid_dataloader(cfg, text_encoder)
    steps_per_epoch = max(1, len(loader) // cfg.optim.grad_accum)
    total_steps = steps_per_epoch * cfg.phase1.epochs

    optimizer = build_optimizer(generator, cfg)
    scheduler = build_scheduler(optimizer, total_steps, cfg.optim.warmup_ratio)
    stopper = EarlyStopping(patience=cfg.run.patience)

    record = LossRecord()
    global_step = 0
    history: list[dict] = []
    best_metrics: dict = {}

    logger.info(
        "Phase 1: %d sentences, %d steps/epoch, style_dim=%d content_dim=%d fusion=%s, "
        "selecting on %s",
        len(loader.dataset),
        steps_per_epoch,
        cfg.model.style_dim,
        cfg.model.content_dim,
        cfg.model.fusion,
        cfg.phase1.select_metric,
    )

    for epoch in range(cfg.phase1.epochs):
        generator.train()
        record.reset()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            progress = global_step / max(1, total_steps)
            grl_lambda = grl_lambda_schedule(
                progress, cfg.phase1.grl_max, cfg.phase1.grl_warmup_ratio
            )
            total, stats = compute_phase1_losses(
                generator, semantic, batch, cfg, grl_lambda, device
            )
            (total / cfg.optim.grad_accum).backward()

            if (step + 1) % cfg.optim.grad_accum == 0:
                clip_gradients([generator], cfg.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            for name, value in stats.items():
                record.add(name, value)
            if global_step % cfg.run.log_every == 0 and (step + 1) % cfg.optim.grad_accum == 0:
                logger.info("epoch %d step %d | %s", epoch, global_step, record.format())

        tagger = getattr(loader, "pos_tagger", None)
        if tagger is not None:
            tagger.flush()

        metrics = {"epoch": epoch, "train": record.mean()}
        if cfg.run.eval_every_epoch:
            val = quick_validation_sari(
                generator,
                text_encoder,
                cfg,
                device,
                cfg.data.valid_paired,
                cfg.run.max_eval_samples,
            )
            if valid_loader is not None:
                val["recon"] = validate_reconstruction(generator, valid_loader, device)
            metrics["valid"] = val

            # EarlyStopping maximises, so reconstruction loss enters negated.
            if cfg.phase1.select_metric == "recon" and "recon" in val:
                score = -val["recon"]
            else:
                score = val["sari"]
            improved = stopper.update(score, epoch)

            logger.info(
                "epoch %d | train %s | valid SARI %.2f FKGL %.2f recon %.4f%s",
                epoch,
                record.format(),
                val["sari"],
                val["fkgl"],
                val.get("recon", float("nan")),
                "  <- best" if improved else "",
            )
            if improved:
                best_metrics = {"epoch": epoch, **val}
                save_checkpoint(
                    output_dir / "best.pt", generator, cfg, "phase1", epoch, val
                )
            # Always keep the most recent epoch as well. The two selection
            # metrics disagree in this phase, so the alternative choice must
            # remain available without retraining.
            save_checkpoint(output_dir / "last.pt", generator, cfg, "phase1", epoch, val)
            if not cfg.run.save_best_only:
                save_checkpoint(
                    output_dir / f"epoch{epoch}.pt", generator, cfg, "phase1", epoch, val
                )
            if stopper.should_stop:
                logger.info("early stopping at epoch %d (best epoch %d)", epoch, stopper.best_epoch)
                history.append(metrics)
                break
        else:
            save_checkpoint(output_dir / "best.pt", generator, cfg, "phase1", epoch, {})
            best_metrics = {"epoch": epoch}

        history.append(metrics)

    write_metrics(output_dir / "phase1_history.json", {"history": history, "best": best_metrics})
    if not (output_dir / "best.pt").exists():
        save_checkpoint(output_dir / "best.pt", generator, cfg, "phase1", -1, {})
    return best_metrics
