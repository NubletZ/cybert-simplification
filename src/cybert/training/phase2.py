"""Phase 2: cycle-consistent adversarial style transfer (paper 3.2.3).

    L(2) = lambda_cycle * L_cycle
         + lambda_adv^sim * L_GAN^{G,sim} + lambda_adv^com * L_GAN^{G,com}
         + lambda_style * L_cls(s) + lambda_content * L_GRL(c)
         + lambda_emb   * L_emb(c)                                    (Eq. 16)

One shared generator provides both mappings; the direction is chosen by the
target style vector drawn from the EMA style bank:

    G(x) = decode(fuse(c(x), s_bar[simple]))     complex -> simple
    F(y) = decode(fuse(c(y), s_bar[complex]))    simple  -> complex

Per step, discriminators are updated first on hard greedy samples with the
encoder detached (4.3), then the generator is updated with the discriminator
heads frozen.

Differentiability. Eq. 11 stop-gradients the first-pass greedy decode, so the
gradient in Eq. 9 has to travel through the *second* pass, and phi needs a
sentence. Both paths therefore emit per-step distributions and feed phi soft
embeddings (``probs @ E``) through ``inputs_embeds``:

``cycle_mode: teacher_forced``
    The second pass is teacher-forced on the original sentence. Low variance,
    and it matches Eq. 11 literally -- nothing backpropagates through a
    discrete decode.

``cycle_mode: gumbel``
    The second pass free-runs with straight-through Gumbel-softmax, so the
    round trip is a genuine free-running reconstruction at the cost of sampling
    variance. ``tau`` anneals from 2.0 to 0.5.

The generator half of the adversarial loss needs the same treatment, and uses
a differentiable free-running decode in both modes.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import CyBERTConfig, save_config
from ..data.datasets import (
    PairedCollator,
    PairedDataset,
    TextEncoder,
    UnpairedCollator,
    UnpairedDataset,
)
from ..data.readability import COMPLEX, SIMPLE
from ..eval.evaluate import quick_validation_sari
from ..inference.generate import (
    greedy_decode,
    prepend_cls_embedding,
    soft_decode_free_running,
    soft_decode_teacher_forced,
    soft_to_embeddings,
    strip_after_eos,
)
from ..models.discriminator import DomainDiscriminator
from ..models.encoder import FrozenSemanticEncoder
from ..models.generator import CyBERTGenerator
from .losses import (
    LossRecord,
    classification_accuracy,
    content_embedding_loss,
    cycle_consistency_loss,
    grl_lambda_schedule,
    gumbel_tau_schedule,
    reconstruction_loss,
    style_classification_loss,
)
from .phase1 import build_components
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


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def build_unpaired_loader(cfg: CyBERTConfig, text_encoder: TextEncoder) -> DataLoader:
    dataset = UnpairedDataset(cfg.data.train_complex, cfg.data.train_simple, seed=cfg.data.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=UnpairedCollator(text_encoder),
        drop_last=True,
    )


def build_paired_loader(cfg: CyBERTConfig, text_encoder: TextEncoder) -> DataLoader | None:
    """Supervised batches interleaved with the unpaired stream (4.2)."""
    if not cfg.data.train_paired or cfg.phase2.lambda_paired <= 0:
        return None
    dataset = PairedDataset(cfg.data.train_paired)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=PairedCollator(text_encoder),
        drop_last=True,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pooled_from_ids(
    generator: CyBERTGenerator, ids: torch.Tensor, pad_id: int, grad: bool = False
) -> torch.Tensor:
    """Pooled [CLS] feature of hard token ids, via the generator's encoder."""
    mask = (ids != pad_id).long()
    # A fully padded row would produce NaNs in attention; keep its first slot.
    mask[:, 0] = 1
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        _, pooled = generator.encoder(input_ids=ids, attention_mask=mask)
    return pooled


def _pooled_from_soft(
    generator: CyBERTGenerator, probs: torch.Tensor
) -> torch.Tensor:
    """Pooled feature of a soft sequence, keeping the graph intact."""
    embeds = soft_to_embeddings(probs, generator.encoder.word_embeddings.weight)
    cls_vector = generator.encoder.word_embeddings.weight[generator.bos_id]
    embeds, mask = prepend_cls_embedding(embeds, cls_vector)
    _, pooled = generator.encoder(inputs_embeds=embeds, attention_mask=mask)
    return pooled


def _phi_from_soft(
    semantic: FrozenSemanticEncoder,
    probs: torch.Tensor,
    bos_id: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """phi(soft sequence): the differentiable side of Eq. 9."""
    embeds = soft_to_embeddings(probs, semantic.word_embeddings.weight)
    cls_vector = semantic.word_embeddings.weight[bos_id]
    embeds, attn = prepend_cls_embedding(embeds, cls_vector, mask)
    return semantic(inputs_embeds=embeds, attention_mask=attn)


def _shift_right(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split token ids into ``(decoder_input, labels)`` for teacher forcing."""
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _cycle_term(
    generator: CyBERTGenerator,
    semantic: FrozenSemanticEncoder,
    cfg: CyBERTConfig,
    pseudo_ids: torch.Tensor,
    original_ids: torch.Tensor,
    original_mask: torch.Tensor,
    target_label: int,
    tau: float,
) -> torch.Tensor:
    """One direction of Eq. 9.

    ``pseudo_ids`` is the detached first-pass output; the second pass maps it
    back towards ``original_ids``' domain and is scored against phi(original).
    """
    pad_id = generator.pad_id
    pseudo_mask = (pseudo_ids != pad_id).long()
    pseudo_mask[:, 0] = 1

    encoded = generator.encode(input_ids=pseudo_ids, attention_mask=pseudo_mask)
    style = generator.target_style(target_label, pseudo_ids.size(0), encoded.style)

    soft_mask: torch.Tensor | None = None
    if cfg.phase2.cycle_mode == "teacher_forced":
        decoder_input, _ = _shift_right(original_ids)
        probs = soft_decode_teacher_forced(
            generator,
            encoded,
            decoder_input,
            style=style,
            temperature=cfg.phase2.soft_temperature,
        )
        # The distributions predict original_ids[:, 1:]; reuse that slice of the
        # mask so padded target positions stay out of phi's pooling.
        soft_mask = original_mask[:, 1:].contiguous()
    else:
        probs = soft_decode_free_running(
            generator,
            encoded,
            style=style,
            max_length=cfg.phase2.cycle_max_len,
            tau=tau,
            gumbel=True,
            hard=True,
        )

    round_trip = _phi_from_soft(semantic, probs, generator.bos_id, soft_mask)
    original = semantic.encode_ids(original_ids, original_mask)
    return cycle_consistency_loss(
        round_trip, original, p=1, normalize=cfg.phase2.cycle_normalize
    )


def _latent_terms(
    generator: CyBERTGenerator,
    semantic: FrozenSemanticEncoder,
    cfg: CyBERTConfig,
    encoded_complex,
    encoded_simple,
    complex_ids: torch.Tensor,
    complex_mask: torch.Tensor,
    simple_ids: torch.Tensor,
    simple_mask: torch.Tensor,
    grl_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Eq. 16's ``L_cls(s)``, ``L_GRL(c)`` and ``L_emb(c)``, over both domains."""
    device = complex_ids.device
    batch = complex_ids.size(0)
    labels = torch.cat(
        [
            torch.full((batch,), COMPLEX, dtype=torch.long, device=device),
            torch.full((simple_ids.size(0),), SIMPLE, dtype=torch.long, device=device),
        ]
    )
    style = torch.cat([encoded_complex.style, encoded_simple.style], dim=0)
    content = torch.cat([encoded_complex.content, encoded_simple.content], dim=0)

    style_logits = generator.style_logits(style)
    l_style = style_classification_loss(style_logits, labels)

    content_logits = generator.content_adv_logits(content, grl_lambda)
    l_content = style_classification_loss(content_logits, labels)

    target = torch.cat(
        [
            semantic.encode_ids(complex_ids, complex_mask),
            semantic.encode_ids(simple_ids, simple_mask),
        ],
        dim=0,
    )
    l_emb = content_embedding_loss(generator.project_content(content), target)

    generator.update_style_bank(style, labels)

    stats = {
        "acc_style": classification_accuracy(style_logits, labels),
        "acc_content_adv": classification_accuracy(content_logits, labels),
    }
    return l_style, l_content, l_emb, stats


def train(cfg: CyBERTConfig, init_from: str | None = None) -> dict:
    """Run Phase 2 and return the best validation metrics."""
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    _, generator, semantic, text_encoder = build_components(cfg, device)
    if init_from:
        checkpoint = load_checkpoint(init_from, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        logger.info("initialised generator from %s (phase=%s)", init_from, checkpoint.get("phase"))
    else:
        logger.warning(
            "Phase 2 without --init-from: the decoder has not been trained as a "
            "language model, so adversarial training will start from noise."
        )
    generator.style_bank.momentum = cfg.phase2.style_bank_momentum

    hidden = generator.encoder.hidden_size
    d_sim = DomainDiscriminator(hidden).to(device)
    d_com = DomainDiscriminator(hidden).to(device)

    unpaired = build_unpaired_loader(cfg, text_encoder)
    paired = build_paired_loader(cfg, text_encoder)
    paired_iter = itertools.cycle(paired) if paired is not None else None

    # Phase 2 alternates discriminator and generator updates every batch, so
    # gradient accumulation is not applied here; the schedule counts one
    # generator step per batch.
    if cfg.optim.grad_accum > 1:
        logger.warning("optim.grad_accum is ignored in Phase 2 (alternating GAN updates)")
    steps_per_epoch = max(1, len(unpaired))
    total_steps = steps_per_epoch * cfg.phase2.epochs

    gen_optimizer = build_optimizer(generator, cfg)
    gen_scheduler = build_scheduler(gen_optimizer, total_steps, cfg.optim.warmup_ratio)
    disc_optimizer = torch.optim.AdamW(
        list(d_sim.parameters()) + list(d_com.parameters()),
        lr=cfg.optim.disc_lr,
        weight_decay=cfg.optim.weight_decay,
    )

    stopper = EarlyStopping(patience=cfg.run.patience)
    record = LossRecord()
    global_step = 0
    history: list[dict] = []
    best_metrics: dict = {}
    pad_id = generator.pad_id

    logger.info(
        "Phase 2: %d unpaired items, %d steps/epoch, cycle_mode=%s lambda_cycle=%.2f",
        len(unpaired.dataset),
        steps_per_epoch,
        cfg.phase2.cycle_mode,
        cfg.phase2.lambda_cycle,
    )

    for epoch in range(cfg.phase2.epochs):
        generator.train()
        d_sim.train()
        d_com.train()
        record.reset()

        for step, batch in enumerate(unpaired):
            progress = global_step / max(1, total_steps)
            grl_lambda = grl_lambda_schedule(progress, cfg.phase2.grl_max, warmup_ratio=0.0)
            tau = gumbel_tau_schedule(
                progress, cfg.phase2.gumbel_tau_start, cfg.phase2.gumbel_tau_end
            )

            complex_ids = batch["complex_input_ids"].to(device)
            complex_mask = batch["complex_attention_mask"].to(device)
            simple_ids = batch["simple_input_ids"].to(device)
            simple_mask = batch["simple_attention_mask"].to(device)

            # ---- first pass: detached greedy pseudo-sentences (Eq. 11) ---- #
            generator.eval()
            with torch.no_grad():
                enc_x = generator.encode(input_ids=complex_ids, attention_mask=complex_mask)
                enc_y = generator.encode(input_ids=simple_ids, attention_mask=simple_mask)
                style_simple = generator.target_style(SIMPLE, complex_ids.size(0), enc_x.style)
                style_complex = generator.target_style(COMPLEX, simple_ids.size(0), enc_y.style)
                y_hat = greedy_decode(
                    generator, enc_x, style_simple, cfg.phase2.cycle_max_len
                )
                x_hat = greedy_decode(
                    generator, enc_y, style_complex, cfg.phase2.cycle_max_len
                )
            y_hat = strip_after_eos(y_hat, generator.eos_id, pad_id).detach()
            x_hat = strip_after_eos(x_hat, generator.eos_id, pad_id).detach()
            generator.train()

            # ---- discriminator update (Eqs. 12, 14) ----------------------- #
            for _ in range(cfg.phase2.d_steps):
                disc_optimizer.zero_grad(set_to_none=True)
                real_simple = _pooled_from_ids(generator, simple_ids, pad_id)
                fake_simple = _pooled_from_ids(generator, y_hat, pad_id)
                real_complex = _pooled_from_ids(generator, complex_ids, pad_id)
                fake_complex = _pooled_from_ids(generator, x_hat, pad_id)

                d_loss = d_sim.loss_real_fake(real_simple, fake_simple) + d_com.loss_real_fake(
                    real_complex, fake_complex
                )
                d_loss.backward()
                clip_gradients([d_sim, d_com], cfg.optim.grad_clip)
                disc_optimizer.step()

            record.add("d_loss", d_loss)
            record.add("d_acc_sim", d_sim.accuracy(real_simple, fake_simple))
            record.add("d_acc_com", d_com.accuracy(real_complex, fake_complex))

            # ---- generator update (Eqs. 9, 13, 15, 16) -------------------- #
            gen_optimizer.zero_grad(set_to_none=True)
            for param in itertools.chain(d_sim.parameters(), d_com.parameters()):
                param.requires_grad_(False)

            enc_x = generator.encode(input_ids=complex_ids, attention_mask=complex_mask)
            enc_y = generator.encode(input_ids=simple_ids, attention_mask=simple_mask)
            style_simple = generator.target_style(SIMPLE, complex_ids.size(0), enc_x.style)
            style_complex = generator.target_style(COMPLEX, simple_ids.size(0), enc_y.style)

            # Differentiable fakes for the generator half of the GAN loss.
            probs_simple = soft_decode_free_running(
                generator, enc_x, style_simple, cfg.phase2.cycle_max_len, tau, gumbel=True
            )
            probs_complex = soft_decode_free_running(
                generator, enc_y, style_complex, cfg.phase2.cycle_max_len, tau, gumbel=True
            )
            l_adv_sim = d_sim.loss_generator(_pooled_from_soft(generator, probs_simple))
            l_adv_com = d_com.loss_generator(_pooled_from_soft(generator, probs_complex))

            # Cycle consistency, both directions (Eq. 9).
            l_cycle = _cycle_term(
                generator, semantic, cfg, y_hat, complex_ids, complex_mask, COMPLEX, tau
            ) + _cycle_term(
                generator, semantic, cfg, x_hat, simple_ids, simple_mask, SIMPLE, tau
            )

            l_style, l_content, l_emb, latent_stats = _latent_terms(
                generator,
                semantic,
                cfg,
                enc_x,
                enc_y,
                complex_ids,
                complex_mask,
                simple_ids,
                simple_mask,
                grl_lambda,
            )

            total = (
                cfg.phase2.lambda_cycle * l_cycle
                + cfg.phase2.lambda_adv_sim * l_adv_sim
                + cfg.phase2.lambda_adv_com * l_adv_com
                + cfg.phase2.lambda_style * l_style
                + cfg.phase2.lambda_content * l_content
                + cfg.phase2.lambda_emb * l_emb
            )

            # Interleaved paired supervision (4.2).
            l_paired = torch.zeros((), device=device)
            if paired_iter is not None and step % cfg.phase2.paired_every == 0:
                pbatch = next(paired_iter)
                p_ids = pbatch["input_ids"].to(device)
                p_mask = pbatch["attention_mask"].to(device)
                p_dec = pbatch["decoder_input_ids"].to(device)
                p_labels = pbatch["labels"].to(device)
                enc_p = generator.encode(input_ids=p_ids, attention_mask=p_mask)
                style_p = generator.target_style(SIMPLE, p_ids.size(0), enc_p.style)
                p_logits = generator.decode_teacher_forced(enc_p, p_dec, style_p)
                l_paired = reconstruction_loss(p_logits, p_labels)
                total = total + cfg.phase2.lambda_paired * l_paired

            total.backward()
            clip_gradients([generator], cfg.optim.grad_clip)
            gen_optimizer.step()
            gen_scheduler.step()
            global_step += 1

            for param in itertools.chain(d_sim.parameters(), d_com.parameters()):
                param.requires_grad_(True)
            disc_optimizer.zero_grad(set_to_none=True)

            record.add("loss", total)
            record.add("l_cycle", l_cycle)
            record.add("l_adv_sim", l_adv_sim)
            record.add("l_adv_com", l_adv_com)
            record.add("l_style", l_style)
            record.add("l_content", l_content)
            record.add("l_emb", l_emb)
            record.add("l_paired", l_paired)
            record.add("tau", tau)
            for name, value in latent_stats.items():
                record.add(name, value)

            if global_step % cfg.run.log_every == 0:
                logger.info("epoch %d step %d | %s", epoch, global_step, record.format())

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
            metrics["valid"] = val
            improved = stopper.update(val["sari"], epoch)
            logger.info(
                "epoch %d | train %s | valid SARI %.2f FKGL %.2f%s",
                epoch,
                record.format(),
                val["sari"],
                val["fkgl"],
                "  <- best" if improved else "",
            )
            if improved:
                best_metrics = {"epoch": epoch, **val}
                save_checkpoint(
                    output_dir / "best.pt",
                    generator,
                    cfg,
                    "phase2",
                    epoch,
                    val,
                    discriminators={"d_sim": d_sim, "d_com": d_com},
                )
            if not cfg.run.save_best_only:
                save_checkpoint(
                    output_dir / f"epoch{epoch}.pt", generator, cfg, "phase2", epoch, val
                )
            if stopper.should_stop:
                logger.info("early stopping at epoch %d (best epoch %d)", epoch, stopper.best_epoch)
                history.append(metrics)
                break
        else:
            save_checkpoint(output_dir / "best.pt", generator, cfg, "phase2", epoch, {})
            best_metrics = {"epoch": epoch}

        history.append(metrics)

    write_metrics(output_dir / "phase2_history.json", {"history": history, "best": best_metrics})
    if not (output_dir / "best.pt").exists():
        save_checkpoint(output_dir / "best.pt", generator, cfg, "phase2", -1, {})
    return best_metrics
