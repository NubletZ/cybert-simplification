"""Shared training machinery: optimisers, checkpoints, early stopping.

4.3 specifies AdamW with *separate* learning rates -- 2e-5 for BERT and 1e-3
for the LSTM and projection layers -- weight decay 0.01, and gradient clipping
at a global norm of 1.0. :func:`build_optimizer` implements that split, and
keeps biases and LayerNorm parameters out of weight decay, which is the
standard AdamW convention for transformers.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from ..config import CyBERTConfig, to_dict

logger = logging.getLogger(__name__)

_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _is_no_decay(param_name: str) -> bool:
    return any(marker in param_name for marker in _NO_DECAY)


def build_optimizer(
    model: nn.Module,
    cfg: CyBERTConfig,
    extra_modules: Iterable[nn.Module] = (),
) -> torch.optim.Optimizer:
    """AdamW with the paper's two learning rates (4.3).

    Anything under ``model.encoder`` moves at ``bert_lr``; the decoder, the
    latent projections and the auxiliary heads move at ``head_lr``.
    """
    groups: list[dict[str, Any]] = [
        {"params": [], "lr": cfg.optim.bert_lr, "weight_decay": cfg.optim.weight_decay},
        {"params": [], "lr": cfg.optim.bert_lr, "weight_decay": 0.0},
        {"params": [], "lr": cfg.optim.head_lr, "weight_decay": cfg.optim.weight_decay},
        {"params": [], "lr": cfg.optim.head_lr, "weight_decay": 0.0},
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bert = name.startswith("encoder.")
        base = 0 if is_bert else 2
        groups[base + int(_is_no_decay(name))]["params"].append(param)

    for module in extra_modules:
        for name, param in module.named_parameters():
            if param.requires_grad:
                groups[2 + int(_is_no_decay(name))]["params"].append(param)

    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup then linear decay, per-group."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        remaining = max(1, total_steps - warmup)
        return max(0.0, (total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_gradients(modules: Iterable[nn.Module], max_norm: float) -> float:
    params = [p for module in modules for p in module.parameters() if p.grad is not None]
    if not params or max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


@dataclass
class EarlyStopping:
    """Stop when validation SARI has not improved for ``patience`` epochs (4.3)."""

    patience: int = 5
    best: float = float("-inf")
    bad_epochs: int = 0
    best_epoch: int = -1

    def update(self, score: float, epoch: int) -> bool:
        """Return True when ``score`` is a new best."""
        if score > self.best:
            self.best = score
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    cfg: CyBERTConfig,
    phase: str,
    epoch: int,
    metrics: dict[str, float] | None = None,
    discriminators: dict[str, nn.Module] | None = None,
) -> None:
    """Persist weights, config and the style bank together.

    The style bank is part of ``generator.state_dict()`` because it lives in
    registered buffers; without it, inference could not reproduce the target
    style vectors that training converged on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generator": generator.state_dict(),
        "config": to_dict(cfg),
        "phase": phase,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if discriminators:
        payload["discriminators"] = {k: v.state_dict() for k, v in discriminators.items()}
    torch.save(payload, path)
    logger.info("saved checkpoint to %s (phase=%s epoch=%d)", path, phase, epoch)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_generator_weights(generator: nn.Module, checkpoint: dict[str, Any], strict: bool = True) -> None:
    missing, unexpected = generator.load_state_dict(checkpoint["generator"], strict=strict)
    if missing:
        logger.warning("missing keys when loading generator: %s", missing)
    if unexpected:
        logger.warning("unexpected keys when loading generator: %s", unexpected)


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(1, self.count)
