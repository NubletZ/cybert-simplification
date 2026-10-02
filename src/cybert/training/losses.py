"""Objective terms.

Equation numbers refer to the paper:

* Eq. 4  ``L_cls(z)``   -- style cross-entropy from a latent vector
* Eq. 5  ``L_GRL(c)``   -- the same objective on ``c`` behind gradient reversal
* Eq. 6  ``L_emb(c)``   -- cosine distance between ``P_c(c)`` and ``E_BERT(x)``
* Eq. 7  ``L_gen``      -- token-level reconstruction cross-entropy
* Eq. 8  ``L(1)``       -- the Phase 1 total
* Eq. 9  ``L_cycle``    -- round-trip distance in the frozen phi feature space
* Eqs. 12-15            -- discriminator and generator adversarial terms
* Eq. 16 ``L(2)``       -- the Phase 2 total
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def reconstruction_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. 7: token-level cross-entropy, padding excluded via ``-100``."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
    )


def style_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. 4. Applied to ``s`` directly and to ``GRL(c)`` for Eq. 5."""
    return F.cross_entropy(logits, labels)


def content_embedding_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. 6: ``1 - cos(E_BERT(x), P_c(c))``.

    The target is detached: phi is a fixed semantic anchor, so the gradient
    must move ``P_c`` and the content projection, never the reference.
    """
    return (1.0 - F.cosine_similarity(projected, target.detach(), dim=-1)).mean()


def cycle_consistency_loss(
    round_trip: torch.Tensor, original: torch.Tensor, p: int = 1, normalize: bool = True
) -> torch.Tensor:
    """Eq. 9 with ``p = 1``.

    ``L1`` "encourages sparse deviations in feature space and discourages
    unnecessary rewriting while preserving semantic content" (3.2.3).

    ``normalize`` divides by the feature width. Eq. 9 is a plain norm, and an
    L1 sum over BERT's 768 dimensions lands around 450 while every other term
    in Eq. 16 is order 1 -- with any lambda_cycle near the CycleGAN convention
    the cycle term is ~700x the rest and the decoder collapses to emitting
    [SEP] immediately. Dividing by the width is a constant rescaling of the
    same objective, absorbed into lambda_cycle, and it puts the weights in
    Eq. 16 on a comparable scale. Set ``phase2.cycle_normalize: false`` to
    recover the raw norm (and lower lambda_cycle by ~1/768 to match).
    """
    distance = torch.norm(round_trip - original.detach(), p=p, dim=-1)
    if normalize:
        distance = distance / round_trip.size(-1)
    return distance.mean()


@dataclass
class LossRecord:
    """Accumulates scalar values for logging without holding onto the graph."""

    values: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, value: torch.Tensor | float) -> None:
        scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        self.values[name] = self.values.get(name, 0.0) + scalar
        self.counts[name] = self.counts.get(name, 0) + 1

    def mean(self) -> dict[str, float]:
        return {k: v / max(1, self.counts[k]) for k, v in self.values.items()}

    def reset(self) -> None:
        self.values.clear()
        self.counts.clear()

    def format(self) -> str:
        return " ".join(f"{k}={v:.4f}" for k, v in sorted(self.mean().items()))


def grl_lambda_schedule(progress: float, max_lambda: float = 1.0, warmup_ratio: float = 0.1) -> float:
    """DANN ramp for the gradient-reversal coefficient.

    ``2 / (1 + exp(-10 * p)) - 1`` over the first ``warmup_ratio`` of training.
    The paper does not specify a schedule; ramping in matters because a
    full-strength adversary against a randomly initialised content projection
    destroys the content vector before the reconstruction term can shape it.
    """
    if warmup_ratio <= 0:
        return max_lambda
    p = min(1.0, max(0.0, progress / warmup_ratio))
    import math

    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def gumbel_tau_schedule(progress: float, start: float = 2.0, end: float = 0.5) -> float:
    """Linear anneal of the Gumbel-softmax temperature over training."""
    progress = min(1.0, max(0.0, progress))
    return start + (end - start) * progress


@torch.no_grad()
def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())
