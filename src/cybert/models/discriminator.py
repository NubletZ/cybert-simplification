"""Domain discriminators (paper 3.2.3, Eqs. 12-15).

``D_sim`` separates real simple sentences from ``G(x)``; ``D_com`` separates
real complex sentences from ``F(y)``. Both are binary classifiers over the
pooled ``[CLS]`` representation produced by *the generator's own* BERT encoder.

Only the linear head is trainable. 4.3 is explicit that during discriminator
updates the encoder is detached, so discriminator gradients never modify the
generator; during generator updates the head is frozen and the gradient flows
back through the encoder into the generator that produced the fake.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

REAL = 1
FAKE = 0


class DomainDiscriminator(nn.Module):
    """Trainable linear head over pooled BERT features."""

    def __init__(self, feature_dim: int = 768, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 2))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Logits over ``{fake, real}``."""
        return self.head(pooled)

    def loss_real_fake(self, real_pooled: torch.Tensor, fake_pooled: torch.Tensor) -> torch.Tensor:
        """Discriminator objective, Eqs. 12 and 14.

        Both terms are cross-entropies against the softmax over the real/fake
        classes, which is the binary classifier the paper describes.
        """
        real_logits = self(real_pooled)
        fake_logits = self(fake_pooled)
        real_target = torch.full((real_logits.size(0),), REAL, dtype=torch.long, device=real_logits.device)
        fake_target = torch.full((fake_logits.size(0),), FAKE, dtype=torch.long, device=fake_logits.device)
        return F.cross_entropy(real_logits, real_target) + F.cross_entropy(fake_logits, fake_target)

    def loss_generator(self, fake_pooled: torch.Tensor) -> torch.Tensor:
        """Generator objective, Eqs. 13 and 15: make the fake read as real."""
        logits = self(fake_pooled)
        target = torch.full((logits.size(0),), REAL, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    @torch.no_grad()
    def accuracy(self, real_pooled: torch.Tensor, fake_pooled: torch.Tensor) -> float:
        """Diagnostic: how well the head currently separates the two sets."""
        real_ok = (self(real_pooled).argmax(-1) == REAL).float().sum()
        fake_ok = (self(fake_pooled).argmax(-1) == FAKE).float().sum()
        total = real_pooled.size(0) + fake_pooled.size(0)
        return float((real_ok + fake_ok) / max(1, total))
