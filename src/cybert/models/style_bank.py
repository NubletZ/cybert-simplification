"""Per-domain style vectors.

At transfer time the decoder must be conditioned on the *target* domain's
style, not the source sentence's own. Following John et al. (2019) -- which
3.2.2 names as the basis for the disentanglement scheme -- the target style is
the average style embedding of the training sentences carrying that label.

The average is maintained as an exponential moving average during training so
it tracks the encoder as the encoder moves, and it is stored in the checkpoint
so inference reproduces training exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StyleBank(nn.Module):
    """EMA of encoder style vectors, one slot per domain label."""

    def __init__(self, num_styles: int, style_dim: int, momentum: float = 0.99) -> None:
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim
        self.momentum = momentum
        self.register_buffer("bank", torch.zeros(num_styles, style_dim))
        self.register_buffer("counts", torch.zeros(num_styles))

    @torch.no_grad()
    def update(self, style: torch.Tensor, labels: torch.Tensor) -> None:
        """Fold a batch of style vectors into the bank.

        The first observation for a label initialises the slot outright; an EMA
        started from zeros would otherwise spend many steps crawling away from
        the origin and hand the decoder a meaningless style during early
        training.
        """
        style = style.detach().float()
        for label in labels.unique():
            idx = int(label.item())
            if idx < 0 or idx >= self.num_styles:
                continue
            mean = style[labels == label].mean(dim=0)
            if self.counts[idx] == 0:
                self.bank[idx] = mean
            else:
                self.bank[idx] = self.momentum * self.bank[idx] + (1 - self.momentum) * mean
            self.counts[idx] += 1

    def get(self, label: int, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Return the target style vector broadcast to ``(batch_size, style_dim)``."""
        vec = self.bank[label]
        if device is not None or dtype is not None:
            vec = vec.to(device=device, dtype=dtype)
        return vec.unsqueeze(0).expand(batch_size, -1).contiguous()

    def get_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Gather one style vector per element of ``labels``."""
        return self.bank[labels]

    @property
    def initialised(self) -> bool:
        return bool((self.counts > 0).all().item())

    def extra_repr(self) -> str:
        return f"num_styles={self.num_styles}, style_dim={self.style_dim}, momentum={self.momentum}"
