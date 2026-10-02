"""The content-style latent interface (paper 3.2.2, 3.2.3, Eq. 10).

The sentence representation is projected once and then split: the first
``style_dim`` coordinates are the style vector ``s``, the remainder is the
content vector ``c``. With the paper's 5% ratio and a 768-wide latent that is
38 style dimensions against 730 content dimensions.

``s`` is pushed to be predictive of the complexity label (Eq. 4) while ``c`` is
pushed to be *un*predictive of it through a gradient-reversal layer (Eq. 5).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity forwards, negated-and-scaled gradient backwards (Eq. 5)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # noqa: D102
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module form, with a coefficient the trainer ramps up over warmup."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class ContentStyleSplitter(nn.Module):
    """Project the sentence embedding and split it into ``(content, style)``."""

    def __init__(self, input_dim: int, latent_dim: int, style_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if not 0 < style_dim < latent_dim:
            raise ValueError("style_dim must lie strictly inside the latent width")
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.content_dim = latent_dim - style_dim
        self.proj = nn.Linear(input_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, sentence_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.norm(self.proj(self.dropout(sentence_embedding)))
        style = z[:, : self.style_dim]
        content = z[:, self.style_dim :]
        return content, style


class AdaINFusion(nn.Module):
    """Adaptive instance normalisation, Eq. 10.

    ``AdaIN(z_c, z_s) = sigma(z_s) * (z_c - mu(z_c)) / sigma(z_c) + mu(z_s)``

    The moments are taken per sample across each vector's own coordinates, so
    the 38-wide style vector modulates the 730-wide content vector without any
    padding. The output keeps the content width.
    """

    def __init__(self, content_dim: int, style_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.eps = eps
        self.output_dim = content_dim

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        c_mean = content.mean(dim=-1, keepdim=True)
        c_std = content.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        s_mean = style.mean(dim=-1, keepdim=True)
        s_std = style.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        return s_std * (content - c_mean) / c_std + s_mean


class ConcatFusion(nn.Module):
    """Concatenate then project -- the first fusion variant of 3.2.3."""

    def __init__(self, content_dim: int, style_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        self.output_dim = output_dim or (content_dim + style_dim)
        self.proj = nn.Linear(content_dim + style_dim, self.output_dim)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([content, style], dim=-1))


def build_fusion(kind: str, content_dim: int, style_dim: int) -> nn.Module:
    """Instantiate the fusion module named by ``model.fusion``."""
    if kind == "adain":
        return AdaINFusion(content_dim, style_dim)
    if kind == "concat":
        return ConcatFusion(content_dim, style_dim)
    raise ValueError(f"unknown fusion {kind!r}; expected 'adain' or 'concat'")
