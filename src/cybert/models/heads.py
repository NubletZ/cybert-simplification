"""Auxiliary heads: style classification, content projection, style evaluation.

Eq. 4 defines a cross-entropy over a style-label distribution predicted from a
latent vector ``z``. Two instances are used: one on ``s`` (minimised, so ``s``
stays predictive of the complexity label) and one on ``GRL(c)`` (Eq. 5, so the
encoder is pushed to strip style out of ``c``). They are separate modules --
a shared head would let the adversary's own updates leak into the style term.

``ContentProjection`` is ``P_c`` from Eq. 6, mapping ``c`` back into BERT's
sentence-embedding space so the cosine distance to ``E_BERT(x)`` is defined.

``TextCNNStyleClassifier`` is a sentence-level classifier, not part of the
training objective. It is trained separately and used at evaluation time to
measure how often a transferred sentence lands in the target domain, and to
produce the style-ratio accuracy curve of Figure 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleClassifierHead(nn.Module):
    """MLP over a latent vector, producing style-label logits (Eq. 4)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_styles: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_styles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ContentProjection(nn.Module):
    """``P_c`` from Eq. 6: content vector -> BERT sentence-embedding space."""

    def __init__(self, content_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        return self.net(content)


class TextCNNStyleClassifier(nn.Module):
    """Sentence-level style classifier used for evaluation only.

    Standard Kim-style TextCNN over frozen or learned token embeddings. It
    answers "does this generated sentence read as simple?", which the latent
    heads cannot: they see a latent vector, not the produced text.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_styles: int = 2,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.kernel_sizes = kernel_sizes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_styles)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Pad so the widest kernel always fits, otherwise short sentences crash.
        min_len = max(self.kernel_sizes)
        if input_ids.size(1) < min_len:
            pad = min_len - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad), value=self.embedding.padding_idx or 0)
        x = self.embedding(input_ids).transpose(1, 2)  # (B, E, T)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))
