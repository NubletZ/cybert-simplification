"""Attention-based LSTM decoder (paper 4.3, Appendix D.1).

Two-layer unidirectional LSTM, hidden 512, dropout 0.3, with dot-product
attention over the BERT token states. The decoder is initialised from the fused
content-style vector, so style control passes through the latent interface
rather than through a separate conditioning path.

Encoder states are 768-wide and the decoder is 512-wide, so the memory is
projected once to the decoder width; the attention itself is then a plain dot
product, as the paper specifies. ``use_attention=False`` reproduces the
BERT+LSTM row of Table 9, where the decoder sees only the fused vector.

The step interface accepts either token ids or pre-computed embeddings. Soft
embeddings are what let free-running Gumbel decoding stay differentiable in the
Phase 2 cycle (Eq. 9/11).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DecoderState = tuple[torch.Tensor, torch.Tensor]


class DotProductAttention(nn.Module):
    """Luong dot-product attention over the projected encoder states."""

    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def project_memory(self, encoder_states: torch.Tensor) -> torch.Tensor:
        """Project once per sentence, then reuse across decoding steps."""
        return self.memory_proj(encoder_states)

    def forward(
        self,
        query: torch.Tensor,  # (B, D)
        memory: torch.Tensor,  # (B, T, D) already projected
        memory_mask: torch.Tensor | None = None,  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(memory, query.unsqueeze(2)).squeeze(2)  # (B, T)
        if memory_mask is not None:
            scores = scores.masked_fill(memory_mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)  # (B, D)
        return context, weights


class AttnLSTMDecoder(nn.Module):
    """Autoregressive decoder over the BERT WordPiece vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        encoder_dim: int = 768,
        fused_dim: int = 730,
        dropout: float = 0.3,
        use_attention: bool = True,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

        # The fused content-style vector initialises every layer's (h, c).
        self.init_proj = nn.Linear(fused_dim, 2 * num_layers * hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if use_attention:
            self.attention = DotProductAttention(encoder_dim, hidden_size)
            self.combine = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.attention = None
            self.combine = None

        self.output_proj = nn.Linear(hidden_size, vocab_size)

    # -- state ------------------------------------------------------------- #

    def init_state(self, fused: torch.Tensor) -> DecoderState:
        """Build ``(h0, c0)`` from the fused content-style vector."""
        batch = fused.size(0)
        both = torch.tanh(self.init_proj(fused))
        both = both.view(batch, 2, self.num_layers, self.hidden_size)
        h0 = both[:, 0].permute(1, 0, 2).contiguous()
        c0 = both[:, 1].permute(1, 0, 2).contiguous()
        return h0, c0

    def project_memory(self, encoder_states: torch.Tensor) -> torch.Tensor | None:
        if not self.use_attention:
            return None
        return self.attention.project_memory(encoder_states)

    def tie_embeddings(self, weight: torch.Tensor) -> None:
        """Initialise the input embedding table from BERT's WordPiece table.

        The weights are copied rather than shared: the decoder embedding is
        trained at the head learning rate (1e-3) while BERT moves at 2e-5, and
        sharing the tensor would drag the encoder along at the wrong rate.
        """
        if weight.shape != self.embedding.weight.shape:
            raise ValueError(
                f"embedding shape mismatch: {tuple(weight.shape)} vs "
                f"{tuple(self.embedding.weight.shape)}"
            )
        with torch.no_grad():
            self.embedding.weight.copy_(weight)

    # -- stepping ---------------------------------------------------------- #

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def embed_soft(self, probs: torch.Tensor) -> torch.Tensor:
        """Soft-embedding lookup: ``(B, V) -> (B, E)`` or ``(B, T, V) -> (B, T, E)``."""
        return probs @ self.embedding.weight

    def step(
        self,
        embedded: torch.Tensor,  # (B, E), one time step
        state: DecoderState,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecoderState]:
        """Advance one step, returning ``(logits, new_state)``."""
        out, state = self.lstm(self.dropout(embedded).unsqueeze(1), state)
        hidden = out.squeeze(1)  # (B, H)
        if self.use_attention and memory is not None:
            context, _ = self.attention(hidden, memory, memory_mask)
            hidden = torch.tanh(self.combine(torch.cat([hidden, context], dim=-1)))
        return self.output_proj(self.dropout(hidden)), state

    # -- teacher forcing --------------------------------------------------- #

    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # (B, T)
        fused: torch.Tensor,  # (B, F)
        encoder_states: torch.Tensor | None = None,  # (B, S, encoder_dim)
        memory_mask: torch.Tensor | None = None,  # (B, S)
    ) -> torch.Tensor:
        """Teacher-forced pass, returning logits of shape ``(B, T, V)``."""
        state = self.init_state(fused)
        embedded = self.dropout(self.embed(decoder_input_ids))
        outputs, _ = self.lstm(embedded, state)  # (B, T, H)

        if self.use_attention and encoder_states is not None:
            memory = self.project_memory(encoder_states)  # (B, S, H)
            scores = torch.bmm(outputs, memory.transpose(1, 2))  # (B, T, S)
            if memory_mask is not None:
                mask = memory_mask.unsqueeze(1) == 0
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, memory)  # (B, T, H)
            outputs = torch.tanh(self.combine(torch.cat([outputs, context], dim=-1)))

        return self.output_proj(self.dropout(outputs))
