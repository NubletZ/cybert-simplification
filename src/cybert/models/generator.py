"""The CyBERT generator (paper 3.2.2, 3.2.3, Appendix D.1).

One network serves as both directional mapping. 3.2.3 states that Phase 2
"reuse[s] the encoder, latent disentanglement module, and attention-based LSTM
decoder from Phase 1", so ``G`` (complex -> simple) and ``F`` (simple ->
complex) are the same weights conditioned on different target style vectors:

``G(x) = decode(fuse(c(x), s_bar[simple]))``
``F(y) = decode(fuse(c(y), s_bar[complex]))``

Data flow::

    x_cor -> BERT -> H (token states, for attention)
                  -> [CLS] -> splitter -> s (38)  -> style head    (Eq. 4)
                                       -> c (730) -> GRL, adv head (Eq. 5)
                                                  -> P_c           (Eq. 6)
                           fuse(c, s_target) -> (h0, c0) -> attention LSTM
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import CyBERTConfig
from .decoder import AttnLSTMDecoder
from .encoder import BertEncoder
from .heads import ContentProjection, StyleClassifierHead
from .latent import ContentStyleSplitter, build_fusion, grad_reverse
from .style_bank import StyleBank


@dataclass
class Encoded:
    """Everything one encoder pass produces, reused by every loss term."""

    states: torch.Tensor  # (B, T, 768) token states for attention
    pooled: torch.Tensor  # (B, 768) [CLS]
    content: torch.Tensor  # (B, content_dim)
    style: torch.Tensor  # (B, style_dim)
    attention_mask: torch.Tensor | None = None


class CyBERTGenerator(nn.Module):
    """BERT encoder + content-style latent interface + attention-LSTM decoder."""

    def __init__(self, cfg: CyBERTConfig, tokenizer=None) -> None:
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        self.encoder = BertEncoder(mcfg.bert_name)
        hidden = self.encoder.hidden_size
        vocab = self.encoder.vocab_size

        self.splitter = ContentStyleSplitter(
            input_dim=hidden,
            latent_dim=mcfg.latent_dim,
            style_dim=mcfg.style_dim,
            dropout=mcfg.dropout,
        )
        self.fusion = build_fusion(mcfg.fusion, mcfg.content_dim, mcfg.style_dim)
        fused_dim = getattr(self.fusion, "output_dim")

        self.decoder = AttnLSTMDecoder(
            vocab_size=vocab,
            embed_dim=hidden if mcfg.tie_input_embeddings else mcfg.decoder_hidden,
            hidden_size=mcfg.decoder_hidden,
            num_layers=mcfg.decoder_layers,
            encoder_dim=hidden,
            fused_dim=fused_dim,
            dropout=mcfg.dropout,
            use_attention=mcfg.use_attention,
            padding_idx=tokenizer.pad_token_id if tokenizer is not None else 0,
        )
        if mcfg.tie_input_embeddings:
            self.decoder.tie_embeddings(self.encoder.word_embeddings.weight.data)

        self.style_head = StyleClassifierHead(
            mcfg.style_dim, mcfg.classifier_hidden, mcfg.num_styles, mcfg.dropout
        )
        self.content_adv_head = StyleClassifierHead(
            mcfg.content_dim, mcfg.classifier_hidden, mcfg.num_styles, mcfg.dropout
        )
        self.content_proj = ContentProjection(mcfg.content_dim, hidden)
        self.style_bank = StyleBank(
            mcfg.num_styles, mcfg.style_dim, momentum=cfg.phase1.style_bank_momentum
        )

        if tokenizer is not None:
            self.pad_id = tokenizer.pad_token_id
            self.bos_id = tokenizer.cls_token_id
            self.eos_id = tokenizer.sep_token_id
        else:  # pragma: no cover - only for bare unit tests
            self.pad_id, self.bos_id, self.eos_id = 0, 101, 102

    # -- encoding ---------------------------------------------------------- #

    def encode(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> Encoded:
        states, pooled = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
        )
        content, style = self.splitter(pooled)
        return Encoded(states, pooled, content, style, attention_mask)

    # -- latent interface -------------------------------------------------- #

    def fuse(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.fusion(content, style)

    def style_logits(self, style: torch.Tensor) -> torch.Tensor:
        """Eq. 4 applied to ``s``."""
        return self.style_head(style)

    def content_adv_logits(self, content: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        """Eq. 5: the same objective on ``c``, behind a gradient-reversal layer."""
        return self.content_adv_head(grad_reverse(content, grl_lambda))

    def project_content(self, content: torch.Tensor) -> torch.Tensor:
        """``P_c(c)`` from Eq. 6."""
        return self.content_proj(content)

    def target_style(self, label: int, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        """Target-domain style vector, broadcast over the batch.

        Falls back to the encoder's own style vectors before the bank has seen
        both labels, which only happens on the first steps of Phase 1.
        """
        if self.style_bank.counts[label] == 0:
            return reference
        return self.style_bank.get(
            label, batch_size, device=reference.device, dtype=reference.dtype
        )

    @torch.no_grad()
    def update_style_bank(self, style: torch.Tensor, labels: torch.Tensor) -> None:
        self.style_bank.update(style, labels)

    # -- decoding ---------------------------------------------------------- #

    def decode_teacher_forced(
        self,
        encoded: Encoded,
        decoder_input_ids: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced logits ``(B, T, V)``.

        ``style=None`` reuses the sentence's own style vector, which is the
        Phase 1 reconstruction setting; passing a target-domain style is what
        turns the same call into a transfer.
        """
        fused = self.fuse(encoded.content, encoded.style if style is None else style)
        return self.decoder(
            decoder_input_ids=decoder_input_ids,
            fused=fused,
            encoder_states=encoded.states,
            memory_mask=encoded.attention_mask,
        )

    def prepare_decoding(
        self, encoded: Encoded, style: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """Return the initial LSTM state and the projected attention memory."""
        fused = self.fuse(encoded.content, encoded.style if style is None else style)
        state = self.decoder.init_state(fused)
        memory = self.decoder.project_memory(encoded.states)
        return state, memory

    # -- convenience ------------------------------------------------------- #

    @property
    def vocab_size(self) -> int:
        return self.decoder.vocab_size

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
