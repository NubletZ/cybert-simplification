"""BERT encoder wrapper (paper 4.3).

Three encoder roles exist in CyBERT and they must not be confused:

``BertEncoder`` (trainable)
    The generator's encoder. Fine-tuned in both phases.

``FrozenSemanticEncoder`` (phi)
    A second, frozen ``bert-base-uncased``. It supplies the ``E_BERT(x)``
    target of Eq. 6 and the fixed feature space of Eq. 9. 3.2.3 is explicit
    that phi is fixed "to provide a stable semantic space and to prevent
    representation drift". Training only -- it is never loaded at inference, so
    it does not count against the 185M parameters / 2.11 GFLOPs of Table 5.

The discriminators reuse the *generator's* encoder with detached features
(3.2.3); see :mod:`cybert.models.discriminator`.

Both wrappers accept ``inputs_embeds``, which is what lets a generated sentence
reach BERT as a differentiable soft one-hot mixture rather than as hard ids.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEBUG_MODEL_PREFIX = "debug-tiny"


def build_bert(model_name: str):
    """Load a pretrained BERT, or build a tiny randomly-initialised one.

    ``debug-tiny`` (optionally ``debug-tiny-<vocab_size>``) constructs a small
    random BERT instead of downloading weights. It exists so the architecture,
    the losses and the decoding loops can be exercised offline in seconds; it
    is never a substitute for ``bert-base-uncased`` in a real run.
    """
    if model_name.startswith(DEBUG_MODEL_PREFIX):
        from transformers import BertConfig, BertModel

        suffix = model_name[len(DEBUG_MODEL_PREFIX) :].lstrip("-")
        vocab_size = int(suffix) if suffix.isdigit() else 128
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=128,
        )
        return BertModel(config)

    from transformers import AutoModel

    return AutoModel.from_pretrained(model_name)


class BertEncoder(nn.Module):
    """Trainable contextual encoder."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.bert = build_bert(model_name)
        self.hidden_size: int = self.bert.config.hidden_size
        self.vocab_size: int = self.bert.config.vocab_size

    @property
    def word_embeddings(self) -> nn.Embedding:
        """The WordPiece embedding table, used to build soft embeddings."""
        return self.bert.get_input_embeddings()

    def embed_soft(self, probs: torch.Tensor) -> torch.Tensor:
        """Map a distribution over the vocabulary to embeddings.

        ``probs`` is ``(B, T, V)``; the result is ``(B, T, H)``. Gradients flow
        through ``probs``, which is how the adversarial and cycle terms reach
        the generator without backpropagating through a discrete argmax.
        """
        return probs @ self.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(token_states, pooled)`` where pooled is the ``[CLS]`` state."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        states = out.last_hidden_state
        return states, states[:, 0]


class FrozenSemanticEncoder(nn.Module):
    """Frozen phi: the semantic target of Eq. 6 and the feature space of Eq. 9."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.bert = build_bert(model_name)
        self.hidden_size: int = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad_(False)
        self.bert.eval()

    def train(self, mode: bool = True):  # noqa: D102 - keep phi in eval mode always
        super().train(False)
        return self

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.bert.get_input_embeddings()

    def embed_soft(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the pooled ``[CLS]`` embedding.

        Parameters are frozen but activations stay in the graph: gradients must
        pass *through* phi to reach the generator that produced
        ``inputs_embeds``, which is exactly what Eq. 9 requires.
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return out.last_hidden_state[:, 0]

    @torch.no_grad()
    def encode_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode hard ids without a graph -- used for the fixed targets."""
        return self.forward(input_ids=input_ids, attention_mask=attention_mask)
