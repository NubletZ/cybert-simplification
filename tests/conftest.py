"""Shared fixtures.

The model tests run entirely offline against a tiny randomly-initialised BERT
(``model.bert_name: debug-tiny-<vocab>``) and a hand-built WordPiece
tokenizer, so no weights are downloaded and a full forward/backward pass takes
milliseconds. They check shapes, gradient routing and decoding contracts --
not quality.
"""

from __future__ import annotations

import pytest

TINY_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "the",
    "town",
    "made",
    "a",
    "new",
    "rule",
    "council",
    "subsequently",
    "promulgated",
    "an",
    "ordinance",
    "building",
    "fell",
    "down",
    "water",
    "river",
    "people",
    "law",
    "said",
    ".",
    ",",
]


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    from transformers import BertTokenizerFast

    vocab_dir = tmp_path_factory.mktemp("vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(TINY_VOCAB) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)


@pytest.fixture(scope="session")
def tiny_config(tiny_tokenizer):
    from cybert.config import CyBERTConfig

    cfg = CyBERTConfig()
    cfg.model.bert_name = f"debug-tiny-{len(TINY_VOCAB)}"
    cfg.model.latent_dim = 40
    cfg.model.style_ratio = 0.05  # -> 2 style / 38 content
    cfg.model.decoder_hidden = 24
    cfg.model.decoder_layers = 2
    cfg.model.dropout = 0.0
    cfg.model.max_length = 12
    cfg.model.classifier_hidden = 16
    cfg.decode.max_length = 6
    cfg.decode.beam_size = 2
    cfg.optim.batch_size = 2
    return cfg


@pytest.fixture()
def tiny_generator(tiny_config, tiny_tokenizer):
    from cybert.models.generator import CyBERTGenerator

    return CyBERTGenerator(tiny_config, tiny_tokenizer)


@pytest.fixture()
def tiny_batch(tiny_tokenizer, tiny_config):
    from cybert.data.datasets import TextEncoder

    encoder = TextEncoder(tiny_tokenizer, tiny_config.model.max_length)
    texts = ["the town made a new rule .", "the building fell down ."]
    source = encoder.encode_source(texts)
    target = encoder.encode_target(texts)
    return {"texts": texts, **source, **target}
