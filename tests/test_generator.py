"""Generator wiring: shapes, gradient routing and decoding contracts."""

from __future__ import annotations

import pytest
import torch

from cybert.data.readability import COMPLEX, SIMPLE
from cybert.inference.generate import (
    beam_search,
    greedy_decode,
    prepend_cls_embedding,
    soft_decode_free_running,
    soft_decode_teacher_forced,
    soft_to_embeddings,
    strip_after_eos,
)


def test_encode_splits_into_content_and_style(tiny_generator, tiny_batch, tiny_config):
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    batch = tiny_batch["input_ids"].size(0)
    assert encoded.style.shape == (batch, tiny_config.model.style_dim)
    assert encoded.content.shape == (batch, tiny_config.model.content_dim)
    assert encoded.states.shape[0] == batch
    assert encoded.pooled.shape == (batch, tiny_generator.encoder.hidden_size)


def test_teacher_forced_logits_cover_the_vocabulary(tiny_generator, tiny_batch):
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    logits = tiny_generator.decode_teacher_forced(encoded, tiny_batch["decoder_input_ids"])
    assert logits.shape == (
        tiny_batch["decoder_input_ids"].shape[0],
        tiny_batch["decoder_input_ids"].shape[1],
        tiny_generator.vocab_size,
    )


def test_reconstruction_gradient_reaches_encoder_and_decoder(tiny_generator, tiny_batch):
    from cybert.training.losses import reconstruction_loss

    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    logits = tiny_generator.decode_teacher_forced(encoded, tiny_batch["decoder_input_ids"])
    reconstruction_loss(logits, tiny_batch["labels"]).backward()

    assert tiny_generator.splitter.proj.weight.grad is not None
    assert tiny_generator.decoder.output_proj.weight.grad is not None
    assert tiny_generator.encoder.bert.encoder.layer[0].output.dense.weight.grad is not None


def test_style_head_and_adversary_are_separate_modules(tiny_generator):
    assert tiny_generator.style_head is not tiny_generator.content_adv_head
    style_params = {id(p) for p in tiny_generator.style_head.parameters()}
    adv_params = {id(p) for p in tiny_generator.content_adv_head.parameters()}
    assert style_params.isdisjoint(adv_params)


def test_grl_reverses_gradient_into_the_content_projection(tiny_generator, tiny_batch):
    """Eq. 5: the adversary's gradient must arrive at the splitter negated."""
    from cybert.training.losses import style_classification_loss

    labels = torch.tensor([COMPLEX, SIMPLE])

    def splitter_grad(grl_lambda: float) -> torch.Tensor:
        tiny_generator.zero_grad(set_to_none=True)
        encoded = tiny_generator.encode(
            input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
        )
        logits = tiny_generator.content_adv_logits(encoded.content, grl_lambda)
        style_classification_loss(logits, labels).backward()
        return tiny_generator.splitter.proj.weight.grad.clone()

    torch.manual_seed(0)
    positive = splitter_grad(1.0)
    torch.manual_seed(0)
    negative = splitter_grad(-1.0)
    assert torch.allclose(positive, -negative, atol=1e-5)


def test_greedy_decode_shape_and_padding(tiny_generator, tiny_batch, tiny_config):
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    ids = greedy_decode(tiny_generator, encoded, max_length=tiny_config.decode.max_length)
    assert ids.shape[0] == tiny_batch["input_ids"].size(0)
    assert ids.shape[1] <= tiny_config.decode.max_length
    assert ids.dtype == torch.long


def test_greedy_decode_has_no_gradient(tiny_generator, tiny_batch):
    """Eq. 11: the first pass is a stop-gradient operation."""
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    ids = greedy_decode(tiny_generator, encoded, max_length=4)
    assert not ids.requires_grad


def test_beam_search_returns_one_hypothesis_per_input(tiny_generator, tiny_batch):
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    ids = beam_search(tiny_generator, encoded, beam_size=3, max_length=5)
    assert ids.shape[0] == tiny_batch["input_ids"].size(0)
    assert ids.shape[1] >= 1


def test_beam_size_one_matches_greedy(tiny_generator, tiny_batch):
    tiny_generator.eval()
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    greedy = greedy_decode(tiny_generator, encoded, max_length=5)
    beam = beam_search(tiny_generator, encoded, beam_size=1, max_length=5)
    assert torch.equal(greedy, beam)


@pytest.mark.parametrize("gumbel", [True, False])
def test_free_running_soft_decode_is_differentiable(tiny_generator, tiny_batch, gumbel):
    """The generator half of Eqs. 13/15 needs a gradient path through the fake."""
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    probs = soft_decode_free_running(
        tiny_generator, encoded, max_length=4, tau=1.0, gumbel=gumbel
    )
    assert probs.shape == (tiny_batch["input_ids"].size(0), 4, tiny_generator.vocab_size)
    assert probs.requires_grad

    probs.sum().backward()
    assert tiny_generator.decoder.output_proj.weight.grad is not None


def test_soft_probabilities_are_normalised(tiny_generator, tiny_batch):
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    probs = soft_decode_teacher_forced(
        tiny_generator, encoded, tiny_batch["decoder_input_ids"]
    )
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5)


def test_soft_embeddings_reach_the_encoder(tiny_generator, tiny_batch):
    """Soft one-hot -> embeddings -> BERT keeps the graph connected."""
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    probs = soft_decode_free_running(tiny_generator, encoded, max_length=3, tau=1.0)
    embeds = soft_to_embeddings(probs, tiny_generator.encoder.word_embeddings.weight)
    cls_vector = tiny_generator.encoder.word_embeddings.weight[tiny_generator.bos_id]
    embeds, mask = prepend_cls_embedding(embeds, cls_vector)

    assert embeds.shape[1] == 4  # 3 generated steps plus [CLS]
    assert mask.shape == (embeds.shape[0], 4)

    _, pooled = tiny_generator.encoder(inputs_embeds=embeds, attention_mask=mask)
    pooled.sum().backward()
    assert tiny_generator.decoder.output_proj.weight.grad is not None


def test_prepend_cls_respects_a_supplied_mask(tiny_generator):
    embeds = torch.randn(2, 5, 8)
    cls_vector = torch.randn(8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    out, attn = prepend_cls_embedding(embeds, cls_vector, mask)
    assert out.shape == (2, 6, 8)
    assert attn[:, 0].tolist() == [1, 1]
    assert attn[0].tolist() == [1, 1, 1, 0, 0, 0]


def test_strip_after_eos_keeps_eos_and_pads_the_rest():
    ids = torch.tensor([[5, 6, 3, 7, 8], [5, 6, 7, 8, 9]])
    out = strip_after_eos(ids, eos_id=3, pad_id=0)
    assert out[0].tolist() == [5, 6, 3, 0, 0]
    assert out[1].tolist() == [5, 6, 7, 8, 9]


def test_target_style_uses_the_bank_once_populated(tiny_generator, tiny_batch):
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    # Empty bank -> fall back to the sentence's own style.
    fallback = tiny_generator.target_style(SIMPLE, 2, encoded.style)
    assert torch.equal(fallback, encoded.style)

    tiny_generator.update_style_bank(encoded.style, torch.tensor([COMPLEX, SIMPLE]))
    target = tiny_generator.target_style(SIMPLE, 2, encoded.style)
    assert torch.allclose(target[0], target[1])  # one shared target per domain
    assert torch.allclose(target[0], tiny_generator.style_bank.bank[SIMPLE])


def test_direction_changes_the_decoder_state(tiny_generator, tiny_batch):
    """G and F differ only in the fused style, so the init state must differ."""
    encoded = tiny_generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    tiny_generator.update_style_bank(
        torch.tensor([[-5.0, -5.0], [5.0, 5.0]]), torch.tensor([COMPLEX, SIMPLE])
    )
    to_simple = tiny_generator.target_style(SIMPLE, 2, encoded.style)
    to_complex = tiny_generator.target_style(COMPLEX, 2, encoded.style)
    state_simple, _ = tiny_generator.prepare_decoding(encoded, to_simple)
    state_complex, _ = tiny_generator.prepare_decoding(encoded, to_complex)
    assert not torch.allclose(state_simple[0], state_complex[0])


def test_no_attention_variant_builds_and_decodes(tiny_config, tiny_tokenizer, tiny_batch):
    """Table 9's BERT+LSTM ablation must run without a memory tensor."""
    import copy

    from cybert.models.generator import CyBERTGenerator

    cfg = copy.deepcopy(tiny_config)
    cfg.model.use_attention = False
    generator = CyBERTGenerator(cfg, tiny_tokenizer)
    encoded = generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    logits = generator.decode_teacher_forced(encoded, tiny_batch["decoder_input_ids"])
    assert logits.shape[-1] == generator.vocab_size
    assert greedy_decode(generator, encoded, max_length=4).shape[0] == 2


def test_concat_fusion_variant_runs(tiny_config, tiny_tokenizer, tiny_batch):
    import copy

    from cybert.models.generator import CyBERTGenerator

    cfg = copy.deepcopy(tiny_config)
    cfg.model.fusion = "concat"
    generator = CyBERTGenerator(cfg, tiny_tokenizer)
    encoded = generator.encode(
        input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"]
    )
    logits = generator.decode_teacher_forced(encoded, tiny_batch["decoder_input_ids"])
    assert logits.shape[0] == 2
