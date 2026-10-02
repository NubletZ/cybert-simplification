"""Objective terms (Eqs. 4, 6, 7, 9) and their schedules."""

from __future__ import annotations

import pytest
import torch

from cybert.models.discriminator import FAKE, REAL, DomainDiscriminator
from cybert.training.losses import (
    LossRecord,
    classification_accuracy,
    content_embedding_loss,
    cycle_consistency_loss,
    grl_lambda_schedule,
    gumbel_tau_schedule,
    reconstruction_loss,
    style_classification_loss,
)


def test_reconstruction_ignores_padding():
    """Eq. 7: positions labelled -100 must not contribute."""
    logits = torch.randn(2, 5, 11)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    labels[0, 0] = 3
    loss_masked = reconstruction_loss(logits, labels)

    only = reconstruction_loss(logits[:1, :1], labels[:1, :1])
    assert loss_masked == pytest.approx(float(only), abs=1e-5)


def test_reconstruction_is_minimised_by_confident_correct_logits():
    labels = torch.tensor([[1, 2]])
    confident = torch.zeros(1, 2, 4)
    confident[0, 0, 1] = 20.0
    confident[0, 1, 2] = 20.0
    assert reconstruction_loss(confident, labels) < 1e-4
    assert reconstruction_loss(torch.zeros(1, 2, 4), labels) > 1.0


def test_content_embedding_loss_is_cosine_distance():
    """Eq. 6: zero when aligned, two when opposed, scale invariant."""
    vector = torch.randn(4, 16)
    assert content_embedding_loss(vector, vector) == pytest.approx(0.0, abs=1e-6)
    assert content_embedding_loss(vector, -vector) == pytest.approx(2.0, abs=1e-6)
    assert content_embedding_loss(vector * 7.0, vector) == pytest.approx(0.0, abs=1e-6)


def test_content_embedding_target_is_detached():
    """phi is a fixed anchor; no gradient may reach the target (3.2.3)."""
    projected = torch.randn(2, 8, requires_grad=True)
    target = torch.randn(2, 8, requires_grad=True)
    content_embedding_loss(projected, target).backward()
    assert projected.grad is not None
    assert target.grad is None


def test_cycle_loss_is_l1_in_feature_space():
    """Eq. 9 with p = 1."""
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0, 0.0, 3.0]])
    assert cycle_consistency_loss(a, b, p=1, normalize=False) == pytest.approx(2.0)
    assert cycle_consistency_loss(a, a, p=1, normalize=False) == pytest.approx(0.0)


def test_cycle_loss_normalisation_is_a_constant_rescaling():
    """Dividing by the feature width leaves the objective proportional."""
    a = torch.randn(4, 768)
    b = torch.randn(4, 768)
    raw = float(cycle_consistency_loss(a, b, normalize=False))
    scaled = float(cycle_consistency_loss(a, b, normalize=True))
    assert scaled == pytest.approx(raw / 768, rel=1e-5)
    # The raw L1 over BERT's width is orders of magnitude above the other
    # terms of Eq. 16, which is what the rescaling exists to correct.
    assert raw > 100 * scaled


def test_cycle_loss_gradient_flows_to_the_round_trip_only():
    round_trip = torch.randn(3, 5, requires_grad=True)
    original = torch.randn(3, 5, requires_grad=True)
    cycle_consistency_loss(round_trip, original).backward()
    assert round_trip.grad is not None
    assert original.grad is None


def test_style_classification_and_accuracy():
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    labels = torch.tensor([0, 1])
    assert style_classification_loss(logits, labels) < 1e-3
    assert classification_accuracy(logits, labels) == pytest.approx(1.0)
    assert classification_accuracy(logits, torch.tensor([1, 0])) == pytest.approx(0.0)


def test_grl_schedule_ramps_then_saturates():
    assert grl_lambda_schedule(0.0) == pytest.approx(0.0)
    assert grl_lambda_schedule(0.1, warmup_ratio=0.1) == pytest.approx(1.0, abs=1e-3)
    assert grl_lambda_schedule(0.9, warmup_ratio=0.1) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 < grl_lambda_schedule(0.03, warmup_ratio=0.1) < 1.0
    assert grl_lambda_schedule(0.0, warmup_ratio=0.0) == pytest.approx(1.0)


def test_gumbel_tau_anneals_downwards():
    assert gumbel_tau_schedule(0.0, 2.0, 0.5) == pytest.approx(2.0)
    assert gumbel_tau_schedule(1.0, 2.0, 0.5) == pytest.approx(0.5)
    assert gumbel_tau_schedule(0.5, 2.0, 0.5) == pytest.approx(1.25)


def test_discriminator_losses_push_in_opposite_directions():
    """Eqs. 12-15: D separates the sets, G tries to make the fake read real."""
    disc = DomainDiscriminator(feature_dim=4, hidden_dim=None)
    with torch.no_grad():
        disc.head[1].weight.zero_()
        disc.head[1].bias.copy_(torch.tensor([0.0, 0.0]))

    real = torch.randn(6, 4)
    fake = torch.randn(6, 4)
    d_loss = disc.loss_real_fake(real, fake)
    g_loss = disc.loss_generator(fake)
    # At chance, D pays ln2 per term and G pays ln2 once.
    assert float(d_loss.detach()) == pytest.approx(2 * 0.6931, abs=1e-3)
    assert float(g_loss.detach()) == pytest.approx(0.6931, abs=1e-3)
    assert disc.accuracy(real, fake) == pytest.approx(0.5, abs=0.5)


def test_discriminator_label_constants():
    assert (REAL, FAKE) == (1, 0)


def test_loss_record_averages_and_resets():
    record = LossRecord()
    record.add("l_gen", torch.tensor(2.0))
    record.add("l_gen", 4.0)
    assert record.mean()["l_gen"] == pytest.approx(3.0)
    assert "l_gen=3.0000" in record.format()
    record.reset()
    assert record.mean() == {}
