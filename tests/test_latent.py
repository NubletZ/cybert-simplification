"""Latent interface: the 5% split, gradient reversal, AdaIN (Eqs. 5, 10)."""

from __future__ import annotations

import pytest
import torch

from cybert.config import CyBERTConfig, load_config
from cybert.models.latent import (
    AdaINFusion,
    ConcatFusion,
    ContentStyleSplitter,
    GradientReversal,
    build_fusion,
    grad_reverse,
)
from cybert.models.style_bank import StyleBank


def test_paper_style_ratio_gives_38_of_768():
    """4.3.1 fixes 5%; with a 768-wide latent that is 38 style dimensions."""
    cfg = CyBERTConfig()
    assert cfg.model.latent_dim == 768
    assert cfg.model.style_ratio == 0.05
    assert cfg.model.style_dim == 38
    assert cfg.model.content_dim == 730
    assert cfg.model.style_dim + cfg.model.content_dim == cfg.model.latent_dim


@pytest.mark.parametrize(
    "ratio,expected", [(0.01, 8), (0.05, 38), (0.10, 77), (0.15, 115), (0.20, 154)]
)
def test_sweep_ratios_resolve(ratio, expected):
    """The Table 7 sweep points must all yield a usable split."""
    cfg = CyBERTConfig()
    cfg.model.style_ratio = ratio
    assert cfg.model.style_dim == expected
    assert cfg.model.content_dim == 768 - expected


def test_splitter_shapes_and_order():
    splitter = ContentStyleSplitter(input_dim=64, latent_dim=40, style_dim=2)
    content, style = splitter(torch.randn(5, 64))
    assert style.shape == (5, 2)
    assert content.shape == (5, 38)


def test_splitter_rejects_degenerate_widths():
    with pytest.raises(ValueError):
        ContentStyleSplitter(input_dim=16, latent_dim=8, style_dim=8)


def test_gradient_reversal_flips_the_gradient():
    """Eq. 5: identity forwards, negated gradient backwards."""
    x = torch.randn(4, 6, requires_grad=True)
    plain = x.sum()
    plain.backward()
    reference = x.grad.clone()

    x.grad = None
    reversed_out = grad_reverse(x, 1.0)
    assert torch.allclose(reversed_out, x)
    reversed_out.sum().backward()
    assert torch.allclose(x.grad, -reference)


def test_gradient_reversal_scales_by_lambda():
    x = torch.randn(3, 5, requires_grad=True)
    GradientReversal(0.25)(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.25))


def test_adain_transfers_style_moments():
    """Eq. 10: the output carries the style vector's mean and std."""
    fusion = AdaINFusion(content_dim=64, style_dim=8)
    content = torch.randn(16, 64) * 3.0 + 5.0
    style = torch.randn(16, 8) * 0.5 - 2.0
    out = fusion(content, style)

    assert out.shape == content.shape
    assert torch.allclose(out.mean(-1), style.mean(-1), atol=1e-4)
    assert torch.allclose(
        out.std(-1, unbiased=False), style.std(-1, unbiased=False), atol=1e-3
    )


def test_adain_preserves_content_structure():
    """Normalising and re-scaling must not reorder the content coordinates."""
    fusion = AdaINFusion(content_dim=32, style_dim=4)
    content = torch.randn(1, 32)
    out = fusion(content, torch.randn(1, 4))
    assert torch.equal(content.argsort(dim=-1), out.argsort(dim=-1))


def test_concat_fusion_width():
    fusion = ConcatFusion(content_dim=730, style_dim=38)
    out = fusion(torch.randn(2, 730), torch.randn(2, 38))
    assert out.shape == (2, 768)
    assert fusion.output_dim == 768


def test_build_fusion_dispatch():
    assert isinstance(build_fusion("adain", 10, 2), AdaINFusion)
    assert isinstance(build_fusion("concat", 10, 2), ConcatFusion)
    with pytest.raises(ValueError):
        build_fusion("bilinear", 10, 2)


def test_style_bank_initialises_then_averages():
    """The first batch seeds a slot outright; later batches move it by EMA."""
    bank = StyleBank(num_styles=2, style_dim=4, momentum=0.9)
    labels = torch.tensor([0, 0, 1, 1])
    first = torch.tensor(
        [[1.0, 1, 1, 1], [1.0, 1, 1, 1], [3.0, 3, 3, 3], [3.0, 3, 3, 3]]
    )
    bank.update(first, labels)
    assert torch.allclose(bank.get(0, 1)[0], torch.ones(4))
    assert torch.allclose(bank.get(1, 1)[0], torch.full((4,), 3.0))
    assert bank.initialised

    bank.update(torch.zeros(4, 4), labels)
    assert torch.allclose(bank.get(0, 1)[0], torch.full((4,), 0.9))


def test_style_bank_broadcasts_over_a_batch():
    bank = StyleBank(2, 3)
    bank.update(torch.ones(2, 3), torch.tensor([0, 0]))
    assert bank.get(0, 5).shape == (5, 3)


def test_config_validation_rejects_bad_settings(tmp_path):
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": {"fusion": "wavelet"}}))
    with pytest.raises(ValueError):
        load_config(path)

    path.write_text(yaml.safe_dump({"model": {"style_ratio": 1.5}}))
    with pytest.raises(ValueError):
        load_config(path)

    path.write_text(yaml.safe_dump({"model": {"nonexistent_key": 1}}))
    with pytest.raises(ValueError):
        load_config(path)


def test_config_overrides_are_typed():
    from cybert.config import apply_overrides

    cfg = CyBERTConfig()
    apply_overrides(cfg, ["model.style_ratio=0.10", "noise.use_pos=false", "optim.batch_size=32"])
    assert cfg.model.style_ratio == 0.10
    assert cfg.noise.use_pos is False
    assert cfg.optim.batch_size == 32
