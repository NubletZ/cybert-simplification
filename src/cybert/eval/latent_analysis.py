"""Latent-space diagnostics (paper 5.1, Figures 3-5).

Two things are checked here, and both are diagnostic evidence rather than
proof of disentanglement -- the paper is explicit about that:

* PCA of ``c`` and ``s`` coloured by domain. The expected pattern is clear
  separation in style space and heavy overlap in content space (Figure 3), and
  the same pattern on aligned pairs (Figure 5).
* Style-classification accuracy from ``s`` versus from ``c``. A working split
  gives high accuracy on ``s`` and near-chance accuracy on ``c``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch

from ..config import CyBERTConfig
from ..data.datasets import PairedDataset, TextEncoder, read_lines
from ..data.readability import COMPLEX, SIMPLE
from ..models.generator import CyBERTGenerator

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_latents(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    sentences: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(content, style)`` matrices for a list of sentences."""
    generator.eval()
    contents, styles = [], []
    for start in range(0, len(sentences), batch_size):
        chunk = list(sentences[start : start + batch_size])
        batch = text_encoder.encode_source(chunk)
        encoded = generator.encode(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        contents.append(encoded.content.cpu())
        styles.append(encoded.style.cpu())
    return torch.cat(contents), torch.cat(styles)


def _pca(matrix: torch.Tensor, components: int = 2):
    from sklearn.decomposition import PCA

    return PCA(n_components=components).fit_transform(matrix.numpy())


def _scatter(ax, points, labels, title: str) -> None:
    for value, name, colour in ((COMPLEX, "complex", "#B4472C"), (SIMPLE, "simple", "#2C6FB4")):
        mask = [i for i, label in enumerate(labels) if label == value]
        if not mask:
            continue
        ax.scatter(
            points[mask, 0], points[mask, 1], s=12, alpha=0.6, label=name, color=colour
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)


def plot_latent_spaces(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    cfg: CyBERTConfig,
    device: torch.device,
    output_path: str | Path,
    max_samples: int = 400,
) -> dict[str, float]:
    """Figure 3: PCA of content and style, coloured by readability domain."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complex_sents = read_lines(cfg.data.valid_complex)[:max_samples]
    simple_sents = read_lines(cfg.data.valid_simple)[:max_samples]
    sentences = complex_sents + simple_sents
    labels = [COMPLEX] * len(complex_sents) + [SIMPLE] * len(simple_sents)

    content, style = collect_latents(generator, text_encoder, sentences, device)
    content_pca = _pca(content)
    style_pca = _pca(style)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _scatter(axes[0], content_pca, labels, "Content space")
    _scatter(axes[1], style_pca, labels, "Style space")
    fig.suptitle("CyBERT latent spaces (PCA)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    separation = latent_separability(content, style, labels)
    logger.info("wrote %s | %s", output_path, separation)
    return separation


def plot_paired_latents(
    generator: CyBERTGenerator,
    text_encoder: TextEncoder,
    cfg: CyBERTConfig,
    device: torch.device,
    output_path: str | Path,
    max_samples: int = 200,
) -> None:
    """Figure 5: the same view restricted to aligned complex-simple pairs."""
    if not cfg.data.valid_paired:
        logger.warning("no paired validation file configured; skipping Figure 5")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = PairedDataset(cfg.data.valid_paired, max_items=max_samples)
    sources = [row["source"] for row in dataset]
    targets = [row["target"] for row in dataset]
    sentences = sources + targets
    labels = [COMPLEX] * len(sources) + [SIMPLE] * len(targets)

    content, style = collect_latents(generator, text_encoder, sentences, device)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _scatter(axes[0], _pca(content), labels, "Content space (paired)")
    _scatter(axes[1], _pca(style), labels, "Style space (paired)")
    fig.suptitle("CyBERT latent spaces on aligned pairs (PCA)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", output_path)


def latent_separability(
    content: torch.Tensor, style: torch.Tensor, labels: Sequence[int]
) -> dict[str, float]:
    """Logistic-regression accuracy from each latent, as in Figure 4.

    High on style and near 0.5 on content is the intended outcome. Reported
    with a train/test split so the number reflects separability rather than
    the classifier's capacity to memorise.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    y = list(labels)
    results: dict[str, float] = {}
    for name, matrix in (("style", style), ("content", content)):
        x = matrix.numpy()
        if len(set(y)) < 2 or len(y) < 8:
            results[f"acc_{name}"] = float("nan")
            continue
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0, stratify=y
        )
        clf = LogisticRegression(max_iter=2000).fit(x_train, y_train)
        results[f"acc_{name}"] = float(clf.score(x_test, y_test))
    return results
