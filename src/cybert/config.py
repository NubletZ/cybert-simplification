"""Configuration dataclasses and YAML loading for CyBERT.

Every value that the paper states explicitly carries a comment naming the
section, equation or table it comes from. Values the paper leaves unspecified
are marked ``UNSPECIFIED`` and are documented in ``README.md``.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Encoder / latent / decoder geometry (paper 4.3)."""

    bert_name: str = "bert-base-uncased"  # 4.3: 12L, hidden 768, 12 heads
    max_length: int = 64  # 4.3

    # Latent interface. The paper fixes the *ratio* (4.3.1, Appendix B) but not
    # the total width; 768 keeps the latent in BERT's own hidden space so the
    # content projection P_c is a square map. UNSPECIFIED: latent_dim.
    latent_dim: int = 768
    style_ratio: float = 0.05  # 4.3.1 / Table 7: 5% is the selected setting

    # Decoder (4.3): 2-layer unidirectional LSTM, hidden 512, dropout 0.3,
    # dot-product attention over the BERT token states.
    decoder_hidden: int = 512
    decoder_layers: int = 2
    dropout: float = 0.3
    use_attention: bool = True  # False reproduces the Table 9 ablation rows
    tie_input_embeddings: bool = True  # decoder input embeddings <- BERT wordpiece

    # Latent fusion before decoding (3.2.3): "concat" -> linear, or AdaIN (Eq. 10).
    fusion: str = "adain"  # {"adain", "concat"}

    # Auxiliary classification heads (Eq. 4/5).
    classifier_hidden: int = 256
    num_styles: int = 2  # {complex, simple}

    @property
    def style_dim(self) -> int:
        """Width of the style sub-vector s."""
        return max(1, int(round(self.latent_dim * self.style_ratio)))

    @property
    def content_dim(self) -> int:
        """Width of the content sub-vector c."""
        return self.latent_dim - self.style_dim


@dataclass
class NoiseConfig:
    """POS-guided span-infilling corruption (paper 3.2.1)."""

    enabled: bool = True  # False -> the "BERT w/o noise" rows of Table 8
    use_pos: bool = True  # False -> the "BERT w/ IN" rows of Table 8
    poisson_lambda: float = 3.0  # 3.2.1: L ~ Poisson(lambda = 3)
    # UNSPECIFIED: total corruption budget. 0.30 follows BART, which 3.2.1
    # names as the inspiration for the corruption function.
    mask_ratio: float = 0.30
    max_spans: int = 8  # safety bound on the corruption loop
    # Eq. 1: K_POS
    pos_categories: tuple[str, ...] = ("VERB", "NOUN", "AUX", "ADJ", "ADV")
    spacy_model: str = "en_core_web_sm"
    cache_dir: str = ".cache/pos"


@dataclass
class DataConfig:
    train_complex: str = "data/toy/train.complex.txt"
    train_simple: str = "data/toy/train.simple.txt"
    valid_complex: str = "data/toy/valid.complex.txt"
    valid_simple: str = "data/toy/valid.simple.txt"
    # Paired supervision (4.2). TSV: complex<TAB>simple[<TAB>simple2...].
    train_paired: str | None = "data/toy/train.paired.tsv"
    valid_paired: str | None = "data/toy/valid.paired.tsv"
    test_paired: str | None = "data/toy/valid.paired.tsv"

    # 4.1: Flesch Reading Ease thresholds used to build the unpaired domains.
    fre_complex_max: float = 10.0
    fre_simple_min: float = 70.0

    num_workers: int = 2
    seed: int = 42


@dataclass
class OptimConfig:
    """4.3: AdamW, separate learning rates, weight decay 0.01, clip 1.0."""

    bert_lr: float = 2e-5
    head_lr: float = 1e-3  # LSTM decoder + projection layers
    disc_lr: float = 1e-4  # UNSPECIFIED: discriminator head lr
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 8  # 4.3
    grad_accum: int = 1
    warmup_ratio: float = 0.06  # UNSPECIFIED


@dataclass
class Phase1Config:
    """Eq. 8 objective weights and schedule.

    All four lambdas are UNSPECIFIED in the paper ("fixed after preliminary
    validation trials"); these defaults keep every term at unit scale.
    """

    epochs: int = 20  # 4.3: up to 20 epochs

    # Checkpoint selection metric. 4.3 states validation SARI for *both*
    # phases, but in Phase 1 that selects against the phase's own objective:
    # the training goal is denoising reconstruction, and a model that
    # reconstructs well copies its input, which SARI scores poorly. Observed
    # directly -- l_gen 4.14 -> 1.84 across epochs 0-2 while validation SARI
    # fell 27.74 -> 25.89, so SARI-based selection returns the epoch-0
    # decoder and hands Phase 2 an untrained language model.
    # "recon" selects on validation reconstruction loss; "sari" restores the
    # literal reading of 4.3. Phase 2 always selects on SARI, where transfer
    # quality genuinely is the objective.
    select_metric: str = "recon"  # {"recon", "sari"}

    lambda_gen: float = 1.0
    lambda_cls_style: float = 1.0
    lambda_cls_content: float = 1.0  # weight on L_GRL(c)
    lambda_emb: float = 1.0
    # DANN-style ramp on the gradient-reversal coefficient, over the first
    # `grl_warmup_ratio` of training. UNSPECIFIED; prevents early collapse.
    grl_max: float = 1.0
    grl_warmup_ratio: float = 0.1
    style_bank_momentum: float = 0.99  # EMA over encoder style vectors


@dataclass
class Phase2Config:
    """Eq. 16 objective weights and the cycle/adversarial schedule."""

    epochs: int = 30  # 4.3: up to 30 epochs
    lambda_cycle: float = 10.0  # UNSPECIFIED; CycleGAN convention
    lambda_adv_sim: float = 1.0  # UNSPECIFIED
    lambda_adv_com: float = 1.0  # UNSPECIFIED
    lambda_style: float = 1.0  # UNSPECIFIED
    lambda_content: float = 1.0  # UNSPECIFIED
    lambda_emb: float = 1.0  # UNSPECIFIED
    lambda_paired: float = 1.0  # UNSPECIFIED; 0 -> unpaired-only (Table 2)
    paired_every: int = 1  # interleave paired batches every N steps (4.2)

    # Eq. 9/11. The first pass is always stop-gradient greedy decoding; the
    # second pass carries the gradient and must reach phi differentiably.
    cycle_mode: str = "teacher_forced"  # {"teacher_forced", "gumbel"}
    cycle_max_len: int = 64
    # Divide the Eq. 9 norm by the feature width so lambda_cycle is on the same
    # scale as the other terms of Eq. 16. See losses.cycle_consistency_loss.
    cycle_normalize: bool = True
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    soft_temperature: float = 1.0  # softmax temperature for soft embeddings
    grl_max: float = 1.0
    style_bank_momentum: float = 0.99
    d_steps: int = 1  # discriminator updates per generator update


@dataclass
class DecodeConfig:
    """Appendix G.2: beam size k = 4, temperature tau = 1.0."""

    beam_size: int = 4
    temperature: float = 1.0
    max_length: int = 64
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3  # UNSPECIFIED; guards LSTM repetition loops


@dataclass
class RunConfig:
    output_dir: str = "runs/cybert"
    device: str = "auto"  # {"auto", "cuda", "cpu"}
    fp16: bool = False
    log_every: int = 50
    eval_every_epoch: bool = True
    patience: int = 5  # 4.3: early stopping patience of 5 epochs
    max_eval_samples: int = 500
    save_best_only: bool = True
    seed: int = 42


@dataclass
class CyBERTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    run: RunConfig = field(default_factory=RunConfig)


# --------------------------------------------------------------------------- #
# (De)serialisation
# --------------------------------------------------------------------------- #


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a plain dict, rejecting typos."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    # ``from __future__ import annotations`` turns field.type into a string, so
    # resolve the real annotations before testing for nested dataclasses.
    hints = get_type_hints(cls)
    for name in known:
        if name not in data:
            continue
        value = data[name]
        hint = hints.get(name)
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _from_dict(hint, value)
        elif isinstance(value, list) and name == "pos_categories":
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> CyBERTConfig:
    """Rebuild a config from the dict stored inside a checkpoint."""
    return _from_dict(CyBERTConfig, data)


def to_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a config dataclass to plain Python types."""
    return dataclasses.asdict(cfg)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(current: Any, text: str) -> Any:
    if isinstance(current, bool):
        return text.lower() in {"1", "true", "yes", "y"}
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if current is None:
        return None if text.lower() in {"none", "null"} else text
    return text


def apply_overrides(cfg: CyBERTConfig, overrides: list[str]) -> CyBERTConfig:
    """Apply ``section.key=value`` strings from the command line."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, text = item.split("=", 1)
        parts = dotted.split(".")
        node: Any = cfg
        for part in parts[:-1]:
            node = getattr(node, part)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise ValueError(f"unknown config path: {dotted}")
        setattr(node, leaf, _coerce(getattr(node, leaf), text))
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> CyBERTConfig:
    """Load a YAML config, merging any ``_base_`` parent it names."""
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        base_name = data.pop("_base_", None)
        if base_name:
            base_path = (path.parent / base_name).resolve()
            with base_path.open() as fh:
                base_data = yaml.safe_load(fh) or {}
            base_data.pop("_base_", None)
            data = _deep_merge(base_data, data)
    cfg = _from_dict(CyBERTConfig, data)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    _validate(cfg)
    return cfg


def _validate(cfg: CyBERTConfig) -> None:
    if not 0.0 < cfg.model.style_ratio < 1.0:
        raise ValueError("model.style_ratio must lie in (0, 1)")
    if cfg.model.fusion not in {"adain", "concat"}:
        raise ValueError("model.fusion must be 'adain' or 'concat'")
    if cfg.phase2.cycle_mode not in {"teacher_forced", "gumbel"}:
        raise ValueError("phase2.cycle_mode must be 'teacher_forced' or 'gumbel'")
    if cfg.phase1.select_metric not in {"recon", "sari"}:
        raise ValueError("phase1.select_metric must be 'recon' or 'sari'")
    if cfg.model.style_dim < 1 or cfg.model.content_dim < 1:
        raise ValueError("latent_dim too small for the requested style_ratio")
    if cfg.data.fre_complex_max >= cfg.data.fre_simple_min:
        raise ValueError("fre_complex_max must be below fre_simple_min")


def save_config(cfg: CyBERTConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
