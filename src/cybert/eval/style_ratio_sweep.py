"""Style-ratio sweep (paper Appendix B, Table 7).

Reruns both phases at each style ratio and reports validation SARI and FKGL.
The paper sweeps ``{1, 5, 10, 15, 20}%`` and selects 5%, where SARI peaks on
both ASSET and TurkCorpus while style-classification accuracy is already at its
maximum -- the evidence that simplicity-related variation is compactly
representable.

This is a full retraining per ratio, so it is the most expensive script here.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from ..config import CyBERTConfig
from ..training import phase1, phase2

logger = logging.getLogger(__name__)

DEFAULT_RATIOS = (0.01, 0.05, 0.10, 0.15, 0.20)


def run_sweep(
    cfg: CyBERTConfig,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    output_path: str | Path = "runs/style_ratio_sweep.json",
    run_phase2: bool = True,
) -> dict[str, dict]:
    """Train at each ratio and collect the best validation metrics."""
    results: dict[str, dict] = {}
    base_output = Path(cfg.run.output_dir)

    for ratio in ratios:
        trial = copy.deepcopy(cfg)
        trial.model.style_ratio = ratio
        trial.run.output_dir = str(base_output / f"style_{int(ratio * 100):02d}")
        logger.info(
            "style ratio %.0f%% -> style_dim=%d content_dim=%d",
            ratio * 100,
            trial.model.style_dim,
            trial.model.content_dim,
        )

        p1 = phase1.train(trial)
        entry = {
            "style_dim": trial.model.style_dim,
            "content_dim": trial.model.content_dim,
            "phase1": p1,
        }
        if run_phase2:
            checkpoint = Path(trial.run.output_dir) / "best.pt"
            entry["phase2"] = phase2.train(trial, init_from=str(checkpoint))
        results[f"{ratio:.2f}"] = entry

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            json.dump(results, fh, indent=2)

    _log_table(results)
    return results


def _log_table(results: dict[str, dict]) -> None:
    logger.info("%-12s %-10s %-10s %-10s", "style ratio", "style dim", "SARI", "FKGL")
    for ratio, entry in sorted(results.items()):
        best = entry.get("phase2") or entry.get("phase1") or {}
        logger.info(
            "%-12s %-10d %-10.2f %-10.2f",
            f"{float(ratio) * 100:.0f}%",
            entry["style_dim"],
            best.get("sari", float("nan")),
            best.get("fkgl", float("nan")),
        )
