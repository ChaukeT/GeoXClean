"""
ARBF Target Preparer — validates active block counts and prepares targets.

Enforces thresholds:
  - > 250K active blocks: warning
  - > 500K active blocks: strong warning, suggest preview mode
  - > 750K active blocks: refuse final mode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .arbf_modes import EstimationMode, PREVIEW

logger = logging.getLogger(__name__)

WARN_THRESHOLD = 250_000
STRONG_WARN_THRESHOLD = 500_000
REFUSE_THRESHOLD = 750_000


@dataclass
class TargetPlan:
    centres: np.ndarray
    n_blocks: int
    mode: EstimationMode
    warnings: List[str] = field(default_factory=list)
    downgraded: bool = False


def prepare_targets(
    active_centres: np.ndarray,
    mode: EstimationMode,
) -> TargetPlan:
    """Validate block count and optionally downgrade mode for large jobs."""
    n = active_centres.shape[0]
    warnings: List[str] = []
    downgraded = False

    if n > REFUSE_THRESHOLD and mode.name == "final":
        warnings.append(
            f"Very large job: {n:,} active blocks. "
            f"Final mode refused — downgrading to preview."
        )
        mode = PREVIEW
        downgraded = True
    elif n > STRONG_WARN_THRESHOLD and mode.name != "preview":
        warnings.append(
            f"Large job: {n:,} active blocks. "
            f"Consider preview mode for faster results."
        )
    elif n > WARN_THRESHOLD:
        warnings.append(f"{n:,} active blocks — estimation may take a few minutes.")

    for w in warnings:
        logger.warning(w)

    return TargetPlan(
        centres=active_centres,
        n_blocks=n,
        mode=mode,
        warnings=warnings,
        downgraded=downgraded,
    )
