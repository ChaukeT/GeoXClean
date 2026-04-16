"""
ARBF estimation mode presets — preview, standard, final.

Controls subpoint density, neighbour count, and uncertainty detail level.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EstimationMode:
    name: str
    subpoints_per_axis: int
    max_neighbours: int
    compute_variance: bool
    compute_condition: bool
    partition_stitching: bool


PREVIEW = EstimationMode(
    name="preview",
    subpoints_per_axis=1,    # 1 subpoint (point support)
    max_neighbours=32,
    compute_variance=False,
    compute_condition=False,
    partition_stitching=False,
)

STANDARD = EstimationMode(
    name="standard",
    subpoints_per_axis=2,    # 8 subpoints (2×2×2)
    max_neighbours=48,
    compute_variance=True,
    compute_condition=False,
    partition_stitching=True,
)

FINAL = EstimationMode(
    name="final",
    subpoints_per_axis=3,    # 27 subpoints (3×3×3)
    max_neighbours=64,
    compute_variance=True,
    compute_condition=True,
    partition_stitching=True,
)

MODE_MAP = {
    "preview": PREVIEW,
    "standard": STANDARD,
    "final": FINAL,
}


def get_mode(name: str) -> EstimationMode:
    return MODE_MAP.get(name, STANDARD)
