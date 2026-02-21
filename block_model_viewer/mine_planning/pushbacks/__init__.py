"""
Pushback Visual Designer (STEP 33)

Module for managing pushback design and integration with NPVS optimization.
"""

from .pushback_model import ShellPhase, Pushback, PushbackPlan, compute_pushback_stats
from .pushback_builder import (
    auto_group_shells_by_depth,
    auto_group_shells_by_value,
    reorder_pushbacks
)

__all__ = [
    "ShellPhase",
    "Pushback",
    "PushbackPlan",
    "compute_pushback_stats",
    "auto_group_shells_by_depth",
    "auto_group_shells_by_value",
    "reorder_pushbacks",
]

