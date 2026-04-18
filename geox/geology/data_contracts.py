"""
Geological Data Contracts

Typed data structures for geological modeling pipelines.
These are the canonical type contracts shared across all geological engines
(contact extraction, domain modeling, structural analysis).

All contracts are immutable (frozen dataclasses) to ensure reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositeInterval:
    """A single composited interval from a drillhole."""
    hole_id: str
    from_depth: float
    to_depth: float
    domain: str
    x: float
    y: float
    z: float
    grade: Optional[float] = None
    density: Optional[float] = None
    length: float = 0.0

    def __post_init__(self):
        if self.length == 0.0 and self.to_depth > self.from_depth:
            object.__setattr__(self, "length", self.to_depth - self.from_depth)


@dataclass
class CompositeCollection:
    """Collection of composite intervals with hole-level access."""
    intervals: List[CompositeInterval] = field(default_factory=list)

    @property
    def hole_ids(self) -> Set[str]:
        """Unique hole IDs in the collection."""
        return {iv.hole_id for iv in self.intervals}

    def intervals_for_hole(self, hole_id: str) -> List[CompositeInterval]:
        """Return intervals belonging to a specific hole."""
        return [iv for iv in self.intervals if iv.hole_id == hole_id]

    def __len__(self) -> int:
        return len(self.intervals)


# ---------------------------------------------------------------------------
# Contact data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contact:
    """A geological contact between two domains."""
    x: float
    y: float
    z: float
    above_unit: str
    below_unit: str
    hole_id: str
    depth: float
    contact_type: str = "sharp"  # sharp, gradational, faulted
    confidence: float = 1.0
    source: str = ""


@dataclass(frozen=True)
class ContactSet:
    """Immutable set of geological contacts with domain-pair metadata."""
    contacts: FrozenSet[Contact] = field(default_factory=frozenset)
    domain_pairs: FrozenSet[tuple] = field(default_factory=frozenset)

    def __len__(self) -> int:
        return len(self.contacts)

    def contacts_for_pair(
        self, domain_a: str, domain_b: str
    ) -> FrozenSet[Contact]:
        """Return contacts between two specific domains (order-independent)."""
        return frozenset(
            c
            for c in self.contacts
            if {c.above_unit, c.below_unit} == {domain_a, domain_b}
        )

    def contacts_for_hole(self, hole_id: str) -> FrozenSet[Contact]:
        """Return all contacts from a specific hole."""
        return frozenset(c for c in self.contacts if c.hole_id == hole_id)
