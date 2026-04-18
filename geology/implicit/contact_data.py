"""
Contact Data — Data Model for Geological Contacts and Orientations.
====================================================================

Holds the structured data that enters the scalar field interpolation:
contact points (where drillholes cross geological boundaries) and
orientation measurements (dip/azimuth of bedding or structures).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ContactPoint:
    """A single geological contact observation from a drillhole."""
    x: float
    y: float
    z: float
    unit_above: str
    unit_below: str
    surface_name: str
    hole_id: str = ""
    depth: float = 0.0
    normal: Optional[np.ndarray] = None  # (3,) estimated surface normal


@dataclass
class OrientationMeasurement:
    """A structural measurement constraining the gradient of the scalar field."""
    x: float
    y: float
    z: float
    dip: float          # degrees, 0 = horizontal, 90 = vertical
    azimuth: float      # degrees, 0 = north, clockwise
    normal: Optional[np.ndarray] = None  # (3,) unit normal (computed from dip/azimuth)
    feature_type: str = "bedding"        # bedding, foliation, contact, etc.
    hole_id: str = ""
    confidence: float = 1.0              # weight factor


@dataclass
class ContactSet:
    """A collection of contacts for one geological surface."""
    surface_name: str
    unit_above: str
    unit_below: str
    contacts: List[ContactPoint] = field(default_factory=list)
    orientations: List[OrientationMeasurement] = field(default_factory=list)

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)

    @property
    def n_orientations(self) -> int:
        return len(self.orientations)

    def contact_coords(self) -> np.ndarray:
        """Return (N_c, 3) array of contact point coordinates."""
        if not self.contacts:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(
            [[c.x, c.y, c.z] for c in self.contacts], dtype=np.float64,
        )

    def contact_normals(self) -> np.ndarray:
        """Return (N_c, 3) array of contact normals (may be zero if unset)."""
        if not self.contacts:
            return np.empty((0, 3), dtype=np.float64)
        normals = []
        for c in self.contacts:
            if c.normal is not None:
                normals.append(c.normal)
            else:
                normals.append(np.array([0.0, 0.0, 1.0]))  # default vertical
        return np.array(normals, dtype=np.float64)

    def orientation_coords(self) -> np.ndarray:
        """Return (N_g, 3) array of orientation measurement locations."""
        if not self.orientations:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(
            [[o.x, o.y, o.z] for o in self.orientations], dtype=np.float64,
        )

    def orientation_normals(self) -> np.ndarray:
        """Return (N_g, 3) array of unit normals from orientation data."""
        if not self.orientations:
            return np.empty((0, 3), dtype=np.float64)
        normals = []
        for o in self.orientations:
            if o.normal is not None:
                normals.append(o.normal)
            else:
                normals.append(dip_azimuth_to_normal(o.dip, o.azimuth))
        return np.array(normals, dtype=np.float64)


@dataclass
class StratigraphicColumn:
    """Ordered sequence of modelling units with contact relationships."""
    units: List[str]                       # top (youngest) to bottom (oldest)
    contact_types: List[str] = field(default_factory=list)
    # contact_types[i] = contact between units[i] and units[i+1]
    # Values: "conformable", "unconformable_erosion", "unconformable_onlap", "gradational"
    unit_colors: Dict[str, str] = field(default_factory=dict)
    unit_types: Dict[str, str] = field(default_factory=dict)
    # unit_types: "stratigraphic", "intrusive", "fault_zone", "weathering", "exclude"

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_surfaces(self) -> int:
        """Number of boundary surfaces = n_units - 1."""
        return max(0, len(self.units) - 1)

    def surface_name(self, index: int) -> str:
        """Name of the i-th boundary surface."""
        if index < 0 or index >= self.n_surfaces:
            raise IndexError(f"Surface index {index} out of range [0, {self.n_surfaces})")
        return f"{self.units[index]}_{self.units[index + 1]}"

    def validate(self) -> List[str]:
        """Return list of validation warnings."""
        warnings = []
        if len(self.units) < 2:
            warnings.append("Need at least 2 units to define a surface.")
        if self.contact_types and len(self.contact_types) != self.n_surfaces:
            warnings.append(
                f"Expected {self.n_surfaces} contact types, got {len(self.contact_types)}."
            )
        return warnings


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def dip_azimuth_to_normal(dip: float, azimuth: float) -> np.ndarray:
    """Convert dip/azimuth (degrees) to unit normal vector.

    Convention (Eq. 2.2 / 5.2 of math spec):
        n = [sin(dip)*sin(azimuth), sin(dip)*cos(azimuth), cos(dip)]

    Dip: 0 = horizontal, 90 = vertical downward.
    Azimuth: 0 = north, clockwise.

    Returns
    -------
    np.ndarray, shape (3,)
        Unit normal vector.
    """
    d = np.radians(dip)
    a = np.radians(azimuth)
    nx = np.sin(d) * np.sin(a)
    ny = np.sin(d) * np.cos(a)
    nz = np.cos(d)
    n = np.array([nx, ny, nz], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm > 1e-12:
        n /= norm
    return n


def contacts_dataframe_to_contact_set(
    df: pd.DataFrame,
    surface_name: str,
    unit_above: str,
    unit_below: str,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    hole_id_col: str = "hole_id",
) -> ContactSet:
    """Build a ContactSet from a DataFrame of contact points."""
    cs = ContactSet(
        surface_name=surface_name,
        unit_above=unit_above,
        unit_below=unit_below,
    )
    for _, row in df.iterrows():
        cp = ContactPoint(
            x=float(row[x_col]),
            y=float(row[y_col]),
            z=float(row[z_col]),
            unit_above=unit_above,
            unit_below=unit_below,
            surface_name=surface_name,
            hole_id=str(row.get(hole_id_col, "")),
        )
        cs.contacts.append(cp)
    return cs


def orientations_dataframe_to_list(
    df: pd.DataFrame,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    dip_col: str = "dip",
    azimuth_col: str = "azimuth",
    hole_id_col: str = "hole_id",
    feature_type_col: str = "feature_type",
) -> List[OrientationMeasurement]:
    """Convert a DataFrame of structural measurements to OrientationMeasurement list."""
    measurements = []
    for _, row in df.iterrows():
        dip = float(row[dip_col])
        azimuth = float(row[azimuth_col])
        om = OrientationMeasurement(
            x=float(row[x_col]),
            y=float(row[y_col]),
            z=float(row[z_col]),
            dip=dip,
            azimuth=azimuth,
            normal=dip_azimuth_to_normal(dip, azimuth),
            feature_type=str(row.get(feature_type_col, "bedding")),
            hole_id=str(row.get(hole_id_col, "")),
        )
        measurements.append(om)
    return measurements
