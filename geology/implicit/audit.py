"""
Geological Model Audit — JORC/SAMREC compliant audit trail.
=============================================================

Records every parameter and data hash for the geological model build,
analogous to geostats.arbf.audit for estimation runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeologicalModelAudit:
    """Audit record for an implicit geological model build."""

    # ── Metadata ──
    run_id: str = ""
    timestamp: str = ""
    software_version: str = "GeoX Implicit Modelling 1.0.0"
    operator: str = ""

    # ── Model configuration ──
    model_type: str = ""           # stratiform, vein, intrusive, structural
    kernel_type: str = "spheroidal"
    alpha: float = 1.0
    range_max: float = 100.0
    range_mid: float = 100.0
    range_min: float = 100.0
    azimuth: float = 0.0
    dip: float = 0.0
    pitch: float = 0.0
    nugget: float = 0.0
    accuracy: float = 1e-6
    drift_type: str = "constant"
    constraint_method: str = "gradient"  # gradient or offset
    offset_distance: float = 2.0
    grid_resolution: float = 10.0

    # ── Input data ──
    n_contacts: int = 0
    n_orientations: int = 0
    n_surfaces: int = 0
    stratigraphic_units: List[str] = field(default_factory=list)
    contact_data_hash: str = ""

    # ── Output ──
    n_blocks_classified: int = 0
    surface_names: List[str] = field(default_factory=list)
    surface_vertex_counts: Dict[str, int] = field(default_factory=dict)
    surface_triangle_counts: Dict[str, int] = field(default_factory=dict)

    # ── Validation ──
    contact_honouring_pct: float = 0.0
    mean_contact_misfit: float = 0.0
    max_contact_misfit: float = 0.0

    # ── Performance ──
    elapsed_seconds: float = 0.0
    matrix_size: int = 0
    condition_number: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (NumPy → native Python)."""
        d = asdict(self)
        # Convert any remaining numpy types
        for key, val in d.items():
            if isinstance(val, (np.integer,)):
                d[key] = int(val)
            elif isinstance(val, (np.floating,)):
                d[key] = float(val)
            elif isinstance(val, np.ndarray):
                d[key] = val.tolist()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def validate_completeness(self) -> List[str]:
        """Check for missing fields required for a complete audit."""
        warnings = []
        if not self.run_id:
            warnings.append("Missing run_id")
        if not self.timestamp:
            warnings.append("Missing timestamp")
        if self.n_contacts == 0:
            warnings.append("No contact points recorded")
        if self.contact_honouring_pct < 90.0:
            warnings.append(
                f"Contact honouring {self.contact_honouring_pct:.1f}% is below 90% threshold"
            )
        return warnings


def compute_contact_data_hash(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
) -> str:
    """Compute SHA-256 hash of contact data for provenance tracking."""
    hasher = hashlib.sha256()

    # Sort by coordinates for deterministic ordering
    if coords.shape[0] > 0:
        idx = np.lexsort(coords.T[::-1])
        sorted_coords = coords[idx]
        hasher.update(sorted_coords.tobytes())

        if values is not None:
            sorted_values = values[idx]
            hasher.update(sorted_values.tobytes())

    return hasher.hexdigest()


def generate_run_id() -> str:
    """Generate a unique run ID."""
    now = datetime.now(timezone.utc)
    return f"geomodel_{now.strftime('%Y%m%d_%H%M%S')}_{id(now) % 10000:04d}"
