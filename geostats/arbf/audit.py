"""
ARBF JORC Audit Record Generation.

Generates structured audit records addressing every JORC Table 1
Section 3 criterion for mineral resource estimation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SOFTWARE_VERSION = "GeoX ARBF 1.0.0"


@dataclass
class ARBFAuditRecord:
    """JORC Table 1 Section 3 compliant audit record for ARBF estimation."""

    # --- Metadata ---
    run_id: str = ""
    timestamp: str = ""
    software_version: str = SOFTWARE_VERSION
    operator: str = ""

    # --- Database Integrity ---
    database_description: str = ""
    composite_length: float = 0.0
    composite_method: str = "length-weighted"
    num_composites: int = 0
    num_drillholes: int = 0
    data_hash: str = ""

    # --- Estimation Parameters ---
    kernel_type: str = "spheroidal"
    kernel_alpha: float = 1.0
    sill: float = 0.0
    nugget: float = 0.0
    range_: float = 0.0
    drift_type: str = "constant"
    requested_drift_type: str = "constant"
    effective_drift_type: str = "constant"
    drift_selection_method: str = "manual"
    trend_cv_constant_rmse: float = 0.0
    trend_cv_linear_rmse: float = 0.0
    trend_cv_rmse_improvement: float = 0.0
    accuracy: float = 1e-6

    # --- ARBF-Specific ---
    estimation_mode: str = "local_neighbourhood_gpr"
    n_subdomains: int = 0
    subdomain_method: str = "kmeans"
    avg_samples_per_subdomain: float = 0.0
    use_lva: bool = False
    lva_source: str = "none"
    use_normal_score: bool = False
    use_ilr: bool = False
    use_change_of_support: bool = False

    # --- Search Neighbourhood ---
    max_samples: int = 300
    min_samples: int = 4

    # --- Anisotropy ---
    azimuth: float = 0.0
    dip: float = 0.0
    pitch: float = 0.0
    range_max: float = 0.0
    range_mid: float = 0.0
    range_min: float = 0.0
    domain_policy: str = "warn"
    n_geological_domains: int = 0
    domains_enforced: bool = False
    block_domain_assignment: str = "none"

    # --- Block Model ---
    block_size: Optional[List[float]] = None
    n_blocks_estimated: int = 0
    n_blocks_total: int = 0
    clip_to_drill_footprint: bool = False
    footprint_buffer_ranges: float = 0.0
    discretisation: str = "adaptive"
    discretisation_density: str = "8/27/64"

    # --- Validation ---
    cv_rmse: float = 0.0
    cv_mae: float = 0.0
    cv_r_squared: float = 0.0
    cv_mean_error: float = 0.0
    cv_slope_of_regression: float = 0.0
    cv_normalised_rmse: float = 0.0
    conditional_bias_global_slope: float = 0.0
    conditional_bias_binned_slope: float = 0.0
    conditional_bias_mean_bin_bias: float = 0.0
    conditional_bias_max_abs_bin_bias: float = 0.0
    support_swath_panel_factors: Optional[List[int]] = None
    support_swath_panels_total: int = 0
    support_swath_panels_with_data: int = 0
    support_swath_mean_rmse: float = 0.0
    support_swath_mean_bias: float = 0.0

    # --- Classification ---
    classification_method: str = "dual-criteria (variance + geometric)"
    variance_thresholds: Optional[List[float]] = None
    measured_blocks: int = 0
    indicated_blocks: int = 0
    inferred_blocks: int = 0
    unclassified_blocks: int = 0

    # --- Change of Support ---
    support_ratio: float = 0.0
    sigma_point: float = 0.0
    sigma_block: float = 0.0

    # --- Output ---
    output_attributes: Optional[List[str]] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serialisable dict (NumPy types -> native Python)."""
        d = asdict(self)
        return _convert_numpy(d)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_jorc_table1_section3(self) -> str:
        """Generate JORC Table 1 Section 3 formatted text report."""
        lines = [
            "=" * 72,
            "JORC TABLE 1 — SECTION 3: ESTIMATION AND REPORTING",
            "=" * 72,
            f"Run ID:      {self.run_id}",
            f"Timestamp:   {self.timestamp}",
            f"Software:    {self.software_version}",
            f"Operator:    {self.operator}",
            "",
            "--- DATABASE INTEGRITY ---",
            f"Description:     {self.database_description}",
            f"Composites:      {self.num_composites} samples",
            f"Drillholes:      {self.num_drillholes}",
            f"Composite length:{self.composite_length:.2f} m",
            f"Composite method:{self.composite_method}",
            f"Data hash:       {self.data_hash[:16]}...",
            "",
            "--- ESTIMATION METHOD ---",
            f"Method:          {self.estimation_mode}",
            f"Kernel:          {self.kernel_type} (alpha={self.kernel_alpha:.2f})",
            f"Sill:            {self.sill:.4f}",
            f"Nugget:          {self.nugget:.4f}",
            f"Range:           {self.range_:.1f} m",
            f"Drift:           {self.effective_drift_type}",
            f"Requested drift: {self.requested_drift_type}",
            f"Drift selection: {self.drift_selection_method}",
            f"Accuracy:        {self.accuracy:.2e}",
            "",
            "--- ARBF PARAMETERS ---",
            f"Sub-domains:     {self.n_subdomains} ({self.subdomain_method})",
            f"Avg samples/SD:  {self.avg_samples_per_subdomain:.0f}",
            f"LVA:             {'Yes (' + self.lva_source + ')' if self.use_lva else 'No'}",
            f"Normal-score:    {'Yes' if self.use_normal_score else 'No'}",
            f"ILR transform:   {'Yes' if self.use_ilr else 'No'}",
            f"Change-of-support: {'Yes' if self.use_change_of_support else 'No'}",
            "",
            "--- ANISOTROPY ---",
            f"Azimuth:  {self.azimuth:.1f} deg",
            f"Dip:      {self.dip:.1f} deg",
            f"Pitch:    {self.pitch:.1f} deg",
            f"Ranges:   {self.range_max:.1f} / {self.range_mid:.1f} / {self.range_min:.1f} m",
            f"Domains:  {self.n_geological_domains} enforced={self.domains_enforced}",
            f"Domain policy: {self.domain_policy} ({self.block_domain_assignment})",
            "",
            "--- BLOCK MODEL ---",
            f"Block size:     {self.block_size}",
            f"Blocks est'd:   {self.n_blocks_estimated} / {self.n_blocks_total}",
            f"Footprint clip: {self.clip_to_drill_footprint} (buffer={self.footprint_buffer_ranges:.2f} ranges)",
            f"Discretisation: {self.discretisation} ({self.discretisation_density})",
            "",
            "--- CROSS-VALIDATION ---",
            f"RMSE:            {self.cv_rmse:.4f}",
            f"MAE:             {self.cv_mae:.4f}",
            f"R-squared:       {self.cv_r_squared:.4f}",
            f"Mean Error:      {self.cv_mean_error:.4f}",
            f"Slope:           {self.cv_slope_of_regression:.4f}",
            f"Normalised RMSE: {self.cv_normalised_rmse:.4f}",
            _slope_interpretation(self.cv_slope_of_regression),
            f"Conditional-bias slope: {self.conditional_bias_global_slope:.4f}",
            f"Binned cond-bias slope: {self.conditional_bias_binned_slope:.4f}",
            f"Max abs bin bias:     {self.conditional_bias_max_abs_bin_bias:.4f}",
            f"Support swath panels: {self.support_swath_panels_with_data} / {self.support_swath_panels_total}",
            f"Support swath RMSE:   {self.support_swath_mean_rmse:.4f}",
            "",
            "--- CLASSIFICATION ---",
            f"Method:          {self.classification_method}",
            f"Thresholds:      {self.variance_thresholds}",
            f"Measured:         {self.measured_blocks}",
            f"Indicated:        {self.indicated_blocks}",
            f"Inferred:         {self.inferred_blocks}",
            f"Unclassified:     {self.unclassified_blocks}",
            "",
            "--- CHANGE OF SUPPORT ---",
            f"Support ratio:   {self.support_ratio:.4f}",
            f"Sigma point:     {self.sigma_point:.4f}",
            f"Sigma block:     {self.sigma_block:.4f}",
            "",
            "--- OUTPUT ---",
            f"Attributes:      {self.output_attributes}",
            f"Clip range:      [{self.clip_min}, {self.clip_max}]",
            f"Elapsed:         {self.elapsed_seconds:.1f} s",
            "=" * 72,
        ]
        return "\n".join(lines)

    def validate_completeness(self) -> List[str]:
        """Check for missing fields required by JORC Table 1 Section 3."""
        warnings = []
        if self.num_composites == 0:
            warnings.append("num_composites is 0 — database not described")
        if self.data_hash == "":
            warnings.append("data_hash empty — cannot verify data integrity")
        if self.cv_slope_of_regression == 0.0:
            warnings.append("cv_slope_of_regression is 0 — CV not run")
        if self.n_blocks_estimated == 0:
            warnings.append("n_blocks_estimated is 0 — no blocks estimated")
        if not self.operator:
            warnings.append("operator not specified — Competent Person required")
        return warnings


def compute_data_hash(
    coords: np.ndarray,
    values: np.ndarray,
) -> str:
    """Compute SHA-256 hash of input data for reproducibility.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) coordinates.
    values : np.ndarray
        (N,) values.

    Returns
    -------
    str
        Hex-encoded SHA-256 hash.
    """
    # Sort lexicographically for deterministic ordering
    data = np.column_stack([coords, values.reshape(-1, 1)])
    sort_idx = np.lexsort(data.T[::-1])
    data_sorted = data[sort_idx]
    return hashlib.sha256(data_sorted.tobytes()).hexdigest()


def _slope_interpretation(slope: float) -> str:
    """Interpret the slope of regression for JORC reporting."""
    if 0.90 <= slope <= 1.10:
        return "  -> JORC compliant (0.90 <= slope <= 1.10)"
    elif 0.80 <= slope <= 1.20:
        return "  -> Acceptable (0.80 <= slope <= 1.20)"
    else:
        return f"  -> REQUIRES INVESTIGATION (slope = {slope:.3f})"


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
