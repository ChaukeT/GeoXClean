"""
JORC Table 1 Section 3 audit trail builder.

Every estimation run produces a structured audit record that a
Competent Person can use to answer all JORC Table 1 Section 3
criteria on an 'if not, why not' basis.

Exportable as JSON/YAML/text.  Non-negotiable for investor trust.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class JORCAuditRecord:
    """
    Captures all JORC Table 1 Section 3 required information.

    This record can be exported as JSON for the Competent Person.
    """

    # ── Metadata ──────────────────────────────────────────────
    run_id: str = ""
    timestamp: str = ""
    software_version: str = "GeoX FastRBF 1.0.0"
    operator: str = ""

    # ── Database integrity ────────────────────────────────────
    database_description: str = ""
    composite_length: float = 0.0
    composite_method: str = "length-weighted"
    num_composites: int = 0
    num_drillholes: int = 0
    data_hash: str = ""

    # ── Estimation parameters ─────────────────────────────────
    estimation_method: str = "Radial Basis Function (FastRBF)"
    kernel_type: str = ""
    kernel_parameters: Dict[str, Any] = field(default_factory=dict)
    drift_type: str = ""
    accuracy: float = 0.0

    # ── Search neighbourhood ──────────────────────────────────
    search_max_samples: int = 0
    search_min_samples: int = 0
    search_max_per_octant: int = 0
    search_min_octants: int = 0
    search_radii: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    search_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── Anisotropy ────────────────────────────────────────────
    ellipsoid_ratios: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    trend_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    trend_source: str = ""

    # ── Block model ───────────────────────────────────────────
    block_size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    block_discretisation: int = 4
    num_blocks_estimated: int = 0
    num_blocks_total: int = 0
    parent_block_size: Optional[Tuple[float, float, float]] = None
    sub_block_size: Optional[Tuple[float, float, float]] = None

    # ── Validation results ────────────────────────────────────
    cv_rmse: float = 0.0
    cv_mae: float = 0.0
    cv_r_squared: float = 0.0
    cv_mean_error: float = 0.0
    cv_slope_of_regression: float = 0.0
    global_bias_percent: float = 0.0

    # ── Classification ────────────────────────────────────────
    classification_method: str = "search-pass based"
    measured_blocks: int = 0
    indicated_blocks: int = 0
    inferred_blocks: int = 0
    unclassified_blocks: int = 0
    measured_criteria: str = "Pass 1: half variogram range, min samples & octants met"
    indicated_criteria: str = "Pass 2: full variogram range, min samples & octants met"
    inferred_criteria: str = "Pass 3: double variogram range, min samples & octants met"

    # ── Cut-off ───────────────────────────────────────────────
    cutoff_grade: Optional[float] = None
    cutoff_basis: Optional[str] = None

    # ── Outputs ───────────────────────────────────────────────
    output_attributes: List[str] = field(default_factory=list)
    evaluation_limits: Optional[Tuple[float, float]] = None

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def from_estimation(
        cls,
        config: "RBFConfig",
        fitted: "FittedRBF",
        result: "EstimationResult",
        cv_result: Optional["CVResult"] = None,
        bias_result: Optional["BiasResult"] = None,
        slope_result: Optional["SlopeResult"] = None,
        operator: str = "",
        database_description: str = "",
        composite_length: float = 0.0,
        num_drillholes: int = 0,
        block_size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "JORCAuditRecord":
        """Build an audit record from estimation components."""
        from .config import RBFConfig
        from .fastrbf_engine import FittedRBF
        from .block_estimator import EstimationResult

        # Classification counts
        classifications = result.classification
        measured = int(np.sum(classifications == "Measured"))
        indicated = int(np.sum(classifications == "Indicated"))
        inferred = int(np.sum(classifications == "Inferred"))
        unclassified = int(np.sum(classifications == "Unclassified"))

        record = cls(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            operator=operator,
            database_description=database_description,
            composite_length=composite_length,
            num_composites=fitted.n_samples,
            num_drillholes=num_drillholes,
            data_hash=fitted.data_hash,
            kernel_type=config.kernel_type.value,
            kernel_parameters={
                "total_sill": config.total_sill,
                "nugget": config.nugget,
                "base_range": config.base_range,
                "alpha": config.alpha,
            },
            drift_type=config.drift.value,
            accuracy=fitted.accuracy_used,
            search_max_samples=config.search_max_samples,
            search_min_samples=config.search_min_samples,
            search_max_per_octant=config.search_max_per_octant,
            search_min_octants=config.search_min_octants,
            search_radii=config.search_radii,
            search_angles=config.search_angles,
            ellipsoid_ratios=(config.ratio_major, config.ratio_semi, config.ratio_minor),
            trend_angles=(config.azimuth, config.dip, config.pitch),
            block_size=block_size,
            block_discretisation=config.discretisation_points,
            num_blocks_estimated=result.n_blocks_estimated,
            num_blocks_total=result.n_blocks_total,
            measured_blocks=measured,
            indicated_blocks=indicated,
            inferred_blocks=inferred,
            unclassified_blocks=unclassified,
            output_attributes=["estimated_grade", "classification", "num_samples", "search_pass"],
        )

        if config.clip_min is not None and config.clip_max is not None:
            record.evaluation_limits = (config.clip_min, config.clip_max)

        if cv_result is not None:
            record.cv_rmse = cv_result.rmse
            record.cv_mae = cv_result.mae
            record.cv_r_squared = cv_result.r_squared
            record.cv_mean_error = cv_result.mean_error

        if slope_result is not None:
            record.cv_slope_of_regression = slope_result.slope

        if bias_result is not None:
            record.global_bias_percent = bias_result.bias_percent

        return record

    # ── Serialisation ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON export."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_jorc_table1_section3(self) -> str:
        """
        Generate a formatted JORC Table 1 Section 3 text report.

        Each criterion is addressed on an 'if not, why not' basis.
        """
        lines = [
            "=" * 72,
            "JORC TABLE 1 — SECTION 3: ESTIMATION AND REPORTING OF MINERAL RESOURCES",
            "=" * 72,
            "",
            f"Run ID:    {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Operator:  {self.operator}",
            f"Software:  {self.software_version}",
            "",
            "─" * 72,
            "DATABASE INTEGRITY",
            "─" * 72,
            f"  Description:      {self.database_description}",
            f"  Composite length: {self.composite_length}m",
            f"  Composite method: {self.composite_method}",
            f"  Num composites:   {self.num_composites}",
            f"  Num drillholes:   {self.num_drillholes}",
            f"  Data hash (SHA256): {self.data_hash[:16]}...",
            "",
            "─" * 72,
            "ESTIMATION AND MODELLING TECHNIQUES",
            "─" * 72,
            f"  Method:           {self.estimation_method}",
            f"  Kernel type:      {self.kernel_type}",
            f"  Kernel params:    {self.kernel_parameters}",
            f"  Drift type:       {self.drift_type}",
            f"  Accuracy (reg.):  {self.accuracy:.6e}",
            "",
            "─" * 72,
            "SEARCH PARAMETERS",
            "─" * 72,
            f"  Max samples:      {self.search_max_samples}",
            f"  Min samples:      {self.search_min_samples}",
            f"  Max per octant:   {self.search_max_per_octant}",
            f"  Min octants:      {self.search_min_octants}",
            f"  Search radii:     {self.search_radii}",
            f"  Search angles:    {self.search_angles}",
            "",
            "─" * 72,
            "ANISOTROPY",
            "─" * 72,
            f"  Ellipsoid ratios: {self.ellipsoid_ratios}",
            f"  Trend angles:     {self.trend_angles}",
            f"  Trend source:     {self.trend_source or 'Not specified'}",
            "",
            "─" * 72,
            "BLOCK MODEL",
            "─" * 72,
            f"  Block size:       {self.block_size}",
            f"  Discretisation:   {self.block_discretisation}^3 = {self.block_discretisation**3} points",
            f"  Blocks estimated: {self.num_blocks_estimated} / {self.num_blocks_total}",
            f"  Parent block:     {self.parent_block_size or 'N/A'}",
            f"  Sub-block:        {self.sub_block_size or 'N/A'}",
            "",
            "─" * 72,
            "VALIDATION",
            "─" * 72,
            f"  CV RMSE:          {self.cv_rmse:.4f}",
            f"  CV MAE:           {self.cv_mae:.4f}",
            f"  CV R²:            {self.cv_r_squared:.4f}",
            f"  CV Mean Error:    {self.cv_mean_error:.4f}",
            f"  Slope of regr.:   {self.cv_slope_of_regression:.4f}",
            f"  Global bias:      {self.global_bias_percent:.1f}%",
            "",
            "─" * 72,
            "CLASSIFICATION",
            "─" * 72,
            f"  Method:           {self.classification_method}",
            f"  Measured:         {self.measured_blocks} blocks — {self.measured_criteria}",
            f"  Indicated:        {self.indicated_blocks} blocks — {self.indicated_criteria}",
            f"  Inferred:         {self.inferred_blocks} blocks — {self.inferred_criteria}",
            f"  Unclassified:     {self.unclassified_blocks} blocks",
            "",
            "─" * 72,
            "CUT-OFF GRADE",
            "─" * 72,
            f"  Cut-off:          {self.cutoff_grade or 'Not applied'}",
            f"  Basis:            {self.cutoff_basis or 'N/A'}",
            "",
            "─" * 72,
            "OUTPUTS",
            "─" * 72,
            f"  Attributes:       {', '.join(self.output_attributes)}",
            f"  Eval. limits:     {self.evaluation_limits or 'None'}",
            "",
            "=" * 72,
            "END OF JORC TABLE 1 SECTION 3 REPORT",
            "=" * 72,
        ]
        return "\n".join(lines)

    def validate_completeness(self) -> List[str]:
        """
        Check that all JORC-mandatory fields are populated.

        Returns list of warnings for missing or suspicious fields.
        """
        warnings = []

        if not self.run_id:
            warnings.append("Missing: run_id")
        if not self.timestamp:
            warnings.append("Missing: timestamp")
        if not self.operator:
            warnings.append("Missing: operator name")
        if not self.data_hash:
            warnings.append("Missing: data_hash (reproducibility)")
        if self.num_composites == 0:
            warnings.append("Missing: num_composites is 0")
        if not self.kernel_type:
            warnings.append("Missing: kernel_type")
        if not self.drift_type:
            warnings.append("Missing: drift_type")
        if self.search_max_samples == 0:
            warnings.append("Missing: search_max_samples")
        if self.num_blocks_estimated == 0:
            warnings.append("Warning: no blocks estimated")
        if self.cv_rmse == 0 and self.cv_r_squared == 0:
            warnings.append("Warning: cross-validation not performed")
        if self.cv_slope_of_regression == 0:
            warnings.append("Warning: slope of regression not computed")
        if abs(self.cv_slope_of_regression - 1.0) > 0.1 and self.cv_slope_of_regression != 0:
            warnings.append(
                f"Warning: slope of regression ({self.cv_slope_of_regression:.3f}) "
                "deviates from 1.0 — conditional bias"
            )
        if self.global_bias_percent > 5.0:
            warnings.append(
                f"Warning: global bias ({self.global_bias_percent:.1f}%) exceeds 5%"
            )
        if self.measured_blocks + self.indicated_blocks + self.inferred_blocks == 0:
            warnings.append("Warning: no blocks classified (all unclassified)")
        if all(s == 0 for s in self.block_size):
            warnings.append("Missing: block_size")
        if not self.database_description:
            warnings.append("Missing: database_description")

        return warnings
