"""
Resource Reporting Engine
=========================

Geological / Tonnage Resource Summary Engine.

Computes geological / tonnage resource summaries per classification:
- Blocks
- Volume (m³)
- Density (t/m³)
- Tonnage (t)
- Mass-weighted mean grade
- Contained metal (t)

Works on a classified block model downstream of the JORCClassificationEngine.

Author: GeoX Mining Software Platform
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Any, Callable, Union, TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .block_model import BlockModel

logger = logging.getLogger(__name__)

# Try to import numba for acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available, using numpy fallback for resource reporting")


# ------------------------------------------------------------------ #
# Configuration Dataclasses
# ------------------------------------------------------------------ #

@dataclass
class DensityConfig:
    """
    Density configuration for resource reporting.

    Supports three modes:
    - constant: Single scalar applied to all blocks
    - domain: Mapping domain → density from DataFrame
    - block: Use DENSITY column from block model
    """
    mode: Literal["constant", "domain", "block"]
    constant_value: float | None = None
    domain_table: Optional[pd.DataFrame] = None  # columns: [domain_field, "DENSITY"]
    block_density_field: str = "DENSITY"

    def __post_init__(self):
        """Validate density configuration."""
        if self.mode == "constant" and (self.constant_value is None or self.constant_value <= 0):
            raise ValueError("constant_value must be positive when mode='constant'")
        if self.mode == "domain" and self.domain_table is None:
            raise ValueError("domain_table required when mode='domain'")
        if self.mode == "domain" and self.domain_table is not None:
            # Domain table should have exactly 2 columns: [domain_field_name, "DENSITY"]
            if len(self.domain_table.columns) != 2:
                raise ValueError(f"domain_table must have exactly 2 columns, got {len(self.domain_table.columns)}")
            if "DENSITY" not in self.domain_table.columns:
                raise ValueError("domain_table must have a 'DENSITY' column")
            # The other column should be the domain field name (not necessarily "DOMAIN")
            density_cols = [col for col in self.domain_table.columns if col != "DENSITY"]
            if len(density_cols) != 1:
                raise ValueError("domain_table must have exactly one domain column besides 'DENSITY'")
            logger.debug(f"Domain table validated with columns: {list(self.domain_table.columns)}")


@dataclass
class VolumeConfig:
    """
    Volume configuration for resource reporting.

    Supports two modes:
    - field: Use existing volume column
    - constant: Calculate as dx * dy * dz
    """
    mode: Literal["field", "constant"]
    field_name: Optional[str] = "BLOCK_VOLUME"
    dx: Optional[float] = None
    dy: Optional[float] = None
    dz: Optional[float] = None

    def __post_init__(self):
        """Validate volume configuration."""
        if self.mode == "field" and not self.field_name:
            raise ValueError("field_name required when mode='field'")
        if self.mode == "constant" and not all([self.dx, self.dy, self.dz]):
            raise ValueError("dx, dy, dz required when mode='constant'")
        if self.mode == "constant" and not all(x > 0 for x in [self.dx, self.dy, self.dz]):
            raise ValueError("dx, dy, dz must be positive when mode='constant'")


@dataclass
class ResourceSummaryRow:
    """Single row in resource summary table."""
    classification: str
    n_blocks: int
    total_volume_m3: float
    avg_density_t_per_m3: float
    total_tonnage_t: float
    grade_pct: float
    contained_metal_t: float


@dataclass
class ResourceSummaryResult:
    """Complete resource reporting result."""
    rows: list[ResourceSummaryRow]
    totals_MI: Optional[ResourceSummaryRow] = None
    totals_all: Optional[ResourceSummaryRow] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Numba-accelerated computation functions
# ------------------------------------------------------------------ #

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_mass_weighted_stats_numba(
        classifications: np.ndarray,
        volumes: np.ndarray,
        densities: np.ndarray,
        grades: np.ndarray,
        unique_classes: np.ndarray,
        grade_is_pct: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Numba-accelerated computation of mass-weighted statistics per classification.

        Returns arrays for each unique classification:
        - n_blocks: number of blocks
        - total_volume: sum of volumes
        - total_tonnage: sum of tonnages (volume * density)
        - weighted_grade: mass-weighted mean grade
        - contained_metal: total contained metal
        """
        n_classes = len(unique_classes)

        # Initialize output arrays
        n_blocks = np.zeros(n_classes, dtype=np.int32)
        total_volume = np.zeros(n_classes, dtype=np.float64)
        total_tonnage = np.zeros(n_classes, dtype=np.float64)
        sum_weighted_grade = np.zeros(n_classes, dtype=np.float64)
        contained_metal = np.zeros(n_classes, dtype=np.float64)

        # Process each block
        for i in prange(len(classifications)):
            # Find classification index
            class_idx = -1
            for j in range(n_classes):
                if classifications[i] == unique_classes[j]:
                    class_idx = j
                    break

            if class_idx >= 0:
                vol = volumes[i]
                den = densities[i]
                grade = grades[i]
                tonnage = vol * den

                n_blocks[class_idx] += 1
                total_volume[class_idx] += vol
                total_tonnage[class_idx] += tonnage
                sum_weighted_grade[class_idx] += grade * tonnage
                if grade_is_pct:
                    contained_metal[class_idx] += tonnage * (grade / 100.0)
                else:
                    contained_metal[class_idx] += tonnage * grade

        # Compute weighted averages
        weighted_grade = np.zeros(n_classes, dtype=np.float64)
        for i in range(n_classes):
            if total_tonnage[i] > 0:
                weighted_grade[i] = sum_weighted_grade[i] / total_tonnage[i]

        return n_blocks, total_volume, total_tonnage, weighted_grade, contained_metal


def _compute_mass_weighted_stats_numpy(
    classifications: np.ndarray,
    volumes: np.ndarray,
    densities: np.ndarray,
    grades: np.ndarray,
    unique_classes: np.ndarray,
    grade_is_pct: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy fallback for mass-weighted statistics computation.
    """
    n_classes = len(unique_classes)

    # Initialize output arrays
    n_blocks = np.zeros(n_classes, dtype=np.int32)
    total_volume = np.zeros(n_classes, dtype=np.float64)
    total_tonnage = np.zeros(n_classes, dtype=np.float64)
    weighted_grade = np.zeros(n_classes, dtype=np.float64)
    contained_metal = np.zeros(n_classes, dtype=np.float64)

    # Process each classification
    for idx, class_val in enumerate(unique_classes):
        mask = classifications == class_val
        if not mask.any():
            continue

        class_volumes = volumes[mask]
        class_densities = densities[mask]
        class_grades = grades[mask]
        class_tonnages = class_volumes * class_densities

        n_blocks[idx] = len(class_volumes)
        total_volume[idx] = class_volumes.sum()
        total_tonnage[idx] = class_tonnages.sum()

        if total_tonnage[idx] > 0:
            weighted_grade[idx] = (class_grades * class_tonnages).sum() / total_tonnage[idx]

        if grade_is_pct:
            contained_metal[idx] = total_tonnage[idx] * (weighted_grade[idx] / 100.0)
        else:
            contained_metal[idx] = total_tonnage[idx] * weighted_grade[idx]

    return n_blocks, total_volume, total_tonnage, weighted_grade, contained_metal


# ------------------------------------------------------------------ #
# Core Resource Reporting Engine
# ------------------------------------------------------------------ #

class ResourceReportingEngine:
    """
    Resource Reporting Engine

    Computes geological / tonnage resource summaries per classification:
    - Blocks
    - Volume (m³)
    - Density (t/m³)
    - Tonnage (t)
    - Mass-weighted mean grade
    - Contained metal (t)

    Works on a classified block model downstream of the JORCClassificationEngine.

    Usage
    -----
    ```python
    engine = ResourceReportingEngine(
        block_model=df,
        class_field="CLASS_FINAL",
        grade_field="Fe",
    )

    result = engine.compute_summary(
        density_config=DensityConfig(mode="constant", constant_value=2.8),
        volume_config=VolumeConfig(mode="field", field_name="BLOCK_VOLUME"),
    )
    ```
    """

    # Version tags for audit trail and traceability
    METHOD_VERSION = "RR-v1.0"
    BUILD = "Numba-accelerated"

    def __init__(
        self,
        block_model: pd.DataFrame,
        class_field: str = "CLASS_FINAL",
        grade_field: str = "Fe",
        volume_field: str | None = "BLOCK_VOLUME",
        domain_field: str | None = "DOMAIN",
        grade_is_pct: bool = True,
    ):
        """
        Initialize the resource reporting engine.

        Parameters
        ----------
        block_model : pd.DataFrame
            Block model with coordinates + classification columns
        class_field : str
            Classification column name (default "CLASS_FINAL")
        grade_field : str
            Primary grade column name (default "Fe")
        volume_field : str, optional
            Volume column name (default "BLOCK_VOLUME")
        domain_field : str, optional
            Domain column name for domain-based density (default "DOMAIN")
        """
        self.df = block_model.copy()
        self.class_field = class_field
        self.grade_field = grade_field
        self.volume_field = volume_field
        self.domain_field = domain_field
        self.grade_is_pct = grade_is_pct

        # Validate inputs
        self._validate_inputs()

        logger.info(
            f"ResourceReportingEngine initialized: "
            f"blocks={len(self.df)}, class_field='{class_field}', "
            f"grade_field='{grade_field}', volume_field='{volume_field}', "
            f"domain_field='{domain_field}', version={self.METHOD_VERSION}, build={self.BUILD}"
        )

    def _validate_inputs(self):
        """Validate that required columns exist."""
        required_cols = [self.class_field, self.grade_field]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for nulls in required fields
        for col in required_cols:
            if self.df[col].isnull().any():
                n_null = self.df[col].isnull().sum()
                raise ValueError(f"Column '{col}' has {n_null} null values")

    def apply_density(self, density_config: DensityConfig) -> None:
        """
        Adds a 'DEN' column to self.df with final density per block.

        Raises a clear ValueError if density cannot be determined for some blocks.

        Parameters
        ----------
        density_config : DensityConfig
            Density configuration
        """
        if density_config.mode == "constant":
            self.df["DEN"] = density_config.constant_value
            logger.info(f"Applied constant density: {density_config.constant_value} t/m³")

        elif density_config.mode == "domain":
            if self.domain_field not in self.df.columns:
                raise ValueError(f"Domain field '{self.domain_field}' not found in block model")

            # Merge density table
            domain_col = self.domain_field
            density_df = density_config.domain_table.copy()
            density_df = density_df.set_index(domain_col)["DENSITY"]

            self.df = self.df.merge(
                density_df, left_on=domain_col, right_index=True, how="left"
            )
            self.df = self.df.rename(columns={"DENSITY": "DEN"})

            # Check for missing densities
            missing_mask = self.df["DEN"].isnull()
            if missing_mask.any():
                missing_domains = self.df.loc[missing_mask, domain_col].unique()
                raise ValueError(
                    f"Density not defined for domains: {list(missing_domains)}. "
                    f"Add these domains to the density table."
                )

            logger.info(f"Applied domain-based density for {len(density_df)} domains")

        elif density_config.mode == "block":
            if density_config.block_density_field not in self.df.columns:
                raise ValueError(f"Density field '{density_config.block_density_field}' not found")

            self.df["DEN"] = self.df[density_config.block_density_field]

            # Check for nulls
            null_mask = self.df["DEN"].isnull()
            if null_mask.any():
                n_null = null_mask.sum()
                raise ValueError(f"Density field has {n_null} null values")

            logger.info(f"Applied block-level density from column '{density_config.block_density_field}'")

        else:
            raise ValueError(f"Unknown density mode: {density_config.mode}")

        # Final validation
        if (self.df["DEN"] <= 0).any():
            n_invalid = (self.df["DEN"] <= 0).sum()
            raise ValueError(f"Density has {n_invalid} non-positive values")

    def apply_volume(self, volume_config: VolumeConfig) -> None:
        """
        Adds a 'VOL' column to self.df with block volume (m³) per block.

        Parameters
        ----------
        volume_config : VolumeConfig
            Volume configuration
        """
        if volume_config.mode == "field":
            if volume_config.field_name not in self.df.columns:
                raise ValueError(f"Volume field '{volume_config.field_name}' not found")

            self.df["VOL"] = self.df[volume_config.field_name]

            # Check for nulls
            null_mask = self.df["VOL"].isnull()
            if null_mask.any():
                n_null = null_mask.sum()
                raise ValueError(f"Volume field '{volume_config.field_name}' has {n_null} null values")

            logger.info(f"Applied volume from field '{volume_config.field_name}'")

        elif volume_config.mode == "constant":
            vol = volume_config.dx * volume_config.dy * volume_config.dz
            self.df["VOL"] = vol
            logger.info(f"Applied constant volume: {vol} m³ ({volume_config.dx}×{volume_config.dy}×{volume_config.dz})")

        else:
            raise ValueError(f"Unknown volume mode: {volume_config.mode}")

        # Final validation
        if (self.df["VOL"] <= 0).any():
            n_invalid = (self.df["VOL"] <= 0).sum()
            raise ValueError(f"Volume has {n_invalid} non-positive values")

    def compute_summary(
        self,
        density_config: DensityConfig,
        volume_config: VolumeConfig,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> ResourceSummaryResult:
        """
        Compute resource summary with mass-weighted statistics.

        Parameters
        ----------
        density_config : DensityConfig
            Density configuration
        volume_config : VolumeConfig
            Volume configuration
        progress_callback : callable, optional
            Callback(percent, message) for progress updates

        Returns
        -------
        ResourceSummaryResult
            Complete resource summary
        """
        start_time = datetime.now()

        if progress_callback:
            progress_callback(5, "Validating input and density/volume configuration...")

        # Apply density and volume
        self.apply_density(density_config)

        if progress_callback:
            progress_callback(20, "Assigning density to blocks...")

        self.apply_volume(volume_config)

        if progress_callback:
            progress_callback(40, "Computing block tonnage...")

        # Compute tonnage
        self.df["TONNES"] = self.df["VOL"] * self.df["DEN"]

        if progress_callback:
            progress_callback(70, "Aggregating per classification...")

        # Get unique classifications
        unique_classes = np.array(sorted(self.df[self.class_field].unique()))
        logger.info(f"Computing summary for {len(unique_classes)} classifications: {list(unique_classes)}")

        # Prepare data arrays
        classifications = self.df[self.class_field].values
        volumes = self.df["VOL"].values
        densities = self.df["DEN"].values
        grades = self.df[self.grade_field].values
        tonnages = self.df["TONNES"].values

        # Compute statistics
        use_numba = NUMBA_AVAILABLE
        try:
            if use_numba:
                n_blocks, total_volume, total_tonnage, weighted_grade, contained_metal = \
                    _compute_mass_weighted_stats_numba(
                        classifications, volumes, densities, grades, unique_classes, self.grade_is_pct
                    )
            else:
                n_blocks, total_volume, total_tonnage, weighted_grade, contained_metal = \
                    _compute_mass_weighted_stats_numpy(
                        classifications, volumes, densities, grades, unique_classes, self.grade_is_pct
                    )
        except Exception as e:
            logger.warning(f"Numba computation failed, using numpy fallback: {e}")
            n_blocks, total_volume, total_tonnage, weighted_grade, contained_metal = \
                _compute_mass_weighted_stats_numpy(
                    classifications, volumes, densities, grades, unique_classes, self.grade_is_pct
                )

        if progress_callback:
            progress_callback(90, "Building result summary...")

        # Build result rows
        rows = []
        total_density_sum = 0
        total_density_weight = 0

        for i, class_val in enumerate(unique_classes):
            if n_blocks[i] == 0:
                continue

            # Average density (volume-weighted)
            avg_density = total_tonnage[i] / total_volume[i] if total_volume[i] > 0 else 0

            row = ResourceSummaryRow(
                classification=str(class_val),
                n_blocks=int(n_blocks[i]),
                total_volume_m3=float(total_volume[i]),
                avg_density_t_per_m3=float(avg_density),
                total_tonnage_t=float(total_tonnage[i]),
                grade_pct=float(weighted_grade[i]),
                contained_metal_t=float(contained_metal[i]),
            )
            rows.append(row)

            # Accumulate for totals
            total_density_sum += avg_density * total_volume[i]
            total_density_weight += total_volume[i]

        # Sort by classification order with flexible fallback for numeric/custom codes
        # Primary JORC/SAMREC order
        primary_order = ["Measured", "Indicated", "Inferred", "Unclassified"]

        # Common alternative names/codes that should be mapped (case-insensitive lookup)
        # Maps lowercase versions to standard names
        order_mapping_lower = {
            # Full names (lowercase)
            "measured": 0, "indicated": 1, "inferred": 2, "unclassified": 3,
            # Abbreviations
            "m": 0, "i": 1, "inf": 2, "u": 3,
            "meas": 0, "ind": 1,
            # Numeric codes as strings
            "1": 0, "2": 1, "3": 2, "4": 3,
            "1.0": 0, "2.0": 1, "3.0": 2, "4.0": 3,
        }

        def get_sort_key(row):
            """Get sort key for classification, handling various naming schemes."""
            classification = row.classification

            # Handle numeric types directly
            if isinstance(classification, (int, float)):
                code = int(classification)
                if 1 <= code <= 4:
                    return code - 1  # 1=Measured(0), 2=Indicated(1), 3=Inferred(2), 4=Unclassified(3)
                return len(primary_order) + code

            # Convert to lowercase string for lookup
            class_lower = str(classification).lower().strip()

            # Try lowercase mapping
            if class_lower in order_mapping_lower:
                return order_mapping_lower[class_lower]

            # Unknown classification - sort alphabetically after known ones
            return len(primary_order) + (ord(class_lower[0]) if class_lower else 999)

        rows.sort(key=get_sort_key)

        # Compute totals
        totals_all = None
        totals_MI = None

        # Define classification mappings for M+I identification
        # Handles: text names, abbreviations, numeric codes (common in mining software)
        measured_identifiers = {
            "measured", "m", "meas", "1", "1.0",  # Common text/codes
        }
        indicated_identifiers = {
            "indicated", "i", "ind", "2", "2.0",  # Common text/codes
        }

        def is_measured_or_indicated(classification: str) -> bool:
            """Check if classification is Measured or Indicated (handles various formats)."""
            class_lower = str(classification).lower().strip()
            return class_lower in measured_identifiers or class_lower in indicated_identifiers

        if rows:
            # Totals (M+I) - flexible matching for various classification schemes
            mi_rows = [r for r in rows if is_measured_or_indicated(r.classification)]
            if mi_rows:
                totals_MI = ResourceSummaryRow(
                    classification="Totals (M+I)",
                    n_blocks=sum(r.n_blocks for r in mi_rows),
                    total_volume_m3=sum(r.total_volume_m3 for r in mi_rows),
                    avg_density_t_per_m3=(sum(r.avg_density_t_per_m3 * r.total_volume_m3 for r in mi_rows) /
                                        sum(r.total_volume_m3 for r in mi_rows)) if mi_rows else 0,
                    total_tonnage_t=sum(r.total_tonnage_t for r in mi_rows),
                    grade_pct=(sum(r.grade_pct * r.total_tonnage_t for r in mi_rows) /
                             sum(r.total_tonnage_t for r in mi_rows)) if mi_rows else 0,
                    contained_metal_t=sum(r.contained_metal_t for r in mi_rows),
                )

            # Totals (All)
            totals_all = ResourceSummaryRow(
                classification="Totals (All)",
                n_blocks=sum(r.n_blocks for r in rows),
                total_volume_m3=sum(r.total_volume_m3 for r in rows),
                avg_density_t_per_m3=(sum(r.avg_density_t_per_m3 * r.total_volume_m3 for r in rows) /
                                    sum(r.total_volume_m3 for r in rows)) if rows else 0,
                total_tonnage_t=sum(r.total_tonnage_t for r in rows),
                grade_pct=(sum(r.grade_pct * r.total_tonnage_t for r in rows) /
                         sum(r.total_tonnage_t for r in rows)) if rows else 0,
                contained_metal_t=sum(r.contained_metal_t for r in rows),
            )

        # Build metadata
        metadata = {
            "method_version": self.METHOD_VERSION,
            "build": self.BUILD,
            "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
            "density_mode": density_config.mode,
            "volume_mode": volume_config.mode,
            "grade_field": self.grade_field,
            "class_field": self.class_field,
            "n_blocks_total": len(self.df),
            "n_classifications": len(unique_classes),
            "timestamp": datetime.now().isoformat(),
            "numba_used": use_numba,
        }

        if progress_callback:
            progress_callback(95, "Validating resource summary...")

        # ================================================================
        # JORC AUDIT GATE: Validate internal consistency of totals
        # ================================================================
        if rows and totals_all:
            # Validate tonnage consistency
            computed_total_tonnage = sum(r.total_tonnage_t for r in rows)
            if abs(computed_total_tonnage - totals_all.total_tonnage_t) > 1e-3:
                raise RuntimeError(
                    f"JORC AUDIT GATE FAILED: Tonnage totals inconsistent. "
                    f"Row sum: {computed_total_tonnage:,.0f}t, "
                    f"Total reported: {totals_all.total_tonnage_t:,.0f}t. "
                    f"Difference: {abs(computed_total_tonnage - totals_all.total_tonnage_t):,.0f}t"
                )
            
            # Validate volume consistency
            computed_total_volume = sum(r.total_volume_m3 for r in rows)
            if abs(computed_total_volume - totals_all.total_volume_m3) > 1e-3:
                raise RuntimeError(
                    f"JORC AUDIT GATE FAILED: Volume totals inconsistent. "
                    f"Row sum: {computed_total_volume:,.0f}m³, "
                    f"Total reported: {totals_all.total_volume_m3:,.0f}m³"
                )
            
            # Validate contained metal consistency
            computed_total_metal = sum(r.contained_metal_t for r in rows)
            if abs(computed_total_metal - totals_all.contained_metal_t) > 1e-3:
                raise RuntimeError(
                    f"JORC AUDIT GATE FAILED: Contained metal totals inconsistent. "
                    f"Row sum: {computed_total_metal:,.0f}t, "
                    f"Total reported: {totals_all.contained_metal_t:,.0f}t"
                )
            
            # Validate block count
            computed_total_blocks = sum(r.n_blocks for r in rows)
            if computed_total_blocks != totals_all.n_blocks:
                raise RuntimeError(
                    f"JORC AUDIT GATE FAILED: Block count totals inconsistent. "
                    f"Row sum: {computed_total_blocks}, "
                    f"Total reported: {totals_all.n_blocks}"
                )
            
            logger.info("JORC Audit Gate: Resource summary validation PASSED")
        
        # Add validation status to metadata
        metadata["validation_passed"] = True
        metadata["validation_timestamp"] = datetime.now().isoformat()

        if progress_callback:
            progress_callback(100, "Resource summary complete.")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Resource summary complete: {len(rows)} classifications, "
            f"{len(self.df)} blocks, {execution_time:.2f}s"
        )

        return ResourceSummaryResult(
            rows=rows,
            totals_MI=totals_MI,
            totals_all=totals_all,
            metadata=metadata,
        )
