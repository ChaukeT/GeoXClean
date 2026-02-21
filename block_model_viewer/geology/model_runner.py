"""
GeologicalModelRunner - Unified CP-Grade Pipeline for JORC/SAMREC Compliance.

This module provides a single, authoritative entry point for geological modeling
that encapsulates:
- Coordinate stabilization
- FDI/PLI interpolation
- Compliance auditing
- Mesh smoothing and repair
- Audit trail generation

GeoX Invariant Compliance:
- All operations are atomic and logged
- Provenance metadata for every output
- JORC/SAMREC audit gates
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .chronos_engine import (
    ChronosEngine,
    validate_stratigraphy_sequence,
    analyze_unit_thickness,
    StratigraphicValidationResult,
    ResolutionAnalysis,
)
from .compliance_manager import ComplianceManager, AuditReport
from .contact_deviation_report import ContactDeviationAnalyzer, ContactDeviationReport
from .mesh_validator import validate_all_units_continuity, ContinuityValidationReport
from .gradient_estimator import (
    compute_contact_gradients,
    ContactGradient,
    GradientEstimationReport,
)

logger = logging.getLogger(__name__)

# Check optional dependencies
try:
    import pyvista as pv
    PV_AVAILABLE = True
except ImportError:
    PV_AVAILABLE = False
    pv = None


def validate_and_filter_drillhole_data(
    df: pd.DataFrame,
    extent: Optional[Dict[str, float]] = None,
    z_score_threshold: float = 3.0,
    min_valid_coordinate: float = 100.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    CRITICAL FIX: Filter drillhole data to remove outliers before modeling.
    
    This prevents the renderer coordinate shift from being skewed by invalid
    points (e.g., 0,0,0 collars or coordinates far outside the main cluster).
    
    The Bug (Pre-Fix):
    - If drillhole_df contains even one outlier point (e.g., a collar at 0,0,0
      while others are at 500000, 7000000), the StandardScaler will stretch the
      model space to span hundreds of kilometers.
    - The resulting model surfaces become massive compared to valid drillholes.
    - The renderer's global_shift is locked to the wrong center.
    
    Args:
        df: Input DataFrame with X, Y, Z columns
        extent: Optional extent dict to use as reference bounds. If provided,
                points outside extent + 20% buffer are filtered.
        z_score_threshold: Points with Z-score > this value are flagged as outliers
        min_valid_coordinate: Coordinates below this absolute value are flagged
                              (catches 0,0,0 dummy coordinates)
    
    Returns:
        Tuple of (filtered_df, warnings_list)
    """
    if df is None or len(df) == 0:
        return df, ["Empty DataFrame provided"]
    
    warnings = []
    original_count = len(df)
    
    # Ensure required columns exist
    required_cols = ['X', 'Y', 'Z']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return df, [f"Missing required columns: {missing}"]
    
    # Make a copy to avoid modifying original
    df_filtered = df.copy()
    
    # =========================================================================
    # FILTER 1: Remove rows with null/NaN coordinates
    # =========================================================================
    null_mask = df_filtered[['X', 'Y', 'Z']].isnull().any(axis=1)
    null_count = null_mask.sum()
    if null_count > 0:
        df_filtered = df_filtered[~null_mask]
        warnings.append(f"Removed {null_count} rows with null coordinates")
    
    if len(df_filtered) == 0:
        return df_filtered, warnings + ["All rows removed after null filter"]
    
    # =========================================================================
    # FILTER 2: Remove rows with suspiciously small coordinates (0,0,0 trap)
    # =========================================================================
    # In UTM systems, valid coordinates are typically > 100m from origin
    coords = df_filtered[['X', 'Y', 'Z']].values
    
    # Check for points where X and Y are both very small (likely dummy data)
    small_xy_mask = (np.abs(coords[:, 0]) < min_valid_coordinate) & \
                    (np.abs(coords[:, 1]) < min_valid_coordinate)
    small_count = small_xy_mask.sum()
    
    if small_count > 0 and small_count < len(df_filtered) * 0.5:  # Only filter if < 50% of data
        df_filtered = df_filtered[~small_xy_mask]
        warnings.append(
            f"Removed {small_count} rows with suspiciously small coordinates "
            f"(X and Y both < {min_valid_coordinate}m - likely dummy/placeholder data)"
        )
    
    if len(df_filtered) == 0:
        return df_filtered, warnings + ["All rows removed after small coordinate filter"]
    
    # =========================================================================
    # FILTER 3: Remove statistical outliers using Z-score
    # =========================================================================
    coords = df_filtered[['X', 'Y', 'Z']].values
    
    # Calculate Z-scores for each dimension
    means = np.mean(coords, axis=0)
    stds = np.std(coords, axis=0)
    
    # Avoid division by zero
    stds = np.where(stds > 0, stds, 1.0)
    
    z_scores = np.abs((coords - means) / stds)
    max_z_scores = np.max(z_scores, axis=1)
    
    outlier_mask = max_z_scores > z_score_threshold
    outlier_count = outlier_mask.sum()
    
    if outlier_count > 0 and outlier_count < len(df_filtered) * 0.1:  # Only filter if < 10% of data
        # Log the outliers for debugging
        outlier_coords = coords[outlier_mask]
        logger.warning(
            f"Detected {outlier_count} statistical outliers (Z-score > {z_score_threshold}):\n"
            f"  Sample outlier coordinates: {outlier_coords[:3]}"
        )
        df_filtered = df_filtered[~outlier_mask]
        warnings.append(
            f"Removed {outlier_count} statistical outliers "
            f"(Z-score > {z_score_threshold})"
        )
    
    if len(df_filtered) == 0:
        return df_filtered, warnings + ["All rows removed after outlier filter"]
    
    # =========================================================================
    # FILTER 4: Remove points outside extent bounds (if provided)
    # =========================================================================
    if extent is not None:
        buffer = 0.2  # 20% buffer outside extent
        x_range = extent.get('xmax', 0) - extent.get('xmin', 0)
        y_range = extent.get('ymax', 0) - extent.get('ymin', 0)
        z_range = extent.get('zmax', 0) - extent.get('zmin', 0)
        
        x_buffer = x_range * buffer
        y_buffer = y_range * buffer
        z_buffer = z_range * buffer
        
        coords = df_filtered[['X', 'Y', 'Z']].values
        in_bounds_mask = (
            (coords[:, 0] >= extent.get('xmin', -np.inf) - x_buffer) &
            (coords[:, 0] <= extent.get('xmax', np.inf) + x_buffer) &
            (coords[:, 1] >= extent.get('ymin', -np.inf) - y_buffer) &
            (coords[:, 1] <= extent.get('ymax', np.inf) + y_buffer) &
            (coords[:, 2] >= extent.get('zmin', -np.inf) - z_buffer) &
            (coords[:, 2] <= extent.get('zmax', np.inf) + z_buffer)
        )
        
        out_of_bounds_count = (~in_bounds_mask).sum()
        if out_of_bounds_count > 0:
            df_filtered = df_filtered[in_bounds_mask]
            warnings.append(
                f"Removed {out_of_bounds_count} rows outside model extent + 20% buffer"
            )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    final_count = len(df_filtered)
    removed_count = original_count - final_count
    
    if removed_count > 0:
        logger.info(
            f"Data validation: {original_count} → {final_count} rows "
            f"({removed_count} removed, {100*removed_count/original_count:.1f}%)"
        )
        
        # Log the final coordinate bounds
        coords = df_filtered[['X', 'Y', 'Z']].values
        logger.info(
            f"Filtered data bounds: "
            f"X=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], "
            f"Y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}], "
            f"Z=[{coords[:, 2].min():.1f}, {coords[:, 2].max():.1f}]"
        )
    else:
        logger.info("Data validation: No outliers detected, all rows retained")
    
    return df_filtered, warnings


@dataclass
class ModelResult:
    """Complete result package from a geological model run."""
    surfaces: List[Dict[str, Any]]
    solids: List[Dict[str, Any]]
    audit_report: Optional[AuditReport]
    provenance: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    # INDUSTRY-STANDARD: Single unified mesh with Formation_ID (eliminates Z-fighting)
    unified_mesh: Optional[Dict[str, Any]] = None
    # Enhanced validation reports
    contact_deviation_report: Optional[ContactDeviationReport] = None
    continuity_report: Optional[ContinuityValidationReport] = None
    stratigraphy_validation: Optional[StratigraphicValidationResult] = None
    resolution_analysis: Optional[ResolutionAnalysis] = None
    # Gradient estimation tracking (for QC and audit)
    computed_gradients: Optional[List[ContactGradient]] = None
    gradient_source: str = "unknown"  # "user_provided", "computed", "synthetic"

    @property
    def total_volume_m3(self) -> float:
        """Total volume of all solids in cubic meters."""
        return sum(s.get('volume_m3', 0) or 0 for s in self.solids)

    @property
    def is_jorc_compliant(self) -> bool:
        """Check if model meets JORC requirements."""
        if self.audit_report is None:
            return False
        return self.audit_report.is_jorc_compliant

    @property
    def validation_summary(self) -> Dict[str, str]:
        """Summary of all validation statuses."""
        summary = {}
        if self.audit_report:
            summary['misfit'] = self.audit_report.status
        if self.contact_deviation_report:
            summary['contact_deviation'] = self.contact_deviation_report.status
        if self.continuity_report:
            summary['continuity'] = self.continuity_report.status
        if self.stratigraphy_validation:
            summary['stratigraphy'] = self.stratigraphy_validation.recommendation
        if self.resolution_analysis:
            summary['resolution'] = 'ADEQUATE' if self.resolution_analysis.resolution_adequate else 'NEEDS_REVIEW'
        return summary


class GeologicalModelRunner:
    """
    THE AUTHORITATIVE PIPELINE for CP-Grade Geological Modeling.
    
    Runs a full model build, compliance audit, and mesh repair in one atomic call.
    Produces industry-standard outputs suitable for JORC/SAMREC reporting.
    
    Features:
    - High-resolution grid (80³ default for CP-grade)
    - Taubin mesh smoothing to remove voxel artifacts
    - Automatic compliance validation
    - Full audit trail with provenance
    
    Usage:
        runner = GeologicalModelRunner(extent, resolution=80)
        result = runner.run_full_stack(drillhole_df, chronology, faults)
        
        # Export for auditor
        runner.export_audit_package(result, "model_exports/")
    """
    
    def __init__(
        self, 
        extent: Dict[str, float], 
        resolution: int = 80,
        cgw: float = 0.005,
        smoothing_iterations: int = 50,
        smoothing_passband: float = 0.05,
        boundary_padding: float = 0.1
    ):
        """
        Initialize the unified pipeline.
        
        Args:
            extent: Bounding box dict with xmin, xmax, ymin, ymax, zmin, zmax
            resolution: Grid resolution (80+ recommended for CP-grade)
            cgw: Regularization weight.
                 0.005 (default) → tight fit, surfaces honour contacts.
                 0.001 → very tight, may oscillate with sparse data.
                 0.01  → moderate smoothing.
                 0.1   → heavy smoothing — rarely appropriate for resources.
            smoothing_iterations: Taubin smoothing iterations (50 = CP-grade standard)
            smoothing_passband: Smoothing strength (0.05 = removes blocky look)
            boundary_padding: Padding fraction to prevent edge tearing.
                0.1 (default, 10%) prevents isosurfaces from being clipped
                at model boundaries, which is the primary cause of
                discontinuous surfaces.
        """
        self.extent = extent
        self.resolution = resolution
        self.cgw = cgw
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_passband = smoothing_passband
        self.boundary_padding = boundary_padding
        
        # Initialize engine with higher resolution AND boundary padding
        # The boundary_padding prevents the "edge tearing" artifact
        self.engine = ChronosEngine(
            extent, 
            resolution=resolution, 
            boundary_padding=boundary_padding
        )
        self.compliance = ComplianceManager()
        
        # Track run metadata
        self._run_timestamp: Optional[datetime] = None
        self._run_duration_seconds: float = 0.0
        
        logger.info(
            f"GeologicalModelRunner initialized: resolution={resolution}³, "
            f"cgw={cgw}, smoothing={smoothing_iterations} iterations, "
            f"boundary_padding={boundary_padding}"
        )
    
    def run_full_stack(
        self,
        contacts_df: pd.DataFrame,
        chronology: List[str],
        orientations_df: Optional[pd.DataFrame] = None,
        faults: Optional[List[Dict[str, Any]]] = None,
        extract_solids: bool = True,
        formation_values: Optional[Dict[str, float]] = None,
        compute_gradients: bool = True,
        allow_synthetic_fallback: bool = True,
    ) -> ModelResult:
        """
        Execute the complete modeling pipeline.

        This is the SINGLE entry point for CP-grade geological modeling.
        It performs:
        1. Coordinate stabilization (UTM → [0,1])
        2. Gradient computation from contact geometry (if enabled)
        3. FDI interpolation solve
        4. SAMREC compliance audit
        5. Surface extraction with Taubin smoothing
        6. Optional solid volume extraction

        Args:
            contacts_df: DataFrame with X, Y, Z, val, formation columns
            chronology: List of unit names from oldest to youngest
            orientations_df: Optional orientation data (gx, gy, gz). If provided,
                these orientations are used directly (overrides compute_gradients).
            faults: Optional list of fault parameter dicts
            extract_solids: Whether to extract 3D solid volumes
            formation_values: Dict mapping formation names to scalar values.
                If None, will be extracted from contacts_df 'formation' and 'val' columns.
            compute_gradients: If True and orientations_df not provided, compute
                gradient vectors from contact point clouds using PCA plane fitting.
                This produces geologically realistic orientations instead of
                assuming flat-lying beds. Default: True.
            allow_synthetic_fallback: If True and gradient computation fails,
                fall back to synthetic horizontal orientations (0,0,1).
                If False, raise an error if no valid orientations available.
                Default: True.

        Returns:
            ModelResult with surfaces, solids, audit report, provenance,
            and gradient_source indicating how orientations were obtained.
        """
        import time
        start_time = time.time()
        self._run_timestamp = datetime.now()
        
        warnings = []
        
        logger.info("=" * 60)
        logger.info("GEOLOGICAL MODEL RUNNER - CP-GRADE PIPELINE")
        logger.info("=" * 60)
        
        # =====================================================================
        # STAGE 0: DATA VALIDATION (CRITICAL FIX FOR COORDINATE MISMATCH BUG)
        # =====================================================================
        # This stage filters out outlier drillholes that would skew the model
        # coordinate system and cause renderer displacement issues.
        # See: FAULT_INTEGRATION_FIX.md for detailed explanation.
        # =====================================================================
        logger.info("STAGE 0: Data validation and outlier filtering...")
        
        # Log original data bounds for debugging
        if len(contacts_df) > 0:
            logger.info(
                f"  Original data bounds: "
                f"X=[{contacts_df['X'].min():.1f}, {contacts_df['X'].max():.1f}], "
                f"Y=[{contacts_df['Y'].min():.1f}, {contacts_df['Y'].max():.1f}], "
                f"Z=[{contacts_df['Z'].min():.1f}, {contacts_df['Z'].max():.1f}]"
            )
        
        # Validate and filter the contact data
        validated_contacts, validation_warnings = validate_and_filter_drillhole_data(
            contacts_df,
            extent=self.extent,
            z_score_threshold=3.0,
            min_valid_coordinate=100.0
        )
        
        # Add any validation warnings to the result
        warnings.extend(validation_warnings)
        
        if len(validated_contacts) == 0:
            raise ValueError(
                "All contact data was filtered out during validation. "
                "Check for invalid coordinates (0,0,0 or extreme outliers)."
            )

        # =====================================================================
        # EXTRACT FORMATION VALUES IF NOT PROVIDED
        # =====================================================================
        # Formation values map formation names to their scalar values.
        # These are critical for:
        # - Contact deviation analysis (boundary isovalue calculation)
        # - Solid extraction (isosurface midpoint calculation)
        # =====================================================================
        if formation_values is None and 'formation' in validated_contacts.columns and 'val' in validated_contacts.columns:
            formation_values = {}
            for formation in chronology:
                # Get the scalar value for this formation from the contacts data
                formation_data = validated_contacts[validated_contacts['formation'] == formation]
                if len(formation_data) > 0:
                    # Use the first val (should be consistent for same formation)
                    formation_values[formation] = float(formation_data['val'].iloc[0])
            logger.info(f"  Extracted formation_values from contacts: {formation_values}")
        
        if len(validated_contacts) < len(contacts_df):
            logger.warning(
                f"DATA VALIDATION: Filtered {len(contacts_df) - len(validated_contacts)} outlier points. "
                f"This is important for correct coordinate alignment with drillholes."
            )
        
        # =====================================================================
        # STAGE 1: Coordinate Stabilization & Gradient Computation
        # =====================================================================
        logger.info("STAGE 1: Coordinate stabilization and gradient computation...")

        # Use validated contacts instead of raw contacts
        scaled_contacts = self.engine.prepare_data(validated_contacts.copy())

        # Track gradient source for audit
        gradient_source = "unknown"
        computed_gradients_list: Optional[List[ContactGradient]] = None

        if orientations_df is not None and len(orientations_df) > 0:
            # User-provided orientations take priority
            scaled_orientations = self.engine.prepare_data(orientations_df.copy())
            gradient_source = "user_provided"
            logger.info(f"  Using {len(scaled_orientations)} user-provided orientations")

        elif compute_gradients:
            # Compute real gradients from contact geometry using multi-strategy pipeline
            logger.info("  Computing gradients: boundary PCA + drillhole sequences + cross-hole...")
            try:
                computed_orientations, computed_gradients_list, gradient_warnings = compute_contact_gradients(
                    contacts_df=validated_contacts,
                    stratigraphy=chronology,
                    min_points_per_formation=3,
                    min_confidence=0.25,
                    formation_values=formation_values,
                    use_drillhole_sequence=True,
                    use_crosshole=True,
                    use_local_knn=False,  # Enable for folded geology
                )
                warnings.extend(gradient_warnings)

                if len(computed_orientations) > 0:
                    scaled_orientations = self.engine.prepare_data(computed_orientations.copy())
                    gradient_source = "computed"
                    logger.info(
                        f"  Computed {len(scaled_orientations)} gradient vectors "
                        f"({len(computed_gradients_list)} boundaries)"
                    )
                    for cg in computed_gradients_list:
                        logger.info(f"    {cg.summary}")
                elif allow_synthetic_fallback:
                    # Gradient computation failed — fall back to synthetic orientations.
                    # Place a vertical gradient (0,0,1) at EVERY contact point so that
                    # the solver at least has dense constraints, even if the dip is wrong.
                    gradient_source = "synthetic"
                    warnings.append(
                        "ORIENTATION WARNING: Could not compute gradients from contacts. "
                        "Using synthetic horizontal orientations (0,0,1) at every contact. "
                        "Model assumes FLAT-LYING beds — review surfaces against drillholes. "
                        "To fix: ensure each formation has ≥ 3 contacts with 'formation' column."
                    )
                    logger.warning(
                        "Gradient computation yielded no results — "
                        "falling back to synthetic (0,0,1) at each contact"
                    )
                    # Build orientations at every contact location
                    synth = validated_contacts[['X', 'Y', 'Z']].copy()
                    synth['gx'] = 0.0
                    synth['gy'] = 0.0
                    synth['gz'] = 1.0
                    scaled_orientations = self.engine.prepare_data(synth)
                else:
                    raise ValueError(
                        "No orientations provided and gradient computation failed. "
                        "Either provide orientations_df or set allow_synthetic_fallback=True."
                    )
            except Exception as e:
                if allow_synthetic_fallback:
                    gradient_source = "synthetic"
                    warnings.append(f"Gradient computation error: {e}. Using synthetic fallback.")
                    logger.error(f"Gradient computation failed: {e}")
                    synth = validated_contacts[['X', 'Y', 'Z']].copy()
                    synth['gx'] = 0.0
                    synth['gy'] = 0.0
                    synth['gz'] = 1.0
                    scaled_orientations = self.engine.prepare_data(synth)
                else:
                    raise
        else:
            # compute_gradients=False, use synthetic
            if allow_synthetic_fallback:
                gradient_source = "synthetic"
                warnings.append(
                    "ORIENTATION WARNING: Using synthetic horizontal orientations (0,0,1). "
                    "Consider enabling compute_gradients=True for geologically realistic results."
                )
                logger.warning("Using synthetic orientations (compute_gradients=False)")
                synth = validated_contacts[['X', 'Y', 'Z']].copy()
                synth['gx'] = 0.0
                synth['gy'] = 0.0
                synth['gz'] = 1.0
                scaled_orientations = self.engine.prepare_data(synth)
            else:
                raise ValueError(
                    "No orientations provided and synthetic fallback disabled. "
                    "Provide orientations_df or enable compute_gradients."
                )

        logger.info(f"  Contacts: {len(scaled_contacts)} points")
        logger.info(f"  Orientations: {len(scaled_orientations)} points (source: {gradient_source})")
        
        # =====================================================================
        # STAGE 2: FDI Mathematical Solve
        # =====================================================================
        logger.info("STAGE 2: FDI interpolation solve...")
        
        self.engine.build_model(
            stratigraphy=chronology,
            contacts=scaled_contacts,
            orientations=scaled_orientations,
            faults=faults,
            cgw=self.cgw,
            interpolator_type="FDI"
        )
        
        # =====================================================================
        # STAGE 3: Compliance Audit (SAMREC Gate)
        # =====================================================================
        logger.info("STAGE 3: SAMREC compliance audit...")
        
        audit_report = None
        try:
            # Create clean audit data (avoid duplicate column issue)
            audit_data = pd.DataFrame({
                'X': scaled_contacts['X_s'].values,
                'Y': scaled_contacts['Y_s'].values,
                'Z': scaled_contacts['Z_s'].values,
                'val': scaled_contacts['val'].values,
            })
            if 'formation' in scaled_contacts.columns:
                audit_data['formation'] = scaled_contacts['formation'].values
            
            audit_report = ComplianceManager.generate_misfit_report(
                self.engine.model,
                audit_data,
                self.engine.scaler,
                feature_name=self.engine.FEATURE_NAME
            )
            
            logger.info(f"  Mean residual: {audit_report.mean_residual:.2f}m")
            logger.info(f"  P90 error: {audit_report.p90_error:.2f}m")
            logger.info(f"  Status: {audit_report.status}")
            logger.info(f"  Classification: {audit_report.classification_recommendation}")
            
            if audit_report.status == "Critical Failure":
                warnings.append(
                    f"COMPLIANCE WARNING: Mean residual {audit_report.mean_residual:.2f}m "
                    "exceeds threshold. Review data quality."
                )
                
        except Exception as e:
            logger.error(f"Compliance audit failed: {e}")
            warnings.append(f"Compliance audit failed: {e}")
        
        # =====================================================================
        # STAGE 4: Surface Extraction with Smoothing
        # =====================================================================
        logger.info("STAGE 4: Surface extraction with Taubin smoothing...")
        
        raw_surfaces = self.engine.extract_meshes()
        surfaces = self._smooth_surfaces(raw_surfaces)
        
        logger.info(f"  Extracted {len(surfaces)} smoothed surfaces")
        
        # =====================================================================
        # STAGE 5: Solid Volume Extraction (Optional)
        # =====================================================================
        solids = []
        if extract_solids:
            logger.info("STAGE 5: Solid volume extraction...")
            raw_solids = self.engine.extract_solids(chronology)
            solids = self._smooth_surfaces(raw_solids)  # Also smooth solids
            
            total_vol = sum(s.get('volume_m3', 0) or 0 for s in solids)
            logger.info(f"  Extracted {len(solids)} solids, total volume: {total_vol:,.0f} m³")
        
        # =====================================================================
        # STAGE 5B: UNIFIED PARTITION MESH (INDUSTRY-STANDARD Z-FIGHTING FIX)
        # =====================================================================
        # This is the Leapfrog/Micromine approach: ONE mesh with Formation_ID
        # Eliminates Z-fighting by ensuring no overlapping geometry
        logger.info("STAGE 5B: Extracting unified partition mesh (Z-fighting elimination)...")
        unified_mesh = None
        try:
            unified_mesh = self.engine.extract_unified_geology_mesh(chronology)
            if unified_mesh:
                logger.info(f"  Unified mesh: {unified_mesh['n_units']} formations, no overlap")
        except Exception as e:
            logger.warning(f"Unified mesh extraction failed (falling back to solids): {e}")
            warnings.append(f"Unified mesh extraction failed: {e}")

        # =====================================================================
        # STAGE 5C: CONTACT DEVIATION ANALYSIS (NEW)
        # =====================================================================
        # Compare modelled surfaces to original logged contacts
        logger.info("STAGE 5C: Contact deviation analysis...")
        contact_deviation_report = None
        try:
            analyzer = ContactDeviationAnalyzer(
                model=self.engine.model,
                scaler=self.engine.scaler,
                stratigraphy=chronology,
                feature_name=self.engine.FEATURE_NAME,
                formation_values=formation_values,
            )
            contact_deviation_report = analyzer.compute_deviations(contacts_df)
            logger.info(
                f"  Contact deviation: mean={contact_deviation_report.mean_deviation_m:.2f}m, "
                f"p90={contact_deviation_report.p90_deviation_m:.2f}m, "
                f"status={contact_deviation_report.status}"
            )
            if contact_deviation_report.status == "CRITICAL":
                warnings.append(
                    f"Contact deviation critical: P90={contact_deviation_report.p90_deviation_m:.2f}m"
                )
        except Exception as e:
            logger.warning(f"Contact deviation analysis failed: {e}")
            warnings.append(f"Contact deviation analysis failed: {e}")

        # =====================================================================
        # STAGE 5D: CONTINUITY VALIDATION (NEW)
        # =====================================================================
        # Detect isolated "floating" volumes that may be modelling artifacts
        logger.info("STAGE 5D: Continuity validation (isolated volume detection)...")
        continuity_report = None
        if solids:
            try:
                continuity_report = validate_all_units_continuity(
                    solids=solids,
                    min_artifact_volume_m3=100.0,
                    max_isolation_ratio=0.10,
                    model_extent=self.extent,
                )
                logger.info(
                    f"  Continuity: {len(continuity_report.unit_results)} units, "
                    f"{continuity_report.total_artifact_count} artifacts, "
                    f"status={continuity_report.status}"
                )
                if continuity_report.status == "CRITICAL":
                    warnings.append(
                        f"Continuity critical: {continuity_report.worst_unit} has "
                        f"{continuity_report.worst_isolation_ratio*100:.1f}% isolated"
                    )
            except Exception as e:
                logger.warning(f"Continuity validation failed: {e}")
                warnings.append(f"Continuity validation failed: {e}")

        # =====================================================================
        # STAGE 5E: STRATIGRAPHIC & RESOLUTION VALIDATION
        # =====================================================================
        logger.info("STAGE 5E: Stratigraphic and resolution validation...")
        stratigraphy_validation = None
        resolution_analysis = None
        try:
            stratigraphy_validation = validate_stratigraphy_sequence(
                stratigraphy=chronology,
                contacts_df=contacts_df,
            )
            if stratigraphy_validation.inversions_found > 0:
                logger.warning(
                    f"  Stratigraphy: {stratigraphy_validation.inversions_found} inversions found, "
                    f"recommendation={stratigraphy_validation.recommendation}"
                )
                if stratigraphy_validation.recommendation == "REJECT":
                    warnings.append(
                        f"Stratigraphic ordering issues: {stratigraphy_validation.inversions_found} inversions"
                    )
        except Exception as e:
            logger.warning(f"Stratigraphic validation failed: {e}")

        try:
            resolution_analysis = analyze_unit_thickness(
                contacts_df=contacts_df,
                stratigraphy=chronology,
                resolution=self.resolution,
                extent=self.extent,
            )
            if not resolution_analysis.resolution_adequate:
                logger.warning(
                    f"  Resolution: {len(resolution_analysis.thin_units)} thin units detected, "
                    f"recommend resolution {resolution_analysis.recommended_resolution}"
                )
                warnings.extend(resolution_analysis.warnings)
        except Exception as e:
            logger.warning(f"Resolution analysis failed: {e}")

        # =====================================================================
        # STAGE 6: Compile Results
        # =====================================================================
        self._run_duration_seconds = time.time() - start_time

        provenance = self._build_provenance(
            contacts_df, chronology, faults, warnings
        )

        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE in {self._run_duration_seconds:.1f}s")
        logger.info("=" * 60)

        return ModelResult(
            surfaces=surfaces,
            solids=solids,
            audit_report=audit_report,
            provenance=provenance,
            warnings=warnings,
            unified_mesh=unified_mesh,
            contact_deviation_report=contact_deviation_report,
            continuity_report=continuity_report,
            stratigraphy_validation=stratigraphy_validation,
            resolution_analysis=resolution_analysis,
            computed_gradients=computed_gradients_list,
            gradient_source=gradient_source,
        )
    
    def build_audit_package(
        self,
        drillhole_df: pd.DataFrame,
        chronology: List[str],
        faults: Optional[List[Dict[str, Any]]] = None,
        extract_volume: bool = False  # Deprecated - kept for API compatibility
    ) -> Dict[str, Any]:
        """
        AUTHORITATIVE ENTRY POINT for CP-Grade Geological Modeling.
        
        Runs the full stack and returns a JORC-compliant package ready
        for auditor review AND direct injection into the main renderer.
        This is the SINGLE method you should call for production workflows.
        
        Args:
            drillhole_df: DataFrame with X, Y, Z, val, formation columns
            chronology: List of unit names from oldest to youngest
            faults: Optional list of fault parameter dicts
            extract_volume: Deprecated parameter (ignored)
            
        Returns:
            Dict containing:
            - "surfaces": Smoothed isosurfaces (Taubin polished)
            - "solids": 3D volume meshes with volume_m3 (individual units)
            - "report": AuditReport object
            - "log": Build provenance log
            - "markdown": Pre-generated JORC/SAMREC audit document
        """
        # Run the full pipeline with gradient computation enabled
        result = self.run_full_stack(
            contacts_df=drillhole_df,
            chronology=chronology,
            orientations_df=None,  # Will compute from contact geometry
            faults=faults,
            extract_solids=True,
            compute_gradients=True,  # Compute real gradients from contacts
            allow_synthetic_fallback=True,
        )
        
        # Generate the markdown report
        markdown = self.generate_markdown_report(result)
        
        return {
            "surfaces": result.surfaces,
            "solids": result.solids,
            "report": result.audit_report,
            "log": result.provenance,
            "markdown": markdown,
            "warnings": result.warnings,
            "total_volume_m3": result.total_volume_m3,
            "is_jorc_compliant": result.is_jorc_compliant,
            # INDUSTRY-STANDARD: Single unified mesh with Formation_ID (eliminates Z-fighting)
            "unified_mesh": result.unified_mesh,
            # Enhanced validation reports
            "contact_deviation_report": result.contact_deviation_report,
            "continuity_report": result.continuity_report,
            "stratigraphy_validation": result.stratigraphy_validation,
            "resolution_analysis": result.resolution_analysis,
            "validation_summary": result.validation_summary,
            # Gradient estimation tracking
            "gradient_source": result.gradient_source,
            "computed_gradients": result.computed_gradients,
        }
    
    def generate_markdown_report(self, result: ModelResult) -> str:
        """
        Generate a professional JORC/SAMREC audit document as a Markdown string.
        
        This is the document that gets submitted to mining stock exchanges
        (ASX, JSE, TSX) to prove model reliability.
        
        Args:
            result: ModelResult from run_full_stack() or internal package
            
        Returns:
            Complete Markdown document as string
        """
        report = result.audit_report
        prov = result.provenance
        
        # Build timestamp
        timestamp = prov.get('timestamp', 'Unknown')
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Status emoji
        status_str = "Unknown"
        status_emoji = "⚠️"
        mean_res = 0.0
        p90_err = 0.0
        classification = "Not Evaluated"
        
        if report:
            status_str = report.status
            status_emoji = "✅" if report.status == "Acceptable" else "⚠️"
            mean_res = report.mean_residual
            p90_err = report.p90_error
            classification = report.classification_recommendation
        
        # Formation list
        formations = prov.get('data', {}).get('formations', [])
        formation_list = "\n".join([f"{i+1}. {f}" for i, f in enumerate(formations)])
        
        # Volume table
        volume_rows = []
        for solid in result.solids:
            vol = solid.get('volume_m3', 0) or 0
            name = solid.get('unit_name', solid.get('name', 'Unknown'))
            volume_rows.append(f"| {name} | {vol:,.0f} |")
        volume_table = "\n".join(volume_rows) if volume_rows else "| No solids extracted | - |"
        
        # Warnings list
        warnings_list = "\n".join([f"- ⚠️ {w}" for w in result.warnings]) if result.warnings else "*No warnings*"

        # Contact deviation section
        deviation_section = ""
        if result.contact_deviation_report:
            dev = result.contact_deviation_report
            dev_emoji = "✅" if dev.status == "ACCEPTABLE" else "⚠️"
            deviation_section = f"""
## 1b. Contact Deviation Analysis

| Metric | Value | Threshold |
|--------|-------|-----------|
| **Mean Deviation** | {dev.mean_deviation_m:.2f} m | < 2.0 m |
| **P90 Deviation** | {dev.p90_deviation_m:.2f} m | < 5.0 m |
| **Max Deviation** | {dev.max_deviation_m:.2f} m | |
| **Contacts Evaluated** | {dev.contacts_evaluated} / {dev.total_contacts} | |
| **Status** | {dev_emoji} {dev.status} | |

"""

        # Continuity section
        continuity_section = ""
        if result.continuity_report:
            cont = result.continuity_report
            cont_emoji = "✅" if cont.status == "ACCEPTABLE" else "⚠️"
            continuity_section = f"""
## 4b. Unit Continuity Analysis

| Metric | Value |
|--------|-------|
| **Units Analyzed** | {len(cont.unit_results)} |
| **Total Artifacts Detected** | {cont.total_artifact_count} |
| **Worst Isolation** | {cont.worst_unit or 'N/A'} ({cont.worst_isolation_ratio*100:.1f}%) |
| **Status** | {cont_emoji} {cont.status} |

"""

        # Resolution section
        resolution_section = ""
        if result.resolution_analysis:
            res = result.resolution_analysis
            res_emoji = "✅" if res.resolution_adequate else "⚠️"
            thin_units_list = ", ".join([t.unit_name for t in res.thin_units]) if res.thin_units else "None"
            resolution_section = f"""
## 2b. Resolution Analysis

| Metric | Value |
|--------|-------|
| **Voxel Size** | {res.voxel_size_m:.2f} m |
| **Min Unit Thickness** | {res.min_unit_thickness_m:.2f} m |
| **Thin Units** | {thin_units_list} |
| **Current Resolution** | {res.current_resolution}³ |
| **Recommended Resolution** | {res.recommended_resolution}³ |
| **Status** | {res_emoji} {'ADEQUATE' if res.resolution_adequate else 'NEEDS_REVIEW'} |

"""

        # Stratigraphy validation section
        strat_section = ""
        if result.stratigraphy_validation:
            strat = result.stratigraphy_validation
            strat_emoji = "✅" if strat.recommendation == "ACCEPT" else "⚠️"
            strat_section = f"""
## 3b. Stratigraphic Validation

| Metric | Value |
|--------|-------|
| **Drillholes Checked** | {strat.holes_checked} |
| **Inversions Found** | {strat.inversions_found} |
| **Recommendation** | {strat_emoji} {strat.recommendation} |

"""

        md = f"""
# GEOLOGICAL MODEL AUDIT REPORT

**Status:** {status_emoji} {status_str}
**Date:** {timestamp}
**Engine:** LoopStructural 1.6 (FDI Solver)

---

## 1. Misfit Analysis (JORC/SAMREC Compliance)

| Metric | Value | Threshold |
|--------|-------|-----------|
| **Mean Residual** | {mean_res:.2f} meters | < 2.0 m |
| **P90 Error** | {p90_err:.2f} meters | < 5.0 m |
| **Total Contacts Validated** | {prov.get('data', {}).get('n_contacts', 0)} | |
| **Recommendation** | **{classification}** | |

{deviation_section}---

## 2. Model Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | {prov.get('parameters', {}).get('resolution', 80)}³ |
| Smoothing (CGW) | {prov.get('parameters', {}).get('cgw', 0.01)} |
| Taubin Iterations | {prov.get('parameters', {}).get('smoothing_iterations', 50)} |
| Pass Band | {prov.get('parameters', {}).get('smoothing_passband', 0.05)} |
| Interpolator | FDI (Finite Difference) |

{resolution_section}---

## 3. Stratigraphic Sequence (Oldest → Youngest)

{formation_list}

{strat_section}---

## 4. Extracted Volumes

| Unit | Volume (m³) |
|------|-------------|
{volume_table}

**Total Volume:** {result.total_volume_m3:,.0f} m³

{continuity_section}---

## 5. Warnings

{warnings_list}

---

*This report was automatically generated by GeoX GeologicalModelRunner.*
*JORC-Compliant: {result.is_jorc_compliant}*
"""
        return md.strip()
    
    def _smooth_surfaces(self, surfaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Taubin smoothing to remove voxel artifacts.
        
        Taubin smoothing is the industry standard for geological meshes because
        it removes high-frequency noise (voxel edges) while preserving volume
        and overall shape.
        
        Args:
            surfaces: List of surface dicts with 'vertices' and 'faces'
            
        Returns:
            Smoothed surfaces
        """
        if not PV_AVAILABLE:
            logger.warning("PyVista not available - skipping mesh smoothing")
            return surfaces
        
        if self.smoothing_iterations <= 0:
            return surfaces
        
        smoothed = []
        for s in surfaces:
            try:
                verts = s.get('vertices')
                faces = s.get('faces')
                
                if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                    smoothed.append(s)
                    continue
                
                # Convert to PyVista format
                faces_pv = np.hstack([
                    np.full((len(faces), 1), 3, dtype=np.int64),
                    np.asarray(faces, dtype=np.int64)
                ]).flatten()
                
                mesh = pv.PolyData(np.asarray(verts, dtype=np.float64), faces_pv)
                
                # Apply Taubin smoothing (volume-preserving)
                # This is the industry standard for mining meshes
                smooth_mesh = mesh.smooth_taubin(
                    n_iter=self.smoothing_iterations,
                    pass_band=self.smoothing_passband,
                    normalize_coordinates=True
                )
                
                # Update surface with smoothed vertices
                s_copy = s.copy()
                s_copy['vertices'] = np.asarray(smooth_mesh.points, dtype=np.float64)
                s_copy['smoothed'] = True
                s_copy['smoothing_params'] = {
                    'method': 'taubin',
                    'iterations': self.smoothing_iterations,
                    'pass_band': self.smoothing_passband
                }
                
                smoothed.append(s_copy)
                
            except Exception as e:
                logger.warning(f"Smoothing failed for surface '{s.get('name', 'unknown')}': {e}")
                smoothed.append(s)
        
        return smoothed
    
    def _build_provenance(
        self,
        contacts_df: pd.DataFrame,
        chronology: List[str],
        faults: Optional[List[Dict[str, Any]]],
        warnings: List[str]
    ) -> Dict[str, Any]:
        """Build complete provenance metadata for audit trail."""
        
        return {
            "timestamp": self._run_timestamp.isoformat() if self._run_timestamp else None,
            "duration_seconds": self._run_duration_seconds,
            "engine": {
                "name": "ChronosEngine",
                "version": "1.0.0",
                "backend": "LoopStructural",
                "feature_name": self.engine.FEATURE_NAME,
            },
            "parameters": {
                "resolution": self.resolution,
                "cgw": self.cgw,
                "interpolator": "FDI",
                "smoothing_iterations": self.smoothing_iterations,
                "smoothing_passband": self.smoothing_passband,
            },
            "data": {
                "n_contacts": len(contacts_df),
                "n_formations": len(chronology),
                "formations": chronology,
                "n_faults": len(faults) if faults else 0,
            },
            "extent": self.extent,
            "coordinate_transform": self.engine.build_log.coordinate_transform,
            "warnings": warnings,
        }
    
    def export_audit_package(
        self,
        result: ModelResult,
        output_dir: str,
        include_meshes: bool = True
    ) -> str:
        """
        Export a complete audit package for JORC/SAMREC compliance.
        
        Creates the following structure:
        /output_dir/
          ├── manifest.json        (Provenance metadata)
          ├── audit_report.md      (Human-readable JORC report)
          ├── misfit_data.csv      (Spatial residuals for QC)
          └── meshes/              (Watertight mesh files)
               ├── Surface_val_0.00.vtk
               └── Solid_Unit_A.vtk
        
        Args:
            result: ModelResult from run_full_stack()
            output_dir: Output directory path
            include_meshes: Whether to export mesh files
            
        Returns:
            Path to the created audit package directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting audit package to: {output_path}")
        
        # 1. Manifest (JSON provenance)
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(result.provenance, f, indent=2, default=str)
        logger.info(f"  Written: manifest.json")
        
        # 2. Audit Report (Markdown)
        report_path = output_path / "audit_report.md"
        self._write_audit_markdown(result, report_path)
        logger.info(f"  Written: audit_report.md")
        
        # 3. Misfit Data (CSV)
        if result.audit_report and len(result.audit_report.misfit_data) > 0:
            misfit_path = output_path / "misfit_data.csv"
            result.audit_report.misfit_data.to_csv(misfit_path, index=False)
            logger.info(f"  Written: misfit_data.csv")
        
        # 4. Mesh Files (VTK)
        if include_meshes and PV_AVAILABLE:
            mesh_dir = output_path / "meshes"
            mesh_dir.mkdir(exist_ok=True)
            
            for surface in result.surfaces:
                self._export_mesh_vtk(surface, mesh_dir)
            
            for solid in result.solids:
                self._export_mesh_vtk(solid, mesh_dir)
            
            logger.info(f"  Written: {len(result.surfaces)} surfaces + {len(result.solids)} solids")
        
        logger.info(f"Audit package complete: {output_path}")
        return str(output_path)
    
    def _write_audit_markdown(self, result: ModelResult, filepath: Path) -> None:
        """Generate professional JORC/SAMREC audit report in Markdown."""
        
        lines = [
            "# Geological Model Audit Report",
            "",
            f"**Generated:** {result.provenance.get('timestamp', 'Unknown')}",
            f"**Engine:** {result.provenance['engine']['name']} ({result.provenance['engine']['backend']})",
            "",
            "---",
            "",
            "## 1. Model Summary",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Resolution | {result.provenance['parameters']['resolution']}³ |",
            f"| Regularization (CGW) | {result.provenance['parameters']['cgw']} |",
            f"| Interpolator | {result.provenance['parameters']['interpolator']} |",
            f"| Smoothing | Taubin ({result.provenance['parameters']['smoothing_iterations']} iterations) |",
            f"| Contacts | {result.provenance['data']['n_contacts']} |",
            f"| Formations | {result.provenance['data']['n_formations']} |",
            f"| Faults | {result.provenance['data']['n_faults']} |",
            "",
            "### Stratigraphic Sequence (Oldest → Youngest)",
            "",
        ]
        
        for i, unit in enumerate(result.provenance['data']['formations']):
            lines.append(f"{i+1}. {unit}")
        
        lines.extend([
            "",
            "---",
            "",
            "## 2. Compliance Audit (JORC/SAMREC)",
            "",
        ])
        
        if result.audit_report:
            status_emoji = "✅" if result.audit_report.status == "Acceptable" else "⚠️"
            lines.extend([
                f"| Metric | Value | Threshold |",
                f"|--------|-------|-----------|",
                f"| Mean Residual | {result.audit_report.mean_residual:.2f}m | < 2.0m |",
                f"| P90 Error | {result.audit_report.p90_error:.2f}m | < 5.0m |",
                f"| Status | {status_emoji} {result.audit_report.status} | |",
                f"| Classification | **{result.audit_report.classification_recommendation}** | |",
                "",
            ])
        else:
            lines.append("*Audit report not available*\n")
        
        lines.extend([
            "---",
            "",
            "## 3. Extracted Volumes",
            "",
            f"| Unit | Volume (m³) | Cells |",
            f"|------|-------------|-------|",
        ])
        
        for solid in result.solids:
            vol = solid.get('volume_m3', 0) or 0
            cells = solid.get('n_cells', 0) or 0
            lines.append(f"| {solid.get('unit_name', 'Unknown')} | {vol:,.0f} | {cells:,} |")
        
        lines.extend([
            "",
            f"**Total Volume:** {result.total_volume_m3:,.0f} m³",
            "",
            "---",
            "",
            "## 4. Warnings",
            "",
        ])
        
        if result.warnings:
            for w in result.warnings:
                lines.append(f"- ⚠️ {w}")
        else:
            lines.append("*No warnings*")
        
        lines.extend([
            "",
            "---",
            "",
            "*This report was automatically generated by GeoX GeologicalModelRunner.*",
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    def _export_mesh_vtk(self, mesh_data: Dict[str, Any], output_dir: Path) -> None:
        """Export a single mesh to VTK format."""
        try:
            name = mesh_data.get('name', mesh_data.get('unit_name', 'mesh'))
            safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
            
            verts = mesh_data.get('vertices')
            faces = mesh_data.get('faces')
            
            if verts is None or faces is None:
                return
            
            # Convert to PyVista format
            faces_pv = np.hstack([
                np.full((len(faces), 1), 3, dtype=np.int64),
                np.asarray(faces, dtype=np.int64)
            ]).flatten()
            
            mesh = pv.PolyData(np.asarray(verts, dtype=np.float64), faces_pv)
            
            # Save as VTK
            filepath = output_dir / f"{safe_name}.vtk"
            mesh.save(str(filepath))
            
        except Exception as e:
            logger.warning(f"Failed to export mesh '{name}': {e}")

