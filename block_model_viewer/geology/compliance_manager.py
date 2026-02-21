"""
ComplianceManager - JORC/SAMREC Compliant Validation.

Computes spatial misfit between drillhole data and model surfaces,
generates audit reports, and validates compliance thresholds.

GeoX Invariant Compliance:
- Spatial misfit converted to real-world meters
- P90 threshold checks for resource classification
- Full audit trail with visualization support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

# Import ChronosEngine for feature name constant (avoid circular import)
if TYPE_CHECKING:
    from .chronos_engine import ChronosEngine

logger = logging.getLogger("GeoX_Compliance")

# Default feature name - should match ChronosEngine.FEATURE_NAME
# This is duplicated here to avoid import issues, but ChronosEngine is authoritative
DEFAULT_FEATURE_NAME = "Stratigraphy"


@dataclass
class JORCThresholds:
    """
    Configurable JORC/SAMREC compliance thresholds.

    These thresholds define the accuracy requirements for different
    resource classification categories. Values are in meters.

    Default values are based on typical JORC (2012) requirements:
    - Measured: Highest confidence, requires excellent model fit
    - Indicated: Good confidence, allows moderate deviation
    - Inferred: Lower confidence, allows larger deviation

    Users can customize these based on deposit type, data density,
    and project-specific requirements (as allowed by JORC/SAMREC guidelines).
    """
    # Measured category thresholds
    measured_p90: float = 2.0  # P90 error must be below this
    measured_mean: float = 1.0  # Mean residual must be below this

    # Indicated category thresholds
    indicated_p90: float = 5.0
    indicated_mean: float = 2.0

    # Inferred category thresholds
    inferred_p90: float = 10.0
    inferred_mean: float = 5.0

    def get_category_thresholds(self, category: str) -> Dict[str, float]:
        """Get P90 and mean thresholds for a category."""
        thresholds = {
            "Measured": {"p90": self.measured_p90, "mean": self.measured_mean},
            "Indicated": {"p90": self.indicated_p90, "mean": self.indicated_mean},
            "Inferred": {"p90": self.inferred_p90, "mean": self.inferred_mean},
        }
        return thresholds.get(category, {"p90": float('inf'), "mean": float('inf')})

    def classify(self, p90_error: float, mean_residual: float) -> str:
        """Classify based on P90 and mean residual values."""
        if p90_error < self.measured_p90 and mean_residual < self.measured_mean:
            return "Measured"
        elif p90_error < self.indicated_p90 and mean_residual < self.indicated_mean:
            return "Indicated"
        elif p90_error < self.inferred_p90:
            return "Inferred"
        else:
            return "Unclassified"

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary for export."""
        return {
            "measured_p90": self.measured_p90,
            "measured_mean": self.measured_mean,
            "indicated_p90": self.indicated_p90,
            "indicated_mean": self.indicated_mean,
            "inferred_p90": self.inferred_p90,
            "inferred_mean": self.inferred_mean,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'JORCThresholds':
        """Create from dictionary."""
        return cls(
            measured_p90=data.get("measured_p90", 2.0),
            measured_mean=data.get("measured_mean", 1.0),
            indicated_p90=data.get("indicated_p90", 5.0),
            indicated_mean=data.get("indicated_mean", 2.0),
            inferred_p90=data.get("inferred_p90", 10.0),
            inferred_mean=data.get("inferred_mean", 5.0),
        )


# Global default thresholds instance
DEFAULT_JORC_THRESHOLDS = JORCThresholds()


@dataclass
class AuditReport:
    """
    JORC/SAMREC compliant validation data.

    This report is the primary output for auditors reviewing
    geological model quality for Mineral Resource statements.

    Enhanced with optional references to additional validation reports
    for comprehensive auditing.
    """
    mean_residual: float  # Mean error in meters
    p90_error: float  # 90th percentile error in meters
    total_contacts: int  # Number of contact points validated
    status: str  # 'Acceptable', 'Needs Review', 'Critical Failure'
    misfit_data: pd.DataFrame  # Spatial coordinates + error magnitude
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional audit metadata
    model_engine: str = "LoopStructural"
    validation_type: str = "contact_misfit"

    # NEW: Optional extended validation summaries
    # These are lightweight summaries - full reports are in ModelResult
    contact_deviation_status: Optional[str] = None  # 'ACCEPTABLE', 'NEEDS_REVIEW', 'CRITICAL'
    continuity_status: Optional[str] = None  # 'ACCEPTABLE', 'NEEDS_REVIEW', 'CRITICAL'
    stratigraphy_status: Optional[str] = None  # 'ACCEPT', 'REVIEW', 'REJECT'
    resolution_status: Optional[str] = None  # 'ADEQUATE', 'NEEDS_REVIEW'

    # Configurable JORC thresholds
    thresholds: JORCThresholds = field(default_factory=lambda: DEFAULT_JORC_THRESHOLDS)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        result = {
            "mean_residual": self.mean_residual,
            "p90_error": self.p90_error,
            "total_contacts": self.total_contacts,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "model_engine": self.model_engine,
            "validation_type": self.validation_type,
            "misfit_summary": {
                "min": float(self.misfit_data['residual_m'].min()) if len(self.misfit_data) > 0 else 0,
                "max": float(self.misfit_data['residual_m'].max()) if len(self.misfit_data) > 0 else 0,
                "median": float(self.misfit_data['residual_m'].median()) if len(self.misfit_data) > 0 else 0,
            }
        }
        # Add extended validation summaries if present
        if self.contact_deviation_status:
            result["contact_deviation_status"] = self.contact_deviation_status
        if self.continuity_status:
            result["continuity_status"] = self.continuity_status
        if self.stratigraphy_status:
            result["stratigraphy_status"] = self.stratigraphy_status
        if self.resolution_status:
            result["resolution_status"] = self.resolution_status
        return result

    @property
    def all_validations_pass(self) -> bool:
        """Check if all extended validations pass (if present)."""
        passing_statuses = {'ACCEPTABLE', 'ACCEPT', 'ADEQUATE', None}
        return (
            self.contact_deviation_status in passing_statuses and
            self.continuity_status in passing_statuses and
            self.stratigraphy_status in {'ACCEPT', None} and
            self.resolution_status in {'ADEQUATE', None}
        )
    
    @property
    def is_jorc_compliant(self) -> bool:
        """Check if model meets JORC requirements for Measured Resources."""
        return (
            self.p90_error < self.thresholds.measured_p90 and
            self.mean_residual < self.thresholds.measured_mean and
            self.status == "Acceptable"
        )

    @property
    def classification_recommendation(self) -> str:
        """Recommend resource classification based on misfit metrics and thresholds."""
        return self.thresholds.classify(self.p90_error, self.mean_residual)


class ComplianceManager:
    """
    Computes spatial misfit between drillholes and model surfaces.
    
    This is the critical QA/QC gate for geological models under
    JORC (2012) and SAMREC codes.
    
    Key Features:
    - Converts scalar misfit to real-world meters using gradient magnitude
    - Generates spatial misfit data for 3D visualization
    - Produces audit reports with classification recommendations
    
    Usage:
        report = ComplianceManager.generate_misfit_report(model, scaled_data, scaler)
        if report.is_jorc_compliant:
            print(f"Model passes for {report.classification_recommendation}")
    """
    
    @staticmethod
    def generate_misfit_report(
        model: Any,  # LoopStructural GeologicalModel
        scaled_data: pd.DataFrame,
        scaler: Any,  # sklearn scaler with inverse_transform
        feature_name: str = DEFAULT_FEATURE_NAME
    ) -> AuditReport:
        """
        Generate JORC/SAMREC compliant misfit report.
        
        The key insight is that scalar field misfit must be converted
        to real-world meters using the gradient magnitude. This gives
        a physically meaningful error metric for auditors.
        
        Args:
            model: Solved GeologicalModel with stratigraphic feature
            scaled_data: DataFrame with X, Y, Z (scaled), val, formation columns
            scaler: Scaler object used to transform coordinates
            feature_name: Name of the stratigraphic feature (default 'Main_Sequence')
            
        Returns:
            AuditReport with spatial misfit data and compliance status
        """
        try:
            # 1. Get predicted scalar values from the model at drillhole locations
            points = scaled_data[['X', 'Y', 'Z']].values.astype(float)
            actual_isovalues = scaled_data['val'].values.astype(float)
            
            # FATAL ERROR PREVENTION: Ensure 2D array [N, 3]
            # LoopStructural requires Nx3 array for evaluate_value
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            if points.shape[1] != 3:
                raise ValueError(f"Points array must have shape (N, 3), got {points.shape}")
            
            logger.debug(f"Evaluating {len(points)} points, shape: {points.shape}")
            
            # Evaluate the main stratigraphic feature
            try:
                strat_feature = model[feature_name]
            except KeyError:
                logger.error(f"Feature '{feature_name}' not found in model")
                # Try alternative feature names for backwards compatibility
                alt_names = ['Stratigraphy', 'Main_Sequence', 'stratigraphy']
                for alt in alt_names:
                    if alt != feature_name:
                        try:
                            strat_feature = model[alt]
                            logger.warning(f"Using alternative feature name '{alt}' instead of '{feature_name}'")
                            break
                        except KeyError:
                            continue
                else:
                    raise ValueError(f"Feature '{feature_name}' not found in model")
            
            predicted_isovalues = strat_feature.evaluate_value(points)
            
            # 2. Convert scalar error to meters (Residuals)
            # =================================================================
            # CRITICAL FIX: Convert scaled-space gradient to world-space
            # =================================================================
            # The gradient is evaluated in [0,1] scaled space.  To get a
            # physically meaningful residual in METERS we must transform the
            # gradient back to world coordinates.
            #
            # Chain rule:  ∂f/∂x_world = ∂f/∂x_scaled * ∂x_scaled/∂x_world
            #            = ∂f/∂x_scaled * scale_[i]
            #
            # For MinMaxScaler, scale_[i] = 1 / (x_max - x_min).
            #
            # residual_m = scalar_misfit / |∇f_world|
            # =================================================================
            grad_scaled = strat_feature.evaluate_gradient(points)
            
            # Transform gradient to world space using scaler's scale factors
            scale_factors = scaler.scale_   # shape (3,) = 1/(max-min) per axis
            grad_world = grad_scaled * scale_factors[np.newaxis, :]
            grad_mag_world = np.linalg.norm(grad_world, axis=1)
            
            # Avoid division by zero
            grad_mag_world = np.where(grad_mag_world < 1e-15, 1e-15, grad_mag_world)
            
            # Residual (m) = Scalar Misfit / World-Space Gradient
            scalar_misfit = np.abs(actual_isovalues - predicted_isovalues)
            residuals_m = scalar_misfit / grad_mag_world
            
            # Sanity clamp: residuals > 10× model extent are numerical artefacts
            extent_diag = np.sqrt(
                sum((1.0 / s) ** 2 for s in scale_factors if s > 0)
            )
            max_sane_residual = extent_diag * 10.0
            clamped = np.sum(residuals_m > max_sane_residual)
            if clamped > 0:
                logger.warning(
                    f"Clamped {clamped} residuals exceeding {max_sane_residual:.0f}m "
                    f"(likely numerical artefacts near model boundaries)"
                )
                residuals_m = np.clip(residuals_m, 0, max_sane_residual)
            
            # 3. Compile Spatial Misfit Data for 3D View
            # Reset index to avoid reindexing errors
            scaled_data_reset = scaled_data.reset_index(drop=True)
            
            world_coords = scaler.inverse_transform(points)
            misfit_df = pd.DataFrame(world_coords, columns=['X', 'Y', 'Z'])
            misfit_df['residual_m'] = residuals_m
            misfit_df['scalar_misfit'] = scalar_misfit
            
            # Add formation if available (use values to avoid index alignment)
            if 'formation' in scaled_data_reset.columns:
                misfit_df['unit'] = scaled_data_reset['formation'].values
            elif 'feature_name' in scaled_data_reset.columns:
                misfit_df['unit'] = scaled_data_reset['feature_name'].values
            else:
                misfit_df['unit'] = 'Unknown'
            
            # 4. Statistical Summary for Auditor
            mean_err = float(np.mean(residuals_m))
            p90 = float(np.percentile(residuals_m, 90))
            
            # Determine compliance status
            status = "Acceptable"
            if p90 > 5.0:
                status = "Needs Review"
            if mean_err > 2.0:
                status = "Critical Failure"
            
            report = AuditReport(
                mean_residual=mean_err,
                p90_error=p90,
                total_contacts=len(scaled_data),
                status=status,
                misfit_data=misfit_df,
            )
            
            logger.info(
                f"Compliance Report: Mean={mean_err:.2f}m, P90={p90:.2f}m, "
                f"Status={status}, Classification={report.classification_recommendation}"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate misfit report: {e}")
            # Return a minimal error report
            return AuditReport(
                mean_residual=-1,
                p90_error=-1,
                total_contacts=len(scaled_data),
                status="Error",
                misfit_data=pd.DataFrame(columns=['X', 'Y', 'Z', 'residual_m', 'unit']),
            )
    
    @staticmethod
    def validate_model_for_classification(
        report: AuditReport,
        target_class: str = "Indicated"
    ) -> Dict[str, Any]:
        """
        Validate if model meets requirements for a specific classification.
        
        JORC/SAMREC Classification Requirements:
        - Measured: P90 < 2.0m, Mean < 1.0m
        - Indicated: P90 < 5.0m, Mean < 2.0m
        - Inferred: P90 < 10.0m
        
        Args:
            report: AuditReport from generate_misfit_report
            target_class: Target classification level
            
        Returns:
            Dict with validation results and recommendations
        """
        thresholds = {
            "Measured": {"p90": 2.0, "mean": 1.0},
            "Indicated": {"p90": 5.0, "mean": 2.0},
            "Inferred": {"p90": 10.0, "mean": 5.0},
        }
        
        if target_class not in thresholds:
            target_class = "Indicated"
        
        thresh = thresholds[target_class]
        
        passes_p90 = report.p90_error < thresh["p90"]
        passes_mean = report.mean_residual < thresh["mean"]
        passes = passes_p90 and passes_mean
        
        return {
            "target_class": target_class,
            "passes": passes,
            "p90_threshold": thresh["p90"],
            "p90_actual": report.p90_error,
            "p90_passes": passes_p90,
            "mean_threshold": thresh["mean"],
            "mean_actual": report.mean_residual,
            "mean_passes": passes_mean,
            "recommendation": report.classification_recommendation,
            "jorc_compliant": report.is_jorc_compliant,
        }
    
    @staticmethod
    def export_audit_report(
        report: AuditReport,
        filepath: str,
        include_spatial_data: bool = True
    ) -> None:
        """
        Export audit report to files for regulatory submission.
        
        Args:
            report: AuditReport to export
            filepath: Base path for export (without extension)
            include_spatial_data: If True, also exports CSV of misfit locations
        """
        import json
        
        # Export JSON summary
        json_path = f"{filepath}_audit.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Exported audit summary to {json_path}")
        
        # Export spatial data CSV
        if include_spatial_data and len(report.misfit_data) > 0:
            csv_path = f"{filepath}_misfit_data.csv"
            report.misfit_data.to_csv(csv_path, index=False)
            logger.info(f"Exported spatial misfit data to {csv_path}")

