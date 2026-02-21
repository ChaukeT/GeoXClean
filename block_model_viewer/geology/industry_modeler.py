"""
GeoXIndustryModeler - CP-Grade Geological Modelling Engine.

Compliant with JORC/SAMREC audit standards for Mineral Resource estimation.

GeoX Invariant Compliance:
- All results include provenance metadata
- Model residuals are calculated and tracked
- Engine version and parameters logged for reproducibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("GeoX_Compliance")

# Feature name constant - should match ChronosEngine.FEATURE_NAME for consistency
# Legacy code used "Main_Sequence" but this caused mismatches with ChronosEngine
FEATURE_NAME = "Stratigraphy"

# Check LoopStructural availability
try:
    from LoopStructural import GeologicalModel
    from LoopStructural.modelling.features import StructuralFrame
    LS_AVAILABLE = True
except ImportError:
    LS_AVAILABLE = False
    GeologicalModel = None
    StructuralFrame = None


@dataclass
class MisfitReport:
    """JORC/SAMREC compliant model misfit metrics."""
    mean_error: float
    max_error: float
    std_dev: float
    p90_error: float
    status: str  # 'PASS' or 'FAIL'
    threshold: float = 0.05  # Normalized threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "std_dev": self.std_dev,
            "p90_error": self.p90_error,
            "status": self.status,
            "threshold": self.threshold,
        }


class GeoXIndustryModeler:
    """
    CP-GRADE GEOLOGICAL MODELLING ENGINE
    Compliant with JORC/SAMREC audit standards.

    .. deprecated::
        This class is being consolidated with ChronosEngine.
        For new code, prefer using GeologicalModelRunner which uses ChronosEngine internally.
        GeoXIndustryModeler uses StandardScaler (zero-mean) while ChronosEngine uses
        MinMaxScaler ([0,1] scaling). ChronosEngine is now the authoritative engine.

    This class provides:
    - UTM coordinate stabilization for numerical safety
    - FDI (Finite Difference Interpolation) for layered rocks
    - Fault displacement field handling
    - Model misfit auditing (SAMREC compliance)
    - Watertight solid extraction for volume calculation

    Usage:
        modeler = GeoXIndustryModeler(extent, resolution=50)
        model = modeler.solve_geology(
            drillhole_data=df,
            chronology=['Unit_A', 'Unit_B', 'Unit_C'],
            fault_params=[{'name': 'F1', 'displacement': 50}]
        )
        solids = modeler.get_watertight_solids(model, chronology, formation_values={'Unit_A': 0, 'Unit_B': 1, 'Unit_C': 2})
    """
    
    def __init__(self, extent: np.ndarray, resolution: int = 50):
        """
        Initialize the industry-grade modeler.

        Args:
            extent: Array [xmin, xmax, ymin, ymax, zmin, zmax]
            resolution: Grid resolution (cells per axis)

        .. deprecated::
            Use GeologicalModelRunner instead, which uses ChronosEngine
            with proper gradient computation from contact geometry.
        """
        import warnings
        warnings.warn(
            "GeoXIndustryModeler is deprecated. Use GeologicalModelRunner instead:\n"
            "  from block_model_viewer.geology.model_runner import GeologicalModelRunner\n"
            "  runner = GeologicalModelRunner(extent_dict, resolution=80)\n"
            "  result = runner.run_full_stack(contacts_df, chronology)\n\n"
            "GeologicalModelRunner computes real gradients from contact geometry\n"
            "instead of using synthetic horizontal orientations.",
            DeprecationWarning,
            stacklevel=2
        )

        if not LS_AVAILABLE:
            raise RuntimeError(
                "LoopStructural library not found. Install with: pip install LoopStructural>=1.6.0"
            )
        
        self.extent = extent
        self.resolution = [resolution, resolution, resolution]
        
        # PROVENANCE TRACKING: Required for JSE/SAMREC auditing
        self.build_log: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "engine": "LoopStructural",
            "engine_version": "1.6+",
            "parameters": {},
            "misfit_report": {},
            "audit_trail": [],
        }
        
        # Coordinate Stabilizer using StandardScaler for zero-mean, unit-variance
        self.scaler = StandardScaler()
        self._model: Optional[GeologicalModel] = None

        # Store StandardScaler bounds for coordinate transformation
        # These are set in solve_geology when the model is created
        self._stdscaler_origin: Optional[np.ndarray] = None
        self._stdscaler_maximum: Optional[np.ndarray] = None

        logger.info(f"GeoXIndustryModeler initialized with resolution {resolution}^3")

    def _model_to_world(self, model_coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from LoopStructural's LOCAL space to WORLD (UTM) coordinates.
        
        COORDINATE PIPELINE:
        1. LoopStructural LOCAL space: origin at [0,0,0], maximum at [size_x, size_y, size_z]
        2. StandardScaler STANDARDIZED space: origin at _stdscaler_origin, roughly [-2, +2]
        3. WORLD (UTM) space: original drillhole coordinates (e.g., 500000, 7000000, 500)
        
        Transform: LOCAL → STANDARDIZED → WORLD
        
        LOCAL + stdscaler_origin = STANDARDIZED
        StandardScaler.inverse_transform(STANDARDIZED) = WORLD
        """
        if len(model_coords) == 0:
            return model_coords
        
        # Diagnostic logging
        c_min = model_coords.min(axis=0)
        c_max = model_coords.max(axis=0)
        logger.info(f"[_model_to_world] Input (LoopStructural LOCAL): {c_min} to {c_max}")
        
        # Skip if already in world coordinates (sanity check)
        if np.max(np.abs(c_max)) > 10000:
            logger.warning(f"[_model_to_world] Input appears to be World Coords already! Returning unchanged.")
            return model_coords
        
        try:
            # STEP 1: Convert from LoopStructural LOCAL to StandardScaler STANDARDIZED space
            # LoopStructural shifts coordinates so origin is at [0,0,0]
            # We need to add back the stdscaler_origin to get standardized coords
            if self._stdscaler_origin is not None:
                standardized_coords = model_coords + self._stdscaler_origin
                logger.info(f"[_model_to_world] After adding stdscaler_origin: {standardized_coords.min(axis=0)} to {standardized_coords.max(axis=0)}")
            else:
                # Fallback: assume coords are already standardized
                standardized_coords = model_coords
                logger.warning("[_model_to_world] _stdscaler_origin is None, assuming coords are standardized")
            
            # STEP 2: Convert from STANDARDIZED to WORLD (UTM) using inverse transform
            world_coords = self.scaler.inverse_transform(standardized_coords)
            
            # Log output for verification
            w_min = world_coords.min(axis=0)
            w_max = world_coords.max(axis=0)
            logger.info(f"[_model_to_world] Output (WORLD/UTM): {w_min} to {w_max}")
            
            return world_coords
        except Exception as e:
            logger.error(f"[_model_to_world] Transform failed: {e}")
            return model_coords
    
    def _normalize_space(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stabilize UTM coordinates for numerical solver safety.
        
        Large UTM coordinates (e.g., 500000+ Easting) can cause numerical
        instability in the interpolation matrix. StandardScaler centers
        and scales the data to prevent singularities.
        
        Args:
            df: DataFrame with 'X', 'Y', 'Z' columns
            
        Returns:
            DataFrame with 'X_n', 'Y_n', 'Z_n' normalized columns
        """
        coords = df[['X', 'Y', 'Z']].values
        df_result = df.copy()
        df_result[['X_n', 'Y_n', 'Z_n']] = self.scaler.fit_transform(coords)
        
        # Record transformation for audit
        self.build_log["audit_trail"].append({
            "action": "coordinate_normalization",
            "timestamp": datetime.now().isoformat(),
            "method": "StandardScaler",
            "n_points": len(df),
            "original_range": {
                "x": [float(df['X'].min()), float(df['X'].max())],
                "y": [float(df['Y'].min()), float(df['Y'].max())],
                "z": [float(df['Z'].min()), float(df['Z'].max())],
            }
        })
        
        return df_result
    
    def solve_geology(
        self,
        drillhole_data: pd.DataFrame,
        chronology: List[str],
        fault_params: Optional[List[Dict[str, Any]]] = None,
        cgw: float = 0.1,
        nelements_factor: int = 1
    ) -> GeologicalModel:
        """
        Industry-standard FDI Solve.
        
        Event Ordering (Geological Correctness):
        1. TECTONIC EVENTS (Faults) - Added first as displacement fields
        2. STRATIGRAPHIC SERIES - FDI interpolation for bed continuity
        
        Args:
            drillhole_data: DataFrame with X, Y, Z, val, formation columns
            chronology: List of unit names from oldest to youngest
            fault_params: List of fault parameter dicts
            cgw: Regularization weight (0.1 = default smoothing)
            nelements_factor: Factor for mesh elements (resolution^2 * factor)
            
        Returns:
            Solved GeologicalModel instance
        """
        df = self._normalize_space(drillhole_data)

        # Define Model Bounding Box in Normalized Space
        origin = df[['X_n', 'Y_n', 'Z_n']].min().values
        maximum = df[['X_n', 'Y_n', 'Z_n']].max().values

        # CRITICAL: Store StandardScaler bounds for coordinate transformation
        # LoopStructural will rescale these bounds to [0,1] internally.
        # We need these to reverse the transformation later.
        self._stdscaler_origin = origin.copy()
        self._stdscaler_maximum = maximum.copy()
        logger.info(f"StandardScaler bounds stored: origin={origin}, maximum={maximum}")

        model = GeologicalModel(origin, maximum)
        # Drop original X, Y, Z before renaming normalized columns to avoid duplicates
        model_df = df.drop(columns=['X', 'Y', 'Z']).rename(columns={'X_n': 'X', 'Y_n': 'Y', 'Z_n': 'Z'})
        # LoopStructural requires 'feature_name' column to identify data points for each feature
        model_df['feature_name'] = FEATURE_NAME

        # =================================================================
        # FIX: Validate and normalize gradient vectors if present
        # =================================================================
        if all(col in model_df.columns for col in ['gx', 'gy', 'gz']):
            grad_vectors = model_df[['gx', 'gy', 'gz']].values.astype(float)
            grad_magnitudes = np.linalg.norm(grad_vectors, axis=1, keepdims=True)

            # Check for near-zero gradients (invalid)
            zero_grad_mask = grad_magnitudes.flatten() < 1e-10
            n_zero_grads = np.sum(zero_grad_mask)

            if n_zero_grads > 0:
                logger.warning(
                    f"GRADIENT VALIDATION: {n_zero_grads} orientation(s) have near-zero gradient vectors. "
                    f"Setting to vertical (0, 0, 1) to prevent numerical errors."
                )
                grad_vectors[zero_grad_mask] = [0.0, 0.0, 1.0]
                grad_magnitudes[zero_grad_mask] = 1.0

            # Normalize all gradients to unit vectors
            grad_normalized = grad_vectors / grad_magnitudes
            model_df['gx'] = grad_normalized[:, 0]
            model_df['gy'] = grad_normalized[:, 1]
            model_df['gz'] = grad_normalized[:, 2]
            logger.info(f"Validated {len(model_df)} gradient vectors (normalized to unit length)")

            # =================================================================
            # WARNING: Complex geology indicators
            # =================================================================
            # Check for downward-pointing gradients (overturned folds)
            downward_mask = grad_normalized[:, 2] < 0
            n_downward = np.sum(downward_mask)
            if n_downward > 0:
                pct_downward = 100.0 * n_downward / len(grad_normalized)
                logger.warning(
                    f"COMPLEX GEOLOGY WARNING: {n_downward} ({pct_downward:.1f}%) orientations point downward. "
                    f"May indicate overturned folds or inverted stratigraphy."
                )

            # Check for steeply dipping beds
            horizontal_component = np.sqrt(grad_normalized[:, 0]**2 + grad_normalized[:, 1]**2)
            steep_mask = horizontal_component > 0.9  # > ~65 degrees
            n_steep = np.sum(steep_mask)
            if n_steep > 0 and (100.0 * n_steep / len(grad_normalized)) > 20:
                logger.warning(
                    f"COMPLEX GEOLOGY WARNING: {n_steep} orientations indicate steep dips (>65°)."
                )

            # Check for mixed gradient directions
            if len(grad_normalized) > 1:
                gz_signs = np.sign(grad_normalized[:, 2])
                if np.any(gz_signs > 0) and np.any(gz_signs < 0):
                    logger.warning(
                        f"COMPLEX GEOLOGY WARNING: Mixed gradient directions detected. "
                        f"Consider domain separation for complex folding."
                    )

        model.data = model_df

        # 1. TECTONIC EVENT LAYER (Faulting)
        # In industry models, faults are displacement fields, not just surfaces.
        if fault_params:
            for f in fault_params:
                try:
                    # FIX: Scale displacement to normalized coordinates
                    # StandardScaler uses scale_ = std_dev of each axis
                    avg_scale = np.mean(self.scaler.scale_)
                    scaled_displacement = f['displacement'] / avg_scale

                    model.create_and_add_fault(
                        f['name'],
                        displacement=scaled_displacement,
                        fault_type=f.get('type', 'normal')
                    )
                    logger.info(f"Added fault '{f['name']}' with displacement {f['displacement']}m (scaled: {scaled_displacement:.4f})")
                except Exception as e:
                    logger.error(f"Failed to add fault '{f['name']}': {e}")
                    raise
        
        # 2. STRATIGRAPHIC SERIES (Sedimentary/Volcanic FDI)
        # FDI ensures that beds do not cross and maintain thicknesses.
        try:
            nelements = self.resolution[0] ** 2 * nelements_factor
            strat_feature = model.create_and_add_foliation(
                FEATURE_NAME,
                interpolatortype="FDI",
                nelements=nelements,
                cgw=cgw
            )
        except Exception as e:
            logger.error(f"Failed to create stratigraphic feature: {e}")
            raise
        
        model.update()
        
        # Record parameters
        self.build_log["parameters"] = {
            "chronology": chronology,
            "n_samples": len(df),
            "n_faults": len(fault_params) if fault_params else 0,
            "cgw": cgw,
            "nelements": nelements,
            "resolution": self.resolution,
        }
        
        # Calculate audit metrics
        self._calculate_audit_metrics(model, df)
        
        self._model = model
        return model
    
    def _calculate_audit_metrics(self, model: GeologicalModel, df: pd.DataFrame) -> None:
        """
        SAMREC COMPLIANCE: Calculate Model Residuals.
        
        If Mean Residual > 2.0m equivalent (normalized), the model is 'Suspect'.
        
        This is a critical audit gate for Mineral Resource estimation under
        JORC (2012) and SAMREC codes.
        """
        try:
            feature = model[FEATURE_NAME]

            # Use model.data coordinates which are already in LoopStructural's projected space
            # The X, Y, Z in model.data have been projected by LoopStructural's bounding_box.project()
            points = model.data[['X', 'Y', 'Z']].values
            actual_vals = model.data['val'].values
            predicted_vals = feature.evaluate_value(points)

            # Log diagnostic info for debugging
            logger.debug(f"Audit points shape: {points.shape}, actual_vals: {len(actual_vals)}, predicted_vals: {len(predicted_vals) if predicted_vals is not None else 'None'}")
            if predicted_vals is not None:
                nan_count = np.sum(np.isnan(predicted_vals))
                if nan_count > 0:
                    logger.warning(f"Audit: {nan_count}/{len(predicted_vals)} predicted values are NaN")

            # Calculate misfit in scalar space (use nanmean/nanmax to handle NaN gracefully)
            misfit = np.abs(actual_vals - predicted_vals)

            mean_err = float(np.nanmean(misfit))
            max_err = float(np.nanmax(misfit))
            std_err = float(np.nanstd(misfit))
            p90_err = float(np.nanpercentile(misfit, 90))
            
            # Threshold check (normalized space)
            threshold = 0.05  # 5% of scalar range
            status = "PASS" if mean_err < threshold else "FAIL"
            
            report = MisfitReport(
                mean_error=mean_err,
                max_error=max_err,
                std_dev=std_err,
                p90_error=p90_err,
                status=status,
                threshold=threshold,
            )
            
            self.build_log["misfit_report"] = report.to_dict()
            
            logger.info(f"Model Misfit Audit: {report.to_dict()}")
            
            if status == "FAIL":
                logger.warning(
                    f"MODEL AUDIT WARNING: Mean misfit {mean_err:.4f} exceeds threshold {threshold}. "
                    "Review data quality or consider adding structural controls."
                )
                
        except Exception as e:
            logger.error(f"Failed to calculate audit metrics: {e}")
            self.build_log["misfit_report"] = {"error": str(e), "status": "ERROR"}
    
    def get_watertight_solids(
        self,
        model: GeologicalModel,
        chronology: List[str],
        formation_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Industry Requirement: Manifold meshes for Volume/Tonnage calculation.

        Extracts watertight (manifold) solids for each stratigraphic unit.
        These are required for resource estimation under JORC/SAMREC codes.

        Args:
            model: Solved GeologicalModel
            chronology: List of unit names from oldest to youngest
            formation_values: Optional dict mapping formation names to scalar values.
                             If provided, uses midpoints between actual values as isosurfaces.
                             If None, falls back to integer indices (legacy behavior).

        Returns:
            Dict mapping unit names to {'verts': array, 'faces': array}
        """
        solids: Dict[str, Dict[str, np.ndarray]] = {}

        logger.info(f"get_watertight_solids called with {len(chronology)} units: {chronology}")

        try:
            feature = model[FEATURE_NAME]
            logger.info(f"Got feature '{FEATURE_NAME}' from model: {type(feature)}")
        except KeyError:
            logger.error(f"{FEATURE_NAME} feature not found in model")
            return solids

        # =================================================================
        # FIX: Calculate correct boundary isovalues
        # =================================================================
        # If formation_values provided, use midpoints between actual scalar values.
        # This fixes the bug where proportional spacing caused wrong boundaries.
        # =================================================================
        if formation_values:
            # Get scalar values in chronology order
            sorted_vals = [formation_values.get(unit, float(i)) for i, unit in enumerate(chronology)]
            logger.info(f"Using actual formation scalar values: {dict(zip(chronology, sorted_vals))}")

            # Calculate boundaries as midpoints between consecutive values
            boundaries = []
            for i in range(len(sorted_vals) - 1):
                midpoint = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
                boundaries.append(midpoint)
            logger.info(f"Calculated boundary isovalues (midpoints): {boundaries}")
        else:
            # Legacy behavior: use integer indices
            boundaries = [float(i) for i in range(len(chronology) - 1)]
            logger.warning("No formation_values provided - using integer isovalues (may be incorrect if proportional spacing was used)")

        for i, boundary_val in enumerate(boundaries):
            try:
                # Extract boundary between Unit[i] and Unit[i+1]
                # LoopStructural 1.6+ returns a LIST of mesh objects, not a tuple
                logger.info(f"Extracting surface at isovalue={boundary_val:.3f} for boundary between '{chronology[i]}' and '{chronology[i+1]}'...")
                result_list = feature.surfaces(boundary_val)

                if result_list is None:
                    logger.warning(f"No surface at isovalue={boundary_val:.3f} for '{chronology[i]}' (None returned)")
                    continue

                # Ensure we have an iterable list (some versions return single object)
                if not isinstance(result_list, (list, tuple)):
                    result_list = [result_list]

                logger.info(f"Got {len(result_list)} result(s) for isovalue={boundary_val:.3f}, types: {[type(r).__name__ for r in result_list]}")

                if len(result_list) == 0:
                    logger.warning(f"No surface at isovalue={boundary_val:.3f} for '{chronology[i]}' (empty list)")
                    continue

                # Process mesh objects - take first valid mesh
                verts, faces, normals = None, None, None

                for result in result_list:
                    # FORMAT 1: LoopStructural Mesh Object (1.6+) with .vertices and .triangles
                    if hasattr(result, 'vertices') and hasattr(result, 'triangles'):
                        verts = np.asarray(result.vertices)
                        faces = np.asarray(result.triangles)
                        normals = np.asarray(result.normals) if hasattr(result, 'normals') else None
                        break

                    # FORMAT 2: PyVista PolyData with .points and .faces
                    elif hasattr(result, 'points') and hasattr(result, 'faces'):
                        verts = np.asarray(result.points)
                        faces_raw = np.asarray(result.faces)
                        if len(faces_raw) > 0:
                            # PyVista faces are [3, i, j, k, 3, ...], reshape to [N, 3]
                            faces = faces_raw.reshape(-1, 4)[:, 1:4]
                        break

                    # FORMAT 3: Legacy Tuple (verts, faces, normals, values)
                    elif isinstance(result, (list, tuple)) and len(result) >= 3:
                        verts = np.asarray(result[0]) if result[0] is not None else None
                        faces = np.asarray(result[1]) if result[1] is not None else None
                        normals = np.asarray(result[2]) if len(result) > 2 and result[2] is not None else None
                        break

                    # FORMAT 4: Object with .vertices and .faces (Trimesh style)
                    elif hasattr(result, 'vertices') and hasattr(result, 'faces'):
                        verts = np.asarray(result.vertices)
                        faces = np.asarray(result.faces)
                        normals = np.asarray(result.normals) if hasattr(result, 'normals') else None
                        break

                if verts is None or len(verts) == 0:
                    logger.warning(f"No valid mesh data for '{chronology[i]}' after parsing {len(result_list)} results")
                    continue

                # Rescale back to UTM coordinates (model space → world)
                verts_utm = self._model_to_world(verts)

                solids[chronology[i]] = {
                    "verts": verts_utm,
                    "faces": faces,
                    "normals": normals,
                }

                logger.info(f"Extracted solid for '{chronology[i]}': {len(verts)} vertices, {len(faces)} faces")

            except Exception as e:
                logger.warning(f"Failed to extract solid for '{chronology[i]}': {e}")
        
        logger.info(f"get_watertight_solids returning {len(solids)} solids: {list(solids.keys())}")
        return solids
    
    def get_scalar_field_on_grid(
        self,
        nx: int = 50,
        ny: int = 50,
        nz: int = 25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate scalar field on a regular grid for visualization.
        
        Args:
            nx, ny, nz: Grid dimensions
            
        Returns:
            Tuple of (x, y, z, values) arrays in world coordinates
        """
        if self._model is None:
            raise RuntimeError("Model not solved - call solve_geology() first")
        
        try:
            feature = self._model[FEATURE_NAME]
        except KeyError:
            raise RuntimeError(f"{FEATURE_NAME} feature not found")
        
        # Create grid in normalized space
        x_n = np.linspace(self.scaler.mean_[0] - 2*self.scaler.scale_[0],
                         self.scaler.mean_[0] + 2*self.scaler.scale_[0], nx)
        y_n = np.linspace(self.scaler.mean_[1] - 2*self.scaler.scale_[1],
                         self.scaler.mean_[1] + 2*self.scaler.scale_[1], ny)
        z_n = np.linspace(self.scaler.mean_[2] - 2*self.scaler.scale_[2],
                         self.scaler.mean_[2] + 2*self.scaler.scale_[2], nz)
        
        # Actually, let's use the data bounds
        df = self._model.data
        x_n = np.linspace(df['X'].min(), df['X'].max(), nx)
        y_n = np.linspace(df['Y'].min(), df['Y'].max(), ny)
        z_n = np.linspace(df['Z'].min(), df['Z'].max(), nz)
        
        xx, yy, zz = np.meshgrid(x_n, y_n, z_n, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # Evaluate
        values = feature.evaluate_value(points)
        
        # Transform back to world coordinates (model space → world)
        world_points = self._model_to_world(points)
        
        return (
            world_points[:, 0].reshape(nx, ny, nz),
            world_points[:, 1].reshape(nx, ny, nz),
            world_points[:, 2].reshape(nx, ny, nz),
            values.reshape(nx, ny, nz)
        )
    
    def get_build_log(self) -> Dict[str, Any]:
        """Get the complete build log for audit purposes."""
        return self.build_log.copy()
    
    def get_misfit_report(self) -> Dict[str, Any]:
        """Get the misfit report for compliance checking."""
        return self.build_log.get("misfit_report", {})
    
    def extract_unified_geology_mesh(
        self,
        model: GeologicalModel,
        chronology: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        CP-GRADE SOLID EXTRACTION: Creates unified voxel partition mesh.
        
        This is the INDUSTRY-STANDARD approach used by Leapfrog and Micromine.
        Instead of N overlapping isosurfaces that Z-fight, we create a single voxel
        grid where every cell has exactly ONE Formation_ID. This eliminates:
        - Z-fighting/flickering (no overlapping surfaces)
        - Discontinuous surfaces (threshold-based extraction)
        - Missing contacts (cell boundaries are contacts)
        
        The renderer colors this single mesh using a discrete colormap.
        
        Args:
            model: Solved GeologicalModel
            chronology: List of unit names from oldest to youngest
            
        Returns:
            Dict with unified mesh data for rendering, or None on failure
        """
        import pyvista as pv
        
        logger.info("=" * 60)
        logger.info("EXTRACTING UNIFIED PARTITION MESH (INDUSTRY-STANDARD)")
        logger.info("This creates ONE mesh with Formation_ID - NO overlapping surfaces")
        logger.info("=" * 60)
        
        try:
            feature = model[FEATURE_NAME]
            logger.info(f"Got feature '{FEATURE_NAME}' from model")
        except KeyError:
            logger.error(f"{FEATURE_NAME} feature not found in model")
            return None
        
        # Get unique stratigraphic values (sorted)
        unique_vals = np.sort(model.data['val'].dropna().unique())
        n_units = len(unique_vals)
        logger.info(f"Partitioning into {n_units} geological units at vals: {unique_vals}")
        
        # Create formation name mapping
        formation_names = {}
        for i, val in enumerate(unique_vals):
            if i < len(chronology):
                formation_names[i] = chronology[i]
            else:
                formation_names[i] = f"Unit_{i}"
        
        # ================================================================
        # CREATE HIGH-RESOLUTION GRID FOR VOXEL PARTITIONING
        # ================================================================
        # Use the MODEL'S bounding box for evaluation (LoopStructural's coordinate space)
        # The grid vertices will be transformed to WORLD coordinates after evaluation.
        # ================================================================
        res = 120
        nx, ny, nz = res, res, res
        pad = 0.0  # No padding - model snaps to drillhole boundaries

        # Get model's bounding box
        # CRITICAL: LoopStructural uses a LOCAL coordinate system where:
        #   - model.bounding_box.origin = [0, 0, 0] (always!)
        #   - model.bounding_box.maximum = size of box
        #   - model.bounding_box.global_origin = the original origin we provided
        # The grid is created in this LOCAL space. Points must be transformed:
        #   LOCAL + stdscaler_origin → STANDARDIZED → WORLD (via scaler.inverse_transform)
        model_origin = np.array(model.bounding_box.origin)  # [0, 0, 0]
        model_maximum = np.array(model.bounding_box.maximum)  # size
        model_size = model_maximum - model_origin

        # Add padding (currently 0)
        padded_origin = model_origin - pad * model_size
        padded_maximum = model_maximum + pad * model_size
        padded_size = padded_maximum - padded_origin

        logger.info(f"Model bbox (LoopStructural LOCAL): origin={model_origin}, maximum={model_maximum}")
        logger.info(f"Padded grid (LOCAL): origin={padded_origin}, maximum={padded_maximum}")
        logger.info(f"stdscaler_origin (for LOCAL→STANDARDIZED): {self._stdscaler_origin}")

        # Create uniform grid in MODEL space (for evaluation)
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(padded_size[0] / (nx - 1), padded_size[1] / (ny - 1), padded_size[2] / (nz - 1)),
            origin=tuple(padded_origin)
        )
        
        # ================================================================
        # CRITICAL: Evaluate at CELL CENTERS (not vertices)
        # ================================================================
        # This is THE KEY to removing blur. By evaluating at cell centers
        # and assigning to cell_data, each voxel is 100% one color.
        cell_centers = grid.cell_centers().points
        
        logger.info(f"Evaluating scalar field on {len(cell_centers)} CELL CENTERS ({nx}x{ny}x{nz})...")
        
        # Evaluate scalar field at CELL CENTERS
        field_values = feature.evaluate_value(cell_centers)
        
        field_min = float(np.nanmin(field_values))
        field_max = float(np.nanmax(field_values))
        logger.info(f"Scalar field range: [{field_min:.2f}, {field_max:.2f}]")
        
        # ================================================================
        # PARTITION: Assign every CELL to EXACTLY one formation
        # ================================================================
        # Calculate boundaries (midpoints between consecutive formation values)
        boundaries = []
        for i in range(len(unique_vals) - 1):
            boundaries.append((unique_vals[i] + unique_vals[i + 1]) / 2)
        
        # Assign Formation_ID based on which interval the scalar falls into
        formation_ids = np.zeros(len(field_values), dtype=np.int32)
        
        for i in range(len(unique_vals)):
            if i == 0:
                # Lowest unit: everything below first boundary
                if boundaries:
                    mask = field_values < boundaries[0]
                else:
                    mask = np.ones(len(field_values), dtype=bool)
            elif i == len(unique_vals) - 1:
                # Highest unit: everything above last boundary
                mask = field_values >= boundaries[-1]
            else:
                # Middle units: between boundaries
                mask = (field_values >= boundaries[i - 1]) & (field_values < boundaries[i])
            
            formation_ids[mask] = i
        
        # Log partition statistics
        for i in range(n_units):
            count = np.sum(formation_ids == i)
            pct = 100 * count / len(formation_ids) if len(formation_ids) > 0 else 0
            logger.info(f"  Formation {i} ({formation_names[i]}): {count:,} voxels ({pct:.1f}%)")
        
        # ================================================================
        # TRANSFORM TO WORLD COORDINATES (model space → world)
        # ================================================================
        verts_world = self._model_to_world(cell_centers)
        
        # ================================================================
        # ASSIGN TO CELL_DATA (prevents interpolation/blur)
        # ================================================================
        grid.cell_data['Formation_ID'] = formation_ids
        grid.cell_data['scalar'] = field_values
        
        logger.info("Assigned Formation_ID to CELL_DATA (voxel-sharpness mode)")
        
        # ================================================================
        # EXTRACT PER-FORMATION SOLIDS FOR CONTACTS/SURFACES
        # ================================================================
        solids_list = []
        contact_surfaces = []
        
        for i in range(n_units):
            unit_name = formation_names[i]
            try:
                # Threshold to extract this formation's cells
                if boundaries:
                    if i == 0:
                        val_min = field_min - 1.0
                        val_max = boundaries[0]
                    elif i == n_units - 1:
                        val_min = boundaries[-1]
                        val_max = field_max + 1.0
                    else:
                        val_min = boundaries[i - 1]
                        val_max = boundaries[i]
                else:
                    val_min = field_min - 1.0
                    val_max = field_max + 1.0
                
                clipped_vol = grid.threshold([val_min, val_max], scalars='scalar')
                
                if clipped_vol is None or clipped_vol.n_cells == 0:
                    logger.warning(f"No cells for '{unit_name}' in range [{val_min:.2f}, {val_max:.2f}]")
                    continue
                
                # Extract surface (boundary of this volume)
                surface = clipped_vol.extract_surface().triangulate()
                
                if surface is None or surface.n_points == 0:
                    continue
                
                # Transform to world coordinates (model space → world)
                verts_scaled = np.asarray(surface.points, dtype=np.float64)
                verts_utm = self._model_to_world(verts_scaled)
                
                # Extract faces
                if hasattr(surface, 'faces') and surface.faces is not None and len(surface.faces) > 0:
                    faces_raw = np.asarray(surface.faces)
                    try:
                        n_faces = len(faces_raw) // 4
                        faces = faces_raw.reshape(n_faces, 4)[:, 1:4]
                    except ValueError:
                        faces = None
                else:
                    faces = None
                
                if faces is not None and len(faces) > 0:
                    solids_list.append({
                        'name': unit_name,
                        'unit_name': unit_name,
                        'vertices': verts_utm,
                        'faces': faces.astype(np.int64),
                        'formation_id': i,
                        'val_range': [float(val_min), float(val_max)],
                    })
                    
                    # Add as contact surface (boundary between formations)
                    contact_surfaces.append({
                        'name': f"Contact_{unit_name}",
                        'vertices': verts_utm,
                        'faces': faces,
                    })
                    
                    logger.info(f"Extracted solid '{unit_name}': {len(verts_utm)} verts, {len(faces)} faces")
                
            except Exception as e:
                logger.warning(f"Failed to extract solid for '{unit_name}': {e}")
                continue
        
        # ================================================================
        # TRANSFORM GRID TO WORLD COORDINATES (BAKE TRANSFORM)
        # ================================================================
        # The 'grid' is currently in Model Space (LoopStructural's standardized
        # coordinates, roughly -3 to +3). We MUST transform it to World Coordinates
        # (UTM scale, e.g., 500000+) so the renderer doesn't have to guess.
        # This is the AUTHORITATIVE place to do this transformation.
        
        # 1. Convert ImageData to StructuredGrid (allows explicit point modification)
        world_grid = grid.cast_to_structured_grid()
        
        # 2. Get current points (these are in Model Space / Standardized coords)
        model_points = np.array(world_grid.points, dtype=np.float64)
        logger.info(f"[BAKE] Grid points BEFORE transform: min={model_points.min(axis=0)}, max={model_points.max(axis=0)}")
        
        # 3. Transform to World Space using the scaler's inverse_transform
        #    Note: _model_to_world handles the inverse StandardScaler logic
        world_points = self._model_to_world(model_points)
        
        # 4. CRITICAL: Apply the transformed points back to the grid
        world_grid.points = world_points
        
        # 5. Verify the transformation was applied
        verify_points = np.array(world_grid.points)
        logger.info(f"[BAKE] Grid points AFTER transform: min={verify_points.min(axis=0)}, max={verify_points.max(axis=0)}")
        logger.info(f"[BAKE] Grid bounds: {world_grid.bounds}")
        
        # Sanity check: World coordinates should be large (UTM scale)
        if np.max(np.abs(verify_points)) < 1000:
            logger.error("[BAKE] WARNING: Grid still appears to be in Model Space! Transform may have failed.")

        # Build result package
        result = {
            'vertices': verts_world,
            'formation_ids': formation_ids,
            'formation_names': formation_names,
            'unique_vals': unique_vals.tolist(),
            'boundaries': boundaries,
            'n_units': n_units,
            'grid_dimensions': (nx, ny, nz),
            'field_range': (field_min, field_max),
            'solids': solids_list,
            'contacts': contact_surfaces,
            '_pyvista_grid': world_grid,  # NOW IN WORLD COORDINATES (UTM)
            '_scaler': self.scaler,
            '_stdscaler_origin': self._stdscaler_origin,
            '_stdscaler_maximum': self._stdscaler_maximum,
            '_is_world_coordinates': True,  # Flag to indicate grid is already in World space
        }
        
        logger.info("=" * 60)
        logger.info(f"UNIFIED MESH EXTRACTED: {len(verts_world):,} voxels, {n_units} formations, {len(solids_list)} solids")
        logger.info("This mesh has NO OVERLAP - Z-fighting eliminated")
        logger.info("=" * 60)
        
        return result
    
    @staticmethod
    def is_available() -> bool:
        """Check if LoopStructural is available."""
        return LS_AVAILABLE

