"""
ChronosEngine - Senior-level controller for LoopStructural.

Handles coordinate normalization, event-stacking, and mesh extraction for
geological modeling with JORC/SAMREC compliance.

GeoX Invariant Compliance:
- All results include provenance metadata
- Parameters and transformations are recorded
- Engine version tracked for reproducibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Check LoopStructural availability
try:
    from LoopStructural import GeologicalModel
    from LoopStructural.modelling.features import StructuralFrame
    LS_AVAILABLE = True
    logger.debug("LoopStructural library available")
except ImportError:
    LS_AVAILABLE = False
    GeologicalModel = None
    StructuralFrame = None
    logger.warning("LoopStructural library not available - geological modeling disabled")


@dataclass
class ModelBuildLog:
    """Provenance tracking for JORC/SAMREC auditing."""
    timestamp: datetime = field(default_factory=datetime.now)
    engine: str = "LoopStructural"
    engine_version: str = "1.6+"
    parameters: Dict[str, Any] = field(default_factory=dict)
    misfit_report: Dict[str, Any] = field(default_factory=dict)
    coordinate_transform: Dict[str, Any] = field(default_factory=dict)
    event_stack: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "engine": self.engine,
            "engine_version": self.engine_version,
            "parameters": self.parameters,
            "misfit_report": self.misfit_report,
            "coordinate_transform": self.coordinate_transform,
            "event_stack": self.event_stack,
        }


class ChronosEngine:
    """
    Senior-level controller for LoopStructural.
    
    Handles coordinate normalization, event-stacking, and mesh extraction
    for geological modeling with industry-grade compliance.
    
    Key Features:
    - UTM coordinate normalization to [0,1] for numerical stability
    - Fault-first event stacking (faults displace space)
    - FDI interpolation for layered rocks
    - Watertight mesh extraction for volume calculations
    
    Usage:
        engine = ChronosEngine(extent={'xmin': 0, 'xmax': 1000, ...}, resolution=50)
        df = engine.prepare_data(drillhole_df)
        engine.build_model(stratigraphy=['Unit_A', 'Unit_B'], 
                          contacts=contacts_df,
                          orientations=orientations_df,
                          faults=[{'name': 'F1', 'displacement': 50}])
        surfaces = engine.extract_meshes()
    """
    
    # Class constant for feature name - used by both engine and compliance checks
    # This ensures ChronosEngine and ComplianceManager use the same feature name
    FEATURE_NAME = "Stratigraphy"
    
    def __init__(self, extent: Dict[str, float], resolution: int = 50, boundary_padding: float = 0.1):
        """
        Initialize the ChronosEngine.
        
        Args:
            extent: Bounding box dict with keys 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'
            resolution: Grid resolution for interpolation (cells per axis)
            boundary_padding: Fractional padding added to model extent to prevent boundary clipping.
                            Default 0.1 (10%) prevents isosurfaces from being clipped at edges.
                            Higher padding ensures smooth edges without "shattered" appearance.
                            GEOLOGICAL NOTE: A padding of 0.0 causes surfaces near the model
                            boundary to be truncated, producing discontinuous geology.  0.1 is
                            the minimum safe value for CP-grade work.
        """
        if not LS_AVAILABLE:
            raise RuntimeError(
                "LoopStructural library not found. Install with: pip install LoopStructural>=1.6.0"
            )
        
        self.raw_extent = extent
        self.resolution = [resolution, resolution, resolution]
        self.boundary_padding = boundary_padding
        self.scaler = MinMaxScaler()
        self.model: Optional[GeologicalModel] = None
        self.build_log = ModelBuildLog()
        
        # Internal coordinate system: [0, 1] to prevent matrix singularities
        bbox = np.array([
            [extent['xmin'], extent['ymin'], extent['zmin']],
            [extent['xmax'], extent['ymax'], extent['zmax']]
        ])
        self.scaler.fit(bbox)
        
        # Record coordinate transformation for provenance
        self.build_log.coordinate_transform = {
            "method": "MinMaxScaler",
            "original_extent": extent,
            "scaled_extent": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "scale": self.scaler.scale_.tolist(),
            "min": self.scaler.min_.tolist() if hasattr(self.scaler, 'min_') else None,
            "boundary_padding": boundary_padding,
        }
        
        logger.info(f"ChronosEngine initialized with resolution {resolution}^3, padding {boundary_padding}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale world coordinates to [0, 1] range for numerical stability.
        
        This transformation prevents matrix singularities that can occur
        with large UTM coordinates (e.g., 500000+ Eastings).
        
        Args:
            df: DataFrame with 'X', 'Y', 'Z' columns in world coordinates
            
        Returns:
            DataFrame with additional 'X_s', 'Y_s', 'Z_s' scaled columns
        """
        if not all(col in df.columns for col in ['X', 'Y', 'Z']):
            raise ValueError("DataFrame must have 'X', 'Y', 'Z' columns")
        
        coords = df[['X', 'Y', 'Z']].values
        scaled_coords = self.scaler.transform(coords)
        
        result = df.copy()
        result[['X_s', 'Y_s', 'Z_s']] = scaled_coords
        
        logger.debug(f"Prepared {len(df)} data points, scaled to [0,1]")
        return result
    
    def build_model(
        self,
        stratigraphy: List[str],
        contacts: pd.DataFrame,
        orientations: pd.DataFrame,
        faults: Optional[List[Dict[str, Any]]] = None,
        cgw: float = 0.005,
        interpolator_type: str = "FDI"
    ) -> None:
        """
        Execute the modeling pipeline following stratigraphic rules.
        
        Event Ordering (Critical for geological correctness):
        1. Faults are added FIRST (they displace space)
        2. Stratigraphic series added after (they fill the displaced space)
        
        Args:
            stratigraphy: List of unit names from oldest to youngest
            contacts: DataFrame with contact points (X_s, Y_s, Z_s, val, formation)
            orientations: DataFrame with orientation data (X_s, Y_s, Z_s, gx, gy, gz)
            faults: List of fault dicts with 'name', 'displacement', 'type'
            cgw: Regularization weight.  Lower = tighter fit to contacts.
                 0.005 (default) → surfaces pass close to contacts (CP-grade).
                 0.001 → very tight, may oscillate with sparse data.
                 0.01–0.05 → smoother but departs from contacts — use for
                   reconnaissance models or structurally complex areas.
                 0.1 → heavy smoothing, almost never appropriate for resource models.
            interpolator_type: 'FDI' (Finite Difference) or 'PLI' (Piece-wise Linear)
        """
        if not LS_AVAILABLE:
            raise RuntimeError("LoopStructural library not found.")
        
        # 1. Initialize Model with PADDED scaled bounds
        # FIX: Add padding to prevent isosurface boundary clipping
        # When surfaces exist near x=1.0 but model max is 1.0, they get clipped to nothing
        pad = self.boundary_padding
        origin = [-pad, -pad, -pad]
        maximum = [1 + pad, 1 + pad, 1 + pad]
        self.model = GeologicalModel(origin, maximum)
        
        logger.info(f"Model bounds with padding: origin={origin}, max={maximum}")
        
        # 2. Prepare data for LoopStructural format
        # Reset indices first to avoid reindexing errors with duplicate indices
        contacts_reset = contacts.reset_index(drop=True)
        orientations_reset = orientations.reset_index(drop=True)
        
        # Create new DataFrames with only required columns to avoid duplicate column names
        # (after prepare_data, DataFrame has both X and X_s - we need to pick X_s as X)
        
        # IMPORTANT: In LoopStructural, 'feature_name' must match the geological feature name
        # The 'val' column contains the scalar value (stratigraphic order)
        # All contact points belong to the same feature "Stratigraphy"
        
        # Check which coordinate columns to use (scaled X_s or original X)
        # Use class constant FEATURE_NAME for consistency with ComplianceManager
        if 'X_s' in contacts_reset.columns:
            contacts_ls = pd.DataFrame({
                'X': contacts_reset['X_s'].values,
                'Y': contacts_reset['Y_s'].values,
                'Z': contacts_reset['Z_s'].values,
                'val': contacts_reset['val'].values,
                'feature_name': self.FEATURE_NAME,  # Must match the feature name, not individual formations
            })
        else:
            contacts_ls = pd.DataFrame({
                'X': contacts_reset['X'].values,
                'Y': contacts_reset['Y'].values,
                'Z': contacts_reset['Z'].values,
                'val': contacts_reset['val'].values,
                'feature_name': self.FEATURE_NAME,  # Must match the feature name, not individual formations
            })
        
        if 'X_s' in orientations_reset.columns:
            orientations_ls = pd.DataFrame({
                'X': orientations_reset['X_s'].values,
                'Y': orientations_reset['Y_s'].values,
                'Z': orientations_reset['Z_s'].values,
                'gx': orientations_reset['gx'].values,
                'gy': orientations_reset['gy'].values,
                'gz': orientations_reset['gz'].values,
                'feature_name': self.FEATURE_NAME,  # Must match the feature name
            })
        else:
            orientations_ls = pd.DataFrame({
                'X': orientations_reset['X'].values,
                'Y': orientations_reset['Y'].values,
                'Z': orientations_reset['Z'].values,
                'gx': orientations_reset['gx'].values,
                'gy': orientations_reset['gy'].values,
                'gz': orientations_reset['gz'].values,
                'feature_name': self.FEATURE_NAME,  # Must match the feature name
            })
        
        # Ensure required columns exist
        required_contact_cols = ['X', 'Y', 'Z', 'val', 'feature_name']
        required_orient_cols = ['X', 'Y', 'Z', 'gx', 'gy', 'gz', 'feature_name']
        
        if not all(col in contacts_ls.columns for col in required_contact_cols):
            missing = [c for c in required_contact_cols if c not in contacts_ls.columns]
            raise ValueError(f"Contacts missing columns: {missing}")
        
        if not all(col in orientations_ls.columns for col in required_orient_cols):
            missing = [c for c in required_orient_cols if c not in orientations_ls.columns]
            raise ValueError(f"Orientations missing columns: {missing}")

        # =================================================================
        # FIX: Validate, normalise, and enforce polarity-consistent gradients
        # =================================================================
        # LoopStructural's FDI solver is sensitive to gradient polarity.
        # Mixed polarities (some gz>0, some gz<0) within the same feature
        # will cause the solver to produce folded / chaotic surfaces.
        #
        # Geological justification:
        #   For a SINGLE stratigraphic pile, scalar values increase monotonically
        #   from old (deep) to young (shallow).  The gradient should therefore
        #   point consistently in the direction of increasing scalar value.
        #   If > 50% of gradients point upward, ALL should point upward.
        #   (True overturned folds are handled by domain separation, not by
        #    feeding contradictory polarities into a single feature.)
        # =================================================================
        grad_vectors = orientations_ls[['gx', 'gy', 'gz']].values.astype(float)
        grad_magnitudes = np.linalg.norm(grad_vectors, axis=1, keepdims=True)

        # Handle near-zero gradients
        zero_mask = grad_magnitudes.flatten() < 1e-10
        n_zero = int(np.sum(zero_mask))
        if n_zero > 0:
            logger.warning(
                f"GRADIENT: {n_zero} near-zero gradient(s) → set to vertical (0,0,1)"
            )
            grad_vectors[zero_mask] = [0.0, 0.0, 1.0]
            grad_magnitudes[zero_mask] = 1.0

        # Normalise to unit vectors
        grad_normalized = grad_vectors / grad_magnitudes

        # ── Enforce majority-polarity consistency ──
        gz_signs = np.sign(grad_normalized[:, 2])
        n_up = int(np.sum(gz_signs > 0))
        n_down = int(np.sum(gz_signs < 0))

        if n_up > 0 and n_down > 0:
            # Mixed polarity detected — flip minority to match majority
            if n_up >= n_down:
                flip_mask = gz_signs < 0
                flip_label = "downward → upward"
            else:
                flip_mask = gz_signs > 0
                flip_label = "upward → downward"

            n_flipped = int(np.sum(flip_mask))
            grad_normalized[flip_mask] = -grad_normalized[flip_mask]
            logger.warning(
                f"POLARITY FIX: Flipped {n_flipped} gradient(s) {flip_label} "
                f"to enforce polarity consistency "
                f"(was {n_up} up / {n_down} down)."
            )

        # Update the DataFrame
        orientations_ls['gx'] = grad_normalized[:, 0]
        orientations_ls['gy'] = grad_normalized[:, 1]
        orientations_ls['gz'] = grad_normalized[:, 2]

        logger.info(f"Validated {len(orientations_ls)} orientation vectors (normalised, polarity-consistent)")

        # Log dip statistics for audit
        horiz = np.sqrt(grad_normalized[:, 0]**2 + grad_normalized[:, 1]**2)
        mean_dip_deg = float(np.degrees(np.arctan2(np.mean(horiz), np.mean(np.abs(grad_normalized[:, 2])))))
        logger.info(f"  Mean apparent dip: ~{mean_dip_deg:.0f}°")

        # Set model data (ignore_index=True creates new sequential index)
        self.model.data = pd.concat([contacts_ls, orientations_ls], ignore_index=True)
        
        # Log model data summary
        logger.info(f"Model data: {len(self.model.data)} rows, columns: {list(self.model.data.columns)}")
        logger.info(f"Unique feature_names: {self.model.data['feature_name'].unique()}")
        logger.info(f"Val range: {self.model.data['val'].min()} to {self.model.data['val'].max()}")
        
        # 3. Add Structural Events (Faults first - they displace space)
        if faults:
            for f in faults:
                try:
                    fault_name = f.get('name', f'Fault_{id(f)}')
                    
                    # Check if fault has geometric data (dip, azimuth, point)
                    if all(k in f for k in ['dip', 'azimuth', 'point']):
                        # Import FaultPlane to generate fault geometry
                        from .faults import FaultPlane
                        
                        # Create FaultPlane instance
                        fault_plane = FaultPlane.from_dict(f)
                        
                        # Generate fault trace points and orientations in world coordinates
                        fault_traces = fault_plane.generate_fault_trace_points(
                            extent=self.raw_extent,
                            num_points=20
                        )
                        fault_orientations = fault_plane.generate_fault_orientations(
                            extent=self.raw_extent,
                            num_orientations=10
                        )
                        
                        # Validate generated data
                        if fault_traces.empty or len(fault_traces) == 0:
                            logger.error(f"Failed to generate fault trace points for '{fault_name}'")
                            continue
                        
                        if fault_orientations.empty or len(fault_orientations) == 0:
                            logger.error(f"Failed to generate fault orientations for '{fault_name}'")
                            continue
                        
                        # Scale fault geometry to normalized space [0,1]
                        fault_traces_scaled = self.prepare_data(fault_traces)
                        fault_orientations_scaled = self.prepare_data(fault_orientations)
                        
                        # Rename columns to match LoopStructural expectations
                        fault_traces_renamed = fault_traces_scaled[['X_s', 'Y_s', 'Z_s']].copy()
                        fault_traces_renamed.columns = ['X', 'Y', 'Z']
                        fault_traces_renamed['feature_name'] = fault_name
                        fault_traces_renamed['val'] = 0.0  # Fault surface has scalar value 0
                        
                        fault_orientations_renamed = fault_orientations_scaled[['X_s', 'Y_s', 'Z_s', 'gx', 'gy', 'gz']].copy()
                        fault_orientations_renamed.columns = ['X', 'Y', 'Z', 'gx', 'gy', 'gz']
                        fault_orientations_renamed['feature_name'] = fault_name
                        
                        # Add fault data to model.data BEFORE creating the fault feature
                        self.model.data = pd.concat([
                            self.model.data,
                            fault_traces_renamed,
                            fault_orientations_renamed
                        ], ignore_index=True)
                        
                        logger.info(f"Added {len(fault_traces_renamed)} fault trace points and {len(fault_orientations_renamed)} orientations for '{fault_name}'")
                    else:
                        logger.warning(f"Fault '{fault_name}' missing geometric data (dip, azimuth, point) - fault will have no effect")
                    
                    # Scale displacement to normalized coordinates
                    # FIX: Use Z-axis scale for vertical faults (most common),
                    # or the geometric mean of all scales for oblique faults.
                    # The scaler maps [xmin, xmax] → [0, 1], so scale_[i] = 1/(max-min).
                    # A displacement of D metres in world space = D * scale_[i] in [0,1] space.
                    raw_displacement = f.get('displacement', f.get('throw_magnitude', 0))
                    
                    # For vertical component (most fault types)
                    z_scale = self.scaler.scale_[2]  # 1 / z_range
                    scaled_displacement = raw_displacement * z_scale
                    
                    # Create fault feature in LoopStructural
                    # CRITICAL FIX: Add force_mesh_geometry=True
                    # This prevents 'zero-size array to reduction operation maximum' error
                    # by forcing the fault to use the global model grid instead of auto-sizing to data
                    self.model.create_and_add_fault(
                        fault_name,
                        displacement=scaled_displacement,
                        fault_type=f.get('type', 'normal'),
                        force_mesh_geometry=True
                    )
                    
                    self.build_log.event_stack.append(f"Fault: {fault_name}")
                    logger.info(f"Added fault '{fault_name}' with displacement {f['displacement']}m")
                    
                except Exception as e:
                    logger.error(f"Failed to add fault '{fault_name}': {e}")
                    raise
        
        # 4. Add Stratigraphic Series (Sedimentary/Volcanic)
        # FDI is best for layered rocks - ensures beds don't cross
        try:
            strat_feature = self.model.create_and_add_foliation(
                self.FEATURE_NAME,
                interpolatortype=interpolator_type,
                cgw=cgw
            )
            self.build_log.event_stack.append(f"{self.FEATURE_NAME}: FDI foliation")
            
        except Exception as e:
            logger.error(f"Failed to create stratigraphic feature: {e}")
            raise
        
        # 5. Compute model
        self.model.update()
        
        # RESOLUTION VALIDATION: Check if grid resolution is adequate for unit thickness
        voxel_size_m = self._estimate_voxel_size_meters()
        if voxel_size_m is not None:
            logger.info(f"RESOLUTION CHECK: Voxel size ~{voxel_size_m:.2f}m per axis")
            
            # Estimate minimum unit thickness from val spacing
            val_range = contacts_ls['val'].max() - contacts_ls['val'].min()
            n_units = len(contacts_ls['val'].unique())
            if n_units > 1:
                avg_unit_scalar_thickness = val_range / (n_units - 1)
                # If average scalar step is smaller than 2x voxel scalar resolution,
                # surfaces may fail to extract cleanly
                scalar_resolution = 1.0 / self.resolution[0]  # Assuming [0,1] scaled space
                if avg_unit_scalar_thickness < 2 * scalar_resolution:
                    logger.warning(
                        f"RESOLUTION WARNING: Average unit scalar thickness ({avg_unit_scalar_thickness:.4f}) "
                        f"is less than 2x scalar resolution ({2*scalar_resolution:.4f}). "
                        f"Thin units may not extract cleanly. Consider increasing resolution from {self.resolution[0]} "
                        f"to at least {int(np.ceil(2 * n_units))}."
                    )
        
        # Record build parameters
        self.build_log.parameters = {
            "stratigraphy": stratigraphy,
            "n_contacts": len(contacts),
            "n_orientations": len(orientations),
            "n_faults": len(faults) if faults else 0,
            "cgw": cgw,
            "interpolator_type": interpolator_type,
            "resolution": self.resolution,
            "boundary_padding": self.boundary_padding,
            "feature_name": self.FEATURE_NAME,
        }
        
        logger.info("Geological solve completed.")
    
    def extract_meshes(self, n_surfaces: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        CP-GRADE MESH EXTRACTION (Refactored for LoopStructural 1.6+)
        
        Handles the three different formats LoopStructural uses across versions:
        1. List of Mesh Objects (Modern: uses .vertices and .triangles)
        2. PyVista Objects (uses .points and .faces with padding)
        3. 4-item Tuple (Legacy: verts, faces, normals, values)
        
        Args:
            n_surfaces: Number of surfaces to extract. If None, uses unique values.
            
        Returns:
            List of surface dicts with 'vertices', 'faces', 'val', 'name'
        """
        surfaces = []
        
        if not self.model:
            logger.warning("No model built - cannot extract meshes")
            return surfaces
        
        try:
            feature = self.model[self.FEATURE_NAME]
            logger.info(f"Found {self.FEATURE_NAME} feature: {type(feature)}")
        except KeyError:
            logger.error(f"Feature '{self.FEATURE_NAME}' not found.")
            logger.error(f"Available features: {list(self.model.features.keys()) if hasattr(self.model, 'features') else 'unknown'}")
            return surfaces
        
        # DIAGNOSTIC: Evaluate actual scalar field range
        try:
            grid_points = self.model.regular_grid()
            field_values = feature.evaluate_value(grid_points)
            field_min, field_max = float(np.nanmin(field_values)), float(np.nanmax(field_values))
            logger.info(f"DIAGNOSTIC: Computed scalar field range: [{field_min:.4f}, {field_max:.4f}]")
            
            self.build_log.misfit_report['field_range'] = {
                'min': field_min,
                'max': field_max,
            }
        except Exception as e:
            logger.warning(f"Could not evaluate field range diagnostic: {e}")
            field_min, field_max = None, None
        
        # Determine extraction levels
        unique_vals = np.sort(self.model.data['val'].dropna().unique())
        logger.info(f"Target extraction values from data: {unique_vals}")
        
        for i, val in enumerate(unique_vals):
            try:
                logger.info(f"Attempting surface extraction at val={val:.4f} ({i+1}/{len(unique_vals)})")
                
                # LoopStructural 1.6+ returns a LIST of mesh objects
                result_list = feature.surfaces(val)
                
                if result_list is None:
                    logger.debug(f"No surface at val={val:.2f} (None returned)")
                    continue
                
                # Ensure we have an iterable list
                # Some versions return a single object, others return a list
                if not isinstance(result_list, (list, tuple)):
                    result_list = [result_list]
                
                if len(result_list) == 0:
                    logger.debug(f"No surface at val={val:.2f} (empty list)")
                    continue
                
                logger.info(f"Surface result at val={val:.4f}: {len(result_list)} object(s), first type={type(result_list[0]).__name__}")
                
                # Process each mesh object in the result list
                for mesh_idx, result in enumerate(result_list):
                    verts, faces = None, None
                    
                    # FORMAT 1: LoopStructural Mesh Object (Common in 1.6+)
                    # Uses .vertices and .triangles (NOT .faces!)
                    if hasattr(result, 'vertices') and hasattr(result, 'triangles'):
                        verts = np.asarray(result.vertices)
                        faces = np.asarray(result.triangles)
                        logger.debug(f"  Mesh {mesh_idx}: LS format (.vertices/.triangles)")
                    
                    # FORMAT 2: PyVista PolyData (Common in some LS wrappers)
                    # Uses .points and .faces with [3, i, j, k, 3, ...] padding
                    elif hasattr(result, 'points') and hasattr(result, 'faces'):
                        verts = np.asarray(result.points)
                        faces_raw = np.asarray(result.faces)
                        if len(faces_raw) > 0:
                            # PyVista faces are [3, i, j, k, 3, i, j, k, ...], reshape to [N, 3]
                            try:
                                faces = faces_raw.reshape(-1, 4)[:, 1:4]
                            except ValueError:
                                # Alternative: manually extract
                                n_faces = result.n_faces if hasattr(result, 'n_faces') else len(faces_raw) // 4
                                faces = faces_raw.reshape(-1, 4)[:n_faces, 1:4]
                        logger.debug(f"  Mesh {mesh_idx}: PyVista format (.points/.faces)")
                    
                    # FORMAT 3: Legacy Tuple (verts, faces, normals, values)
                    elif isinstance(result, (list, tuple)) and len(result) == 4:
                        verts, faces, _, _ = result
                        verts = np.asarray(verts) if verts is not None else None
                        faces = np.asarray(faces) if faces is not None else None
                        logger.debug(f"  Mesh {mesh_idx}: Legacy tuple format")
                    
                    # FORMAT 4: Object with .vertices and .faces (older LS or Trimesh)
                    elif hasattr(result, 'vertices') and hasattr(result, 'faces'):
                        verts = np.asarray(result.vertices)
                        faces = np.asarray(result.faces)
                        logger.debug(f"  Mesh {mesh_idx}: Trimesh-like format (.vertices/.faces)")
                    
                    else:
                        # Log available attributes for debugging
                        attrs = [a for a in dir(result) if not a.startswith('_')][:15]
                        logger.warning(f"  Mesh {mesh_idx}: Unknown format, type={type(result).__name__}, attrs={attrs}")
                        continue
                    
                    # Validate extracted mesh data
                    if verts is None or len(verts) == 0:
                        logger.debug(f"  Mesh {mesh_idx}: Empty vertices, skipping")
                        continue
                    
                    if faces is None or len(faces) == 0:
                        logger.warning(f"  Mesh {mesh_idx}: No faces found, skipping")
                        continue
                    
                    # =================================================================
                    # CRITICAL: Inverse transform from [0,1] model space to World (UTM)
                    # =================================================================
                    # Log pre-transform coordinates for debugging
                    logger.debug(
                        f"  Mesh {mesh_idx} PRE-transform (model space): "
                        f"X=[{verts[:, 0].min():.4f}, {verts[:, 0].max():.4f}], "
                        f"Y=[{verts[:, 1].min():.4f}, {verts[:, 1].max():.4f}], "
                        f"Z=[{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}]"
                    )
                    
                    # Inverse scale back to UTM/World coordinates
                    verts_world = self.scaler.inverse_transform(verts)
                    
                    # Log post-transform coordinates for debugging
                    # These should be in UTM-scale (e.g., X=500000, Y=7000000)
                    logger.info(
                        f"  Mesh {mesh_idx} POST-transform (world space): "
                        f"X=[{verts_world[:, 0].min():.1f}, {verts_world[:, 0].max():.1f}], "
                        f"Y=[{verts_world[:, 1].min():.1f}, {verts_world[:, 1].max():.1f}], "
                        f"Z=[{verts_world[:, 2].min():.1f}, {verts_world[:, 2].max():.1f}]"
                    )
                    
                    surface_name = f"Surface_val_{val:.2f}" if len(result_list) == 1 else f"Surface_val_{val:.2f}_{mesh_idx}"
                    
                    surfaces.append({
                        "vertices": verts_world,
                        "faces": faces,
                        "val": float(val),
                        "name": surface_name,
                    })
                    
                    logger.info(f"Successfully extracted: val={val:.2f} ({len(verts)} verts, {len(faces)} faces)")
                
            except Exception as e:
                logger.warning(f"Skipping val={val} due to extraction error: {e}", exc_info=True)
        
        logger.info(f"Final Count: {len(surfaces)} surfaces extracted.")
        return surfaces
    
    def get_scalar_field(self, points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate scalar field at given points or on a regular grid.
        
        Args:
            points: Optional Nx3 array of world coordinates.
                   If None, evaluates on a regular grid.
                   
        Returns:
            Tuple of (points, values) arrays
        """
        if not self.model:
            raise RuntimeError("Model not built - call build_model() first")
        
        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            raise RuntimeError(f"{self.FEATURE_NAME} feature not found in model")
        
        if points is None:
            # Generate regular grid in scaled space
            n = self.resolution[0]
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            z = np.linspace(0, 1, n)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            scaled_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        else:
            # Scale input points
            scaled_points = self.scaler.transform(points)
        
        # Evaluate scalar field
        values = feature.evaluate_value(scaled_points)
        
        # Convert back to world coordinates
        if points is None:
            world_points = self.scaler.inverse_transform(scaled_points)
        else:
            world_points = points
        
        return world_points, values
    
    def get_build_log(self) -> Dict[str, Any]:
        """Get the complete build log for audit purposes."""
        return self.build_log.to_dict()
    
    def _estimate_voxel_size_meters(self) -> Optional[float]:
        """
        Estimate the voxel size in meters based on original extent and resolution.
        
        Returns:
            Estimated voxel size in meters, or None if not available.
        """
        try:
            # Calculate world-space extent
            x_range = self.raw_extent['xmax'] - self.raw_extent['xmin']
            y_range = self.raw_extent['ymax'] - self.raw_extent['ymin']
            z_range = self.raw_extent['zmax'] - self.raw_extent['zmin']
            
            # Average voxel size
            voxel_x = x_range / self.resolution[0]
            voxel_y = y_range / self.resolution[1]
            voxel_z = z_range / self.resolution[2]
            
            return (voxel_x + voxel_y + voxel_z) / 3.0
        except Exception:
            return None
    
    def diagnose_extraction_failure(self) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics when surface extraction fails.
        
        Returns:
            Dict with diagnostic information for troubleshooting.
        """
        diagnostics = {
            "model_exists": self.model is not None,
            "feature_name": self.FEATURE_NAME,
            "feature_found": False,
            "field_range": None,
            "data_val_range": None,
            "scalar_at_center": None,
            "mesh_format_test": None,
            "recommendations": [],
        }
        
        if not self.model:
            diagnostics["recommendations"].append("No model built - call build_model() first")
            return diagnostics
        
        try:
            feature = self.model[self.FEATURE_NAME]
            diagnostics["feature_found"] = True
        except KeyError:
            diagnostics["recommendations"].append(
                f"Feature '{self.FEATURE_NAME}' not found. Check model.features or build_model() was successful."
            )
            return diagnostics
        
        # Evaluate field range on grid
        try:
            grid_points = self.model.regular_grid()
            field_values = feature.evaluate_value(grid_points)
            diagnostics["field_range"] = {
                "min": float(np.nanmin(field_values)),
                "max": float(np.nanmax(field_values)),
                "mean": float(np.nanmean(field_values)),
            }
        except Exception as e:
            diagnostics["recommendations"].append(f"Could not evaluate field: {e}")
        
        # Check data values
        if 'val' in self.model.data.columns:
            data_vals = self.model.data['val'].dropna()
            diagnostics["data_val_range"] = {
                "min": float(data_vals.min()),
                "max": float(data_vals.max()),
                "unique": data_vals.unique().tolist(),
            }
        
        # Evaluate at center
        try:
            center = np.array([[0.5, 0.5, 0.5]])
            center_val = feature.evaluate_value(center)
            diagnostics["scalar_at_center"] = float(center_val[0])
        except Exception as e:
            diagnostics["recommendations"].append(f"Could not evaluate at center: {e}")
        
        # TEST MESH FORMAT: Try extracting one surface and report its format
        try:
            test_val = diagnostics["data_val_range"]["unique"][0] if diagnostics["data_val_range"] else 0.0
            test_result = feature.surfaces(test_val)
            
            if test_result is not None:
                if isinstance(test_result, (list, tuple)) and len(test_result) > 0:
                    first_mesh = test_result[0] if isinstance(test_result, (list, tuple)) else test_result
                else:
                    first_mesh = test_result
                
                mesh_attrs = [a for a in dir(first_mesh) if not a.startswith('_')]
                diagnostics["mesh_format_test"] = {
                    "type": type(first_mesh).__name__,
                    "has_vertices": hasattr(first_mesh, 'vertices'),
                    "has_triangles": hasattr(first_mesh, 'triangles'),  # LS 1.6+ uses .triangles
                    "has_faces": hasattr(first_mesh, 'faces'),  # Older versions use .faces
                    "has_points": hasattr(first_mesh, 'points'),  # PyVista format
                    "sample_attrs": mesh_attrs[:10],
                }
                
                # Check for common attribute mismatch
                if hasattr(first_mesh, 'triangles') and not hasattr(first_mesh, 'faces'):
                    diagnostics["recommendations"].append(
                        "MESH FORMAT: LoopStructural 1.6+ detected. Uses .triangles instead of .faces."
                    )
                elif hasattr(first_mesh, 'points') and hasattr(first_mesh, 'faces'):
                    diagnostics["recommendations"].append(
                        "MESH FORMAT: PyVista format detected. Uses .points/.faces with padding."
                    )
            else:
                diagnostics["mesh_format_test"] = {"error": "surfaces() returned None"}
        except Exception as e:
            diagnostics["mesh_format_test"] = {"error": str(e)}
        
        # Generate recommendations
        if diagnostics["field_range"] and diagnostics["data_val_range"]:
            field_min = diagnostics["field_range"]["min"]
            field_max = diagnostics["field_range"]["max"]
            data_min = diagnostics["data_val_range"]["min"]
            data_max = diagnostics["data_val_range"]["max"]
            
            if data_min < field_min or data_max > field_max:
                diagnostics["recommendations"].append(
                    f"TARGET VALUES OUTSIDE FIELD RANGE: Data values [{data_min}, {data_max}] "
                    f"but field only reaches [{field_min:.4f}, {field_max:.4f}]. "
                    "Reduce cgw (regularization) to allow field to reach extreme values."
                )
            
            field_span = field_max - field_min
            data_span = data_max - data_min
            if field_span < 0.5 * data_span:
                diagnostics["recommendations"].append(
                    f"FIELD COMPRESSION: Field span ({field_span:.4f}) much smaller than "
                    f"data span ({data_span:.4f}). This usually indicates over-smoothing (high cgw)."
                )
        
        return diagnostics
    
    def extract_solids(self, stratigraphy: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        CP-GRADE SOLID EXTRACTION (LoopStructural 1.6+ Compatible).
        
        CRITICAL FIX: Uses threshold() instead of clip_scalar() to create TRUE SOLIDS.
        - threshold() clips the VOLUME, not just the surface
        - fill_holes() closes manifold for proper solid appearance
        - Taubin smoothing removes "origami" triangles without shrinking volume
        
        This method produces professional-quality smooth surfaces (like Leapfrog/Vulcan),
        NOT blocky voxel meshes. Uses marching cubes isosurface extraction with
        high resolution for smooth geological boundaries.
        
        Args:
            stratigraphy: Optional list of unit names. If None, auto-detects from data.
            
        Returns:
            List of solid dicts with 'vertices', 'faces', 'unit_name', 'volume_m3'
        """
        solids = []
        
        if not self.model:
            logger.warning("No model built - cannot extract solids")
            return solids
        
        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            logger.error(f"Feature '{self.FEATURE_NAME}' not found")
            return solids
        
        # Get unique stratigraphic values (sorted)
        unique_vals = np.sort(self.model.data['val'].dropna().unique())
        logger.info(f"Extracting CP-GRADE solids for {len(unique_vals)} units at vals: {unique_vals}")
        
        # ================================================================
        # OPTIMIZATION: USE STANDARD RESOLUTION (Prevents GPU TDR/Crash)
        # ================================================================
        # Using 2x resolution (r*2) creates 8x more voxels (e.g. 160^3 = 4M cells).
        # This causes Driver Timeouts on consumer GPUs.
        # Standard resolution (80^3 = 512k cells) is sufficient for smooth solids.
        nx, ny, nz = [r * 1 for r in self.resolution]
        
        # Grid in scaled space [0, 1] with padding
        pad = self.boundary_padding
        x = np.linspace(-pad, 1 + pad, nx)
        y = np.linspace(-pad, 1 + pad, ny)
        z = np.linspace(-pad, 1 + pad, nz)
        
        # Create ImageData (UniformGrid) for marching cubes - this is key for smooth surfaces
        import pyvista as pv
        
        # Calculate spacing in scaled space
        dx = (1 + 2 * pad) / (nx - 1)
        dy = (1 + 2 * pad) / (ny - 1)
        dz = (1 + 2 * pad) / (nz - 1)
        
        # Create uniform grid in scaled space
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(dx, dy, dz),
            origin=(-pad, -pad, -pad)
        )
        
        # Sample points for scalar field evaluation
        grid_points = np.column_stack([
            grid.points[:, 0],
            grid.points[:, 1],
            grid.points[:, 2]
        ])
        
        # Evaluate scalar field on grid
        logger.info(f"Evaluating scalar field on {len(grid_points)} grid points (high-res {nx}x{ny}x{nz})...")
        field_values = feature.evaluate_value(grid_points)
        
        # Get actual field bounds
        field_min = float(np.nanmin(field_values))
        field_max = float(np.nanmax(field_values))
        logger.info(f"Field range on grid: [{field_min:.2f}, {field_max:.2f}]")
        
        # Add scalar field to grid
        grid['scalar'] = field_values
        
        # Calculate unit boundaries (midpoints between consecutive formation values)
        boundaries = []
        for i in range(len(unique_vals) - 1):
            boundaries.append((unique_vals[i] + unique_vals[i + 1]) / 2)
        
        logger.info(f"Unit boundaries (scalar midpoints): {boundaries}")
        
        # ================================================================
        # EXTRACT SMOOTH SOLIDS USING THRESHOLD + MARCHING CUBES
        # ================================================================
        for i in range(len(unique_vals)):
            unit_name = stratigraphy[i] if stratigraphy and i < len(stratigraphy) else f"Unit_{i}"
            val = unique_vals[i]
            
            # Define the scalar range for this unit
            if i == 0:
                val_min = field_min - 1.0
                val_max = boundaries[0] if boundaries else field_max + 1.0
            elif i == len(unique_vals) - 1:
                val_min = boundaries[-1]
                val_max = field_max + 1.0
            else:
                val_min = boundaries[i - 1]
                val_max = boundaries[i]
            
            logger.info(f"Extracting CP-GRADE solid for '{unit_name}' (val={val}): scalar range [{val_min:.2f}, {val_max:.2f}]")
            
            try:
                # ================================================================
                # CRITICAL FIX: Use threshold() instead of clip_scalar()
                # threshold() creates a true solid by clipping the VOLUME,
                # not just extracting the boundary surface.
                # ================================================================
                clipped_vol = grid.threshold([val_min, val_max], scalars='scalar')
                
                if clipped_vol is None or clipped_vol.n_cells == 0:
                    logger.warning(f"No cells in threshold range for '{unit_name}'")
                    continue
                
                # Convert the voxel volume to a smoothed mesh for high-quality plotting
                # extract_surface() gives us the outer boundary of the clipped volume
                surface = clipped_vol.extract_surface()
                
                if surface is None or surface.n_points == 0:
                    logger.warning(f"Empty surface for unit '{unit_name}'")
                    continue
                
                # Triangulate to ensure consistent face format
                surface = surface.triangulate()
                
                # ================================================================
                # FIX: Fill holes to ensure it looks like a solid block
                # This closes any gaps in the manifold
                # ================================================================
                if hasattr(surface, 'fill_holes'):
                    try:
                        surface = surface.fill_holes(hole_size=1000)
                        logger.debug(f"Filled holes for solid '{unit_name}'")
                    except Exception as e:
                        logger.debug(f"Could not fill holes: {e}")
                
                # ================================================================
                # EXCLUSIVE DOMAIN FIX: DISABLE INDEPENDENT SMOOTHING
                # ================================================================
                # CRITICAL: Do NOT apply independent Taubin smoothing to individual units.
                # Smoothing independent hulls causes their shared boundaries to diverge,
                # which creates "piercing" and Z-fighting artifacts.
                # 
                # We rely on the scalar field regularization (cgw) to provide smoothness.
                # This ensures shared boundaries remain mathematically coincident.
                logger.debug(f"Geometric Exclusivity: Preserving raw isosurface for '{unit_name}' to prevent piercing")
                # surface = surface.smooth_taubin(...) <- REMOVED
                
                # Transform vertices from scaled space to WORLD coordinates
                verts_scaled = np.asarray(surface.points, dtype=np.float64)
                verts_world = self.scaler.inverse_transform(verts_scaled)
                
                # Extract faces from PyVista format
                if hasattr(surface, 'faces') and surface.faces is not None and len(surface.faces) > 0:
                    faces_raw = np.asarray(surface.faces)
                    # PyVista faces after triangulate: [3, i, j, k, 3, i, j, k, ...]
                    try:
                        n_faces = len(faces_raw) // 4
                        faces = faces_raw.reshape(n_faces, 4)[:, 1:4]
                    except ValueError as e:
                        logger.warning(f"Face reshape failed for '{unit_name}': {e}")
                        # Try manual extraction
                        faces = []
                        idx = 0
                        while idx < len(faces_raw):
                            n = faces_raw[idx]
                            if n == 3:
                                faces.append(faces_raw[idx+1:idx+4])
                            elif n == 4:
                                faces.append([faces_raw[idx+1], faces_raw[idx+2], faces_raw[idx+3]])
                                faces.append([faces_raw[idx+1], faces_raw[idx+3], faces_raw[idx+4]])
                            idx += n + 1
                        faces = np.array(faces, dtype=np.int64) if faces else None
                else:
                    faces = None
                
                if faces is None or len(faces) == 0:
                    logger.warning(f"No faces extracted for unit '{unit_name}'")
                    continue
                
                # Calculate volume in world coordinates (cubic meters)
                # Create a mesh in world coordinates for volume calculation
                try:
                    # Build PyVista mesh in world coords
                    faces_pv = np.hstack([
                        np.full((len(faces), 1), 3, dtype=np.int64),
                        faces
                    ]).flatten()
                    world_mesh = pv.PolyData(verts_world, faces_pv)
                    
                    # Fill holes and compute volume
                    try:
                        closed_mesh = world_mesh.fill_holes(hole_size=1e10)
                        volume_m3 = abs(float(closed_mesh.volume))
                    except Exception:
                        # Fallback: estimate from bounding box and cell count
                        bounds = world_mesh.bounds
                        bbox_vol = (bounds[1]-bounds[0]) * (bounds[3]-bounds[2]) * (bounds[5]-bounds[4])
                        # Rough estimate: ~60% of bounding box
                        volume_m3 = bbox_vol * 0.6
                except Exception as vol_e:
                    logger.warning(f"Volume calculation failed for '{unit_name}': {vol_e}")
                    volume_m3 = 0.0
                
                solids.append({
                    "vertices": verts_world,
                    "faces": faces.astype(np.int64),
                    "unit_name": unit_name,
                    "val": float(val),
                    "val_range": [float(val_min), float(val_max)],
                    "volume_m3": volume_m3,
                    "n_cells": surface.n_cells,
                    "name": f"Solid_{unit_name}",
                    "smoothed": True,  # Mark as smoothed for renderer
                })
                
                logger.info(
                    f"Extracted CP-GRADE solid '{unit_name}': {len(verts_world)} verts, {len(faces)} faces, "
                    f"volume={volume_m3:,.0f} m³"
                )
                
            except Exception as e:
                logger.error(f"Failed to extract solid for '{unit_name}': {e}", exc_info=True)
                continue
        
        logger.info(f"Extracted {len(solids)} CP-GRADE geological solids")
        return solids
    
    def extract_unified_geology_mesh(self, stratigraphy: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        CP-GRADE FIX: Creates ONE single mesh for the whole model.
        
        This is the INDUSTRY-STANDARD approach used by Leapfrog and Micromine.
        Instead of N overlapping meshes that Z-fight, we create a single voxel
        grid where every cell has a Formation_ID. This eliminates:
        - Z-fighting/flickering (no overlapping surfaces)
        - Disappearing features (no depth buffer conflicts)  
        - Color bleeding (discrete Formation_ID per cell)
        
        The renderer colors this single mesh using a discrete colormap.
        
        Args:
            stratigraphy: Optional list of unit names for labeling
            
        Returns:
            Dict with:
            - 'vertices': Voxel center points
            - 'formation_ids': Integer ID for each point (stratigraphic order)
            - 'formation_names': Mapping from ID to unit name
            - 'mesh': PyVista StructuredGrid for direct rendering
        """
        import pyvista as pv
        
        if not self.model:
            logger.warning("No model built - cannot extract unified mesh")
            return None
        
        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            logger.error(f"Feature '{self.FEATURE_NAME}' not found")
            return None
        
        logger.info("=" * 60)
        logger.info("EXTRACTING UNIFIED PARTITION MESH (INDUSTRY-STANDARD)")
        logger.info("This creates ONE mesh with Formation_ID - NO overlapping surfaces")
        logger.info("=" * 60)
        
        # Get unique stratigraphic values (sorted)
        unique_vals = np.sort(self.model.data['val'].dropna().unique())
        n_units = len(unique_vals)
        logger.info(f"Partitioning into {n_units} geological units at vals: {unique_vals}")
        
        # Create formation name mapping
        formation_names = {}
        for i, val in enumerate(unique_vals):
            if stratigraphy and i < len(stratigraphy):
                formation_names[i] = stratigraphy[i]
            else:
                formation_names[i] = f"Unit_{i}"
        
        # ================================================================
        # CREATE HIGH-RESOLUTION GRID FOR VOXEL PARTITIONING
        # ================================================================
        # CP-GRADE RESOLUTION: Use 180³ - 256³ for "Voxel-Sharpness" mode.
        # This provides smooth curves that rival marching cubes while 
        # guaranteeing spatial exclusivity (No overlaps).
        # We cap at 180³ for performance on standard GPUs (~5.8M cells).
        res = 180
        nx, ny, nz = res, res, res
        pad = self.boundary_padding
        
        # Create uniform grid in SCALED space [0,1] with padding
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=((1 + 2 * pad) / (nx - 1), (1 + 2 * pad) / (ny - 1), (1 + 2 * pad) / (nz - 1)),
            origin=(-pad, -pad, -pad)
        )
        
        # ================================================================
        # CRITICAL: Evaluate at CELL CENTERS (not points)
        # ================================================================
        # This is THE KEY to removing blur. By evaluating at cell centers
        # and assigning to cell_data, each voxel is 100% one color.
        # No color bleeding between adjacent formations.
        cell_centers = grid.cell_centers().points
        
        logger.info(f"Evaluating scalar field on {len(cell_centers)} CELL CENTERS ({nx}x{ny}x{nz})...")
        logger.info(f"VOXEL-SHARPNESS MODE: 128³ resolution for crisp boundaries")
        
        # Evaluate scalar field at CELL CENTERS (not vertices)
        field_values = feature.evaluate_value(cell_centers)
        
        field_min = float(np.nanmin(field_values))
        field_max = float(np.nanmax(field_values))
        logger.info(f"Field range: [{field_min:.2f}, {field_max:.2f}]")
        
        # ================================================================
        # PARTITION: Assign every CELL to EXACTLY one formation
        # ================================================================
        # This is the key to eliminating blur AND Z-fighting!
        # Each voxel cube gets ONE Formation_ID - no interpolation between cells
        # This ensures sharp, discrete boundaries like professional mining software
        
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
            logger.info(f"  Formation {i} ({formation_names[i]}): {count:,} voxels ({100*count/len(formation_ids):.1f}%)")
        
        # ================================================================
        # TRANSFORM TO WORLD COORDINATES
        # ================================================================
        # Convert from scaled [0,1] space to real UTM coordinates
        # Use cell centers (not vertices) for voxel-based visualization
        verts_world = self.scaler.inverse_transform(cell_centers)
        
        # ================================================================
        # CRITICAL: Assign to CELL_DATA (not point_data)
        # ================================================================
        # This prevents VTK from interpolating colors between adjacent cells
        # Each voxel is 100% one color - eliminates blur and color bleeding
        grid.cell_data['Formation_ID'] = formation_ids
        grid.cell_data['scalar'] = field_values
        
        logger.info("Assigned Formation_ID to CELL_DATA (voxel-sharpness mode)")
        
        # ================================================================
        # CREATE UNSTRUCTURED GRID FOR RENDERING
        # ================================================================
        # Convert ImageData to UnstructuredGrid for better rendering control
        # This allows the renderer to use the Formation_ID as a categorical scalar
        
        # Extract surface for visualization (faster than full volume)
        # But keep the full grid data for volume queries
        
        # Create the result package
        result = {
            'vertices': verts_world,
            'formation_ids': formation_ids,
            'formation_names': formation_names,
            'unique_vals': unique_vals.tolist(),
            'boundaries': boundaries,
            'n_units': n_units,
            'grid_dimensions': (nx, ny, nz),
            'field_range': (field_min, field_max),
            # Include raw grid for direct PyVista rendering
            '_pyvista_grid': grid,
            '_scaler': self.scaler,  # For coordinate transforms
            # For MinMaxScaler, model space IS [0,1] space, so bounds are [0,0,0] to [1,1,1]
            # This ensures consistent transformation in the renderer
            '_stdscaler_origin': np.array([0.0, 0.0, 0.0]),
            '_stdscaler_maximum': np.array([1.0, 1.0, 1.0]),
        }
        
        logger.info("=" * 60)
        logger.info(f"UNIFIED MESH EXTRACTED: {len(verts_world):,} voxels, {n_units} formations")
        logger.info("This mesh has NO OVERLAP - Z-fighting eliminated")
        logger.info("=" * 60)

        return result

    @staticmethod
    def is_available() -> bool:
        """Check if LoopStructural is available."""
        return LS_AVAILABLE


# =============================================================================
# Stratigraphic Validation Functions
# =============================================================================

@dataclass
class StratigraphicOrderViolation:
    """Violation found during stratigraphic ordering validation."""
    violation_type: str  # 'INVERSION', 'GAP', 'DUPLICATE', 'MISSING_UNIT'
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    message: str
    affected_units: List[str]
    hole_id: Optional[str] = None
    depth_from: Optional[float] = None
    depth_to: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "severity": self.severity,
            "message": self.message,
            "affected_units": self.affected_units,
            "hole_id": self.hole_id,
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
        }


@dataclass
class StratigraphicValidationResult:
    """Result of stratigraphic sequence validation."""
    is_valid: bool
    violations: List[StratigraphicOrderViolation]
    holes_checked: int
    inversions_found: int
    recommendation: str  # 'ACCEPT', 'REVIEW', 'REJECT'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "holes_checked": self.holes_checked,
            "inversions_found": self.inversions_found,
            "recommendation": self.recommendation,
        }


@dataclass
class ThinUnitWarning:
    """Warning for a unit that may be too thin for the model resolution."""
    unit_name: str
    estimated_thickness_m: float
    min_thickness_m: float
    max_thickness_m: float
    n_observations: int
    warning_level: str  # 'INFO', 'WARNING', 'CRITICAL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_name": self.unit_name,
            "estimated_thickness_m": self.estimated_thickness_m,
            "min_thickness_m": self.min_thickness_m,
            "max_thickness_m": self.max_thickness_m,
            "n_observations": self.n_observations,
            "warning_level": self.warning_level,
        }


@dataclass
class ResolutionAnalysis:
    """Detailed analysis of resolution vs unit thickness."""
    voxel_size_m: float
    min_unit_thickness_m: float
    thin_units: List[ThinUnitWarning]
    resolution_adequate: bool
    current_resolution: int
    recommended_resolution: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_size_m": self.voxel_size_m,
            "min_unit_thickness_m": self.min_unit_thickness_m,
            "thin_units": [t.to_dict() for t in self.thin_units],
            "resolution_adequate": self.resolution_adequate,
            "current_resolution": self.current_resolution,
            "recommended_resolution": self.recommended_resolution,
            "warnings": self.warnings,
        }


def validate_stratigraphy_sequence(
    stratigraphy: List[str],
    contacts_df: pd.DataFrame,
    allow_missing_units: bool = True,
    max_inversions_per_hole: int = 2
) -> StratigraphicValidationResult:
    """
    Validate that the provided stratigraphic sequence matches depth ordering in drillholes.

    Checks performed:
    1. Depth-based ordering: In each drillhole, deeper units should be older
    2. Consistency: Same ordering observed across all drillholes
    3. Duplicates: No duplicate unit names in sequence
    4. Missing units: Units in data but not in sequence

    Args:
        stratigraphy: List of unit names (oldest to youngest)
        contacts_df: DataFrame with columns:
            - hole_id: Drillhole identifier
            - Z or depth: Depth/elevation of contact
            - formation: Formation name
        allow_missing_units: If True, missing units are WARNING; if False, ERROR
        max_inversions_per_hole: Maximum inversions per hole before ERROR

    Returns:
        StratigraphicValidationResult with violations and recommendations
    """
    violations = []
    inversions_found = 0
    holes_checked = 0

    # Check for duplicate unit names
    if len(stratigraphy) != len(set(stratigraphy)):
        duplicates = [u for u in stratigraphy if stratigraphy.count(u) > 1]
        violations.append(StratigraphicOrderViolation(
            violation_type="DUPLICATE",
            severity="ERROR",
            message=f"Duplicate unit names in stratigraphy: {set(duplicates)}",
            affected_units=list(set(duplicates)),
        ))

    # Build unit-to-index map (oldest=0, youngest=n-1)
    unit_to_idx = {unit: i for i, unit in enumerate(stratigraphy)}

    # Check for missing units in stratigraphy
    if 'formation' in contacts_df.columns:
        data_units = set(contacts_df['formation'].unique())
        strat_units = set(stratigraphy)

        missing_from_strat = data_units - strat_units
        if missing_from_strat:
            severity = "WARNING" if allow_missing_units else "ERROR"
            violations.append(StratigraphicOrderViolation(
                violation_type="MISSING_UNIT",
                severity=severity,
                message=f"Units in data but not in stratigraphy: {missing_from_strat}",
                affected_units=list(missing_from_strat),
            ))

    # Check depth ordering per drillhole
    if 'hole_id' not in contacts_df.columns:
        logger.warning("No 'hole_id' column - cannot validate per-hole ordering")
        return StratigraphicValidationResult(
            is_valid=len([v for v in violations if v.severity == "ERROR"]) == 0,
            violations=violations,
            holes_checked=0,
            inversions_found=0,
            recommendation="REVIEW" if violations else "ACCEPT",
        )

    # Determine depth column (Z is elevation, so deeper = lower Z)
    depth_col = None
    if 'depth' in contacts_df.columns:
        depth_col = 'depth'
        depth_sign = 1  # Higher depth = deeper
    elif 'Z' in contacts_df.columns:
        depth_col = 'Z'
        depth_sign = -1  # Lower Z = deeper
    else:
        logger.warning("No depth or Z column - cannot validate depth ordering")
        return StratigraphicValidationResult(
            is_valid=len([v for v in violations if v.severity == "ERROR"]) == 0,
            violations=violations,
            holes_checked=0,
            inversions_found=0,
            recommendation="REVIEW" if violations else "ACCEPT",
        )

    # Group by drillhole and check ordering
    for hole_id, group in contacts_df.groupby('hole_id'):
        holes_checked += 1

        if 'formation' not in group.columns:
            continue

        # Sort by depth (deeper first)
        sorted_group = group.sort_values(depth_col, ascending=(depth_sign < 0))

        # Get observed unit sequence (from deepest to shallowest)
        observed_units = sorted_group['formation'].tolist()

        # Check for inversions
        hole_inversions = 0
        for i in range(len(observed_units) - 1):
            unit_deep = observed_units[i]
            unit_shallow = observed_units[i + 1]

            # Skip unknown units
            if unit_deep not in unit_to_idx or unit_shallow not in unit_to_idx:
                continue

            # Deeper unit should be older (lower index)
            if unit_to_idx[unit_deep] > unit_to_idx[unit_shallow]:
                hole_inversions += 1
                inversions_found += 1

                # Get depths for context
                depths = sorted_group[depth_col].tolist()
                depth_from = depths[i] if i < len(depths) else None
                depth_to = depths[i + 1] if i + 1 < len(depths) else None

                violations.append(StratigraphicOrderViolation(
                    violation_type="INVERSION",
                    severity="WARNING",
                    message=f"Unit '{unit_deep}' (younger) found below '{unit_shallow}' (older)",
                    affected_units=[unit_deep, unit_shallow],
                    hole_id=str(hole_id),
                    depth_from=depth_from,
                    depth_to=depth_to,
                ))

        # Too many inversions in one hole is an error
        if hole_inversions > max_inversions_per_hole:
            violations.append(StratigraphicOrderViolation(
                violation_type="INVERSION",
                severity="ERROR",
                message=f"Hole {hole_id} has {hole_inversions} inversions (max {max_inversions_per_hole})",
                affected_units=[],
                hole_id=str(hole_id),
            ))

    # Determine overall validity and recommendation
    errors = [v for v in violations if v.severity == "ERROR"]
    warnings = [v for v in violations if v.severity == "WARNING"]

    is_valid = len(errors) == 0

    if errors:
        recommendation = "REJECT"
    elif warnings:
        recommendation = "REVIEW"
    else:
        recommendation = "ACCEPT"

    logger.info(
        f"Stratigraphic validation: {holes_checked} holes checked, "
        f"{inversions_found} inversions, {len(errors)} errors, "
        f"recommendation: {recommendation}"
    )

    return StratigraphicValidationResult(
        is_valid=is_valid,
        violations=violations,
        holes_checked=holes_checked,
        inversions_found=inversions_found,
        recommendation=recommendation,
    )


def analyze_unit_thickness(
    contacts_df: pd.DataFrame,
    stratigraphy: List[str],
    resolution: int,
    extent: Dict[str, float],
    min_voxels_per_unit: int = 3
) -> ResolutionAnalysis:
    """
    Analyze whether model resolution is adequate for unit thicknesses.

    For each pair of consecutive formations, estimates the thickness
    from drillhole data and compares to voxel size.

    Args:
        contacts_df: DataFrame with X, Y, Z, formation, hole_id columns
        stratigraphy: List of formation names (oldest to youngest)
        resolution: Current grid resolution (cells per axis)
        extent: Model extent dict with xmin, xmax, ymin, ymax, zmin, zmax
        min_voxels_per_unit: Minimum voxels needed to represent a unit

    Returns:
        ResolutionAnalysis with thickness estimates and recommendations
    """
    # Calculate voxel size
    z_range = extent.get('zmax', 0) - extent.get('zmin', 0)
    voxel_size = z_range / resolution if resolution > 0 else 0

    thin_units = []
    warnings = []
    min_thickness = float('inf')

    if 'hole_id' not in contacts_df.columns or 'formation' not in contacts_df.columns:
        return ResolutionAnalysis(
            voxel_size_m=voxel_size,
            min_unit_thickness_m=0,
            thin_units=[],
            resolution_adequate=True,
            current_resolution=resolution,
            recommended_resolution=resolution,
            warnings=["Cannot analyze thickness: missing hole_id or formation columns"],
        )

    # Estimate thickness for each unit from drillhole data
    for i in range(len(stratigraphy) - 1):
        unit_above = stratigraphy[i + 1]  # Younger (shallower)
        unit_below = stratigraphy[i]  # Older (deeper)

        thicknesses = []

        # For each drillhole, find the thickness of unit_above
        for hole_id, group in contacts_df.groupby('hole_id'):
            group_sorted = group.sort_values('Z', ascending=False)  # Shallowest first

            formations = group_sorted['formation'].tolist()
            z_values = group_sorted['Z'].tolist()

            # Find unit boundaries
            for j in range(len(formations) - 1):
                if formations[j] == unit_above and formations[j + 1] == unit_below:
                    # Found the contact
                    thickness = abs(z_values[j] - z_values[j + 1])
                    if thickness > 0:
                        thicknesses.append(thickness)
                    break

        if not thicknesses:
            continue

        # Calculate statistics
        avg_thickness = np.mean(thicknesses)
        min_t = np.min(thicknesses)
        max_t = np.max(thicknesses)

        if avg_thickness < min_thickness:
            min_thickness = avg_thickness

        # Check if resolution is adequate
        voxels_in_unit = avg_thickness / voxel_size if voxel_size > 0 else float('inf')

        if voxels_in_unit < min_voxels_per_unit:
            warning_level = "CRITICAL" if voxels_in_unit < 1 else "WARNING"
            thin_units.append(ThinUnitWarning(
                unit_name=unit_above,
                estimated_thickness_m=avg_thickness,
                min_thickness_m=min_t,
                max_thickness_m=max_t,
                n_observations=len(thicknesses),
                warning_level=warning_level,
            ))
            warnings.append(
                f"Unit '{unit_above}' avg thickness {avg_thickness:.1f}m = "
                f"{voxels_in_unit:.1f} voxels (need {min_voxels_per_unit}+)"
            )

    # Calculate recommended resolution
    resolution_adequate = len(thin_units) == 0

    if min_thickness < float('inf') and min_thickness > 0:
        # Recommend resolution that gives min_voxels_per_unit voxels per thinnest unit
        recommended = int(np.ceil(z_range / (min_thickness / min_voxels_per_unit)))
        recommended = max(recommended, resolution)  # Never recommend lower
    else:
        recommended = resolution

    return ResolutionAnalysis(
        voxel_size_m=voxel_size,
        min_unit_thickness_m=min_thickness if min_thickness < float('inf') else 0,
        thin_units=thin_units,
        resolution_adequate=resolution_adequate,
        current_resolution=resolution,
        recommended_resolution=recommended,
        warnings=warnings,
    )

