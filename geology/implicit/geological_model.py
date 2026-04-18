"""
Geological Model Builder — Main Orchestrator.
===============================================

Coordinates the full implicit geological modelling workflow:
contacts → SDF constraints → augmented interpolation → surface extraction
→ domain assignment → validation → audit record.

Usage::

    builder = GeologicalModelBuilder(config)
    builder.set_contacts(contacts_df)
    builder.set_orientations(orientations_df)        # optional
    builder.set_stratigraphic_column(column_def)

    result = builder.build()

    # result contains:
    #   surfaces: dict of {name: pyvista.PolyData}
    #   evaluate_fn: callable
    #   contact_misfit: QC DataFrame
    #   audit_record: dict
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from geostats.arbf.utils import rotation_matrix, scale_matrix

from .audit import (
    GeologicalModelAudit,
    compute_contact_data_hash,
    generate_run_id,
)
from .contact_data import (
    ContactSet,
    StratigraphicColumn,
    dip_azimuth_to_normal,
    contacts_dataframe_to_contact_set,
    orientations_dataframe_to_list,
)
from .signed_distance import (
    construct_sdf_constraints,
    estimate_contact_normals,
    extract_contacts_from_lithology,
)
from .scalar_field import (
    assemble_augmented_matrix,
    solve_augmented_system,
    evaluate_scalar_field,
    make_evaluate_fn,
)
from .surface_extraction import (
    evaluate_field_on_grid,
    extract_isosurface,
    cleanup_mesh,
    field_to_pyvista_mesh,
)
from .validation import (
    check_contact_honouring,
    contact_honouring_summary,
)
from .domain_model import assign_domains_potential_field
from .stratigraphy import (
    StratigraphicModelColumn,
    build_potential_field_constraints,
    extract_stratigraphic_surfaces,
)
from .vein_model import build_vein_model, extract_vein_intersections
from .fault_model import (
    FaultDefinition,
    build_faulted_geological_model,
)
from .fold_frame import FoldFrame, FoldFrameConfig

logger = logging.getLogger(__name__)


class GeologicalModelBuilder:
    """Main orchestrator for implicit geological modelling.

    Parameters
    ----------
    config : dict
        Configuration with keys:

        - ``model_type`` : str (``"stratiform"``, ``"vein"``, ``"intrusive"``, ``"structural"``)
        - ``kernel_type`` : str (default ``"spheroidal"``)
        - ``alpha`` : float (default 1.0)
        - ``range_max``, ``range_mid``, ``range_min`` : float
        - ``azimuth``, ``dip``, ``pitch`` : float (degrees)
        - ``nugget`` : float
        - ``accuracy`` : float (regularisation)
        - ``constraint_method`` : str (``"gradient"`` or ``"offset"``)
        - ``offset_distance`` : float (for offset method)
        - ``grid_resolution`` : float (metres)
        - ``grid_extent`` : dict or ``"auto"``
        - ``drift_type`` : str (``"constant"`` or ``"linear"``)
        - ``tolerance`` : float (contact honouring tolerance, metres)
        - ``operator`` : str
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

        # Kernel parameters
        self._kernel_type = self._config.get("kernel_type", "spheroidal")
        self._alpha = float(self._config.get("alpha", 1.0))
        self._range_max = float(self._config.get("range_max", 300.0))
        self._range_mid = float(self._config.get("range_mid", 200.0))
        self._range_min = float(self._config.get("range_min", 150.0))
        self._azimuth = float(self._config.get("azimuth", 0.0))
        self._dip = float(self._config.get("dip", 0.0))
        self._pitch = float(self._config.get("pitch", 0.0))
        self._nugget = float(self._config.get("nugget", 0.0))
        self._accuracy = float(self._config.get("accuracy", 1e-8))
        self._drift_type = self._config.get("drift_type", "constant")
        self._constraint_method = self._config.get("constraint_method", "gradient")
        self._offset_distance = float(self._config.get("offset_distance", 2.0))
        self._grid_resolution = float(self._config.get("grid_resolution", 10.0))
        self._grid_extent = self._config.get("grid_extent", "auto")
        self._tolerance = float(self._config.get("tolerance", 1.0))
        self._model_type = self._config.get("model_type", "stratiform")
        self._operator = self._config.get("operator", "")

        # Rotation/scaling matrices
        self._R = rotation_matrix(self._azimuth, self._dip, self._pitch)
        self._S = scale_matrix(self._range_max, self._range_mid, self._range_min)

        # Data (set by user)
        self._contacts_df: Optional[pd.DataFrame] = None
        self._orientations_df: Optional[pd.DataFrame] = None
        self._surveys_df: Optional[pd.DataFrame] = None
        self._stratigraphic_column: Optional[StratigraphicColumn] = None
        self._grouping: Optional[Dict[str, List[str]]] = None

        # Phase 2-5 extensions
        self._fault_definitions: List[FaultDefinition] = []
        self._fold_config: Optional[FoldFrameConfig] = None
        self._vein_lithology_df: Optional[pd.DataFrame] = None
        self._vein_code: Optional[str] = None

        # Callbacks
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    # ──────────────────────────────────────────────────────────────
    # Data setters
    # ──────────────────────────────────────────────────────────────

    def set_contacts(self, contacts_df: pd.DataFrame) -> None:
        """Set contact points DataFrame.

        Expected columns: hole_id, depth, X, Y, Z, unit_above, unit_below, surface_name
        """
        self._contacts_df = contacts_df

    def set_orientations(self, orientations_df: pd.DataFrame) -> None:
        """Set orientation measurements DataFrame.

        Expected columns: X, Y, Z, dip, azimuth, [feature_type, hole_id]
        """
        self._orientations_df = orientations_df

    def set_surveys(self, surveys_df: pd.DataFrame) -> None:
        """Set drillhole survey (deviation) data for trajectory-based normal estimation.

        Expected columns: hole_id/HOLEID, depth/FROM, dip/DIP, azimuth/AZIMUTH.
        When surveys are set and contacts have hole_id + depth, the builder
        will add drillhole direction vectors at each contact depth and use
        them to estimate surface normals (survey method).  This works even
        without structural measurements.
        """
        self._surveys_df = surveys_df

    def set_stratigraphic_column(self, column: StratigraphicColumn) -> None:
        """Set stratigraphic column definition."""
        self._stratigraphic_column = column

    def set_lithology_grouping(self, grouping: Dict[str, List[str]]) -> None:
        """Set lithology code grouping for contact extraction."""
        self._grouping = grouping

    def set_progress_callback(self, callback: Callable[[int, str], None]) -> None:
        """Set progress callback for UI integration."""
        self._progress_callback = callback

    def set_fault_definitions(self, faults: List[Dict[str, Any]]) -> None:
        """Set fault definitions for faulted modelling (Phase 4).

        Each dict should have: name, contacts (DataFrame), orientations (DataFrame),
        displacement_vector (3-tuple), chronological_order (int).
        """
        self._fault_definitions = []
        for fd in faults:
            self._fault_definitions.append(FaultDefinition(
                name=fd.get("name", "Fault"),
                fault_type=fd.get("fault_type", "normal"),
                displacement=float(fd.get("displacement", 0.0)),
                displacement_vector=np.array(
                    fd.get("displacement_vector", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                ),
                contacts=fd.get("contacts"),
                orientations=fd.get("orientations"),
                chronological_order=int(fd.get("chronological_order", 0)),
            ))

    def set_fold_config(self, config: Dict[str, Any]) -> None:
        """Set fold frame configuration (Phase 5)."""
        self._fold_config = FoldFrameConfig(
            fold_axis_azimuth=float(config.get("fold_axis_azimuth", 0.0)),
            fold_axis_plunge=float(config.get("fold_axis_plunge", 0.0)),
            fold_type=config.get("fold_type", "cylindrical"),
            wavelength=config.get("wavelength"),
            auto_detect=config.get("auto_detect", True),
            regularisation_weight=float(config.get("regularisation_weight", 1.0)),
        )

    def set_vein_data(
        self, lithology_df: pd.DataFrame, vein_code: str,
    ) -> None:
        """Set vein data for vein modelling (Phase 3).

        Parameters
        ----------
        lithology_df : DataFrame with hole_id, from_depth, to_depth, lith_code, X, Y, Z
        vein_code : lithology code for the vein material
        """
        self._vein_lithology_df = lithology_df
        self._vein_code = vein_code

    # ──────────────────────────────────────────────────────────────
    # Build — dispatches to model-type-specific builders
    # ──────────────────────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        """Execute the modelling workflow, auto-dispatching by model_type.

        Returns
        -------
        dict with:
            surfaces : {name: pyvista.PolyData}
            scalar_field : np.ndarray (nx, ny, nz)
            evaluate_fn : callable
            contact_misfit : pd.DataFrame
            audit_record : dict
            domain_codes : np.ndarray (optional)
        """
        # Dispatch to specialized builders when additional data is set
        if self._model_type == "vein" and self._vein_lithology_df is not None:
            return self.build_vein()
        if self._fold_config is not None and self._orientations_df is not None:
            return self.build_with_fold_frame()
        if (self._model_type == "stratiform"
                and self._stratigraphic_column is not None
                and self._stratigraphic_column.units):
            return self.build_stratigraphic()
        if self._fault_definitions:
            return self.build_faulted()
        return self._build_single_surface()

    def _build_single_surface(self) -> Dict[str, Any]:
        """Original single-surface SDF workflow (Phase 1)."""
        t_start = time.perf_counter()
        audit = GeologicalModelAudit(
            run_id=generate_run_id(),
            timestamp=pd.Timestamp.now().isoformat(),
            operator=self._operator,
            model_type=self._model_type,
            kernel_type=self._kernel_type,
            alpha=self._alpha,
            range_max=self._range_max,
            range_mid=self._range_mid,
            range_min=self._range_min,
            azimuth=self._azimuth,
            dip=self._dip,
            pitch=self._pitch,
            nugget=self._nugget,
            accuracy=self._accuracy,
            drift_type=self._drift_type,
            constraint_method=self._constraint_method,
            offset_distance=self._offset_distance,
            grid_resolution=self._grid_resolution,
        )

        self._progress(5, "Preparing contact data")

        if self._contacts_df is None or self._contacts_df.empty:
            raise ValueError("No contact data provided. Call set_contacts() first.")

        # ── Step 1: Estimate contact normals ──
        self._progress(10, "Estimating contact normals")
        contacts_have_dirs = (
            "hole_dir_x" in self._contacts_df.columns
            and "hole_dir_y" in self._contacts_df.columns
        )
        if self._orientations_df is not None and len(self._orientations_df) > 0:
            normal_method = "structural"    # best: measured structural data
        elif contacts_have_dirs:
            normal_method = "survey"        # good: survey-derived apparent dip
        elif len(self._contacts_df) >= 10:
            normal_method = "local_plane"   # ok: PCA from contact positions
        else:
            normal_method = "drillhole"     # last resort: vertical

        logger.info("Normal estimation method: %s", normal_method)

        contact_normals = estimate_contact_normals(
            self._contacts_df,
            method=normal_method,
            structural_data=self._orientations_df,
        )

        contact_coords = self._contacts_df[["X", "Y", "Z"]].values.astype(np.float64)
        audit.n_contacts = len(contact_coords)
        audit.contact_data_hash = compute_contact_data_hash(contact_coords)

        # ── Step 2: Construct SDF constraints ──
        self._progress(15, "Constructing SDF constraints")
        value_coords, value_data, gradient_coords, gradient_normals = \
            construct_sdf_constraints(
                contact_coords, contact_normals,
                method=self._constraint_method,
                offset_distance=self._offset_distance,
            )

        # ── Step 2b: Add orientation gradient constraints ──
        if self._orientations_df is not None and len(self._orientations_df) > 0:
            orient_list = orientations_dataframe_to_list(self._orientations_df)
            orient_coords = np.array(
                [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
            )
            orient_normals = np.array(
                [o.normal for o in orient_list], dtype=np.float64,
            )
            # Append to gradient constraints
            if gradient_coords.size > 0:
                gradient_coords = np.vstack([gradient_coords, orient_coords])
                gradient_normals = np.vstack([gradient_normals, orient_normals])
            else:
                gradient_coords = orient_coords
                gradient_normals = orient_normals

            audit.n_orientations = len(orient_list)

        N_v = value_coords.shape[0]
        N_g = gradient_coords.shape[0] if gradient_coords.size > 0 else 0

        logger.info(
            "Constraints: %d value, %d gradient (total system size: %d)",
            N_v, N_g, N_v + N_g,
        )

        # ── Step 3: Assemble augmented kernel matrix ──
        self._progress(25, f"Assembling {N_v + N_g}x{N_v + N_g} kernel matrix")

        K_aug, N_v, N_g = assemble_augmented_matrix(
            value_coords, gradient_coords, gradient_normals,
            kernel_type=self._kernel_type,
            alpha=self._alpha,
            range_=self._range_max,
            nugget=self._nugget,
            accuracy=self._accuracy,
            drift_type=self._drift_type,
            R=self._R,
            S=self._S,
        )
        audit.matrix_size = K_aug.shape[0]

        # ── Step 4: Solve for weights ──
        self._progress(35, "Solving augmented system")

        gradient_values = np.ones(N_g, dtype=np.float64)  # grad f . n = 1

        value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
            K_aug, value_data, gradient_values, N_v, N_g,
        )

        # Create evaluation function
        evaluate_fn = make_evaluate_fn(
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            self._kernel_type, self._alpha, self._range_max,
            self._R, self._S, self._drift_type,
        )

        # ── Step 5: Evaluate scalar field on grid ──
        self._progress(45, "Evaluating scalar field on grid")

        grid_origin, grid_spacing, grid_dims = self._compute_grid(contact_coords)

        scalar_field = evaluate_field_on_grid(
            grid_origin, grid_spacing, grid_dims,
            evaluate_fn,
            progress_callback=lambda pct, msg: self._progress(45 + pct * 30 // 100, msg),
        )

        # ── Step 6: Extract surfaces ──
        self._progress(75, "Extracting surfaces (marching cubes)")

        surfaces = {}
        surface_names = self._contacts_df["surface_name"].unique() if "surface_name" in self._contacts_df.columns else ["surface_0"]

        for sname in surface_names:
            isovalue = 0.0  # SDF: surface at f=0
            verts, faces = extract_isosurface(scalar_field, grid_origin, grid_spacing, isovalue)
            verts, faces = cleanup_mesh(verts, faces)

            if verts.shape[0] > 0:
                try:
                    mesh = field_to_pyvista_mesh(verts, faces, surface_name=str(sname))
                    surfaces[str(sname)] = mesh
                except ImportError:
                    surfaces[str(sname)] = {"vertices": verts, "faces": faces}

                audit.surface_vertex_counts[str(sname)] = int(verts.shape[0])
                audit.surface_triangle_counts[str(sname)] = int(faces.shape[0])

            # For a single-surface model, only one isosurface at f=0
            break  # TODO: multi-surface potential field in Phase 2

        audit.surface_names = list(surfaces.keys())
        audit.n_surfaces = len(surfaces)

        # ── Step 7: Validate contact honouring ──
        self._progress(85, "Validating contact honouring")

        misfit_df = check_contact_honouring(
            self._contacts_df, evaluate_fn,
            tolerance=self._tolerance,
        )
        summary = contact_honouring_summary(misfit_df)
        audit.contact_honouring_pct = summary["pct_honoured"]
        audit.mean_contact_misfit = summary["mean_misfit"]
        audit.max_contact_misfit = summary["max_misfit"]

        # ── Step 8: Finalize audit ──
        elapsed = time.perf_counter() - t_start
        audit.elapsed_seconds = elapsed
        self._progress(95, "Generating audit record")

        if self._stratigraphic_column:
            audit.stratigraphic_units = self._stratigraphic_column.units

        self._progress(100, f"Complete ({elapsed:.1f}s)")

        # For single-surface model: one isosurface at f=0
        # Two domains: below surface (f<0) and above surface (f>0)
        surface_names = list(surfaces.keys())
        iso_list = [0.0]
        if self._stratigraphic_column and self._stratigraphic_column.units:
            unit_names = list(self._stratigraphic_column.units)
        else:
            unit_names = ["Below_Surface", "Above_Surface"]

        return {
            "surfaces": surfaces,
            "scalar_field": scalar_field,
            "evaluate_fn": evaluate_fn,
            "contact_misfit": misfit_df,
            "contact_honouring_summary": summary,
            "audit_record": audit.to_dict(),
            "grid_origin": grid_origin,
            "grid_spacing": grid_spacing,
            "grid_dims": grid_dims,
            "isovalues": iso_list,
            "unit_names": unit_names,
            "value_weights": value_weights,
            "gradient_weights": gradient_weights,
            "poly_coeffs": poly_coeffs,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _compute_grid(
        self, contact_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """Compute grid parameters for surface extraction."""
        if isinstance(self._grid_extent, dict):
            xmin = self._grid_extent["xmin"]
            xmax = self._grid_extent["xmax"]
            ymin = self._grid_extent["ymin"]
            ymax = self._grid_extent["ymax"]
            zmin = self._grid_extent["zmin"]
            zmax = self._grid_extent["zmax"]
        else:
            # Auto: use data extent with 20% padding
            mins = contact_coords.min(axis=0)
            maxs = contact_coords.max(axis=0)
            extent = maxs - mins
            padding = np.maximum(extent * 0.2, self._grid_resolution * 2)
            xmin, ymin, zmin = mins - padding
            xmax, ymax, zmax = maxs + padding

        res = self._grid_resolution
        nx = max(2, int(np.ceil((xmax - xmin) / res)) + 1)
        ny = max(2, int(np.ceil((ymax - ymin) / res)) + 1)
        nz = max(2, int(np.ceil((zmax - zmin) / res)) + 1)

        grid_origin = np.array([xmin, ymin, zmin], dtype=np.float64)
        grid_spacing = np.array([res, res, res], dtype=np.float64)

        logger.info(
            "Grid: origin=(%.1f, %.1f, %.1f), dims=(%d, %d, %d), res=%.1f m",
            xmin, ymin, zmin, nx, ny, nz, res,
        )

        return grid_origin, grid_spacing, (nx, ny, nz)

    def _progress(self, pct: int, msg: str) -> None:
        """Emit progress update."""
        if self._progress_callback:
            self._progress_callback(pct, msg)
        logger.debug("[%3d%%] %s", pct, msg)

    # ──────────────────────────────────────────────────────────────
    # Phase 2: Stratigraphic (potential field) model
    # ──────────────────────────────────────────────────────────────

    def build_stratigraphic(self) -> Dict[str, Any]:
        """Build multi-surface potential field model (Phase 2).

        Uses StratigraphicModelColumn to assign isovalues, builds a single
        scalar field, extracts isosurfaces at each isovalue, and assigns
        domain codes to block centroids.
        """
        from .stratigraphy import build_stratigraphic_model

        t_start = time.perf_counter()
        self._progress(5, "Building stratigraphic model")

        if self._contacts_df is None or self._contacts_df.empty:
            raise ValueError("No contact data. Call set_contacts() first.")

        # Unit order from stratigraphic column
        unit_order = self._stratigraphic_column.units if self._stratigraphic_column else []

        result = build_stratigraphic_model(
            contacts_df=self._contacts_df,
            unit_order=unit_order,
            kernel_type=self._kernel_type,
            alpha=self._alpha,
            range_max=self._range_max,
            range_mid=self._range_mid,
            range_min=self._range_min,
            azimuth=self._azimuth,
            dip=self._dip,
            pitch=self._pitch,
            nugget=self._nugget,
            accuracy=self._accuracy,
            drift_type=self._drift_type,
            grid_resolution=self._grid_resolution,
            grid_extent=self._grid_extent,
            tolerance=self._tolerance,
            orientations_df=self._orientations_df,
            progress_callback=self._progress_callback,
        )

        elapsed = time.perf_counter() - t_start
        self._progress(100, f"Stratigraphic model complete ({elapsed:.1f}s)")

        result["elapsed_seconds"] = elapsed

        # Ensure isovalues is a sorted list of floats (not a dict)
        raw_iso = result.get("isovalues", {})
        if isinstance(raw_iso, dict):
            # Dict {surface_name: isovalue} → sorted list of floats
            iso_pairs = sorted(raw_iso.items(), key=lambda kv: kv[1])
            result["isovalues"] = [v for _, v in iso_pairs]
        elif isinstance(raw_iso, (list, tuple)):
            result["isovalues"] = sorted(raw_iso)

        # Add unit_names from stratigraphic column
        strat = result.get("strat_column")
        if strat and hasattr(strat, "unit_names"):
            result["unit_names"] = list(strat.unit_names)
        elif self._stratigraphic_column and self._stratigraphic_column.units:
            result["unit_names"] = list(self._stratigraphic_column.units)

        return result

    # ──────────────────────────────────────────────────────────────
    # Phase 3: Vein model
    # ──────────────────────────────────────────────────────────────

    def build_vein(self) -> Dict[str, Any]:
        """Build vein model from lithology logs (Phase 3)."""
        t_start = time.perf_counter()
        self._progress(5, "Building vein model")

        if self._vein_lithology_df is None or self._vein_code is None:
            raise ValueError("No vein data. Call set_vein_data() first.")

        # Step 1: Extract vein intersections from lithology logs
        self._progress(10, "Extracting vein intersections")
        intersections = extract_vein_intersections(
            self._vein_lithology_df,
            self._vein_code,
        )

        if not intersections:
            raise ValueError(
                f"No vein intersections found for code '{self._vein_code}' "
                f"in lithology data ({len(self._vein_lithology_df)} rows)"
            )

        # Step 2: Build vein model from intersections
        self._progress(20, f"Building vein model ({len(intersections)} intersections)")
        result = build_vein_model(
            intersections=intersections,
            kernel_type=self._kernel_type,
            alpha=self._alpha,
            range_=self._range_max,
            nugget=self._nugget,
            accuracy=self._accuracy,
            min_thickness=float(self._config.get("min_thickness", 0.5)),
            method=self._config.get("vein_method", "median_thickness"),
            R=self._R,
            S=self._S,
            grid_resolution=self._grid_resolution,
            grid_extent=self._grid_extent,
            progress_callback=self._progress_callback,
        )

        elapsed = time.perf_counter() - t_start
        self._progress(100, f"Vein model complete ({elapsed:.1f}s)")

        result["elapsed_seconds"] = elapsed
        return result

    # ──────────────────────────────────────────────────────────────
    # Phase 4: Faulted model
    # ──────────────────────────────────────────────────────────────

    def build_faulted(self) -> Dict[str, Any]:
        """Build geological model with fault handling (Phase 4)."""
        t_start = time.perf_counter()
        self._progress(5, "Building faulted geological model")

        if self._contacts_df is None or self._contacts_df.empty:
            raise ValueError("No contact data. Call set_contacts() first.")

        # Pack interpolation parameters into model_config dict
        model_config = {
            "kernel_type": self._kernel_type,
            "alpha": self._alpha,
            "range_max": self._range_max,
            "range_mid": self._range_mid,
            "range_min": self._range_min,
            "azimuth": self._azimuth,
            "dip": self._dip,
            "pitch": self._pitch,
            "nugget": self._nugget,
            "accuracy": self._accuracy,
            "drift_type": self._drift_type,
            "grid_resolution": self._grid_resolution,
            "grid_extent": self._grid_extent,
            "tolerance": self._tolerance,
        }

        # Add unit_order if stratigraphic column is set
        if self._stratigraphic_column and self._stratigraphic_column.units:
            model_config["unit_order"] = self._stratigraphic_column.units

        result = build_faulted_geological_model(
            contacts_df=self._contacts_df,
            orientations_df=self._orientations_df,
            faults=self._fault_definitions,
            model_config=model_config,
            progress_callback=self._progress_callback,
        )

        elapsed = time.perf_counter() - t_start
        self._progress(100, f"Faulted model complete ({elapsed:.1f}s)")

        result["elapsed_seconds"] = elapsed
        return result

    # ──────────────────────────────────────────────────────────────
    # Phase 5: Fold frame model
    # ──────────────────────────────────────────────────────────────

    def build_with_fold_frame(self) -> Dict[str, Any]:
        """Build geological model using fold frame (Phase 5)."""
        t_start = time.perf_counter()
        self._progress(5, "Building fold frame model")

        if self._contacts_df is None or self._contacts_df.empty:
            raise ValueError("No contact data. Call set_contacts() first.")
        if self._orientations_df is None or self._orientations_df.empty:
            raise ValueError("No orientation data. Required for fold frame.")

        contact_coords = self._contacts_df[["X", "Y", "Z"]].values.astype(np.float64)
        grid_origin, grid_spacing, grid_dims = self._compute_grid(contact_coords)

        fold_frame = FoldFrame(self._fold_config)
        ff_result = fold_frame.build_from_orientations(
            orientations_df=self._orientations_df,
            contacts_df=self._contacts_df,
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            grid_dims=grid_dims,
            kernel_type=self._kernel_type,
            alpha=self._alpha,
            range_=self._range_max,
            nugget=self._nugget,
            accuracy=self._accuracy,
            drift_type=self._drift_type,
            progress_callback=self._progress_callback,
        )

        # Use the S0 field (stratigraphy in fold coordinates) as the
        # primary scalar field for surface extraction and domain assignment
        scalar_field = ff_result["s0_field"]
        evaluate_fn = ff_result["s0_evaluate"]

        # Extract surfaces
        self._progress(85, "Extracting surfaces from fold frame")
        surfaces = {}
        verts, faces = extract_isosurface(
            scalar_field, grid_origin, grid_spacing, 0.0,
        )
        verts, faces = cleanup_mesh(verts, faces)
        if verts.shape[0] > 0:
            try:
                mesh = field_to_pyvista_mesh(verts, faces, surface_name="fold_surface_0")
                surfaces["fold_surface_0"] = mesh
            except ImportError:
                surfaces["fold_surface_0"] = {"vertices": verts, "faces": faces}

        # Validate contact honouring
        self._progress(90, "Validating contact honouring")
        misfit_df = check_contact_honouring(
            self._contacts_df, evaluate_fn, tolerance=self._tolerance,
        )
        summary = contact_honouring_summary(misfit_df)

        elapsed = time.perf_counter() - t_start
        self._progress(100, f"Fold frame model complete ({elapsed:.1f}s)")

        # Isovalues and unit names for solid generation
        iso_list = [0.0]
        if self._stratigraphic_column and self._stratigraphic_column.units:
            unit_names = list(self._stratigraphic_column.units)
        else:
            unit_names = ["Below_Fold_Surface", "Above_Fold_Surface"]

        return {
            "surfaces": surfaces,
            "scalar_field": scalar_field,
            "evaluate_fn": evaluate_fn,
            "fold_frame": ff_result,
            "contact_misfit": misfit_df,
            "contact_honouring_summary": summary,
            "grid_origin": grid_origin,
            "grid_spacing": grid_spacing,
            "grid_dims": grid_dims,
            "isovalues": iso_list,
            "unit_names": unit_names,
            "elapsed_seconds": elapsed,
        }

    # ──────────────────────────────────────────────────────────────
    # Domain assignment convenience
    # ──────────────────────────────────────────────────────────────

    def assign_domains(
        self,
        block_centroids: np.ndarray,
        evaluate_fn: Callable,
        isovalues: List[float],
        unit_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign geological domain codes to block centroids.

        Convenience wrapper around domain_model.assign_domains_potential_field().
        """
        return assign_domains_potential_field(
            block_centroids, evaluate_fn, isovalues, unit_names,
        )
