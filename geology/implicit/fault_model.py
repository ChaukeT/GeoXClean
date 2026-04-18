"""
Fault Model -- Fault Surfaces, Domain Splitting, and Displacement.
==================================================================

Models fault surfaces, splits the geological domain at each fault,
and applies fault displacement.  Faults are processed in reverse
chronological order (youngest first) to handle fault-fault interactions.

Mathematical basis: Section 7 of the GeoX Math Specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contact_data import dip_azimuth_to_normal, orientations_dataframe_to_list
from .signed_distance import construct_sdf_constraints, estimate_contact_normals
from .scalar_field import (
    assemble_augmented_matrix,
    solve_augmented_system,
    make_evaluate_fn,
    solve_augmented_system_pum,
    make_evaluate_fn_pum,
    PUM_THRESHOLD,
)
from .surface_extraction import (
    evaluate_field_on_grid,
    extract_isosurface,
    cleanup_mesh,
    field_to_pyvista_mesh,
)

logger = logging.getLogger(__name__)


# =====================================================================
# HW displacement helpers (Fix 4)
# =====================================================================

def _extract_hw_displacement(contacts_df) -> Optional[np.ndarray]:
    """Return the HW restoration displacement if this block was a HW partition."""
    if contacts_df is None or contacts_df.empty:
        return None
    if "_hw_displacement_x" not in contacts_df.columns:
        return None
    disp = np.array([
        float(contacts_df["_hw_displacement_x"].iloc[0]),
        float(contacts_df["_hw_displacement_y"].iloc[0]),
        float(contacts_df["_hw_displacement_z"].iloc[0]),
    ], dtype=np.float64)
    return disp if np.linalg.norm(disp) > 1e-10 else None


def _apply_forward_displacement_to_result(build_result: dict, displacement: np.ndarray) -> None:
    """Translate all surface mesh vertices by +displacement (in-place).

    Reverses the -displacement applied during contact restoration so that
    hanging wall surfaces end up in geographic coordinates.
    """
    surfaces = build_result.get("surfaces", {})
    if isinstance(surfaces, dict):
        for name, mesh in surfaces.items():
            if hasattr(mesh, "points"):
                mesh.points += displacement[np.newaxis, :]
            elif isinstance(mesh, dict) and "vertices" in mesh:
                mesh["vertices"] = mesh["vertices"] + displacement[np.newaxis, :]
    elif isinstance(surfaces, list):
        for surf in surfaces:
            if isinstance(surf, dict) and "vertices" in surf:
                surf["vertices"] = surf["vertices"] + displacement[np.newaxis, :]
    logger.debug(
        "Applied forward fault displacement [%.1f, %.1f, %.1f] m to HW block meshes",
        displacement[0], displacement[1], displacement[2],
    )


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class FaultDefinition:
    """Definition of a geological fault.

    v1.0 Limitation — Constant displacement:
        A single scalar ``displacement`` (or explicit ``displacement_vector``)
        is applied uniformly to the entire hanging wall.  Real faults have
        variable throw along strike and dip, decaying to zero at the fault
        tips.  Consequence: the HW block translates rigidly rather than
        rotating or tipping, which is acceptable for small domains but will
        introduce artefacts for large faults with significant throw gradients.

    v2.0 Roadmap — Displacement field:
        Replace ``displacement_vector`` with a ``displacement_observations``
        DataFrame (columns: X, Y, Z, dx, dy, dz) of observed displacements
        at specific points along the fault.  Between observations, displace-
        ment is interpolated (e.g., via RBF).  At fault tips the displacement
        decays to zero automatically.  This is structurally identical to the
        ARBF grade estimation architecture — only the target variable changes.
    """
    name: str
    fault_type: str = "normal"       # "normal", "reverse", "strike_slip"
    displacement: float = 0.0        # metres (throw for normal/reverse)
    displacement_vector: Optional[np.ndarray] = None  # (3,) explicit
    contacts: Optional[pd.DataFrame] = None   # fault contact points
    orientations: Optional[pd.DataFrame] = None  # fault orientation data
    chronological_order: int = 0     # 0 = youngest (cuts everything)

    def get_displacement_vector(self) -> np.ndarray:
        """Return the fault displacement vector.

        If not explicitly set, infers from fault_type and displacement.
        """
        if self.displacement_vector is not None:
            return self.displacement_vector.copy()

        disp = abs(self.displacement)
        if self.fault_type == "normal":
            return np.array([0.0, 0.0, -disp], dtype=np.float64)
        elif self.fault_type == "reverse":
            return np.array([0.0, 0.0, disp], dtype=np.float64)
        elif self.fault_type == "strike_slip":
            # Default: displacement along X axis for strike-slip
            return np.array([disp, 0.0, 0.0], dtype=np.float64)
        else:
            return np.array([0.0, 0.0, -disp], dtype=np.float64)


# =====================================================================
# Fault surface construction
# =====================================================================

def build_fault_surface(
    fault: FaultDefinition,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_: float = 200.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    drift_type: str = "constant",
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    grid_origin: Optional[np.ndarray] = None,
    grid_spacing: Optional[np.ndarray] = None,
    grid_dims: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """Model a fault surface as a scalar field.

    The fault surface is f_fault(x) = 0.
    Hanging wall: f_fault(x) > 0
    Footwall: f_fault(x) < 0

    Uses the same gradient-augmented RBF system as geological surfaces.

    Parameters
    ----------
    fault : FaultDefinition
        Must have contacts or orientations data.
    kernel_type, alpha, range_, etc. : interpolation parameters
    grid_origin, grid_spacing, grid_dims : optional grid for mesh extraction

    Returns
    -------
    dict with:
        evaluate_fn : callable (evaluates fault scalar field)
        surface_mesh : pyvista.PolyData (if grid provided)
        hw_mask_fn : callable (returns bool array: True for HW)
    """
    if fault.contacts is None or fault.contacts.empty:
        raise ValueError(f"Fault '{fault.name}' has no contact data")

    contacts = fault.contacts
    coords = contacts[["X", "Y", "Z"]].values.astype(np.float64)

    # Estimate fault surface normals
    if fault.orientations is not None and len(fault.orientations) > 0:
        normals = estimate_contact_normals(
            contacts, method="structural",
            structural_data=fault.orientations,
        )
    else:
        normals = estimate_contact_normals(contacts, method="drillhole")

    # Build SDF
    value_coords, value_data, gradient_coords, gradient_normals = \
        construct_sdf_constraints(
            coords, normals, method="gradient",
        )

    # Add fault orientation constraints if available
    if fault.orientations is not None and len(fault.orientations) > 0:
        orient_list = orientations_dataframe_to_list(fault.orientations)
        orient_coords = np.array(
            [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
        )
        orient_normals = np.array(
            [o.normal for o in orient_list], dtype=np.float64,
        )
        if gradient_coords.size > 0:
            gradient_coords = np.vstack([gradient_coords, orient_coords])
            gradient_normals = np.vstack([gradient_normals, orient_normals])
        else:
            gradient_coords = orient_coords
            gradient_normals = orient_normals

    N_v_total = len(value_coords)
    N_g_total = len(gradient_coords) if gradient_coords.size > 0 else 0
    gradient_values = np.ones(N_g_total, dtype=np.float64) if N_g_total > 0 else np.empty(0)

    if (N_v_total + N_g_total) > PUM_THRESHOLD:
        logger.info(
            "Fault SDF PUM mode: %d value + %d gradient constraints",
            N_v_total, N_g_total,
        )
        pum_model = solve_augmented_system_pum(
            value_coords, value_data,
            gradient_coords, gradient_normals, gradient_values,
            kernel_type=kernel_type, alpha=alpha, range_=range_,
            nugget=nugget, accuracy=accuracy, drift_type=drift_type,
            R=R, S=S,
        )
        evaluate_fn = make_evaluate_fn_pum(pum_model)
    else:
        K_aug, N_v, N_g = assemble_augmented_matrix(
            value_coords, gradient_coords, gradient_normals,
            kernel_type=kernel_type,
            alpha=alpha,
            range_=range_,
            nugget=nugget,
            accuracy=accuracy,
            drift_type=drift_type,
            R=R, S=S,
        )

        value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
            K_aug, value_data, gradient_values, N_v, N_g,
        )

        evaluate_fn = make_evaluate_fn(
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            kernel_type, alpha, range_,
            R, S, drift_type,
        )

    def hw_mask_fn(pts):
        """Return True for points on the hanging wall (f > 0)."""
        return evaluate_fn(pts) > 0.0

    result = {
        "evaluate_fn": evaluate_fn,
        "hw_mask_fn": hw_mask_fn,
        "fault_name": fault.name,
    }

    # Extract mesh if grid provided
    if grid_origin is not None and grid_spacing is not None and grid_dims is not None:
        try:
            field_vals = evaluate_field_on_grid(
                grid_origin, grid_spacing, grid_dims, evaluate_fn,
            )
            verts, faces = extract_isosurface(
                field_vals, grid_origin, grid_spacing, 0.0,
            )
            verts, faces = cleanup_mesh(verts, faces)
            if verts.shape[0] > 0:
                try:
                    result["surface_mesh"] = field_to_pyvista_mesh(
                        verts, faces, fault.name,
                    )
                except ImportError:
                    result["surface_mesh"] = {"vertices": verts, "faces": faces}
        except Exception as e:
            logger.warning("Failed to extract fault surface mesh: %s", e)

    return result


# =====================================================================
# Domain splitting
# =====================================================================

def split_domain_at_fault(
    contacts_df: pd.DataFrame,
    fault_evaluate_fn: Callable,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split geological contacts into hanging wall and footwall subsets.

    For each contact point, evaluate the fault scalar field.
    Positive values -> hanging wall subset.
    Negative values -> footwall subset.

    Parameters
    ----------
    contacts_df : pd.DataFrame
    fault_evaluate_fn : callable (B, 3) -> (B,)

    Returns
    -------
    hw_contacts : pd.DataFrame (hanging wall)
    fw_contacts : pd.DataFrame (footwall)
    """
    if contacts_df.empty:
        return contacts_df.copy(), contacts_df.copy()

    coords = contacts_df[[x_col, y_col, z_col]].values.astype(np.float64)
    field_values = fault_evaluate_fn(coords)

    hw_mask = field_values > 0
    fw_mask = ~hw_mask

    hw_contacts = contacts_df[hw_mask].copy()
    fw_contacts = contacts_df[fw_mask].copy()

    logger.info(
        "Split at fault: %d HW, %d FW contacts",
        len(hw_contacts), len(fw_contacts),
    )

    return hw_contacts, fw_contacts


# =====================================================================
# Displacement restoration
# =====================================================================

def apply_fault_displacement(
    coords: np.ndarray,
    fault_evaluate_fn: Callable,
    displacement_vector: np.ndarray,
) -> np.ndarray:
    """Apply fault displacement to restore pre-faulting positions.

    Points on the hanging wall are shifted by -displacement_vector
    to "unfault" them before geological interpolation.

    Parameters
    ----------
    coords : np.ndarray (N, 3)
    fault_evaluate_fn : callable
        Evaluates fault scalar field (positive = HW)
    displacement_vector : np.ndarray (3,)
        Fault displacement vector (from FW to HW)

    Returns
    -------
    restored_coords : np.ndarray (N, 3)
    """
    restored = coords.copy()
    field_values = fault_evaluate_fn(coords)
    hw_mask = field_values > 0

    # Shift HW points back by displacement
    restored[hw_mask] -= displacement_vector

    n_shifted = int(hw_mask.sum())
    logger.info(
        "Displacement restoration: shifted %d/%d HW points by [%.1f, %.1f, %.1f]",
        n_shifted, len(coords),
        displacement_vector[0], displacement_vector[1], displacement_vector[2],
    )

    return restored


def restore_contacts_through_faults(
    contacts_df: pd.DataFrame,
    faults: List[FaultDefinition],
    fault_fields: Dict[str, Callable],
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> pd.DataFrame:
    """Restore contacts through multiple faults (youngest first).

    Parameters
    ----------
    contacts_df : pd.DataFrame
    faults : list of FaultDefinition (sorted youngest first)
    fault_fields : {fault_name: evaluate_fn}

    Returns
    -------
    pd.DataFrame with restored coordinates
    """
    restored = contacts_df.copy()

    for fault in faults:
        if fault.name not in fault_fields:
            continue

        eval_fn = fault_fields[fault.name]
        disp_vec = fault.get_displacement_vector()

        coords = restored[[x_col, y_col, z_col]].values.astype(np.float64)
        new_coords = apply_fault_displacement(coords, eval_fn, disp_vec)

        restored[x_col] = new_coords[:, 0]
        restored[y_col] = new_coords[:, 1]
        restored[z_col] = new_coords[:, 2]

    return restored


# =====================================================================
# Faulted geological model
# =====================================================================

def build_faulted_geological_model(
    contacts_df: pd.DataFrame,
    orientations_df: Optional[pd.DataFrame],
    faults: List[FaultDefinition],
    model_config: dict,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Build a complete faulted geological model.

    Algorithm:
    1. Sort faults by chronological order (youngest first)
    2. For each fault (youngest to oldest):
       a. Build fault surface
       b. Split remaining contacts into HW and FW
       c. Apply displacement to restore HW contacts
    3. For each fault block (region between faults):
       a. Build geological model from local contacts
    4. Combine all fault blocks into final model

    Parameters
    ----------
    contacts_df : pd.DataFrame
    orientations_df : optional
    faults : list of FaultDefinition
    model_config : dict with interpolation parameters

    Returns
    -------
    dict with:
        fault_surfaces : {name: mesh}
        fault_evaluate_fns : {name: callable}
        fault_blocks : list of {contacts, model_result}
        domain_count : int
    """
    from .stratigraphy import build_stratigraphic_model

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # Sort faults: youngest first (lowest chronological_order)
    sorted_faults = sorted(faults, key=lambda f: f.chronological_order)

    kernel_type = model_config.get("kernel_type", "spheroidal")
    alpha = model_config.get("alpha", 1.0)
    range_ = model_config.get("range_max", 200.0)
    nugget = model_config.get("nugget", 0.0)
    accuracy = model_config.get("accuracy", 1e-6)
    drift_type = model_config.get("drift_type", "constant")

    from geostats.arbf.utils import rotation_matrix, scale_matrix
    R = rotation_matrix(
        model_config.get("azimuth", 0.0),
        model_config.get("dip", 0.0),
        model_config.get("pitch", 0.0),
    )
    S = scale_matrix(
        model_config.get("range_max", 100.0),
        model_config.get("range_mid", 100.0),
        model_config.get("range_min", 100.0),
    )

    # Phase 1: Build fault surfaces
    fault_surfaces = {}
    fault_evaluate_fns = {}
    n_faults = len(sorted_faults)

    for i, fault in enumerate(sorted_faults):
        _progress(
            5 + i * 20 // max(1, n_faults),
            f"Building fault surface: {fault.name}",
        )

        if fault.contacts is None or fault.contacts.empty:
            logger.warning("Fault '%s' has no contacts, skipping", fault.name)
            continue

        result = build_fault_surface(
            fault,
            kernel_type=kernel_type,
            alpha=alpha,
            range_=range_,
            nugget=nugget,
            accuracy=accuracy,
            drift_type=drift_type,
            R=R, S=S,
        )

        fault_evaluate_fns[fault.name] = result["evaluate_fn"]
        if "surface_mesh" in result:
            fault_surfaces[fault.name] = result["surface_mesh"]

    # Phase 2: Split contacts and restore displacements
    _progress(30, "Splitting contacts at faults")

    # Build fault blocks: regions bounded by faults
    # Start with all contacts, split by each fault
    fault_blocks = []
    remaining_contacts = [contacts_df.copy()]
    remaining_orientations = [orientations_df.copy() if orientations_df is not None else None]

    for fault in sorted_faults:
        if fault.name not in fault_evaluate_fns:
            continue

        eval_fn = fault_evaluate_fns[fault.name]
        disp_vec = fault.get_displacement_vector()

        new_remaining = []
        new_orient = []

        for j, block_contacts in enumerate(remaining_contacts):
            if block_contacts is None or block_contacts.empty:
                continue

            hw_contacts, fw_contacts = split_domain_at_fault(
                block_contacts, eval_fn,
            )

            # Restore HW contacts
            if not hw_contacts.empty and np.linalg.norm(disp_vec) > 1e-10:
                hw_coords = hw_contacts[["X", "Y", "Z"]].values.astype(np.float64)
                restored = apply_fault_displacement(
                    hw_coords, eval_fn, disp_vec,
                )
                hw_contacts = hw_contacts.copy()
                hw_contacts["X"] = restored[:, 0]
                hw_contacts["Y"] = restored[:, 1]
                hw_contacts["Z"] = restored[:, 2]
                # Tag the restored HW block with displacement metadata so
                # Phase 3 can apply the forward displacement to resulting meshes.
                hw_contacts["_hw_displacement_x"] = fault.displacement_vector[0] if fault.displacement_vector is not None else disp_vec[0]
                hw_contacts["_hw_displacement_y"] = fault.displacement_vector[1] if fault.displacement_vector is not None else disp_vec[1]
                hw_contacts["_hw_displacement_z"] = fault.displacement_vector[2] if fault.displacement_vector is not None else disp_vec[2]

            new_remaining.append(hw_contacts)
            new_remaining.append(fw_contacts)

            # Split orientations similarly
            orient = remaining_orientations[j] if j < len(remaining_orientations) else None
            if orient is not None and not orient.empty:
                hw_orient, fw_orient = split_domain_at_fault(orient, eval_fn)
                new_orient.append(hw_orient)
                new_orient.append(fw_orient)
            else:
                new_orient.append(None)
                new_orient.append(None)

        remaining_contacts = new_remaining
        remaining_orientations = new_orient

    # Phase 3: Build geological model for each fault block
    _progress(50, "Building geological models per fault block")

    unit_order = model_config.get("unit_order", [])
    block_results = []

    n_blocks = len(remaining_contacts)
    for i, block_contacts in enumerate(remaining_contacts):
        if block_contacts is None or block_contacts.empty:
            continue

        if len(block_contacts) < 3:
            logger.warning(
                "Fault block %d has only %d contacts, skipping",
                i, len(block_contacts),
            )
            continue

        _progress(
            50 + i * 40 // max(1, n_blocks),
            f"Modelling fault block {i + 1}/{n_blocks}",
        )

        block_orient = remaining_orientations[i] if i < len(remaining_orientations) else None

        try:
            if unit_order:
                block_result = build_stratigraphic_model(
                    block_contacts, unit_order,
                    kernel_type=kernel_type,
                    alpha=alpha,
                    range_max=model_config.get("range_max", 100.0),
                    range_mid=model_config.get("range_mid", 100.0),
                    range_min=model_config.get("range_min", 100.0),
                    nugget=nugget,
                    accuracy=accuracy,
                    drift_type=drift_type,
                    grid_resolution=model_config.get("grid_resolution", 10.0),
                    grid_extent=model_config.get("grid_extent", "auto"),
                    tolerance=model_config.get("tolerance", 1.0),
                    orientations_df=block_orient,
                )
            else:
                # Fall back to single SDF model
                from .geological_model import GeologicalModelBuilder
                builder = GeologicalModelBuilder(model_config)
                builder.set_contacts(block_contacts)
                if block_orient is not None:
                    builder.set_orientations(block_orient)
                block_result = builder.build()

            # Apply forward fault displacement to HW block meshes.
            # Contacts were shifted by -displacement during restoration (restored
            # to pre-faulted space for interpolation). Now shift vertices back
            # by +displacement to geographic coordinates.
            _hw_disp = _extract_hw_displacement(block_contacts)
            if _hw_disp is not None:
                _apply_forward_displacement_to_result(block_result, _hw_disp)

            block_results.append({
                "contacts": block_contacts,
                "model_result": block_result,
                "block_index": i,
            })
        except Exception as e:
            logger.warning("Failed to model fault block %d: %s", i, e)

    _progress(95, "Combining results")

    return {
        "fault_surfaces": fault_surfaces,
        "fault_evaluate_fns": fault_evaluate_fns,
        "fault_blocks": block_results,
        "n_fault_blocks": len(block_results),
        "n_faults": len(fault_evaluate_fns),
        "faults": sorted_faults,
    }
