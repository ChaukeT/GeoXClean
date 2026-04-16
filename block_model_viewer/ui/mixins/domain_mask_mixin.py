"""
Domain Masking UI Mixin for estimation and simulation panels.

Provides the "Domain Masking" group box, helper methods for building
block centroids and extracting conditioning coordinates, and the
post-processing step that masks results outside the data support.

Usage in a panel:
    class MyPanel(BaseAnalysisPanel, DomainMaskMixin):
        ...
        def setup_ui(self):
            ...
            mask_group = self._build_domain_mask_group(default_enabled=True)
            left_layout.addWidget(mask_group)

        def on_results(self, payload):
            ...
            results = self._apply_domain_masking(results)
            ...
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QCheckBox, QRadioButton, QButtonGroup, QDoubleSpinBox,
    QLabel, QMessageBox,
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class DomainMaskMixin:
    """Mixin class that adds domain masking UI and logic to estimation panels."""

    # Subclasses should override these
    def _grade_result_keys(self) -> list[str]:
        return ["grade", "mean", "p10", "p50", "p90", "estimate", "value"]

    def _variance_result_keys(self) -> list[str]:
        return ["variance", "std", "kriging_variance", "posterior_variance", "conditional_variance"]

    # ------------------------------------------------------------------
    # UI builder
    # ------------------------------------------------------------------

    def _build_domain_mask_group(self, default_enabled: bool = True) -> QGroupBox:
        """Build the Domain Masking group box for the panel sidebar."""
        group = QGroupBox("Domain Masking")
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(8, 12, 8, 8)
        vbox.setSpacing(4)

        # Enable checkbox
        self.mask_checkbox = QCheckBox("Mask blocks outside data support")
        self.mask_checkbox.setChecked(default_enabled)
        self.mask_checkbox.setToolTip(
            "Blocks more than one search radius from any drillhole composite "
            "will be set to NaN and rendered transparent. This prevents "
            "meaningless extrapolated values from appearing in the block model."
        )
        vbox.addWidget(self.mask_checkbox)

        # Method radio buttons
        method_layout = QHBoxLayout()
        self.mask_method_distance = QRadioButton("Distance (search ellipsoid)")
        self.mask_method_hull = QRadioButton("Convex hull + buffer")
        self.mask_method_distance.setChecked(True)
        self._mask_method_group = QButtonGroup()
        self._mask_method_group.addButton(self.mask_method_distance, 0)
        self._mask_method_group.addButton(self.mask_method_hull, 1)
        method_layout.addWidget(self.mask_method_distance)
        method_layout.addWidget(self.mask_method_hull)
        vbox.addLayout(method_layout)

        # Buffer spinbox (shown only for convex hull mode)
        buf_layout = QFormLayout()
        self.mask_buffer_spin = QDoubleSpinBox()
        self.mask_buffer_spin.setRange(0.0, 500.0)
        self.mask_buffer_spin.setValue(0.0)
        self.mask_buffer_spin.setSuffix(" m")
        self.mask_buffer_spin.setToolTip("Expand convex hull outward by this distance")
        buf_layout.addRow("Expansion:", self.mask_buffer_spin)
        self._mask_buffer_row_label = buf_layout.labelForField(self.mask_buffer_spin)
        vbox.addLayout(buf_layout)

        # Toggle buffer visibility based on method
        def _on_method_changed():
            is_hull = self.mask_method_hull.isChecked()
            self.mask_buffer_spin.setVisible(is_hull)
            if self._mask_buffer_row_label:
                self._mask_buffer_row_label.setVisible(is_hull)
        self._mask_method_group.buttonToggled.connect(lambda: _on_method_changed())
        _on_method_changed()

        # Coverage readout
        self.coverage_label = QLabel("")
        self.coverage_label.setWordWrap(True)
        vbox.addWidget(self.coverage_label)

        # Show mask in 3D checkbox
        self.show_mask_3d_checkbox = QCheckBox("Show mask in 3D")
        self.show_mask_3d_checkbox.setChecked(False)
        self.show_mask_3d_checkbox.setToolTip(
            "Render masked (NaN) blocks as a transparent grey wireframe overlay"
        )
        vbox.addWidget(self.show_mask_3d_checkbox)

        # Enable/disable sub-widgets based on master checkbox
        def _on_enable_toggled(checked):
            for w in (self.mask_method_distance, self.mask_method_hull,
                      self.mask_buffer_spin, self.show_mask_3d_checkbox):
                w.setEnabled(checked)
        self.mask_checkbox.toggled.connect(_on_enable_toggled)
        _on_enable_toggled(default_enabled)

        return group

    # ------------------------------------------------------------------
    # Shared grid synchronisation
    # ------------------------------------------------------------------

    def _publish_grid_to_registry(self):
        """Write the current panel's grid definition to the registry.

        Call this whenever grid spinboxes change or estimation starts.
        Other panels can then pick up the same grid via
        ``_sync_grid_from_registry()``.
        """
        from ...models.shared_grid import SharedGridDefinition

        registry = getattr(self, "registry", None)
        if registry is None:
            return

        # Read grid params from whichever widget naming this panel uses
        ox = oy = oz = ddx = ddy = ddz = 0.0
        nnx = nny = nnz = 0

        for attr in ("xmin_spin", "grid_x0"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                ox = w.value()
                break
        for attr in ("ymin_spin", "grid_y0"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                oy = w.value()
                break
        for attr in ("zmin_spin", "grid_z0"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                oz = w.value()
                break

        for attr in ("dx_spin", "dx", "grid_dx", "grid_x_spin"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                ddx = w.value()
                break
        for attr in ("dy_spin", "dy", "grid_dy", "grid_y_spin"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                ddy = w.value()
                break
        for attr in ("dz_spin", "dz", "grid_dz", "grid_z_spin"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                ddz = w.value()
                break

        for attr in ("nx_spin", "nx", "grid_nx"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                nnx = int(w.value())
                break
        for attr in ("ny_spin", "ny", "grid_ny"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                nny = int(w.value())
                break
        for attr in ("nz_spin", "nz", "grid_nz"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                nnz = int(w.value())
                break

        if nnx <= 0 or nny <= 0 or nnz <= 0 or ddx <= 0 or ddy <= 0 or ddz <= 0:
            return  # Incomplete grid — nothing to publish

        panel_name = type(self).__name__
        grid_def = SharedGridDefinition(
            nx=nnx, ny=nny, nz=nnz,
            dx=ddx, dy=ddy, dz=ddz,
            x0=ox, y0=oy, z0=oz,
            source_panel=panel_name,
        )
        registry.register_shared_grid(grid_def, source_panel=panel_name)

    def _sync_grid_from_registry(self):
        """Pull the shared grid definition from the registry and update
        this panel's spinboxes to match.

        Call this when the panel becomes visible or when the registry
        emits ``sharedGridChanged``.  Returns True if values were updated.
        """
        registry = getattr(self, "registry", None)
        if registry is None:
            return False

        grid_def = registry.get_shared_grid()
        if grid_def is None:
            return False

        updated = False

        def _set(attr_candidates, value):
            nonlocal updated
            for attr in attr_candidates:
                w = getattr(self, attr, None)
                if w is not None and hasattr(w, "setValue"):
                    # QSpinBox expects int; QDoubleSpinBox expects float
                    from PyQt6.QtWidgets import QSpinBox
                    if isinstance(w, QSpinBox):
                        value = int(round(value))
                    current = w.value()
                    if not np.isclose(current, value):
                        w.blockSignals(True)
                        w.setValue(value)
                        w.blockSignals(False)
                        updated = True
                    return

        _set(("xmin_spin", "grid_x0"), grid_def.x0)
        _set(("ymin_spin", "grid_y0"), grid_def.y0)
        _set(("zmin_spin", "grid_z0"), grid_def.z0)
        _set(("dx_spin", "dx", "grid_dx", "grid_x_spin"), grid_def.dx)
        _set(("dy_spin", "dy", "grid_dy", "grid_y_spin"), grid_def.dy)
        _set(("dz_spin", "dz", "grid_dz", "grid_z_spin"), grid_def.dz)
        _set(("nx_spin", "nx", "grid_nx"), int(grid_def.nx))
        _set(("ny_spin", "ny", "grid_ny"), int(grid_def.ny))
        _set(("nz_spin", "nz", "grid_nz"), int(grid_def.nz))

        if updated:
            logger.info(
                "%s: Synced grid from registry (source: %s) — "
                "%dx%dx%d  origin (%.1f, %.1f, %.1f)",
                type(self).__name__, grid_def.source_panel,
                grid_def.nx, grid_def.ny, grid_def.nz,
                grid_def.x0, grid_def.y0, grid_def.z0,
            )
        return updated

    def _connect_grid_sync(self):
        """Connect to the registry's sharedGridChanged signal.

        Call this once from the panel's __init__ / setup method, AFTER
        self.registry is assigned.
        """
        registry = getattr(self, "registry", None)
        if registry is None:
            return
        try:
            signals = registry.signals
            if signals and hasattr(signals, "sharedGridChanged"):
                signals.sharedGridChanged.connect(lambda _: self._sync_grid_from_registry())
        except Exception:
            pass  # Headless mode — no signals

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _get_conditioning_coords(self) -> np.ndarray:
        """Return (N, 3) array of X, Y, Z for conditioning data."""
        df = getattr(self, "drillhole_data", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            return np.empty((0, 3))

        for xcol, ycol, zcol in [
            ("X", "Y", "Z"),
            ("EASTING", "NORTHING", "ELEVATION"),
            ("x", "y", "z"),
            ("EAST", "NORTH", "RL"),
        ]:
            if xcol in df.columns and ycol in df.columns and zcol in df.columns:
                return df[[xcol, ycol, zcol]].dropna().values
        return np.empty((0, 3))

    def _get_block_centroids(self) -> np.ndarray:
        """Return (N_blocks, 3) from panel grid spinboxes."""
        # Try standard naming first
        if hasattr(self, "xmin_spin"):
            ox = self.xmin_spin.value()
            oy = self.ymin_spin.value()
            oz = self.zmin_spin.value()
        elif hasattr(self, "grid_x0"):
            ox = self.grid_x0.value()
            oy = self.grid_y0.value()
            oz = self.grid_z0.value()
        else:
            return np.empty((0, 3))

        # Block sizes
        if hasattr(self, "dx_spin"):
            ddx, ddy, ddz = self.dx_spin.value(), self.dy_spin.value(), self.dz_spin.value()
        elif hasattr(self, "dx"):
            ddx = self.dx.value()
            ddy = self.dy.value()
            ddz = self.dz.value()
        elif hasattr(self, "grid_dx"):
            ddx, ddy, ddz = self.grid_dx.value(), self.grid_dy.value(), self.grid_dz.value()
        else:
            return np.empty((0, 3))

        # Grid counts
        if hasattr(self, "nx_spin"):
            nnx, nny, nnz = self.nx_spin.value(), self.ny_spin.value(), self.nz_spin.value()
        elif hasattr(self, "nx"):
            nnx, nny, nnz = self.nx.value(), self.ny.value(), self.nz.value()
        elif hasattr(self, "grid_nx"):
            nnx, nny, nnz = self.grid_nx.value(), self.grid_ny.value(), self.grid_nz.value()
        else:
            return np.empty((0, 3))

        # Build centroids
        xs = ox + (np.arange(nnx) + 0.5) * ddx
        ys = oy + (np.arange(nny) + 0.5) * ddy
        zs = oz + (np.arange(nnz) + 0.5) * ddz
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def _get_search_radii(self) -> tuple:
        """Extract search radii from panel widgets. Override per panel."""
        # Try explicit anisotropic radii widgets first
        if hasattr(self, "rad_major") and hasattr(self, "rad_minor") and hasattr(self, "rad_vert"):
            return (self.rad_major.value(), self.rad_minor.value(), self.rad_vert.value())
        # Isotropic radius from various widget names
        for attr in ("rad", "max_dist_spin", "search_radius", "search_radius_spin"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                r = w.value()
                return (r, r, r)
        # Variogram range fallback — anisotropic (2× ranges)
        # This ensures the domain mask ellipsoid matches the variogram
        # anisotropy rather than defaulting to an isotropic sphere.
        for major_a, minor_a, vert_a in [
            ("range_major_spin", "range_minor_spin", "range_vert_spin"),
        ]:
            wM = getattr(self, major_a, None)
            wm = getattr(self, minor_a, None)
            wv = getattr(self, vert_a, None)
            if all(w is not None and hasattr(w, "value") for w in (wM, wm, wv)):
                rM, rm, rv = wM.value(), wm.value(), wv.value()
                if rM > 0 and rm > 0 and rv > 0:
                    return (rM * 2.0, rm * 2.0, rv * 2.0)
        # Single variogram range fallback — isotropic (2× range)
        for attr in ("range_major_spin", "range_spin"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                r = w.value()
                if r > 0:
                    return (r * 2.0, r * 2.0, r * 2.0)
        return (200.0, 200.0, 200.0)  # ultimate fallback

    def _get_search_azimuth(self) -> float:
        """Extract search azimuth from panel widgets."""
        for attr in ("azim", "azimuth_spin", "azimuth"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                return w.value()
        return 0.0

    def _get_search_dip(self) -> float:
        """Extract search dip from panel widgets."""
        for attr in ("dip_spin", "dip"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                return w.value()
        return 0.0

    def _get_min_neighbours(self) -> int:
        """Extract minimum neighbours from panel widgets."""
        for attr in ("min_n", "min_data_spin", "min_neighbours_spin", "neighbors_spin"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "value"):
                return max(1, w.value() // 2)
        return 1

    # ------------------------------------------------------------------
    # Coverage label update
    # ------------------------------------------------------------------

    def _update_coverage_label(self, n_informed: int, n_total: int, pct: float):
        if not hasattr(self, "coverage_label"):
            return
        colour = "#81c784" if pct >= 50 else "#ff9800" if pct >= 20 else "#ef5350"
        self.coverage_label.setText(
            f"Informed blocks: {n_informed:,} / {n_total:,}  ({pct:.1f}%)"
        )
        self.coverage_label.setStyleSheet(f"color: {colour}; font-weight: bold;")

    # ------------------------------------------------------------------
    # Pre-filter (preferred): compute mask BEFORE estimation
    # ------------------------------------------------------------------

    def _prefilter_block_centroids(
        self,
        block_centroids: np.ndarray,
    ) -> tuple:
        """Pre-filter block centroids to only those inside the domain mask.

        Call this BEFORE sending centroids to the estimation/simulation engine.
        Only informed blocks are returned — the engine never sees or wastes
        computation on blocks outside the data support.

        Parameters
        ----------
        block_centroids : (N, 3) all block centroids in the grid

        Returns
        -------
        inside_centroids : (K, 3) — centroids inside the mask
        inside_idx       : (K,) int — indices into the original array
        mask             : (N,) bool — full boolean mask
        n_total          : int — total number of blocks (N)

        If masking is disabled, returns the original centroids with a full
        True mask (all blocks pass).

        After estimation, call ``_scatter_estimation_results()`` to map the
        reduced K-length results back into the full N-length grid.
        """
        n_total = block_centroids.shape[0]

        if not hasattr(self, "mask_checkbox") or not self.mask_checkbox.isChecked():
            # Masking disabled — pass everything through
            return (
                block_centroids,
                np.arange(n_total, dtype=np.intp),
                np.ones(n_total, dtype=bool),
                n_total,
            )

        from block_model_viewer.geostats.domain_mask import prefilter_centroids

        data_coords = self._get_conditioning_coords()

        # Coordinate alignment (same logic as _apply_domain_masking)
        shifted = False
        if data_coords.shape[0] > 0:
            try:
                from block_model_viewer.ui.sgsim_panel import SGSIMPanel
                if isinstance(self, SGSIMPanel) and hasattr(self, '_get_renderer_global_shift'):
                    gs = self._get_renderer_global_shift()
                    if gs is not None and np.linalg.norm(gs) > 1.0:
                        data_coords = data_coords - gs
                        shifted = True
            except Exception:
                pass
            if not shifted:
                data_center = data_coords.mean(axis=0)
                block_center = block_centroids.mean(axis=0)
                offset = data_center - block_center
                offset_mag = np.linalg.norm(offset)
                block_extent = block_centroids.max(axis=0) - block_centroids.min(axis=0)
                diag = np.linalg.norm(block_extent)
                if diag > 0 and offset_mag > diag * 0.5:
                    data_coords = data_coords - offset

        self._last_mask_data_coords = data_coords

        if hasattr(self, "mask_method_distance") and not self.mask_method_distance.isChecked():
            method = "convex_hull"
        else:
            method = "distance"

        buffer_m = 0.0
        if method == "convex_hull" and hasattr(self, "mask_buffer_spin"):
            buffer_m = self.mask_buffer_spin.value()

        inside_centroids, inside_idx, mask = prefilter_centroids(
            block_centroids,
            data_coords,
            search_radii=self._get_search_radii(),
            azimuth_deg=self._get_search_azimuth(),
            dip_deg=self._get_search_dip(),
            min_neighbours=self._get_min_neighbours(),
            method=method,
            buffer_m=buffer_m,
        )

        n_inside = len(inside_idx)
        pct = 100.0 * n_inside / max(n_total, 1)
        self._update_coverage_label(n_inside, n_total, pct)

        # Store metadata for registry / audit
        self._domain_mask_metadata = {
            "domain_mask_applied": True,
            "prefiltered": True,
            "n_informed": n_inside,
            "n_total": n_total,
            "pct_informed": round(pct, 1),
            "mask_method": method,
            "search_radii": self._get_search_radii(),
            "azimuth_deg": self._get_search_azimuth(),
        }

        # Store for scatter step
        self._prefilter_inside_idx = inside_idx
        self._prefilter_mask = mask
        self._prefilter_n_total = n_total

        if pct < 10.0:
            QMessageBox.warning(
                self,
                "Low Data Coverage",
                f"Only {pct:.1f}% of grid blocks ({n_inside:,} of {n_total:,}) "
                f"are within the data support.\n\nConsider:\n"
                f"  - Reducing the grid extent to match the drillhole footprint\n"
                f"  - Using Auto-detect Grid to snap the grid to the data\n"
                f"  - Reducing block size",
            )

        logger.info(
            "Pre-filter: %d / %d blocks inside domain (%.1f%%) — "
            "engine will only estimate %d blocks",
            n_inside, n_total, pct, n_inside,
        )
        return inside_centroids, inside_idx, mask, n_total

    def _scatter_estimation_results(
        self,
        inside_results: dict,
        grade_keys: list[str] | None = None,
        variance_keys: list[str] | None = None,
    ) -> dict:
        """Scatter reduced estimation results back into full-size arrays.

        Call this AFTER the engine returns results on the pre-filtered centroids.
        Maps K-length arrays back to N-length arrays with NaN for outside blocks.
        """
        from block_model_viewer.geostats.domain_mask import scatter_results

        inside_idx = getattr(self, '_prefilter_inside_idx', None)
        n_total = getattr(self, '_prefilter_n_total', None)
        mask = getattr(self, '_prefilter_mask', None)

        if inside_idx is None or n_total is None:
            logger.warning("scatter called without prior prefilter — returning as-is")
            return inside_results

        gk = grade_keys if grade_keys is not None else self._grade_result_keys()
        vk = variance_keys if variance_keys is not None else self._variance_result_keys()

        return scatter_results(
            inside_results, inside_idx, n_total,
            grade_keys=gk, variance_keys=vk,
            mask=mask,
        )

    # ------------------------------------------------------------------
    # Post-filter (legacy): mask results AFTER estimation
    # ------------------------------------------------------------------

    def _apply_domain_masking(
        self,
        results: dict,
        grade_keys: list[str] | None = None,
        variance_keys: list[str] | None = None,
    ) -> dict:
        """Apply domain masking to results if enabled. Call from on_results()."""
        if not hasattr(self, "mask_checkbox") or not self.mask_checkbox.isChecked():
            return results

        from block_model_viewer.geostats.domain_mask import (
            compute_distance_mask,
            apply_mask_to_results,
        )

        data_coords = self._get_conditioning_coords()
        block_coords = self._get_block_centroids()

        if block_coords.shape[0] == 0:
            logger.warning("Domain masking skipped: no block centroids available")
            return results

        # Coordinate alignment: data may be in UTM while grid is in local coords
        # Prefer the renderer's exact global_shift if available
        shifted = False
        if data_coords.shape[0] > 0:
            try:
                from block_model_viewer.ui.sgsim_panel import SGSIMPanel
                if isinstance(self, SGSIMPanel) and hasattr(self, '_get_renderer_global_shift'):
                    gs = self._get_renderer_global_shift()
                    if gs is not None and np.linalg.norm(gs) > 1.0:
                        data_coords = data_coords - gs
                        shifted = True
                        logger.info(f"Domain mask: applied renderer global_shift to data coords")
            except Exception:
                pass

            if not shifted and block_coords.shape[0] > 0:
                data_center = data_coords.mean(axis=0)
                block_center = block_coords.mean(axis=0)
                offset = data_center - block_center
                offset_mag = np.linalg.norm(offset)
                block_extent = block_coords.max(axis=0) - block_coords.min(axis=0)
                diag = np.linalg.norm(block_extent)
                if diag > 0 and offset_mag > diag * 0.5:
                    logger.info(
                        f"Domain mask: shifting data coords by "
                        f"({-offset[0]:.0f}, {-offset[1]:.0f}, {-offset[2]:.0f}) "
                        f"to align with grid"
                    )
                    data_coords = data_coords - offset

        # Store aligned data coords so _strip_unmasked_cells() can use them
        self._last_mask_data_coords = data_coords

        if self.mask_method_distance.isChecked():
            mask = compute_distance_mask(
                block_coords,
                data_coords,
                search_radii=self._get_search_radii(),
                azimuth_deg=self._get_search_azimuth(),
                dip_deg=self._get_search_dip(),
                min_neighbours=self._get_min_neighbours(),
            )
        else:
            from block_model_viewer.geostats.domain_mask import compute_convex_hull_mask
            mask = compute_convex_hull_mask(
                block_coords, data_coords,
                buffer_m=self.mask_buffer_spin.value(),
            )

        n_informed = int(mask.sum())
        n_total = len(mask)
        pct = 100.0 * n_informed / n_total if n_total > 0 else 0.0

        self._update_coverage_label(n_informed, n_total, pct)

        # Store metadata for registry and audit
        mask_method = "distance" if self.mask_method_distance.isChecked() else "convex_hull"
        self._domain_mask_metadata = {
            "domain_mask_applied": True,
            "n_informed": n_informed,
            "n_total": n_total,
            "pct_informed": round(pct, 1),
            "mask_method": mask_method,
            "search_radii": self._get_search_radii(),
            "azimuth_deg": self._get_search_azimuth(),
        }

        # Publish mask to registry so ALL estimation methods share it
        try:
            from ...models.shared_grid import SharedDomainMask
            panel_name = type(self).__name__
            shared_mask = SharedDomainMask(
                mask=mask,
                method=mask_method,
                n_informed=n_informed,
                n_total=n_total,
                pct_informed=pct,
                search_radii=self._get_search_radii(),
                azimuth_deg=self._get_search_azimuth(),
                dip_deg=self._get_search_dip(),
                min_neighbours=self._get_min_neighbours(),
                buffer_m=self.mask_buffer_spin.value() if mask_method == "convex_hull" else 0.0,
                source_panel=panel_name,
            )
            registry = getattr(self, "registry", None)
            if registry and hasattr(registry, "register_shared_domain_mask"):
                registry.register_shared_domain_mask(shared_mask, source_panel=panel_name)
        except Exception as exc:
            logger.debug("Could not publish shared domain mask: %s", exc)

        if pct < 10.0:
            QMessageBox.warning(
                self,
                "Low Data Coverage",
                f"Only {pct:.1f}% of grid blocks ({n_informed:,} of {n_total:,}) are within "
                f"the data support.\n\nConsider:\n"
                f"  - Reducing the grid extent to match the drillhole footprint\n"
                f"  - Using Auto-detect Grid to snap the grid to the data\n"
                f"  - Reducing block size",
            )

        gk = grade_keys if grade_keys is not None else self._grade_result_keys()
        vk = variance_keys if variance_keys is not None else self._variance_result_keys()

        results = apply_mask_to_results(results, mask, grade_keys=gk, variance_keys=vk)

        logger.info(
            f"Domain masking applied: {n_informed}/{n_total} blocks informed ({pct:.1f}%)"
        )
        return results

    def _get_domain_mask_metadata(self) -> dict | None:
        """Return mask metadata for registry/audit, or None if not applied."""
        return getattr(self, "_domain_mask_metadata", None)

    def _strip_unmasked_cells(self, grid):
        """Remove cells outside the domain mask from a PyVista grid.

        Uses the grid's own cell centers to recompute the mask, avoiding any
        cell-ordering mismatch between the mask array and the engine-built grid.

        Returns the grid unchanged if masking is disabled or no data coords are stored.
        """
        if not hasattr(self, "mask_checkbox") or not self.mask_checkbox.isChecked():
            return grid
        if grid is None:
            return grid

        try:
            # Get actual cell centers from the grid (ordering-agnostic)
            centers = grid.cell_centers().points  # (N_cells, 3)
            if centers.shape[0] == 0:
                return grid

            data_coords = getattr(self, "_last_mask_data_coords", None)
            if data_coords is None or data_coords.shape[0] == 0:
                # Fallback: fetch conditioning coords directly
                data_coords = self._get_conditioning_coords()
                if data_coords.shape[0] == 0:
                    logger.warning("_strip_unmasked_cells: no conditioning data available")
                    return grid
                # Align coordinates if needed (data may be UTM, grid may be local)
                data_center = data_coords.mean(axis=0)
                grid_center = centers.mean(axis=0)
                offset = data_center - grid_center
                diag = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
                if diag > 0 and np.linalg.norm(offset) > diag * 0.5:
                    data_coords = data_coords - offset
                    logger.info(
                        "_strip_unmasked_cells: aligned data coords "
                        "(shift %.0f, %.0f, %.0f)",
                        -offset[0], -offset[1], -offset[2],
                    )

            if self.mask_method_distance.isChecked():
                from block_model_viewer.geostats.domain_mask import compute_distance_mask
                mask = compute_distance_mask(
                    centers,
                    data_coords,
                    search_radii=self._get_search_radii(),
                    azimuth_deg=self._get_search_azimuth(),
                    dip_deg=self._get_search_dip(),
                    min_neighbours=self._get_min_neighbours(),
                )
            else:
                from block_model_viewer.geostats.domain_mask import compute_convex_hull_mask
                mask = compute_convex_hull_mask(
                    centers,
                    data_coords,
                    buffer_m=self.mask_buffer_spin.value(),
                )

            n_valid = int(mask.sum())
            n_total = len(mask)
            if n_valid == n_total:
                return grid  # nothing to strip

            cell_ids = np.where(mask)[0]
            stripped = grid.extract_cells(cell_ids)
            # Preserve coordinate-shift flag if set on original grid
            if getattr(grid, "_coordinate_shifted", False):
                stripped._coordinate_shifted = True
            logger.info(
                "Domain mask: stripped %d / %d cells from grid "
                "(%d inside, %.1f%%)",
                n_total - n_valid, n_total, n_valid,
                100.0 * n_valid / n_total,
            )
            return stripped

        except Exception as e:
            logger.warning("_strip_unmasked_cells failed: %s — returning grid unchanged", e)
            return grid
