"""
Fragmentation Interactive Editor Panel
========================================

Full interactive workspace for point cloud editing:
- Load LAS/DXF/PLY files
- Box, lasso, and brush selection tools
- Delete, crop, invert selection
- Auto cleaning with live preview
- Segmentation with live coloured fragments
- Fragment editing (merge, split, reclassify)
- Undo/redo for all operations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QToolButton, QButtonGroup, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar, QSplitter, QWidget, QComboBox, QLineEdit,
    QSizePolicy,
)
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


class FragEditorPanel(BaseDockPanel):
    """Interactive point cloud editor for fragmentation analysis."""

    PANEL_ID = "FragEditorPanel"
    PANEL_NAME = "Fragmentation Editor"
    PANEL_CATEGORY = PanelCategory.ANALYSIS
    PANEL_ICON = "scan"
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT
    PANEL_DEFAULT_VISIBLE = False
    PANEL_MINIMUM_WIDTH = 320
    PANEL_TOOLTIP = "Interactive point cloud editor for fragmentation analysis"

    def __init__(self, parent=None, **kwargs):
        self._interactor = None
        self._dataset = None
        super().__init__(parent, panel_id=kwargs.get("panel_id"))

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ══════════════════════════════════════════════════════════════
        # TOOLBAR ROW
        # ══════════════════════════════════════════════════════════════
        toolbar = QHBoxLayout()
        toolbar.setSpacing(2)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        self._btn_navigate = self._tool_btn("Navigate", "Navigate / rotate camera", 0)
        self._btn_box = self._tool_btn("Box", "Box select (rubber-band)", 1)
        self._btn_lasso = self._tool_btn("Lasso", "Lasso select (freehand)", 2)
        self._btn_brush = self._tool_btn("Brush", "Brush select (paint)", 3)
        self._btn_navigate.setChecked(True)

        for btn in [self._btn_navigate, self._btn_box, self._btn_lasso, self._btn_brush]:
            toolbar.addWidget(btn)

        toolbar.addSpacing(10)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setToolTip("Delete selected points (Del)")
        self._btn_delete.clicked.connect(self._on_delete_selected)
        toolbar.addWidget(self._btn_delete)

        self._btn_crop = QPushButton("Crop")
        self._btn_crop.setToolTip("Keep only selected points, delete rest")
        self._btn_crop.clicked.connect(self._on_crop_to_selection)
        toolbar.addWidget(self._btn_crop)

        self._btn_invert = QPushButton("Invert")
        self._btn_invert.setToolTip("Invert selection")
        self._btn_invert.clicked.connect(self._on_invert_selection)
        toolbar.addWidget(self._btn_invert)

        toolbar.addSpacing(10)

        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setToolTip("Undo last action (Ctrl+Z)")
        self._btn_undo.clicked.connect(self._on_undo)
        toolbar.addWidget(self._btn_undo)

        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setToolTip("Redo (Ctrl+Y)")
        self._btn_redo.clicked.connect(self._on_redo)
        toolbar.addWidget(self._btn_redo)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Point size + brush radius
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Pt Size:"))
        self._pt_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._pt_size_slider.setRange(1, 15)
        self._pt_size_slider.setValue(3)
        self._pt_size_slider.setFixedWidth(80)
        self._pt_size_slider.valueChanged.connect(self._on_point_size_changed)
        controls_row.addWidget(self._pt_size_slider)

        controls_row.addSpacing(10)
        controls_row.addWidget(QLabel("Brush R:"))
        self._brush_radius_spin = QSpinBox()
        self._brush_radius_spin.setRange(5, 200)
        self._brush_radius_spin.setValue(30)
        self._brush_radius_spin.setSuffix(" px")
        self._brush_radius_spin.valueChanged.connect(self._on_brush_radius_changed)
        controls_row.addWidget(self._brush_radius_spin)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        # Selection info
        self._sel_label = QLabel("No data loaded")
        self._sel_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self._sel_label)

        # ══════════════════════════════════════════════════════════════
        # STEP 1: LOAD
        # ══════════════════════════════════════════════════════════════
        load_group = self._collapsible("1. Load Data")
        load_form = QFormLayout()

        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText("Select LAS, DXF, PLY...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_file)
        file_row = QHBoxLayout()
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(browse_btn)
        load_form.addRow("File:", file_row)

        self._load_btn = QPushButton("Load")
        self._load_btn.clicked.connect(self._on_load)
        self._load_btn.setEnabled(False)
        load_form.addRow(self._load_btn)

        load_group.setLayout(load_form)
        layout.addWidget(load_group)

        # ══════════════════════════════════════════════════════════════
        # STEP 2: CLEAN (auto + interactive)
        # ══════════════════════════════════════════════════════════════
        clean_group = self._collapsible("2. Clean")
        clean_form = QFormLayout()

        self._sor_check = QCheckBox("SOR (k=20, std=2.0)")
        self._sor_check.setChecked(True)
        clean_form.addRow(self._sor_check)

        self._ror_check = QCheckBox("ROR (r=0.5m, min=6)")
        clean_form.addRow(self._ror_check)

        self._voxel_check = QCheckBox("Voxel downsample (0.05m)")
        clean_form.addRow(self._voxel_check)

        preview_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview Removal")
        self._preview_btn.setToolTip("Show points that would be removed in red")
        self._preview_btn.clicked.connect(self._on_preview_clean)
        preview_row.addWidget(self._preview_btn)

        self._apply_clean_btn = QPushButton("Apply Clean")
        self._apply_clean_btn.clicked.connect(self._on_apply_clean)
        preview_row.addWidget(self._apply_clean_btn)
        clean_form.addRow(preview_row)

        clean_group.setLayout(clean_form)
        layout.addWidget(clean_group)

        # ══════════════════════════════════════════════════════════════
        # STEP 3: GROUND REMOVAL
        # ══════════════════════════════════════════════════════════════
        ground_group = self._collapsible("3. Ground Removal")
        ground_form = QFormLayout()

        ground_row = QHBoxLayout()
        self._auto_ground_btn = QPushButton("Auto CSF")
        self._auto_ground_btn.setToolTip("Automatic cloth simulation filter")
        self._auto_ground_btn.clicked.connect(self._on_auto_ground)
        ground_row.addWidget(self._auto_ground_btn)

        self._select_ground_btn = QPushButton("Select & Delete")
        self._select_ground_btn.setToolTip("Use selection tools to mark ground, then delete")
        self._select_ground_btn.clicked.connect(self._on_delete_selected)
        ground_row.addWidget(self._select_ground_btn)
        ground_form.addRow(ground_row)

        ground_group.setLayout(ground_form)
        layout.addWidget(ground_group)

        # ══════════════════════════════════════════════════════════════
        # STEP 4: SEGMENT
        # ══════════════════════════════════════════════════════════════
        seg_group = self._collapsible("4. Segment")
        seg_form = QFormLayout()

        self._seg_method = QComboBox()
        self._seg_method.addItems(["Geometric (Region Growing)", "Watershed (Image-based)"])
        seg_form.addRow("Method:", self._seg_method)

        seg_row = QHBoxLayout()
        self._run_seg_btn = QPushButton("Run Segmentation")
        self._run_seg_btn.clicked.connect(self._on_run_segmentation)
        seg_row.addWidget(self._run_seg_btn)

        self._extract_btn = QPushButton("Extract Sizes")
        self._extract_btn.clicked.connect(self._on_extract_sizes)
        self._extract_btn.setEnabled(False)
        seg_row.addWidget(self._extract_btn)
        seg_form.addRow(seg_row)

        seg_group.setLayout(seg_form)
        layout.addWidget(seg_group)

        # ══════════════════════════════════════════════════════════════
        # STEP 5: EDIT FRAGMENTS
        # ══════════════════════════════════════════════════════════════
        edit_group = self._collapsible("5. Edit Fragments")
        edit_form = QFormLayout()

        self._frag_info = QLabel("Run segmentation first")
        edit_form.addRow(self._frag_info)

        edit_row = QHBoxLayout()
        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.setToolTip("Merge two fragments: click first, Shift+click second")
        self._merge_btn.clicked.connect(self._on_merge_fragments)
        self._merge_btn.setEnabled(False)
        edit_row.addWidget(self._merge_btn)

        self._noise_btn = QPushButton("Mark as Noise")
        self._noise_btn.setToolTip("Reclassify selected points as noise (-1)")
        self._noise_btn.clicked.connect(self._on_mark_noise)
        self._noise_btn.setEnabled(False)
        edit_row.addWidget(self._noise_btn)
        edit_form.addRow(edit_row)

        edit_group.setLayout(edit_form)
        layout.addWidget(edit_group)

        # ══════════════════════════════════════════════════════════════
        # PROGRESS + STATUS
        # ══════════════════════════════════════════════════════════════
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        layout.addWidget(self._status)

        layout.addStretch()

        if self.main_layout:
            self.main_layout.addLayout(layout)
        else:
            self.setLayout(layout)

        # ── Keyboard Shortcuts ──
        QShortcut(QKeySequence("Delete"), self, self._on_delete_selected)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+A"), self, self._on_select_all)
        QShortcut(QKeySequence("Escape"), self, self._on_clear_selection)

        # Connect mode buttons
        self._mode_group.idToggled.connect(self._on_mode_changed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_btn(self, text: str, tooltip: str, mode_id: int) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setMinimumWidth(50)
        self._mode_group.addButton(btn, mode_id)
        return btn

    def _collapsible(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        return group

    def _get_interactor(self):
        """Lazy-init the PointCloudInteractor."""
        if self._interactor is None:
            if not self.controller or not hasattr(self.controller, 'r'):
                return None
            from ..visualization.renderer.point_cloud_interactor import PointCloudInteractor
            self._interactor = PointCloudInteractor(self.controller.r, parent=self)
            self._interactor.selection_changed.connect(self._on_selection_changed)
        return self._interactor

    def _get_dataset(self):
        registry = self.get_registry()
        if registry and registry.has_fragment_dataset():
            return registry.get_fragment_dataset(copy_data=False)
        return None

    def _refresh_renderer(self):
        """Re-render the point cloud after edit operations."""
        ds = self._get_dataset()
        if ds is None or ds.fused_cloud is None:
            return
        from ..visualization.renderer.fragment_renderer import FragmentRenderer, FragColourMode
        renderer = self.controller.r if self.controller else None
        if renderer is None:
            return
        frag_r = FragmentRenderer(renderer)
        shift = getattr(renderer, '_global_shift', None)

        if ds.fragment_labels is not None and ds.fragment_labels.max() >= 0:
            frag_r.render_point_cloud(
                ds.fused_cloud, FragColourMode.FRAGMENT_LABEL,
                ds.fragment_labels, self._pt_size_slider.value(), shift,
            )
        else:
            frag_r.render_point_cloud(
                ds.fused_cloud, FragColourMode.ELEVATION,
                point_size=self._pt_size_slider.value(), global_shift=shift,
            )

        interactor = self._get_interactor()
        if interactor:
            interactor.set_cloud(ds.fused_cloud, shift)

        self._update_info()
        try:
            renderer.plotter.render()
        except Exception:
            pass

    def _update_info(self):
        ds = self._get_dataset()
        if ds is None:
            self._sel_label.setText("No data loaded")
            return

        interactor = self._get_interactor()
        n_sel = interactor.selection_count if interactor else 0
        n_total = ds.point_count
        n_frags = ds.fragment_count

        parts = [f"{n_total:,} points"]
        if n_sel > 0:
            parts.append(f"{n_sel:,} selected")
        if n_frags > 0:
            parts.append(f"{n_frags} fragments")
        if ds.fsd_global:
            parts.append(f"D50={ds.fsd_global.d50:.3f}m")
        self._sel_label.setText(" | ".join(parts))

    def _set_status(self, msg: str):
        self._status.setText(msg)

    # ------------------------------------------------------------------
    # Mode Switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode_id: int, checked: bool):
        if not checked:
            return
        interactor = self._get_interactor()
        if interactor is None:
            return

        from ..visualization.renderer.point_cloud_interactor import SelectionMode
        mode_map = {
            0: SelectionMode.NAVIGATE,
            1: SelectionMode.BOX,
            2: SelectionMode.LASSO,
            3: SelectionMode.BRUSH,
        }
        interactor.set_mode(mode_map.get(mode_id, SelectionMode.NAVIGATE))

    def _on_selection_changed(self, mask):
        self._update_info()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Cloud / CAD File", "",
            "All Supported (*.las *.laz *.ply *.obj *.xyz *.dxf);;"
            "LiDAR (*.las *.laz);;Point Cloud (*.ply *.obj *.xyz);;"
            "DXF (*.dxf);;All (*)"
        )
        if path:
            self._file_edit.setText(path)
            self._load_btn.setEnabled(True)

    def _on_load(self):
        if not self.controller:
            return
        path = self._file_edit.text()
        if not path:
            return

        self._load_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._set_status("Loading...")

        params = {
            "las_path": path,
            "name": Path(path).stem,
        }

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._set_status(msg)

        params["_progress_callback"] = on_progress
        self.controller.run_task("frag_import", params,
                                  callback=self._on_load_complete,
                                  progress_callback=on_progress)

    def _on_load_complete(self, result):
        self._progress.setVisible(False)
        self._load_btn.setEnabled(True)

        if isinstance(result, dict) and "error" not in result:
            self._set_status(f"Loaded: {result.get('point_count', 0):,} points")
            self._refresh_renderer()
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._set_status(f"Error: {error}")
            QMessageBox.critical(self, "Load Failed", str(error))

    # ------------------------------------------------------------------
    # Selection Actions
    # ------------------------------------------------------------------

    def _on_delete_selected(self):
        interactor = self._get_interactor()
        ds = self._get_dataset()
        if not interactor or not ds or not interactor.has_selection:
            self._set_status("Nothing selected")
            return

        mask = interactor.get_selection_mask()
        n_del = int(mask.sum())

        from ..controllers.undo.frag_commands import DeletePointsCommand
        cmd = DeletePointsCommand(ds, mask, self._refresh_renderer)
        if self.controller:
            self.controller.undo_manager.execute(cmd)
        else:
            cmd.execute()

        interactor.clear_selection()
        self._set_status(f"Deleted {n_del:,} points")

    def _on_crop_to_selection(self):
        interactor = self._get_interactor()
        ds = self._get_dataset()
        if not interactor or not ds or not interactor.has_selection:
            self._set_status("Nothing selected")
            return

        mask = interactor.get_selection_mask()
        n_keep = int(mask.sum())

        from ..controllers.undo.frag_commands import CropToSelectionCommand
        cmd = CropToSelectionCommand(ds, mask, self._refresh_renderer)
        if self.controller:
            self.controller.undo_manager.execute(cmd)
        else:
            cmd.execute()

        interactor.clear_selection()
        self._set_status(f"Cropped to {n_keep:,} points")

    def _on_invert_selection(self):
        interactor = self._get_interactor()
        if interactor:
            interactor.invert_selection()

    def _on_select_all(self):
        interactor = self._get_interactor()
        if interactor:
            interactor.select_all()

    def _on_clear_selection(self):
        interactor = self._get_interactor()
        if interactor:
            interactor.clear_selection()
        self._btn_navigate.setChecked(True)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _on_undo(self):
        if self.controller:
            self.controller.undo_manager.undo()

    def _on_redo(self):
        if self.controller:
            self.controller.undo_manager.redo()

    # ------------------------------------------------------------------
    # Point Size / Brush Radius
    # ------------------------------------------------------------------

    def _on_point_size_changed(self, val):
        interactor = self._get_interactor()
        if interactor:
            interactor.set_point_size(float(val))
        self._refresh_renderer()

    def _on_brush_radius_changed(self, val):
        interactor = self._get_interactor()
        if interactor:
            interactor.set_brush_radius(float(val))

    # ------------------------------------------------------------------
    # Auto Clean
    # ------------------------------------------------------------------

    def _on_preview_clean(self):
        """Show points that would be removed in red (preview only)."""
        ds = self._get_dataset()
        if ds is None or ds.fused_cloud is None:
            return

        from ..scans.preprocessing import statistical_outlier_removal
        cloud = ds.fused_cloud
        _, removed_idx = statistical_outlier_removal(cloud, k=20, std_ratio=2.0)

        if len(removed_idx) == 0:
            self._set_status("Preview: no outliers found")
            return

        # Show removed points in red
        renderer = self.controller.r if self.controller else None
        if renderer and renderer.plotter:
            xyz = cloud[removed_idx, :3].copy()
            shift = getattr(renderer, '_global_shift', None)
            if shift is not None:
                xyz -= shift
            import pyvista as pv
            pd = pv.PolyData(xyz)
            renderer.plotter.add_mesh(
                pd, color="red", point_size=5,
                render_points_as_spheres=True,
                name="frag_clean_preview", pickable=False,
            )
            renderer.plotter.render()
            self._set_status(f"Preview: {len(removed_idx):,} points to remove (red)")

    def _on_apply_clean(self):
        if not self.controller:
            return

        self._progress.setVisible(True)
        self._set_status("Cleaning...")

        params = {"config": None}  # Uses defaults

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._set_status(msg)

        params["_progress_callback"] = on_progress

        self.controller.run_task("frag_preprocess", params,
                                  callback=self._on_clean_complete,
                                  progress_callback=on_progress)

    def _on_clean_complete(self, result):
        self._progress.setVisible(False)
        # Remove preview
        renderer = self.controller.r if self.controller else None
        if renderer and renderer.plotter:
            try:
                renderer.plotter.remove_actor("frag_clean_preview")
            except Exception:
                pass
        if isinstance(result, dict) and "error" not in result:
            self._set_status(f"Cleaned: {result.get('point_count', 0):,} points")
            self._refresh_renderer()
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._set_status(f"Error: {error}")

    # ------------------------------------------------------------------
    # Ground Removal
    # ------------------------------------------------------------------

    def _on_auto_ground(self):
        """Run CSF ground filter and delete ground points."""
        ds = self._get_dataset()
        if ds is None or ds.fused_cloud is None:
            return

        from ..scans.preprocessing import ground_filter_csf
        self._set_status("Running ground filter...")

        ground, non_ground = ground_filter_csf(ds.fused_cloud)
        if len(ground) == 0:
            self._set_status("No ground points detected")
            return

        # Mark ground as selected for preview
        from ..scans.point_cloud_selector import box_select
        # Build mask: ground points
        from scipy.spatial import cKDTree
        tree = cKDTree(ds.fused_cloud[:, :3])
        _, idx = tree.query(ground[:, :3], k=1)
        mask = np.zeros(len(ds.fused_cloud), dtype=bool)
        mask[idx] = True

        from ..controllers.undo.frag_commands import DeletePointsCommand
        cmd = DeletePointsCommand(ds, mask, self._refresh_renderer,
                                   description="Remove ground (CSF)")
        if self.controller:
            self.controller.undo_manager.execute(cmd)
        else:
            cmd.execute()

        self._set_status(f"Removed {len(ground):,} ground points")

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _on_run_segmentation(self):
        if not self.controller:
            return

        method = "geometric" if self._seg_method.currentIndex() == 0 else "watershed"
        params = {"method": method}

        self._progress.setVisible(True)
        self._set_status(f"Segmenting ({method})...")

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._set_status(msg)

        params["_progress_callback"] = on_progress
        self.controller.run_task("frag_segment", params,
                                  callback=self._on_seg_complete,
                                  progress_callback=on_progress)

    def _on_seg_complete(self, result):
        self._progress.setVisible(False)
        if isinstance(result, dict) and "error" not in result:
            n = result.get("fragment_count", 0)
            self._set_status(f"Segmented: {n} fragments")
            self._extract_btn.setEnabled(True)
            self._merge_btn.setEnabled(True)
            self._noise_btn.setEnabled(True)
            self._frag_info.setText(f"{n} fragments detected")
            self._refresh_renderer()
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._set_status(f"Error: {error}")

    def _on_extract_sizes(self):
        if not self.controller:
            return

        self._progress.setVisible(True)
        self._set_status("Extracting sizes...")
        params = {}

        def on_progress(pct, msg):
            self._progress.setValue(int(pct))
            self._set_status(msg)

        params["_progress_callback"] = on_progress
        self.controller.run_task("frag_size", params,
                                  callback=self._on_sizes_complete,
                                  progress_callback=on_progress)

    def _on_sizes_complete(self, result):
        self._progress.setVisible(False)
        if isinstance(result, dict) and "error" not in result:
            self._set_status(
                f"D50={result.get('d50', 0):.3f}m  D80={result.get('d80', 0):.3f}m  "
                f"({result.get('fragment_count', 0)} fragments)"
            )
            self._update_info()
        else:
            error = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            self._set_status(f"Error: {error}")

    # ------------------------------------------------------------------
    # Fragment Editing
    # ------------------------------------------------------------------

    def _on_merge_fragments(self):
        """Merge: select points spanning two fragments, merge their labels."""
        interactor = self._get_interactor()
        ds = self._get_dataset()
        if not interactor or not ds or ds.fragment_labels is None or not interactor.has_selection:
            self._set_status("Select points spanning fragments to merge")
            return

        mask = interactor.get_selection_mask()
        selected_labels = ds.fragment_labels[mask]
        unique = np.unique(selected_labels[selected_labels >= 0])

        if len(unique) < 2:
            self._set_status("Select points from at least 2 fragments to merge")
            return

        target = int(unique[0])
        from ..controllers.undo.frag_commands import MergeFragmentsCommand
        cmd = MergeFragmentsCommand(ds, unique.tolist(), target, self._refresh_renderer)
        if self.controller:
            self.controller.undo_manager.execute(cmd)
        self._set_status(f"Merged {len(unique)} fragments -> {target}")
        interactor.clear_selection()

    def _on_mark_noise(self):
        """Reclassify selected points as noise (-1)."""
        interactor = self._get_interactor()
        ds = self._get_dataset()
        if not interactor or not ds or ds.fragment_labels is None or not interactor.has_selection:
            return

        mask = interactor.get_selection_mask()
        from ..controllers.undo.frag_commands import ReclassifyPointsCommand
        cmd = ReclassifyPointsCommand(ds, mask, -1, self._refresh_renderer)
        if self.controller:
            self.controller.undo_manager.execute(cmd)
        self._set_status(f"Reclassified {int(mask.sum()):,} points as noise")
        interactor.clear_selection()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._interactor:
            self._interactor.cleanup()
            self._interactor = None
        super().closeEvent(event)
