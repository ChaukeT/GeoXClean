"""
Indicator RBF Domain Panel
================================
Tabbed panel for indicator domain estimation using RBF
interpolation and marching-cubes isosurface extraction.
"""

from __future__ import annotations

import logging
import math
import traceback
from typing import Optional

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QGroupBox, QTabWidget,
    QWidget, QFrame, QLineEdit,
    QProgressBar, QSizePolicy, QColorDialog, QTextEdit,
    QFileDialog, QSlider, QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    FigureCanvas = QWidget
    Figure = None
    HAS_MPL = False

from .base_analysis_panel import BaseAnalysisPanel
from .mixins.coded_domain_filter_mixin import CodedDomainFilterMixin
from .panel_toolkit import (
    make_form, form_row, make_combo, make_spin, make_int_spin,
    action_button,
)
from .modern_styles import ModernColors, get_theme_colors
from ..utils.variable_utils import get_grade_columns
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)

# ── Layout constants ─────────────────────────────────────────────
_GAP = 4
_BODY_MARGINS = (10, 10, 10, 10)
_ACTION_MARGINS = (10, 6, 10, 6)

# ── Style helpers ────────────────────────────────────────────────
_style_cache: dict = {}

def _s(key: str) -> str:
    if key not in _style_cache:
        C = ModernColors
        _style_cache.update({
            "caption":    f"color:{C.TEXT_HINT}; font-size:8px; font-weight:600; letter-spacing:0.8px;",
            "hint":       f"color:{C.TEXT_HINT}; font-size:10px;",
            "sub":        f"color:{C.TEXT_HINT}; font-size:9px;",
            "success":    "color:#27AE60; font-size:10px;",
            "card":       (f"background:{C.CARD_BG}; border:1px solid {C.DIVIDER};"
                           f" border-radius:5px; padding:4px;"),
            "line":       f"background:{C.DIVIDER};",
        })
    return _style_cache[key]

def _badge(color: str) -> str:
    return (
        f"color:{color}; font-size:10px; font-weight:bold;"
        f"background:{ModernColors.CARD_BG};"
        f"border:1px solid {color};"
        f"border-radius:8px; padding:2px 8px;"
    )

def _wrap(layout, parent_lay):
    """Wrap a layout in a QWidget and add to parent."""
    w = QWidget()
    w.setLayout(layout)
    parent_lay.addWidget(w)

def _group_box(title: str) -> QGroupBox:
    gb = QGroupBox(title)
    gb.setStyleSheet(f"""
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {ModernColors.DIVIDER};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {ModernColors.TEXT_PRIMARY};
        }}
    """)
    return gb

_COORD_TRIPLES = [["X", "Y", "Z"], ["x", "y", "z"], ["XC", "YC", "ZC"]]

def _find_coord_cols(df):
    for cs in _COORD_TRIPLES:
        if all(c in df.columns for c in cs):
            return cs
    return None

def _map_labels_to_index(labels, run_idx, target_index):
    col = pd.Series("Unclassified", index=target_index, dtype=object)
    lab = np.asarray(labels)
    if run_idx is not None and len(lab) == len(run_idx):
        col.loc[run_idx] = lab
    elif len(lab) == len(target_index):
        col[:] = lab
    return col


def _extract_sample_identity_payload(df: Optional[pd.DataFrame]) -> dict:
    """Capture stable sample identifiers for downstream IRBF remapping."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    payload: dict = {}

    def _find_col(candidates):
        lookup = {str(c).strip().lower() for c in candidates}
        for col in df.columns:
            if str(col).strip().lower() in lookup:
                return col
        return None

    gid_col = _find_col(["GLOBAL_INTERVAL_ID", "global_interval_id"])
    if gid_col is not None:
        payload["sample_global_interval_ids"] = (
            df[gid_col].astype(str).tolist()
        )

    iid_col = _find_col(["INTERVAL_ID", "interval_id", "SAMPLE_ID", "sample_id"])
    if iid_col is not None:
        payload["sample_interval_ids"] = (
            df[iid_col].astype(str).tolist()
        )

    hole_col = _find_col(["HOLEID", "holeid", "hole_id", "BHID"])
    from_col = _find_col(["FROM", "from", "depth_from", "start"])
    to_col = _find_col(["TO", "to", "depth_to", "end"])
    if hole_col is not None and from_col is not None and to_col is not None:
        payload["sample_hole_ids"] = df[hole_col].astype(str).tolist()
        payload["sample_from_values"] = pd.to_numeric(df[from_col], errors="coerce").tolist()
        payload["sample_to_values"] = pd.to_numeric(df[to_col], errors="coerce").tolist()

    return payload


# ═══════════════════════════════════════════════════════════════════
#  Workers
# ═══════════════════════════════════════════════════════════════════

class _Worker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            from ..geostats.indicator_rbf_engine import run_indicator_rbf
            r = run_indicator_rbf(
                coords=self.p["coords"], values=self.p["values"],
                cutoff=self.p["cutoff"], iso_value=self.p["iso_value"],
                kernel=self.p["kernel"], smoothing=self.p["smoothing"],
                trend_degree=self.p["trend_degree"],
                resolution=self.p.get("resolution"),
                clip_far_field=self.p.get("clip_far_field", True),
                clip_factor=self.p.get("clip_factor", 2.0),
                smooth_iterations=self.p.get("smooth_iters", 15),
                anisotropy_ranges=self.p.get("anisotropy_ranges"),
                anisotropy_rotation=self.p.get("anisotropy_rotation"),
                progress=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(r)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class _ThresholdWorker(QThread):
    finished = pyqtSignal(object, object, float)
    failed = pyqtSignal(str)

    def __init__(self, prob, x, y, z, iso, smooth_iters):
        super().__init__()
        self.prob, self.x, self.y, self.z = prob, x, y, z
        self.iso = iso
        self.smooth_iters = smooth_iters

    def run(self):
        try:
            from ..geostats.indicator_rbf_engine import _extract_iso, _close_mesh, _smooth
            verts, faces = _extract_iso(self.prob, self.x, self.y, self.z, self.iso)
            if verts is not None:
                verts, faces = _close_mesh(verts, faces)
                verts, faces = _smooth(verts, faces, self.smooth_iters)
            self.finished.emit(verts, faces, self.iso)
        except Exception as e:
            self.failed.emit(f"Threshold update failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  Histogram (compact)
# ═══════════════════════════════════════════════════════════════════

class _Histogram(FigureCanvas):
    def __init__(self, parent=None):
        if not HAS_MPL:
            super().__init__(parent); return
        c = get_theme_colors()
        self.fig = Figure(figsize=(4, 1.8), dpi=100)
        self.fig.patch.set_facecolor(c.CARD_BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(150)
        self._v = self._c = None

    def plot(self, vals, cutoff):
        if not HAS_MPL: return
        self._v, self._c = vals, cutoff
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        c = get_theme_colors()
        ax.set_facecolor(c.CARD_BG)
        ax.tick_params(colors=c.TEXT_PRIMARY, labelsize=7)
        for s in ax.spines.values(): s.set_color(c.DIVIDER)

        v = vals[np.isfinite(vals)]
        if not len(v): self.draw_idle(); return

        # Use shared bins so inside/outside histograms align properly
        bins = np.linspace(v.min(), v.max(), 50)
        lo, hi = v[v <= cutoff], v[v > cutoff]
        if len(lo): ax.hist(lo, bins=bins, color="#E74C3C", alpha=.75, label=f"Out ({len(lo):,})")
        if len(hi): ax.hist(hi, bins=bins, color="#27AE60", alpha=.75, label=f"In ({len(hi):,})")

        ax.axvline(cutoff, color="#F39C12", lw=2, ls="--")
        ax.legend(fontsize=7, loc="upper right", framealpha=.7)
        self.fig.tight_layout(pad=0.3); self.draw_idle()

    def replot(self, cutoff):
        if self._v is not None: self.plot(self._v, cutoff)


class _Tile(QFrame):
    def __init__(self, title, value="--", accent=None):
        super().__init__()
        self.setObjectName("tile")
        self.setStyleSheet(f"QFrame#tile {{ {_s('card')} }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet(_s("caption"))
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        self._val = QLabel(value)
        self._val.setStyleSheet(f"font-size:13px; font-weight:bold; color:{accent or ModernColors.TEXT_PRIMARY};")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._val)

    def set(self, text: str):
        self._val.setText(text)


# ═══════════════════════════════════════════════════════════════════
#  PANEL
# ═══════════════════════════════════════════════════════════════════

class IndicatorRBFPanel(CodedDomainFilterMixin, BaseAnalysisPanel):
    PANEL_ID = "IndicatorRBFPanel"
    PANEL_NAME = "Indicator RBF"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT
    task_name = "indicator_rbf"
    panel_title = "Indicator RBF"

    request_visualization = pyqtSignal(object, str)

    def __init__(self, parent=None, **kw):
        self.drillhole_data: Optional[pd.DataFrame] = None
        self._result = None
        self._worker: Optional[_Worker] = None
        self._thresh_worker: Optional[_ThresholdWorker] = None
        self._actor_names: list = []
        self._inside_color = "#E8A838"
        self._outside_color = "#5DADE2"
        self._ui_ready = False
        self.main_window = None
        super().__init__(parent=parent, **kw)

    def refresh_theme(self): pass

    def _plotter(self):
        mw = self.main_window or getattr(self, "_main_window_ref", None)
        if mw is None:
            p = self.parent()
            while p:
                if hasattr(p, "viewer_widget"): mw = p; break
                p = p.parent() if hasattr(p, "parent") else None
        if mw and hasattr(mw, "viewer_widget") and mw.viewer_widget:
            r = getattr(mw.viewer_widget, "renderer", None)
            if r and hasattr(r, "plotter"):
                return r.plotter, r
        return None, None

    def _global_shift(self):
        _, r = self._plotter()
        return getattr(r, "_global_shift", None) if r else None

    # ══════════════════════════════════════════════════════════════
    #  UI Setup — Tabbed Interface
    # ══════════════════════════════════════════════════════════════

    def _setup_base_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {ModernColors.DIVIDER}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: {ModernColors.CARD_BG}; color: {ModernColors.TEXT_SECONDARY}; padding: 6px 12px; border: 1px solid transparent; border-bottom: 1px solid {ModernColors.DIVIDER}; }}
            QTabBar::tab:selected {{ background: {ModernColors.ELEVATED_BG}; color: {ModernColors.ACCENT_PRIMARY}; border: 1px solid {ModernColors.DIVIDER}; border-bottom: 1px solid transparent; font-weight: bold; }}
            QTabBar::tab:hover:!selected {{ background: {ModernColors.CARD_HOVER}; }}
        """)

        # Data Tab
        tab_data = QWidget()
        lay_data = QVBoxLayout(tab_data)
        lay_data.setContentsMargins(*_BODY_MARGINS)
        self._build_data_classification(lay_data)
        lay_data.addStretch()
        self.tabs.addTab(tab_data, "Data")

        # Interpolation Tab
        tab_interp = QWidget()
        lay_interp = QVBoxLayout(tab_interp)
        lay_interp.setContentsMargins(*_BODY_MARGINS)
        self._build_interpolation(lay_interp)
        lay_interp.addStretch()
        self.tabs.addTab(tab_interp, "Setup")

        # Results Tab
        tab_disp = QWidget()
        lay_disp = QVBoxLayout(tab_disp)
        lay_disp.setContentsMargins(*_BODY_MARGINS)
        self._build_display_stability(lay_disp)
        self._build_results_export(lay_disp)
        lay_disp.addStretch()
        self.tabs.addTab(tab_disp, "Results")

        root.addWidget(self.tabs, stretch=1)
        root.addWidget(self._build_action_bar())

        self._ui_ready = True
        QTimer.singleShot(0, self._connect_registry_signals)

    def _build_data_classification(self, parent_lay):
        gb_src = _group_box("Source & Variables")
        src_lay = QVBoxLayout(gb_src)
        form = make_form()

        # Status
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("No data loaded")
        self.lbl_status.setStyleSheet(_s("hint"))
        status_row.addWidget(self.lbl_status, stretch=1)
        btn_load = action_button("Reload", style="secondary", tooltip="Pull data from registry")
        btn_load.clicked.connect(self._load_data)
        status_row.addWidget(btn_load)
        form.addRow(status_row)

        self.cb_var = make_combo(tooltip="Grade variable to binarise")
        self.cb_var.currentTextChanged.connect(self._var_changed)
        form_row(form, "Variable:", self.cb_var)

        self.domain_combo = make_combo(tooltip="Domain for hard-boundary filtering")
        self.domain_combo.addItem("All Data")
        self.domain_combo.currentTextChanged.connect(self._on_domain_filter_selection_changed)
        form_row(form, "Domain:", self.domain_combo)

        self.cb_filter = make_combo(tooltip="Secondary filter by lithology or zone")
        self.cb_filter.addItem("(No filter)")
        form_row(form, "Filter:", self.cb_filter)
        _wrap(form, src_lay)
        parent_lay.addWidget(gb_src)

        gb_class = _group_box("Classification (Cut-off)")
        c_lay = QVBoxLayout(gb_class)
        c_form = make_form()

        cutoff_row = QHBoxLayout()
        cutoff_row.setSpacing(_GAP)
        self.sp_cutoff = make_spin(-1e9, 1e9, 0.0, 4, tooltip="Threshold: > cut-off = Inside (1)")
        self.sp_cutoff.setSingleStep(1)
        self.sp_cutoff.valueChanged.connect(self._cutoff_changed)
        cutoff_row.addWidget(self.sp_cutoff, stretch=1)

        for txt, fn in [("Med", lambda: self._quick("median")), ("Avg", lambda: self._quick("mean")), ("P75", lambda: self._quick("p75"))]:
            b = QPushButton(txt)
            b.setFixedSize(32, 24)
            b.setStyleSheet(f"QPushButton {{ font-size:9px; border-radius:3px; background:{ModernColors.CARD_BG}; border:1px solid {ModernColors.DIVIDER}; }}")
            b.clicked.connect(fn)
            cutoff_row.addWidget(b)

        form_row(c_form, "Cut-off:", cutoff_row)
        _wrap(c_form, c_lay)

        # Histogram
        self.hist = _Histogram()
        c_lay.addWidget(self.hist)

        # Counts
        row = QHBoxLayout()
        row.setSpacing(_GAP)
        self.c_in = _Tile("INSIDE", "--", "#27AE60")
        self.c_out = _Tile("OUTSIDE", "--", "#E74C3C")
        self.c_pct = _Tile("RATIO", "--", "#3498DB")
        row.addWidget(self.c_in); row.addWidget(self.c_out); row.addWidget(self.c_pct)
        _wrap(row, c_lay)
        parent_lay.addWidget(gb_class)

    def _build_interpolation(self, parent_lay):
        gb_rbf = _group_box("RBF & Gridding")
        lay_rbf = QVBoxLayout(gb_rbf)
        form = make_form()

        self.cb_kernel = make_combo(["thin_plate_spline", "multiquadric", "cubic", "gaussian", "linear"])
        form_row(form, "Kernel:", self.cb_kernel)

        self.sp_smooth = make_spin(0.01, 100, 0.5, 2, tooltip="Smoothing parameter.")
        form_row(form, "Smoothing:", self.sp_smooth)

        self.cb_trend = make_combo(["Constant (0)", "Linear (1)"])
        self.cb_trend.setCurrentIndex(1)
        form_row(form, "Trend:", self.cb_trend)

        self.sp_iso = make_spin(0.01, 0.99, 0.50, 2, tooltip="Iso-surface probability threshold.")
        self.sp_iso.setSingleStep(0.05)
        form_row(form, "Iso Thresh:", self.sp_iso)

        self.chk_auto = QCheckBox("Auto grid resolution")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(lambda on: self.sp_res.setEnabled(not on))
        form.addRow(self.chk_auto)

        self.sp_res = make_spin(0.5, 500, 10, 1)
        self.sp_res.setEnabled(False)
        form_row(form, "Cell Size:", self.sp_res)
        _wrap(form, lay_rbf)
        parent_lay.addWidget(gb_rbf)

        # Anisotropy
        self.gb_aniso = QGroupBox("Ellipsoidal Anisotropy")
        self.gb_aniso.setCheckable(True)
        self.gb_aniso.setChecked(False)
        self.gb_aniso.setStyleSheet(f"""
            QGroupBox {{ font-weight: bold; border: 1px solid {ModernColors.DIVIDER}; border-radius: 4px; margin-top: 10px; padding-top: 15px; }}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: {ModernColors.TEXT_PRIMARY}; }}
            QGroupBox::indicator {{ width: 14px; height: 14px; }}
        """)
        lay_aniso = QVBoxLayout(self.gb_aniso)
        a_form = make_form()

        ranges_row = QHBoxLayout()
        for lbl_text, attr, default in [("Maj", "sp_rmaj", 100), ("Semi", "sp_rsemi", 100), ("Min", "sp_rmin", 50)]:
            col = QVBoxLayout(); col.setSpacing(1)
            lbl = QLabel(lbl_text); lbl.setStyleSheet(_s("sub")); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl)
            sp = make_spin(1, 10000, default, 0)
            setattr(self, attr, sp); col.addWidget(sp)
            ranges_row.addLayout(col)
        form_row(a_form, "Ranges:", ranges_row)

        rot_row = QHBoxLayout()
        for lbl_text, attr, lo, hi, default in [("Az", "sp_az_aniso", 0, 360, 0), ("Dip", "sp_dip_aniso", -90, 90, 0)]:
            col = QVBoxLayout(); col.setSpacing(1)
            lbl = QLabel(lbl_text); lbl.setStyleSheet(_s("sub")); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl)
            sp = make_spin(lo, hi, default, 1)
            setattr(self, attr, sp); col.addWidget(sp)
            rot_row.addLayout(col)
        form_row(a_form, "Rotation:", rot_row)

        btn_import = action_button("Import Variogram", style="secondary")
        btn_import.clicked.connect(self._import_variogram_aniso)
        a_form.addRow("", btn_import)

        self.lbl_vario_status = QLabel("")
        self.lbl_vario_status.setStyleSheet(_s("hint"))
        a_form.addRow("", self.lbl_vario_status)

        _wrap(a_form, lay_aniso)
        parent_lay.addWidget(self.gb_aniso)

    def _build_display_stability(self, parent_lay):
        gb_disp = _group_box("Visuals & Smoothing")
        lay_disp = QVBoxLayout(gb_disp)
        form = make_form()

        colour_row = QHBoxLayout()
        self._btn_cin = QPushButton(); self._btn_cin.setFixedSize(28, 20)
        self._set_btn_color(self._btn_cin, self._inside_color)
        self._btn_cin.clicked.connect(lambda: self._pick("inside"))
        colour_row.addWidget(QLabel("In")); colour_row.addWidget(self._btn_cin); colour_row.addSpacing(10)
        self._btn_cout = QPushButton(); self._btn_cout.setFixedSize(28, 20)
        self._set_btn_color(self._btn_cout, self._outside_color)
        self._btn_cout.clicked.connect(lambda: self._pick("outside"))
        colour_row.addWidget(QLabel("Out")); colour_row.addWidget(self._btn_cout); colour_row.addStretch()
        form.addRow("Colours:", colour_row)

        self.sp_opacity = make_spin(0.05, 1.0, 1.0, 2)
        form_row(form, "Opacity:", self.sp_opacity)

        self.sp_clipf = make_spin(1, 10, 2, 1)
        form_row(form, "Clip Dist:", self.sp_clipf)

        self.sp_smiter = make_int_spin(0, 100, 15)
        form_row(form, "Smoothing Iters:", self.sp_smiter)

        _wrap(form, lay_disp)
        parent_lay.addWidget(gb_disp)

    def _build_results_export(self, parent_lay):
        grid = QGridLayout()
        grid.setSpacing(_GAP)
        self.rc_faces = _Tile("FACES", "--", "#27AE60")
        self.rc_cells = _Tile("CELLS", "--")
        self.rc_acc = _Tile("ACCURACY", "--", "#3498DB")
        self.rc_time = _Tile("RES (m)", "--")
        grid.addWidget(self.rc_faces, 0, 0); grid.addWidget(self.rc_cells, 0, 1)
        grid.addWidget(self.rc_acc, 1, 0); grid.addWidget(self.rc_time, 1, 1)
        _wrap(grid, parent_lay)

        # Dynamic Threshold slider — non-blocking via _ThresholdWorker
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Adjust Iso:"))
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(1, 99); self._thresh_slider.setValue(50)

        self._thresh_debounce = QTimer()
        self._thresh_debounce.setSingleShot(True)
        self._thresh_debounce.setInterval(250)
        self._thresh_debounce.timeout.connect(self._apply_threshold)
        self._thresh_slider.valueChanged.connect(self._on_threshold_changed)

        slider_row.addWidget(self._thresh_slider, stretch=1)
        self._thresh_label = QLabel("0.50")
        self._thresh_label.setFixedWidth(34)
        self._thresh_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thresh_label.setStyleSheet(f"font-weight:bold; font-size:10px; color:{ModernColors.TEXT_PRIMARY}; border:1px solid {ModernColors.DIVIDER}; border-radius:3px; padding:1px;")
        slider_row.addWidget(self._thresh_label)
        _wrap(slider_row, parent_lay)

        # Actions
        form = make_form()
        self.le_in = QLineEdit("IRBF_Inside")
        self.le_out = QLineEdit("IRBF_Outside")
        form_row(form, "In Name:", self.le_in)
        form_row(form, "Out Name:", self.le_out)
        _wrap(form, parent_lay)

        for text, slot in [
            ("Register Domain to DB", self._re_register),
            ("Export Classifications (.csv)", self._export_csv),
            ("Export Probability Field (.csv)", self._export_probability),
        ]:
            btn = action_button(text, style="secondary")
            btn.clicked.connect(slot)
            parent_lay.addWidget(btn)

        self.lbl_reg_status = QLabel("")
        self.lbl_reg_status.setWordWrap(True)
        self.lbl_reg_status.setStyleSheet(_s("hint"))
        parent_lay.addWidget(self.lbl_reg_status)

    def _build_action_bar(self) -> QFrame:
        bar = QFrame()
        bar.setStyleSheet(f"background:{ModernColors.ELEVATED_BG}; border-top:1px solid {ModernColors.DIVIDER};")
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(*_ACTION_MARGINS)

        self._prog_w = QWidget()
        pl = QVBoxLayout(self._prog_w)
        pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(2)
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100); self.pbar.setFixedHeight(4); self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet(f"QProgressBar {{ background:{ModernColors.DIVIDER}; border:none; border-radius:2px; }} QProgressBar::chunk {{ background:{ModernColors.ACCENT_PRIMARY}; border-radius:2px; }}")
        pl.addWidget(self.pbar)
        self.lbl_prog = QLabel(""); self.lbl_prog.setStyleSheet(_s("sub"))
        pl.addWidget(self.lbl_prog)
        self._prog_w.setVisible(False)
        lay.addWidget(self._prog_w)

        row = QHBoxLayout()
        self.btn_clear = action_button("Reset", style="secondary")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_clear.setEnabled(False)
        row.addWidget(self.btn_clear)
        row.addStretch()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(_badge(ModernColors.TEXT_HINT))
        row.addWidget(self.status_label)
        self.btn_run = action_button("Run Model", style="primary")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        lay.addLayout(row)
        return bar

    # ══════════════════════════════════════════════════════════════
    #  Logic Setup
    # ══════════════════════════════════════════════════════════════

    def set_drillhole_data(self, data):
        if isinstance(data, dict):
            for k in ("composites", "composites_df", "assays", "assays_df"):
                v = data.get(k)
                if isinstance(v, pd.DataFrame) and not v.empty:
                    self.drillhole_data = v; break
            else:
                self.drillhole_data = None
        elif isinstance(data, pd.DataFrame):
            self.drillhole_data = data
        else:
            self.drillhole_data = None
        if self.drillhole_data is not None and self._ui_ready:
            self._refresh_ui()

    def _load_data(self):
        reg = self.get_registry()
        if not reg: return
        d = reg.get_drillhole_data(copy_data=True)
        if d: self.set_drillhole_data(d)

    def _refresh_ui(self):
        df = self.drillhole_data
        if df is None or df.empty: return
        gc = get_grade_columns(df)
        self.cb_var.blockSignals(True); self.cb_var.clear()
        for c in gc: self.cb_var.addItem(c)
        self.cb_var.blockSignals(False)

        self._populate_filter()
        has_xyz = _find_coord_cols(df) is not None
        self.lbl_status.setText(f"{len(df):,} samples | {'OK' if has_xyz else 'NO XYZ'}")

        if hasattr(self, "domain_combo"):
            try: self._populate_domain_filter_combo(df, combo=self.domain_combo, all_label="All Data")
            except Exception: pass
        self._var_changed()

    def _populate_filter(self):
        df = self.drillhole_data
        if df is None: return
        self.cb_filter.blockSignals(True); self.cb_filter.clear()
        self.cb_filter.addItem("(No filter)")
        hints = {"domain", "lith", "unit", "rock", "zone", "facies", "ore", "type"}
        for c in df.columns:
            if any(h in c.lower() for h in hints) and 2 <= df[c].nunique() <= 200:
                for v in sorted(df[c].dropna().unique()):
                    self.cb_filter.addItem(f"{c}: {v}")
        self.cb_filter.blockSignals(False)

    def _var_changed(self, *_):
        df = self.drillhole_data
        var = self.cb_var.currentText()
        if df is None or not var or var not in df.columns: return
        v = df[var].to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v): return
        med = float(np.median(v))
        self.sp_cutoff.blockSignals(True)
        self.sp_cutoff.setValue(med)
        self.sp_cutoff.blockSignals(False)
        self._update_cards(v, med)
        self.hist.plot(v, med)

    def _cutoff_changed(self, val):
        df = self.drillhole_data
        var = self.cb_var.currentText()
        if df is None or not var or var not in df.columns: return
        v = df[var].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v):
            self._update_cards(v, val)
            self.hist.replot(val)

    def _update_cards(self, v, cutoff):
        ni = int((v > cutoff).sum())
        no = len(v) - ni
        self.c_in.set(f"{ni:,}")
        self.c_out.set(f"{no:,}")
        self.c_pct.set(f"{ni / (ni + no) * 100:.0f}%" if ni + no else "--")

    def _quick(self, what):
        df = self.drillhole_data
        var = self.cb_var.currentText()
        if df is None or not var or var not in df.columns: return
        v = df[var].dropna().to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v): return
        if what == "median":   self.sp_cutoff.setValue(float(np.median(v)))
        elif what == "mean":   self.sp_cutoff.setValue(float(np.mean(v)))
        elif what == "p75":    self.sp_cutoff.setValue(float(np.percentile(v, 75)))

    def _set_btn_color(self, btn, c):
        btn.setStyleSheet(f"background:{c}; border:1px solid #888; border-radius:3px;")

    def _pick(self, which):
        cur = self._inside_color if which == "inside" else self._outside_color
        c = QColorDialog.getColor(QColor(cur), self)
        if not c.isValid(): return
        if which == "inside":
            self._inside_color = c.name(); self._set_btn_color(self._btn_cin, c.name())
        else:
            self._outside_color = c.name(); self._set_btn_color(self._btn_cout, c.name())

    def _import_variogram_aniso(self):
        reg = self.get_registry()
        if not reg: return
        try:
            vario = None
            if hasattr(reg, "get_variogram_results"): vario = reg.get_variogram_results()
            if not vario and hasattr(reg, "get_results"): vario = reg.get_results("variogram")
            if not vario:
                self.lbl_vario_status.setText("No variogram available"); return
            p = vario if isinstance(vario, dict) else getattr(vario, "__dict__", {})
            r_max = p.get("range_max") or p.get("a_max") or p.get("range1", 100)
            r_mid = p.get("range_mid") or p.get("a_mid") or p.get("range2", 100)
            r_min = p.get("range_min") or p.get("a_min") or p.get("range3", 50)
            az = p.get("azimuth") or p.get("azim", 0)
            dip = p.get("dip", 0)
            self.gb_aniso.setChecked(True)
            self.sp_rmaj.setValue(float(r_max)); self.sp_rsemi.setValue(float(r_mid)); self.sp_rmin.setValue(float(r_min))
            self.sp_az_aniso.setValue(float(az)); self.sp_dip_aniso.setValue(float(dip))
            self.lbl_vario_status.setText(f"Loaded: {r_max:.0f}/{r_mid:.0f}/{r_min:.0f} m")
        except Exception as e:
            self.lbl_vario_status.setText(f"Import failed")

    # ══════════════════════════════════════════════════════════════
    #  Execution
    # ══════════════════════════════════════════════════════════════

    def _run(self):
        df = self.drillhole_data
        var = self.cb_var.currentText()
        if df is None or df.empty:
            self.show_error("No Data", "Load data first."); return
        if not var or var not in df.columns:
            self.show_error("Variable", "Select a valid variable."); return

        filt = df
        ft = self.cb_filter.currentText()
        if ft and ft != "(No filter)" and ": " in ft:
            col, val = ft.split(": ", 1)
            if col in df.columns:
                filt = df[df[col].astype(str) == val]
        if hasattr(self, "_apply_domain_filter"):
            filt, _ = self._apply_domain_filter(filt)

        cc = _find_coord_cols(filt)
        if cc is None:
            self.show_error("Coordinates", "No X/Y/Z columns found."); return

        coords = filt[cc].to_numpy(float)
        values = filt[var].to_numpy(float)
        self._run_index = filt.index

        params = dict(
            coords=coords, values=values,
            cutoff=self.sp_cutoff.value(), iso_value=self.sp_iso.value(),
            kernel=self.cb_kernel.currentText(), smoothing=self.sp_smooth.value(),
            trend_degree=self.cb_trend.currentIndex(),
            resolution=None if self.chk_auto.isChecked() else self.sp_res.value(),
            clip_factor=self.sp_clipf.value(), smooth_iters=self.sp_smiter.value(),
        )
        if self.gb_aniso.isChecked():
            params["anisotropy_ranges"] = (self.sp_rmaj.value(), self.sp_rsemi.value(), self.sp_rmin.value())
            az, dp = math.radians(-self.sp_az_aniso.value()), math.radians(self.sp_dip_aniso.value())
            cz, sz = math.cos(az), math.sin(az); cx, sx = math.cos(dp), math.sin(dp)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
            params["anisotropy_rotation"] = (Rx @ Rz)

        self.btn_run.setEnabled(False)
        self._prog_w.setVisible(True); self.pbar.setValue(0); self.lbl_prog.setText("Starting...")
        self.status_label.setText("Running"); self.status_label.setStyleSheet(_badge("#F39C12"))

        self._worker = _Worker(params)
        self._worker.progress.connect(self._on_prog)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_prog(self, pct, msg):
        self.pbar.setValue(pct); self.lbl_prog.setText(msg)

    def _on_done(self, result):
        self.btn_run.setEnabled(True); self._prog_w.setVisible(False); self.btn_clear.setEnabled(True)
        self.status_label.setText("Complete"); self.status_label.setStyleSheet(_badge("#27AE60"))
        self._result = result
        self._show_results(result)
        self._display()
        self._register_domain(result)
        self._worker = None
        if result is not None:
            iso = result.stats.get("iso_value", 0.5)
            self._thresh_slider.blockSignals(True)
            self._thresh_slider.setValue(int(iso * 100))
            self._thresh_slider.blockSignals(False)
            self._thresh_label.setText(f"{iso:.2f}")

    def _on_fail(self, msg):
        self.btn_run.setEnabled(True); self._prog_w.setVisible(False)
        self.status_label.setText("Failed"); self.status_label.setStyleSheet(_badge("#E74C3C"))
        self.show_error("Indicator RBF", msg)
        self._worker = None

    def _on_threshold_changed(self, value: int):
        self._thresh_label.setText(f"{value / 100.0:.2f}")
        self._thresh_debounce.start()

    def _apply_threshold(self):
        """Re-extract isosurface at new threshold using background thread."""
        if self._result is None or self._result.prob is None: return
        iso = self._thresh_slider.value() / 100.0

        self._thresh_label.setStyleSheet(f"color: #F39C12; font-weight: bold; font-size:10px; border:1px solid {ModernColors.DIVIDER}; border-radius:3px; padding:1px;")
        sm_iters = self._result.stats.get("smooth_iterations", 15)

        if self._thresh_worker is not None and self._thresh_worker.isRunning():
            self._thresh_worker.terminate()
            self._thresh_worker.wait()

        self._thresh_worker = _ThresholdWorker(self._result.prob, self._result.x, self._result.y, self._result.z, iso, sm_iters)
        self._thresh_worker.finished.connect(self._on_threshold_done)
        self._thresh_worker.failed.connect(lambda msg: self.show_error("Threshold Error", msg))
        self._thresh_worker.start()

    def _on_threshold_done(self, verts, faces, iso):
        self._thresh_label.setStyleSheet(f"font-weight:bold; font-size:10px; color:{ModernColors.TEXT_PRIMARY}; border:1px solid {ModernColors.DIVIDER}; border-radius:3px; padding:1px;")
        if self._result is not None:
            self._result.verts = verts
            self._result.faces = faces
            self._result.stats["iso_value"] = iso
            self._display()
            self._result.stats["n_faces"] = len(faces) if faces is not None else 0
            self.rc_faces.set(f"{self._result.stats['n_faces']:,}")

    def _show_results(self, r):
        s = r.stats
        self.rc_faces.set(f"{s.get('n_faces', 0):,}")
        self.rc_cells.set(f"{s.get('inside_cells', 0):,}")
        acc = s.get("holdout_accuracy")
        self.rc_acc.set(f"{acc:.0%}" if acc else "N/A")
        self.rc_time.set(f"{s.get('resolution', '?'):.1f}")
        self.tabs.setCurrentIndex(2)

    # ══════════════════════════════════════════════════════════════
    #  Visualisation & Export
    # ══════════════════════════════════════════════════════════════

    def _display(self):
        r = self._result
        if r is None: return
        plotter, renderer = self._plotter()
        if plotter is None: return

        import pyvista as pv
        shift = self._global_shift()
        def S(c):
            a = np.asarray(c, dtype=float)
            return a - np.asarray(shift, dtype=float) if shift is not None else a

        self._remove_actors(plotter, renderer)
        opacity = self.sp_opacity.value()

        if r.verts is not None and r.faces is not None and len(r.verts):
            sv = S(r.verts)
            fp = np.hstack([np.full((len(r.faces), 1), 3, dtype=np.int64), r.faces]).ravel()

            m1 = pv.PolyData(sv.copy(), fp)
            m1["Domain"] = np.full(m1.n_points, 1.0)
            n1 = f"IRBF: {self.le_in.text()}"
            a1 = plotter.add_mesh(m1, name=n1, color=self._inside_color, opacity=opacity, show_edges=False, smooth_shading=True)
            self._register_layer(renderer, n1, a1, m1, "geology_surface")
            self._actor_names.append(n1)

        # Note: the IRBF probability field is NOT rendered here as a
        # translucent volume. Doing so would add a grey/pink bounding box
        # around the estimation block model whenever Quick-Layer toggles
        # or other visibility sweeps flip its hidden state back to True.
        # The iso-surface mesh above is the only user-visible representation
        # of the IRBF domain. The probability field is still available in
        # the registry payload for downstream tools that need it.
        plotter.reset_camera()

    def _remove_actors(self, plotter, renderer):
        for name in self._actor_names:
            try: plotter.remove_actor(name)
            except Exception: pass
            if renderer:
                for attr in ("active_layers", "scene_layers"):
                    getattr(renderer, attr, {}).pop(name, None)
        self._actor_names.clear()

    def _register_layer(self, renderer, name, actor, mesh, layer_type="geology_surface"):
        if renderer and hasattr(renderer, "add_layer"):
            try: renderer.add_layer(name, actor, mesh, layer_type=layer_type, opacity=self.sp_opacity.value())
            except Exception: pass

    def _clear(self):
        plotter, renderer = self._plotter()
        if plotter: self._remove_actors(plotter, renderer)
        self._result = None
        self.btn_clear.setEnabled(False)
        self.lbl_prog.setText(""); self.status_label.setText("Ready")
        self.status_label.setStyleSheet(_badge(ModernColors.TEXT_HINT))
        for t in (self.rc_faces, self.rc_cells, self.rc_acc, self.rc_time): t.set("--")

    def _register_domain(self, r):
        reg = self.get_registry()
        if not reg:
            logger.warning("IRBF: No registry available for domain registration")
            return
        in_nm = self.le_in.text().strip() or "IRBF_Inside"
        run_idx = getattr(self, "_run_index", None)

        # 1. Register IRBF domain to registry FIRST so downstream queries work
        try:
            identity_payload = _extract_sample_identity_payload(
                self.drillhole_data.loc[run_idx] if self.drillhole_data is not None and run_idx is not None else self.drillhole_data
            )
            payload = {
                "domain_name": in_nm,
                "sample_domain_labels": r.labels,
                "probability_field": r.prob,
                "inside_mask_shared": r.inside_mask,
                "sample_domain_index": list(run_idx) if run_idx is not None else None,
                "sample_domain_column": "IRBF_Domain",
                "x": r.x, "y": r.y, "z": r.z,
                "iso_value": r.stats.get("iso_value", 0.5),
                "iso_surface_verts": r.verts,
                "iso_surface_faces": r.faces,
                "statistics": r.stats,
            }
            payload.update(identity_payload)
            reg.register_indicator_rbf_domain(
                payload, source_panel="IndicatorRBFPanel",
                metadata={"domain_name": in_nm})
            self.lbl_reg_status.setText(f"Registered: {in_nm}")
            self.lbl_reg_status.setStyleSheet(_s("success"))
            logger.info("IRBF: Registered domain '%s' to registry", in_nm)
        except Exception as exc:
            logger.warning("IRBF: register_indicator_rbf_domain failed: %s", exc)

        # 2. Register generic domain model for geological model panel
        try:
            reg.register_domain_model(
                {"type": "indicator_rbf", "name": in_nm,
                 "iso_surface": {"vertices": r.verts, "faces": r.faces}},
                source_panel="IndicatorRBFPanel")
        except Exception:
            pass

        # 3. Inject IRBF_Domain column into composites DataFrame and emit signal
        #    Done AFTER registry registration so downstream _on_composites_refreshed
        #    handlers can query the registered domain via get_indicator_rbf_domain()
        if r.labels is not None:
            try:
                dh = reg.get_drillhole_data(copy_data=False)
                if dh is None:
                    logger.debug("IRBF: No drillhole data in registry, skipping column injection")
                else:
                    cdf = None
                    if isinstance(dh, pd.DataFrame):
                        cdf = dh
                    elif isinstance(dh, dict):
                        cdf = dh.get("composites") or dh.get("assays")
                    if cdf is not None and isinstance(cdf, pd.DataFrame):
                        cdf["IRBF_Domain"] = pd.Categorical(
                            _map_labels_to_index(r.labels, run_idx, cdf.index))
                        logger.info("IRBF: Injected IRBF_Domain column (%d rows)", len(cdf))
                        # Emit compositesLoaded to refresh domain combos in all panels
                        sig = getattr(reg, "signals", None)
                        if sig and hasattr(sig, "compositesLoaded"):
                            sig.compositesLoaded.emit(cdf)
                        elif hasattr(reg, "compositesLoaded"):
                            reg.compositesLoaded.emit(cdf)
                    else:
                        logger.debug("IRBF: No composites/assays DataFrame found in drillhole data")
            except Exception as exc:
                logger.warning("IRBF: Domain column injection failed: %s", exc)

    def _re_register(self):
        if self._result: self._register_domain(self._result)

    def _export_csv(self):
        r = self._result
        if r is None or r.labels is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Classifications", "irbf_classifications.csv", "CSV (*.csv)")
        if not path: return
        df = self.drillhole_data.copy() if self.drillhole_data is not None else pd.DataFrame()
        if not df.empty:
            df["IRBF_Domain"] = _map_labels_to_index(r.labels, getattr(self, "_run_index", None), df.index)
        else: df["Label"] = r.labels
        df.to_csv(path, index=False)
        self.lbl_reg_status.setText(f"Saved: {path}")

    def _export_probability(self):
        r = self._result
        if r is None or r.prob is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Probability", "irbf_probability.csv", "CSV (*.csv)")
        if not path: return
        zz, yy, xx = np.meshgrid(r.z, r.y, r.x, indexing="ij")
        flat = pd.DataFrame({"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel(), "Probability": r.prob.ravel()}).dropna()
        flat.to_csv(path, index=False)
        self.lbl_reg_status.setText(f"Saved: {path} ({len(flat):,} cells)")

    def _connect_registry_signals(self):
        if getattr(self, "_reg_conn", False): return
        reg = self.get_registry()
        if not reg: return
        try:
            for sn in ("drillholeDataLoaded", "compositesLoaded"):
                sig = getattr(reg.signals, sn, None) or getattr(reg, sn, None)
                if sig: sig.connect(lambda _=None: QTimer.singleShot(100, self._load_data))
            self._reg_conn = True
        except Exception: pass

    def showEvent(self, ev):
        super().showEvent(ev); self._connect_registry_signals()
