"""
Cross-Section Widget — 2D geological cross-section viewer.
============================================================

Displays a vertical cross-section through the geological model,
showing coloured domains, drillhole traces, and surface traces.

This is the primary QC tool for implicit geological models:
geologists inspect cross-sections to verify surface geometry,
contact honouring, and domain assignment.

Usage:
    widget = CrossSectionWidget()
    widget.set_model_result(build_result)
    widget.set_section_line(start_xy, end_xy, width=50)
    widget.update_section()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
        QPushButton, QVBoxLayout, QWidget,
    )
    from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QImage, QPixmap
    HAS_QT = True
except ImportError:
    HAS_QT = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Section sampling logic (no Qt dependency)
# ═══════════════════════════════════════════════════════════════════

def sample_section(
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    z_min: float,
    z_max: float,
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    nx: int = 200,
    nz: int = 100,
    width: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a vertical cross-section through a scalar field.

    Parameters
    ----------
    start_xy, end_xy : (x, y) endpoints of the section line
    z_min, z_max : vertical extent
    evaluate_fn : (N, 3) -> (N,) scalar field
    nx, nz : resolution along section and vertically
    width : swath width (0 = single slice)

    Returns
    -------
    section_values : (nz, nx) scalar field values on section
    h_coords : (nx,) horizontal distance along section
    z_coords : (nz,) vertical coordinates
    """
    sx, sy = start_xy
    ex, ey = end_xy

    dx = ex - sx
    dy = ey - sy
    length = np.sqrt(dx**2 + dy**2)

    if length < 1e-10:
        raise ValueError("Section line has zero length")

    # Unit direction along section
    ux, uy = dx / length, dy / length

    h_coords = np.linspace(0, length, nx)
    z_coords = np.linspace(z_min, z_max, nz)

    # Build sample grid
    hh, zz = np.meshgrid(h_coords, z_coords, indexing="ij")
    xx = sx + hh * ux
    yy = sy + hh * uy

    points = np.column_stack([
        xx.ravel(), yy.ravel(), zz.ravel(),
    ])

    values = evaluate_fn(points)
    section_values = values.reshape(nx, nz).T  # (nz, nx), row 0 = top

    # Flip so row 0 = z_max (top)
    section_values = section_values[::-1]

    return section_values, h_coords, z_coords


def classify_section(
    section_values: np.ndarray,
    isovalues: List[float],
) -> np.ndarray:
    """Classify section pixels into domain indices.

    Returns (nz, nx) int array of domain codes (0-based).
    """
    K = len(isovalues)
    codes = np.full(section_values.shape, K, dtype=np.int32)
    sorted_iso = sorted(isovalues)

    for k in range(K):
        mask = section_values < sorted_iso[k]
        codes[mask] = np.minimum(codes[mask], k)

    return np.clip(codes, 0, K)


def drillhole_traces_on_section(
    drillholes_df,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    width: float = 50.0,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> List[Dict[str, Any]]:
    """Extract drillhole collar traces within the section swath.

    Returns list of {hole_id, h_position, z_top, z_bottom} for
    drillholes within `width` metres of the section line.
    """
    if drillholes_df is None or drillholes_df.empty:
        return []

    sx, sy = start_xy
    ex, ey = end_xy
    dx, dy = ex - sx, ey - sy
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return []

    ux, uy = dx / length, dy / length
    # Normal to section line
    nx_dir, ny_dir = -uy, ux

    traces = []
    hole_col = "hole_id" if "hole_id" in drillholes_df.columns else None

    if hole_col:
        for hole_id, group in drillholes_df.groupby(hole_col):
            coords = group[[x_col, y_col, z_col]].values
            # Project onto section
            rel_x = coords[:, 0] - sx
            rel_y = coords[:, 1] - sy
            h_pos = rel_x * ux + rel_y * uy
            perp_dist = np.abs(rel_x * nx_dir + rel_y * ny_dir)

            # Filter: within swath
            in_swath = perp_dist < width / 2
            if not np.any(in_swath):
                continue

            h_mean = np.mean(h_pos[in_swath])
            z_vals = coords[in_swath, 2]

            traces.append({
                "hole_id": str(hole_id),
                "h_position": float(h_mean),
                "z_top": float(z_vals.max()),
                "z_bottom": float(z_vals.min()),
            })

    return traces


# ═══════════════════════════════════════════════════════════════════
# Domain color palette
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DOMAIN_COLORS = [
    (255, 215, 0),     # Gold
    (34, 139, 34),     # Forest green
    (65, 105, 225),    # Royal blue
    (255, 99, 71),     # Tomato
    (148, 103, 189),   # Purple
    (255, 165, 0),     # Orange
    (0, 191, 255),     # Deep sky blue
    (210, 105, 30),    # Chocolate
    (0, 128, 128),     # Teal
    (220, 20, 60),     # Crimson
    (128, 128, 128),   # Grey (unclassified)
]


def render_section_to_image(
    domain_codes: np.ndarray,
    width_px: int = 800,
    height_px: int = 400,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Render domain codes to an RGB image array.

    Parameters
    ----------
    domain_codes : (nz, nx) int array
    width_px, height_px : output image size
    colors : optional color list per domain

    Returns
    -------
    image : (height_px, width_px, 3) uint8 RGB
    """
    if colors is None:
        colors = DEFAULT_DOMAIN_COLORS

    nz, nx_sec = domain_codes.shape

    # Scale domain codes to image
    from scipy.ndimage import zoom
    scale_y = height_px / nz
    scale_x = width_px / nx_sec
    scaled = zoom(domain_codes.astype(np.float64), (scale_y, scale_x), order=0)
    scaled = scaled.astype(np.int32)

    image = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    for code in np.unique(scaled):
        c_idx = int(code) % len(colors)
        mask = scaled == code
        image[mask] = colors[c_idx]

    return image


# ═══════════════════════════════════════════════════════════════════
# Qt Widget
# ═══════════════════════════════════════════════════════════════════

if HAS_QT:
    class CrossSectionWidget(QWidget):
        """2D cross-section viewer for geological models.

        Shows coloured geological domains on a vertical slice,
        with drillhole traces and surface isovalue lines.
        """

        sectionChanged = pyqtSignal()

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._evaluate_fn: Optional[Callable] = None
            self._isovalues: List[float] = [0.0]
            self._drillholes_df = None
            self._section_image: Optional[QPixmap] = None
            self._traces: List[Dict] = []

            # Section parameters
            self._start_xy = (0.0, 0.0)
            self._end_xy = (1000.0, 0.0)
            self._z_min = -500.0
            self._z_max = 500.0
            self._width = 50.0
            self._nx = 200
            self._nz = 100

            self._setup_ui()

        def _setup_ui(self) -> None:
            lay = QVBoxLayout(self)

            # Controls
            grp = QGroupBox("Section Line")
            form = QFormLayout(grp)

            self.start_x_spin = QDoubleSpinBox()
            self.start_x_spin.setRange(-1e6, 1e6)
            form.addRow("Start X:", self.start_x_spin)

            self.start_y_spin = QDoubleSpinBox()
            self.start_y_spin.setRange(-1e6, 1e6)
            form.addRow("Start Y:", self.start_y_spin)

            self.end_x_spin = QDoubleSpinBox()
            self.end_x_spin.setRange(-1e6, 1e6)
            self.end_x_spin.setValue(1000)
            form.addRow("End X:", self.end_x_spin)

            self.end_y_spin = QDoubleSpinBox()
            self.end_y_spin.setRange(-1e6, 1e6)
            form.addRow("End Y:", self.end_y_spin)

            self.z_min_spin = QDoubleSpinBox()
            self.z_min_spin.setRange(-1e6, 1e6)
            self.z_min_spin.setValue(-500)
            form.addRow("Z Min:", self.z_min_spin)

            self.z_max_spin = QDoubleSpinBox()
            self.z_max_spin.setRange(-1e6, 1e6)
            self.z_max_spin.setValue(500)
            form.addRow("Z Max:", self.z_max_spin)

            self.width_spin = QDoubleSpinBox()
            self.width_spin.setRange(0, 10000)
            self.width_spin.setValue(50)
            form.addRow("Swath Width (m):", self.width_spin)

            lay.addWidget(grp)

            btn = QPushButton("Update Section")
            btn.clicked.connect(self.update_section)
            lay.addWidget(btn)

            # Image display
            self.image_label = QLabel("No section computed")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setMinimumHeight(300)
            lay.addWidget(self.image_label, stretch=1)

            self.info_label = QLabel("")
            lay.addWidget(self.info_label)

        def set_model_result(self, result: Dict[str, Any]) -> None:
            """Set the geological model result for section display."""
            self._evaluate_fn = result.get("evaluate_fn")
            self._isovalues = result.get("isovalues", [0.0])

            # Auto-set z range from grid
            origin = result.get("grid_origin")
            spacing = result.get("grid_spacing")
            dims = result.get("grid_dims")
            if origin is not None and spacing is not None and dims is not None:
                self._z_min = origin[2]
                self._z_max = origin[2] + spacing[2] * dims[2]
                self.z_min_spin.setValue(self._z_min)
                self.z_max_spin.setValue(self._z_max)

                # Auto-set section line from grid extent
                self.start_x_spin.setValue(origin[0])
                self.start_y_spin.setValue(origin[1] + spacing[1] * dims[1] / 2)
                self.end_x_spin.setValue(origin[0] + spacing[0] * dims[0])
                self.end_y_spin.setValue(origin[1] + spacing[1] * dims[1] / 2)

        def set_drillholes(self, df) -> None:
            self._drillholes_df = df

        def set_section_line(
            self,
            start_xy: Tuple[float, float],
            end_xy: Tuple[float, float],
            width: float = 50.0,
        ) -> None:
            self._start_xy = start_xy
            self._end_xy = end_xy
            self._width = width
            self.start_x_spin.setValue(start_xy[0])
            self.start_y_spin.setValue(start_xy[1])
            self.end_x_spin.setValue(end_xy[0])
            self.end_y_spin.setValue(end_xy[1])
            self.width_spin.setValue(width)

        def update_section(self) -> None:
            """Recompute and display the cross-section."""
            if self._evaluate_fn is None:
                self.info_label.setText("No model loaded")
                return

            self._start_xy = (self.start_x_spin.value(), self.start_y_spin.value())
            self._end_xy = (self.end_x_spin.value(), self.end_y_spin.value())
            self._z_min = self.z_min_spin.value()
            self._z_max = self.z_max_spin.value()
            self._width = self.width_spin.value()

            try:
                section_vals, h_coords, z_coords = sample_section(
                    self._start_xy, self._end_xy,
                    self._z_min, self._z_max,
                    self._evaluate_fn,
                    nx=self._nx, nz=self._nz,
                    width=self._width,
                )

                # Classify into domains
                codes = classify_section(section_vals, self._isovalues)

                # Render to image
                w = self.image_label.width() or 800
                h = max(300, self.image_label.height() or 400)
                rgb = render_section_to_image(codes, w, h)

                # Convert to QPixmap
                qimg = QImage(
                    rgb.data.tobytes(), w, h, 3 * w,
                    QImage.Format.Format_RGB888,
                )
                self._section_image = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(self._section_image)

                # Get drillhole traces
                self._traces = drillhole_traces_on_section(
                    self._drillholes_df,
                    self._start_xy, self._end_xy,
                    self._width,
                )

                length = np.sqrt(
                    (self._end_xy[0] - self._start_xy[0])**2
                    + (self._end_xy[1] - self._start_xy[1])**2,
                )
                n_domains = len(np.unique(codes))
                self.info_label.setText(
                    f"Section length: {length:.0f} m | "
                    f"Z: {self._z_min:.0f} to {self._z_max:.0f} m | "
                    f"{n_domains} domains | "
                    f"{len(self._traces)} drillhole traces"
                )

                self.sectionChanged.emit()

            except Exception as exc:
                logger.exception("Section update failed")
                self.info_label.setText(f"Error: {exc}")
