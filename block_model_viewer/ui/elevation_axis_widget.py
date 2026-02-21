from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .modern_styles import ModernColors


class ElevationAxisWidget(QWidget):
    """Small left-side vertical elevation axis widget.

    Shows min / mid / max elevation labels stacked vertically. Designed to be
    lightweight and fully in screen-space so it never overlaps scene labels.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ElevationAxisWidget")
        self.setFixedWidth(80)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Labels: top = max, middle = mid, bottom = min
        self.max_label = QLabel("", self)
        self.mid_label = QLabel("", self)
        self.min_label = QLabel("", self)

        # Use theme-aware colors for dark theme compatibility (UX-004 fix)
        for lbl in (self.max_label, self.mid_label, self.min_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            f = QFont()
            f.setPointSize(10)
            lbl.setFont(f)
            lbl.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; background: transparent;")

        layout.addWidget(self.max_label)
        layout.addStretch(1)
        layout.addWidget(self.mid_label)
        layout.addStretch(1)
        layout.addWidget(self.min_label)

        self.hide()  # start hidden; shown when overlays/axes enabled

    def set_values(self, zmin: float, zmid: float, zmax: float, fmt: str = "{:.0f}"):
        try:
            self.max_label.setText(fmt.format(zmax))
            self.mid_label.setText(fmt.format(zmid))
            self.min_label.setText(fmt.format(zmin))
            self.update()
        except Exception:
            # tolerant formatting fallback
            try:
                self.max_label.setText(f"{zmax:.0f}")
                self.mid_label.setText(f"{zmid:.0f}")
                self.min_label.setText(f"{zmin:.0f}")
            except Exception:
                self.max_label.setText("")
                self.mid_label.setText("")
                self.min_label.setText("")

    def show_for_bounds(self, bounds: tuple | None):
        if not bounds:
            self.hide()
            return
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        zmid = 0.5 * (zmin + zmax)
        self.set_values(zmin, zmid, zmax)
        self.show()
