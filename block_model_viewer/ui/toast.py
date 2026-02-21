from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .modern_styles import get_theme_colors


class ToastWidget(QWidget):
    """A minimal, cross-platform toast widget.

    Usage:
        ToastWidget.show_message(parent_widget, "Message text", duration_ms)
    """

    def __init__(self, parent, message: str, duration: int = 3000):
        flags = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        super().__init__(parent=parent, flags=flags)
        # Transparent background and don't take focus
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.duration = int(duration)

        # Get theme colors
        colors = get_theme_colors()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.label = QLabel(message, self)
        self.label.setStyleSheet("color: white;")  # Always white on dark background for readability
        font = QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Semi-transparent elevated background (works in both themes)
        self.setStyleSheet(f"background-color: rgba(45, 45, 45, 220); border-radius: 6px;")
        layout.addWidget(self.label)

        self.adjustSize()

    def _position_over_parent(self):
        parent = self.parent() if self.parent() is not None else self
        try:
            pg = parent.geometry()
            x = pg.x() + (pg.width() - self.width()) // 2
            # place near top (15% down) so it doesn't overlap status bar or central controls
            y = pg.y() + int(pg.height() * 0.15)
            self.move(x, y)
        except Exception:
            # best-effort positioning; ignore failures
            pass

    def show_(self):
        self._position_over_parent()
        try:
            self.show()
        except Exception:
            # If showing fails for any reason, ignore and continue (status bar fallback exists)
            return
        QTimer.singleShot(self.duration, self.close)

    @classmethod
    def show_message(cls, parent, message: str, duration: int = 3000):
        try:
            t = cls(parent, message, duration)
            t.show_()
            return t
        except Exception:
            return None
