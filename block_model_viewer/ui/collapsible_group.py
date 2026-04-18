"""
Collapsible group widget — flat Pictures style.

Appearance is controlled entirely by the app-level QSS via the object
names assigned below:

  - QFrame#CollapsibleGroupTitle
  - QPushButton#CollapsibleGroupToggle
  - QLabel#CollapsibleGroupTitleLabel
  - QWidget#CollapsibleGroupContent

No inline setStyleSheet() calls; no "click to toggle" hint label; no
card gradients or borders — themes are the single source of truth.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon


class CollapsibleGroup(QWidget):
    """
    Collapsible group with a simple arrow + title header.

    API:
        add_widget(widget), add_layout(layout), set_title(str),
        set_collapsed(bool), toggle_collapsed(), is_collapsed().
    Signal:
        collapsed_changed(bool)
    """

    collapsed_changed = pyqtSignal(bool)

    def __init__(self, title: str = "", icon_name: str = "", collapsed: bool = False, parent=None):
        super().__init__(parent)

        self.title = title
        self.icon_name = icon_name
        self._collapsed = collapsed

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 8)
        main_layout.setSpacing(0)

        # ── Title bar ────────────────────────────────────────────
        self.title_frame = QFrame()
        self.title_frame.setObjectName("CollapsibleGroupTitle")
        self.title_frame.setFixedHeight(28)
        self.title_frame.setCursor(Qt.CursorShape.PointingHandCursor)

        title_layout = QHBoxLayout(self.title_frame)
        title_layout.setContentsMargins(4, 0, 8, 0)
        title_layout.setSpacing(6)

        # Arrow toggle — ▼ open / ▶ collapsed. Plain text button so app
        # QSS controls its look.
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("CollapsibleGroupToggle")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setFlat(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        self._update_toggle_icon()
        title_layout.addWidget(self.toggle_button)

        # Optional theme icon
        if self.icon_name:
            try:
                icon = QIcon.fromTheme(self.icon_name)
                if not icon.isNull():
                    icon_label = QLabel()
                    icon_label.setPixmap(icon.pixmap(16, 16))
                    title_layout.addWidget(icon_label)
            except Exception:
                pass

        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("CollapsibleGroupTitleLabel")
        self.title_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.title_label.mousePressEvent = lambda e: self.toggle_collapsed()
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()

        # Header is clickable
        self.title_frame.mousePressEvent = lambda e: self.toggle_collapsed()

        main_layout.addWidget(self.title_frame)

        # ── Content area ─────────────────────────────────────────
        self.content_widget = QWidget()
        self.content_widget.setObjectName("CollapsibleGroupContent")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(8)
        main_layout.addWidget(self.content_widget)

        # Animation
        self.animation = QPropertyAnimation(self.content_widget, b"maximumHeight")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        if self._collapsed:
            self.content_widget.setMaximumHeight(0)
            self.content_widget.setVisible(False)
        else:
            self.content_widget.setMaximumHeight(16777215)
            self.content_widget.setVisible(True)

    # ── API ──────────────────────────────────────────────────────

    def _update_toggle_icon(self):
        self.toggle_button.setText("\u25b6" if self._collapsed else "\u25bc")

    def toggle_collapsed(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool):
        if self._collapsed == collapsed:
            return

        self._collapsed = collapsed
        self._update_toggle_icon()

        if hasattr(self, 'animation'):
            self.animation.stop()
            try:
                self.animation.finished.disconnect()
            except TypeError:
                pass

        if collapsed:
            start_height = self.content_widget.height()
            self.animation.setStartValue(start_height)
            self.animation.setEndValue(0)
            self.animation.finished.connect(self._on_collapse_finished)
            self.animation.start()
        else:
            self.content_widget.setVisible(True)
            self.content_widget.setMaximumHeight(16777215)
            content_height = self.content_widget.sizeHint().height()
            self.animation.setStartValue(0)
            self.animation.setEndValue(content_height if content_height > 0 else 200)
            self.animation.finished.connect(self._on_expand_finished)
            self.animation.start()

        self.collapsed_changed.emit(collapsed)

    def _on_collapse_finished(self):
        self.content_widget.setVisible(False)

    def _on_expand_finished(self):
        self.content_widget.setMaximumHeight(16777215)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_title(self, title: str):
        self.title = title
        self.title_label.setText(title)

    def add_widget(self, widget: QWidget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def refresh_theme(self):
        """App-level QSS handles theming — no-op retained for API compatibility."""
        return None
