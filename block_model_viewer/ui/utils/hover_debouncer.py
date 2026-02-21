"""
Hover Debouncer for high-performance mouse interaction.

Prevents UI lag by only triggering expensive pick operations when the mouse
stops moving for a set interval (e.g., 50-100ms).
"""

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from typing import Optional, Tuple


class HoverDebouncer(QObject):
    """
    Prevents UI lag by only triggering a pick event when the mouse 
    stops moving for a set interval (e.g., 100ms).
    
    Usage:
        debouncer = HoverDebouncer(interval_ms=50, parent=self)
        debouncer.hover_stable.connect(self._on_hover_stable)
        # On mouse move:
        debouncer.mouse_moved((x, y))
    """
    
    hover_stable = pyqtSignal(tuple)  # Emits (x, y) screen coordinates
    
    def __init__(self, interval_ms: int = 100, parent: Optional[QObject] = None):
        """
        Initialize the hover debouncer.
        
        Args:
            interval_ms: Milliseconds to wait after mouse stops moving before emitting signal
            parent: Parent QObject for proper lifecycle management
        """
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._emit_stable)
        self._last_pos: Optional[Tuple[int, int]] = None
    
    def mouse_moved(self, pos: Tuple[int, int]) -> None:
        """
        Call this on every mouse move event.
        
        Args:
            pos: (x, y) screen coordinates
        """
        self._last_pos = pos
        # Reset the timer. If user keeps moving, timeout never fires.
        self._timer.start()
    
    def _emit_stable(self) -> None:
        """Internal: Emit signal when mouse has stopped moving."""
        if self._last_pos is not None:
            self.hover_stable.emit(self._last_pos)

