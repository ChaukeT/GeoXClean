"""
DPI-aware sizing utilities for GeoX.

Provides a `dp()` function that converts logical pixels to physical pixels
based on the current screen's device pixel ratio. Use this for any sizing
that must scale with display DPI.

Usage:
    from .dpi_utils import dp

    widget.setMinimumWidth(dp(280))
    widget.setIconSize(QSize(dp(16), dp(16)))

For static layouts defined in design_tokens.py, the QSS generator
handles DPI automatically via Qt's built-in scaling. `dp()` is needed
for programmatic sizing (e.g., setting icon sizes, custom painting).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

# Cache the DPI ratio to avoid per-call overhead
_cached_ratio: Optional[float] = None


def _get_dpi_ratio() -> float:
    """
    Get the device pixel ratio from the primary screen.

    Returns 1.0 if no QApplication exists yet (safe for module-level imports).
    """
    global _cached_ratio
    if _cached_ratio is not None:
        return _cached_ratio

    try:
        app = QGuiApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            if screen is not None:
                ratio = screen.devicePixelRatio()
                _cached_ratio = ratio
                return ratio
    except Exception as e:
        logger.debug(f"Could not get DPI ratio: {e}")

    return 1.0


def dp(value: int | float) -> int:
    """
    Convert logical pixels to DPI-scaled pixels.

    Args:
        value: Size in logical pixels (design-time value at 100% scaling)

    Returns:
        Scaled size in physical pixels, rounded to nearest integer.

    Examples:
        dp(16)  → 16 at 100%, 24 at 150%, 32 at 200%
        dp(280) → 280 at 100%, 420 at 150%, 560 at 200%
    """
    ratio = _get_dpi_ratio()
    return round(value * ratio)


def dp_f(value: float) -> float:
    """
    Convert logical pixels to DPI-scaled pixels (float version).

    Use when sub-pixel precision matters (e.g., QPainter coordinates).
    """
    return value * _get_dpi_ratio()


def reset_dpi_cache() -> None:
    """
    Reset the cached DPI ratio.

    Call this if the application moves to a screen with a different DPI.
    """
    global _cached_ratio
    _cached_ratio = None


def get_icon_size(size_name: str = "md") -> int:
    """
    Get a DPI-scaled icon size.

    Args:
        size_name: 'sm' (16), 'md' (20), 'lg' (24), 'xl' (48)

    Returns:
        Scaled icon size in pixels.
    """
    from .design_tokens import ICON_SIZE_SM, ICON_SIZE_MD, ICON_SIZE_LG, ICON_SIZE_XL

    sizes = {
        "sm": ICON_SIZE_SM,
        "md": ICON_SIZE_MD,
        "lg": ICON_SIZE_LG,
        "xl": ICON_SIZE_XL,
    }
    base = sizes.get(size_name, ICON_SIZE_MD)
    return dp(base)
