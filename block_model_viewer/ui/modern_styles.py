"""
Modern UI styling module — backward-compatibility SHIM.

Historical note: this file used to hold a 1,000+ line per-widget stylesheet
generator that conflicted with the application-level QSS produced by
`qss_generator.py`.  As of the March Pictures snapshot, the design tokens
+ generated QSS is the SINGLE source of truth for theming.

This shim keeps the public API (`ModernColors`, `get_theme_colors`,
`set_current_theme`, `AnimationHelper`, and all `get_*_stylesheet()`
functions) so existing panel code doesn't break — but every `get_*`
function now returns an empty string so it can never override the
application QSS.

Legacy attribute names (PANEL_BG, CARD_BG, ACCENT_PRIMARY, ...) are
mapped onto the new ColorPalette fields defined in `design_tokens.py`.
"""

from typing import Type

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import QWidget

from .design_tokens import tokens, DARK, LIGHT


# ============================================================================
# THEME STATE
# ============================================================================

_current_theme: str = "dark"


def set_current_theme(theme_name: str) -> None:
    """Set the current theme name (called by ThemeManager).

    Args:
        theme_name: Theme name ('light' or 'dark').  Any other value
            is treated as 'dark'.
    """
    global _current_theme
    _current_theme = theme_name if theme_name in ("light", "dark") else "dark"
    # Also update design_tokens theme so ModernColors and generated QSS stay in sync
    try:
        if hasattr(tokens, "set_theme"):
            tokens.set_theme(_current_theme)
    except Exception:
        pass


def get_current_theme() -> str:
    """Return the current theme name."""
    return _current_theme


# ============================================================================
# LEGACY COLOR CLASS
# ============================================================================
# Maps old attribute names (PANEL_BG, CARD_BG, ACCENT_PRIMARY, etc.) onto
# fields of the new ColorPalette so that existing panel code keeps working
# unchanged while the generated QSS is the single source of truth.


def _make_legacy_color_class(palette) -> type:
    """Build a simple class exposing legacy attribute names from a ColorPalette."""

    class _Legacy:
        # Background
        PANEL_BG = palette.BG_BASE
        CARD_BG = palette.BG_SURFACE
        CARD_HOVER = palette.BG_SURFACE_HOVER
        ELEVATED_BG = palette.BG_ELEVATED

        # Borders
        BORDER = palette.BORDER_DEFAULT
        BORDER_LIGHT = palette.BORDER_STRONG
        DIVIDER = palette.BORDER_SUBTLE

        # Text
        TEXT_PRIMARY = palette.TEXT_PRIMARY
        TEXT_SECONDARY = palette.TEXT_SECONDARY
        TEXT_DISABLED = palette.TEXT_DISABLED
        TEXT_HINT = palette.TEXT_TERTIARY

        # Accent
        ACCENT_PRIMARY = palette.ACCENT
        ACCENT_HOVER = palette.ACCENT_HOVER
        ACCENT_PRESSED = palette.ACCENT_PRESSED
        ACCENT_SECONDARY = palette.ACCENT_SECONDARY

        # Status
        SUCCESS = palette.STATUS_SUCCESS
        WARNING = palette.STATUS_WARNING
        ERROR = palette.STATUS_ERROR
        INFO = palette.STATUS_INFO

        # Special
        HIGHLIGHT = palette.HIGHLIGHT
        SHADOW = palette.SHADOW

    return _Legacy


DarkColors = _make_legacy_color_class(DARK)
LightColors = _make_legacy_color_class(LIGHT)


def get_theme_colors() -> Type:
    """Return the legacy-style color class for the current theme."""
    return LightColors if _current_theme == "light" else DarkColors


class _ModernColorsMeta(type):
    """Metaclass that proxies attribute access to the current theme's
    legacy color class — so ``ModernColors.CARD_BG`` always returns a
    color from the active palette."""

    def __getattr__(cls, name: str) -> str:
        colors = get_theme_colors()
        if hasattr(colors, name):
            return getattr(colors, name)
        raise AttributeError(
            f"'{cls.__name__}' has no attribute '{name}'"
        )


class ModernColors(metaclass=_ModernColorsMeta):
    """Dynamic proxy for legacy color attribute lookups.

    Use as ``ModernColors.CARD_BG`` — returns the current theme's value.
    Backed by the ColorPalette dataclasses in ``design_tokens.py``.
    """
    pass


# ============================================================================
# STYLESHEET GENERATORS — ALL NO-OPS
# ============================================================================
# These functions used to generate per-widget QSS.  The application-level
# theme QSS (from qss_generator.py + assets/themes/*.qss) is now the single
# source of truth, so these all return empty strings.  Existing callers
# see no behaviour change — Qt just applies no inline stylesheet.


def get_panel_stylesheet(*_a, **_kw) -> str:
    return ""


def get_button_stylesheet(*_a, **_kw) -> str:
    return ""


def get_card_stylesheet(*_a, **_kw) -> str:
    return ""


def get_group_box_stylesheet(*_a, **_kw) -> str:
    return ""


def get_collapsible_group_stylesheet(*_a, **_kw) -> str:
    return ""


def get_combo_box_stylesheet(*_a, **_kw) -> str:
    return ""


def get_slider_stylesheet(*_a, **_kw) -> str:
    return ""


def get_spin_box_stylesheet(*_a, **_kw) -> str:
    return ""


def get_checkbox_stylesheet(*_a, **_kw) -> str:
    return ""


def get_label_stylesheet(*_a, **_kw) -> str:
    return ""


def get_line_edit_stylesheet(*_a, **_kw) -> str:
    return ""


def get_progress_bar_stylesheet(*_a, **_kw) -> str:
    return ""


def get_analysis_panel_stylesheet(*_a, **_kw) -> str:
    return ""


def get_table_stylesheet(*_a, **_kw) -> str:
    return ""


def get_complete_panel_stylesheet(*_a, **_kw) -> str:
    return ""


def apply_modern_style(*_a, **_kw) -> None:
    """No-op. Modern styling is handled by application-level QSS (qss_generator)."""
    return None


# ============================================================================
# ANIMATION HELPER
# ============================================================================
# Used by a handful of panels for fade-in / fade-out / slide-in.  Kept as-is.


class AnimationHelper:
    """Helper class for creating smooth widget animations."""

    @staticmethod
    def fade_in(widget: QWidget, duration: int = 200):
        widget.setWindowOpacity(0)
        widget.show()
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        return animation

    @staticmethod
    def fade_out(widget: QWidget, duration: int = 200):
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)
        animation.finished.connect(widget.hide)
        animation.start()
        return animation

    @staticmethod
    def slide_in(widget: QWidget, direction: str = "down", duration: int = 300):
        start_rect = widget.geometry()
        if direction == "down":
            start_rect.moveTop(start_rect.top() - start_rect.height())
        elif direction == "up":
            start_rect.moveTop(start_rect.top() + start_rect.height())
        elif direction == "left":
            start_rect.moveLeft(start_rect.left() + start_rect.width())
        elif direction == "right":
            start_rect.moveLeft(start_rect.left() - start_rect.width())
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setStartValue(start_rect)
        animation.setEndValue(widget.geometry())
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        return animation
