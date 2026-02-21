"""
Modern UI styling module for GeoX panels.

Provides consistent, professional styling for all UI panels with:
- Modern color palette (supports both dark and light themes)
- Card-based layouts
- Smooth transitions
- Professional typography
- Accessibility-focused contrast
"""

from typing import Dict, Optional, Type
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QRect, pyqtProperty
from PyQt6.QtGui import QColor


# ============================================================================
# THEME STATE
# ============================================================================

_current_theme: str = "dark"


def set_current_theme(theme_name: str) -> None:
    """
    Set the current theme. Called by ThemeManager when theme changes.

    Args:
        theme_name: Theme name ('light' or 'dark')
    """
    global _current_theme
    _current_theme = theme_name


def get_current_theme() -> str:
    """Get the current theme name."""
    return _current_theme


# ============================================================================
# COLOR PALETTES - Dark and Light Themes
# ============================================================================

class DarkColors:
    """Dark theme color palette."""

    # Background colors
    PANEL_BG = "#1e1e1e"              # Main panel background
    CARD_BG = "#252525"                # Card/section background
    CARD_HOVER = "#2a2a2a"             # Card hover state
    ELEVATED_BG = "#2d2d2d"            # Elevated elements

    # Borders and dividers
    BORDER = "#3d3d3d"                 # Default border
    BORDER_LIGHT = "#4d4d4d"           # Light border
    DIVIDER = "#333333"                # Section divider

    # Text colors
    TEXT_PRIMARY = "#e8e8e8"           # Primary text
    TEXT_SECONDARY = "#b0b0b0"         # Secondary text
    TEXT_DISABLED = "#6d6d6d"          # Disabled text
    TEXT_HINT = "#8a8a8a"              # Hint/placeholder text

    # Accent colors
    ACCENT_PRIMARY = "#0e7aca"         # Primary accent (blue)
    ACCENT_HOVER = "#1a8cd8"           # Accent hover
    ACCENT_PRESSED = "#0c6ab5"         # Accent pressed
    ACCENT_SECONDARY = "#26a69a"       # Secondary accent (teal)

    # Status colors
    SUCCESS = "#4caf50"                # Success/positive
    WARNING = "#ff9800"                # Warning
    ERROR = "#f44336"                  # Error/danger
    INFO = "#2196f3"                   # Information

    # Special colors
    HIGHLIGHT = "#ffa726"              # Highlight/selection
    SHADOW = "rgba(0, 0, 0, 0.3)"      # Shadow


class LightColors:
    """Light theme color palette."""

    # Background colors
    PANEL_BG = "#f5f5f5"              # Main panel background
    CARD_BG = "#ffffff"                # Card/section background
    CARD_HOVER = "#fafafa"             # Card hover state
    ELEVATED_BG = "#ffffff"            # Elevated elements

    # Borders and dividers
    BORDER = "#e0e0e0"                 # Default border
    BORDER_LIGHT = "#bdbdbd"           # Light border (darker in light theme)
    DIVIDER = "#eeeeee"                # Section divider

    # Text colors
    TEXT_PRIMARY = "#212121"           # Primary text
    TEXT_SECONDARY = "#757575"         # Secondary text
    TEXT_DISABLED = "#bdbdbd"          # Disabled text
    TEXT_HINT = "#9e9e9e"              # Hint/placeholder text

    # Accent colors
    ACCENT_PRIMARY = "#1976d2"         # Primary accent (blue)
    ACCENT_HOVER = "#1e88e5"           # Accent hover
    ACCENT_PRESSED = "#1565c0"         # Accent pressed
    ACCENT_SECONDARY = "#00897b"       # Secondary accent (teal)

    # Status colors
    SUCCESS = "#388e3c"                # Success/positive
    WARNING = "#f57c00"                # Warning
    ERROR = "#d32f2f"                  # Error/danger
    INFO = "#1976d2"                   # Information

    # Special colors
    HIGHLIGHT = "#ff9800"              # Highlight/selection
    SHADOW = "rgba(0, 0, 0, 0.1)"      # Shadow (lighter in light theme)


def get_theme_colors() -> Type[DarkColors] | Type[LightColors]:
    """
    Get the color palette for the current theme.

    Returns:
        DarkColors or LightColors class based on current theme
    """
    return LightColors if _current_theme == "light" else DarkColors


# Keep ModernColors as an alias that points to current theme colors
# This maintains backward compatibility with code using ModernColors.ATTR syntax
class _ModernColorsMeta(type):
    """
    Metaclass that enables class-level attribute access to current theme colors.

    This allows code like `ModernColors.CARD_BG` to work without needing
    an instance, and always returns colors from the current theme.
    """

    def __getattr__(cls, name: str) -> str:
        """Get color attribute from current theme when accessed as class attribute."""
        colors = get_theme_colors()
        if hasattr(colors, name):
            return getattr(colors, name)
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")


class ModernColors(metaclass=_ModernColorsMeta):
    """
    Modern color palette for GeoX UI.

    This class provides dynamic access to theme colors. Use it as:
        ModernColors.CARD_BG  # Returns the current theme's CARD_BG color

    The colors automatically update when the theme changes via set_current_theme().

    Available colors:
    - Background: PANEL_BG, CARD_BG, CARD_HOVER, ELEVATED_BG
    - Borders: BORDER, BORDER_LIGHT, DIVIDER
    - Text: TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DISABLED, TEXT_HINT
    - Accent: ACCENT_PRIMARY, ACCENT_HOVER, ACCENT_PRESSED, ACCENT_SECONDARY
    - Status: SUCCESS, WARNING, ERROR, INFO
    - Special: HIGHLIGHT, SHADOW
    """
    pass


# ============================================================================
# STYLESHEET TEMPLATES
# ============================================================================

def get_panel_stylesheet() -> str:
    """Get stylesheet for main panel container."""
    colors = get_theme_colors()
    return f"""
        QWidget {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            font-size: 12px;
        }}

        QWidget[objectName="PanelContent"] {{
            background-color: {colors.PANEL_BG};
        }}

        QScrollArea {{
            border: none;
            background-color: {colors.PANEL_BG};
        }}

        QScrollBar:vertical {{
            background: {colors.ELEVATED_BG};
            width: 20px;
            border: 1px solid {colors.BORDER};
            border-radius: 8px;
            margin: 4px 2px;
        }}

        QScrollBar::handle:vertical {{
            background: {colors.ACCENT_PRIMARY};
            border-radius: 6px;
            min-height: 50px;
            margin: 3px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {colors.ACCENT_PRIMARY};
        }}

        QScrollBar::handle:vertical:pressed {{
            background: {colors.ACCENT_HOVER};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}

        QScrollBar:horizontal {{
            background: {colors.ELEVATED_BG};
            height: 16px;
            border: 1px solid {colors.BORDER};
            border-radius: 8px;
            margin: 2px 4px;
        }}

        QScrollBar::handle:horizontal {{
            background: {colors.ACCENT_PRIMARY};
            border-radius: 6px;
            min-width: 50px;
            margin: 3px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background: {colors.ACCENT_PRIMARY};
        }}

        QScrollBar::handle:horizontal:pressed {{
            background: {colors.ACCENT_HOVER};
        }}

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}

        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
    """


def get_button_stylesheet(style: str = "primary") -> str:
    """
    Get stylesheet for buttons with different styles.

    Args:
        style: Button style - 'primary', 'secondary', 'icon', 'toggle'
    """
    colors = get_theme_colors()
    if style == "primary":
        return f"""
            QPushButton {{
                background-color: {colors.ACCENT_PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {colors.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {colors.ACCENT_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {colors.BORDER};
                color: {colors.TEXT_DISABLED};
            }}
        """
    elif style == "secondary":
        return f"""
            QPushButton {{
                background-color: {colors.CARD_BG};
                color: {colors.TEXT_PRIMARY};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {colors.CARD_HOVER};
                border-color: {colors.BORDER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {colors.BORDER};
            }}
            QPushButton:disabled {{
                background-color: {colors.PANEL_BG};
                color: {colors.TEXT_DISABLED};
                border-color: {colors.BORDER};
            }}
        """
    elif style == "icon":
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {colors.TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {colors.CARD_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {colors.BORDER};
            }}
            QPushButton:disabled {{
                color: {colors.TEXT_DISABLED};
            }}
        """
    elif style == "toggle":
        return f"""
            QPushButton {{
                background-color: {colors.CARD_BG};
                color: {colors.TEXT_SECONDARY};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: 500;
                font-size: 13px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {colors.CARD_HOVER};
                border-color: {colors.BORDER_LIGHT};
            }}
            QPushButton:checked {{
                background-color: {colors.ACCENT_PRIMARY};
                color: white;
                border-color: {colors.ACCENT_PRIMARY};
                font-weight: 600;
            }}
            QPushButton:checked:hover {{
                background-color: {colors.ACCENT_HOVER};
                border-color: {colors.ACCENT_HOVER};
            }}
            QPushButton:disabled {{
                background-color: {colors.PANEL_BG};
                color: {colors.TEXT_DISABLED};
                border-color: {colors.BORDER};
            }}
        """
    return ""


def get_card_stylesheet() -> str:
    """Get stylesheet for card containers."""
    colors = get_theme_colors()
    return f"""
        QFrame[objectName="Card"] {{
            background-color: {colors.CARD_BG};
            border: 1px solid {colors.BORDER};
            border-radius: 8px;
            padding: 12px;
        }}

        QFrame[objectName="Card"]:hover {{
            border-color: {colors.BORDER_LIGHT};
        }}
    """


def get_group_box_stylesheet() -> str:
    """Get stylesheet for group boxes."""
    colors = get_theme_colors()
    return f"""
        QGroupBox {{
            background-color: {colors.CARD_BG};
            border: 1px solid {colors.BORDER};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 20px;
            font-weight: 600;
            font-size: 13px;
            color: {colors.TEXT_PRIMARY};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 4px;
            padding: 4px 8px;
            background-color: {colors.CARD_BG};
            border-radius: 4px;
        }}
    """


def get_collapsible_group_stylesheet() -> str:
    """Get stylesheet for collapsible groups."""
    colors = get_theme_colors()
    return f"""
        QFrame[objectName="CollapsibleGroupTitle"] {{
            background-color: {colors.CARD_BG};
            border: 1px solid {colors.BORDER};
            border-radius: 6px;
        }}

        QFrame[objectName="CollapsibleGroupTitle"]:hover {{
            background-color: {colors.CARD_HOVER};
            border-color: {colors.BORDER_LIGHT};
        }}

        QLabel[objectName="CollapsibleGroupTitleLabel"] {{
            color: {colors.TEXT_PRIMARY};
            font-weight: 600;
            font-size: 13px;
        }}

        QWidget[objectName="CollapsibleGroupContent"] {{
            background-color: {colors.CARD_BG};
            border: 1px solid {colors.BORDER};
            border-top: none;
            border-bottom-left-radius: 6px;
            border-bottom-right-radius: 6px;
            padding: 12px;
        }}
    """


def get_combo_box_stylesheet() -> str:
    """Get stylesheet for combo boxes."""
    colors = get_theme_colors()
    return f"""
        QComboBox {{
            background-color: {colors.ELEVATED_BG};
            color: {colors.TEXT_PRIMARY};
            border: 1px solid {colors.BORDER};
            border-radius: 6px;
            padding: 6px 12px;
            min-height: 24px;
        }}

        QComboBox:hover {{
            border-color: {colors.BORDER_LIGHT};
            background-color: {colors.CARD_HOVER};
        }}

        QComboBox:focus {{
            border-color: {colors.ACCENT_PRIMARY};
        }}

        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {colors.TEXT_SECONDARY};
            margin-right: 8px;
        }}

        QComboBox:disabled {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_DISABLED};
            border-color: {colors.BORDER};
        }}

        QComboBox QAbstractItemView {{
            background-color: {colors.ELEVATED_BG};
            color: {colors.TEXT_PRIMARY};
            border: 1px solid {colors.BORDER};
            border-radius: 6px;
            padding: 4px;
            selection-background-color: {colors.ACCENT_PRIMARY};
            selection-color: white;
        }}

        QComboBox QAbstractItemView::item {{
            padding: 6px 12px;
            border-radius: 4px;
            min-height: 24px;
        }}

        QComboBox QAbstractItemView::item:hover {{
            background-color: {colors.CARD_HOVER};
        }}
    """


def get_slider_stylesheet() -> str:
    """Get stylesheet for sliders."""
    colors = get_theme_colors()
    return f"""
        QSlider::groove:horizontal {{
            background: {colors.BORDER};
            height: 6px;
            border-radius: 3px;
        }}

        QSlider::handle:horizontal {{
            background: {colors.ACCENT_PRIMARY};
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}

        QSlider::handle:horizontal:hover {{
            background: {colors.ACCENT_HOVER};
        }}

        QSlider::sub-page:horizontal {{
            background: {colors.ACCENT_PRIMARY};
            border-radius: 3px;
        }}
    """


def get_spin_box_stylesheet() -> str:
    """Get stylesheet for spin boxes."""
    colors = get_theme_colors()
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {colors.ELEVATED_BG};
            color: {colors.TEXT_PRIMARY};
            border: 1px solid {colors.BORDER};
            border-radius: 6px;
            padding: 6px 8px;
            min-height: 24px;
        }}

        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {colors.BORDER_LIGHT};
        }}

        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {colors.ACCENT_PRIMARY};
        }}

        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            background-color: transparent;
            border: none;
            width: 20px;
        }}

        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
            background-color: {colors.CARD_HOVER};
        }}

        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: transparent;
            border: none;
            width: 20px;
        }}

        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {colors.CARD_HOVER};
        }}

        QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_DISABLED};
            border-color: {colors.BORDER};
        }}
    """


def get_checkbox_stylesheet() -> str:
    """Get stylesheet for checkboxes."""
    colors = get_theme_colors()
    return f"""
        QCheckBox {{
            color: {colors.TEXT_PRIMARY};
            spacing: 8px;
            padding: 4px;
        }}

        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {colors.BORDER};
            border-radius: 4px;
            background-color: {colors.ELEVATED_BG};
        }}

        QCheckBox::indicator:hover {{
            border-color: {colors.BORDER_LIGHT};
            background-color: {colors.CARD_HOVER};
        }}

        QCheckBox::indicator:checked {{
            background-color: {colors.ACCENT_PRIMARY};
            border-color: {colors.ACCENT_PRIMARY};
            image: none;
        }}

        QCheckBox::indicator:checked:hover {{
            background-color: {colors.ACCENT_HOVER};
            border-color: {colors.ACCENT_HOVER};
        }}

        QCheckBox:disabled {{
            color: {colors.TEXT_DISABLED};
        }}

        QCheckBox::indicator:disabled {{
            background-color: {colors.PANEL_BG};
            border-color: {colors.BORDER};
        }}
    """


def get_label_stylesheet(style: str = "normal") -> str:
    """
    Get stylesheet for labels.

    Args:
        style: Label style - 'normal', 'heading', 'subheading', 'caption', 'hint'
    """
    colors = get_theme_colors()
    if style == "heading":
        return f"""
            QLabel {{
                color: {colors.TEXT_PRIMARY};
                font-size: 16px;
                font-weight: 700;
                padding: 4px 0;
            }}
        """
    elif style == "subheading":
        return f"""
            QLabel {{
                color: {colors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
                padding: 2px 0;
            }}
        """
    elif style == "caption":
        return f"""
            QLabel {{
                color: {colors.TEXT_SECONDARY};
                font-size: 11px;
                padding: 2px 0;
            }}
        """
    elif style == "hint":
        return f"""
            QLabel {{
                color: {colors.TEXT_HINT};
                font-size: 11px;
                font-style: italic;
                padding: 2px 0;
            }}
        """
    else:  # normal
        return f"""
            QLabel {{
                color: {colors.TEXT_PRIMARY};
                font-size: 12px;
            }}
        """


def get_line_edit_stylesheet() -> str:
    """Get stylesheet for line edits."""
    colors = get_theme_colors()
    return f"""
        QLineEdit {{
            background-color: {colors.ELEVATED_BG};
            color: {colors.TEXT_PRIMARY};
            border: 1px solid {colors.BORDER};
            border-radius: 6px;
            padding: 6px 10px;
            min-height: 24px;
        }}

        QLineEdit:hover {{
            border-color: {colors.BORDER_LIGHT};
        }}

        QLineEdit:focus {{
            border-color: {colors.ACCENT_PRIMARY};
        }}

        QLineEdit:disabled {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_DISABLED};
            border-color: {colors.BORDER};
        }}
    """


def get_progress_bar_stylesheet() -> str:
    """Get stylesheet for progress bars."""
    colors = get_theme_colors()
    return f"""
        QProgressBar {{
            background-color: {colors.BORDER};
            border: none;
            border-radius: 6px;
            height: 12px;
            text-align: center;
            color: {colors.TEXT_PRIMARY};
            font-size: 10px;
            font-weight: 600;
        }}

        QProgressBar::chunk {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {colors.ACCENT_PRIMARY},
                stop:1 {colors.ACCENT_HOVER}
            );
            border-radius: 6px;
        }}
    """


def get_analysis_panel_stylesheet() -> str:
    """
    Get comprehensive stylesheet for analysis panels.

    This replaces the DARK_STYLESHEET constants that were previously defined
    in individual panel files. It provides consistent styling across all
    analysis panels and automatically adapts to the current theme.
    """
    colors = get_theme_colors()
    return f"""
        QWidget {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_PRIMARY};
            font-family: "Segoe UI", "Roboto", sans-serif;
            font-size: 10pt;
        }}
        QGroupBox {{
            border: 1px solid {colors.BORDER};
            border-radius: 6px;
            margin-top: 22px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {colors.ACCENT_PRIMARY};
        }}
        QComboBox, QDoubleSpinBox, QSpinBox {{
            background-color: {colors.ELEVATED_BG};
            border: 1px solid {colors.BORDER};
            border-radius: 3px;
            padding: 4px;
            color: {colors.TEXT_PRIMARY};
        }}
        QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
            border: 1px solid {colors.BORDER_LIGHT};
        }}
        QComboBox QAbstractItemView {{
            background-color: {colors.ELEVATED_BG};
            color: {colors.TEXT_PRIMARY};
            selection-background-color: {colors.ACCENT_PRIMARY};
            selection-color: white;
        }}
        QTableWidget {{
            background-color: {colors.ELEVATED_BG};
            gridline-color: {colors.DIVIDER};
            border: none;
            selection-background-color: {colors.ACCENT_PRIMARY};
            selection-color: white;
        }}
        QHeaderView::section {{
            background-color: {colors.CARD_BG};
            color: {colors.TEXT_SECONDARY};
            padding: 4px;
            border: 1px solid {colors.BORDER};
            font-weight: bold;
        }}
        QPushButton {{
            background-color: {colors.CARD_BG};
            border: 1px solid {colors.BORDER_LIGHT};
            border-radius: 4px;
            padding: 6px 12px;
            color: {colors.TEXT_PRIMARY};
        }}
        QPushButton:hover {{
            background-color: {colors.CARD_HOVER};
        }}
        QPushButton:pressed {{
            background-color: {colors.ELEVATED_BG};
        }}
        QPushButton:disabled {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_DISABLED};
            border-color: {colors.BORDER};
        }}
        QPushButton#PrimaryButton {{
            background-color: {colors.ACCENT_PRIMARY};
            border: 1px solid {colors.ACCENT_PRIMARY};
            color: white;
            font-weight: bold;
        }}
        QPushButton#PrimaryButton:hover {{
            background-color: {colors.ACCENT_HOVER};
        }}
        QTabWidget::pane {{
            border: 1px solid {colors.BORDER};
        }}
        QTabBar::tab {{
            background-color: {colors.CARD_BG};
            color: {colors.TEXT_SECONDARY};
            padding: 8px 20px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_PRIMARY};
            border-top: 2px solid {colors.ACCENT_PRIMARY};
        }}
        QTextEdit, QPlainTextEdit {{
            background-color: {colors.ELEVATED_BG};
            border: 1px solid {colors.BORDER};
            font-family: Consolas, monospace;
            font-size: 10px;
            color: {colors.TEXT_PRIMARY};
        }}
        QCheckBox, QRadioButton {{
            spacing: 8px;
            color: {colors.TEXT_PRIMARY};
        }}
        QSlider::groove:horizontal {{
            border: 1px solid {colors.BORDER};
            height: 6px;
            background: {colors.ELEVATED_BG};
            margin: 2px 0;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {colors.BORDER_LIGHT};
            border: 1px solid {colors.BORDER_LIGHT};
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {colors.ACCENT_PRIMARY};
            border: 1px solid {colors.ACCENT_PRIMARY};
        }}
        QLineEdit {{
            background-color: {colors.ELEVATED_BG};
            border: 1px solid {colors.BORDER};
            border-radius: 3px;
            padding: 4px;
            color: {colors.TEXT_PRIMARY};
        }}
        QLineEdit:hover {{
            border: 1px solid {colors.BORDER_LIGHT};
        }}
        QProgressBar {{
            background-color: {colors.BORDER};
            border: none;
            border-radius: 4px;
            text-align: center;
            color: {colors.TEXT_PRIMARY};
        }}
        QProgressBar::chunk {{
            background-color: {colors.ACCENT_PRIMARY};
            border-radius: 4px;
        }}
        QScrollArea {{
            border: none;
            background-color: {colors.PANEL_BG};
        }}
        QScrollBar:vertical {{
            background: {colors.ELEVATED_BG};
            width: 12px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {colors.BORDER_LIGHT};
            border-radius: 6px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {colors.ACCENT_PRIMARY};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QListWidget {{
            background-color: {colors.ELEVATED_BG};
            border: 1px solid {colors.BORDER};
            color: {colors.TEXT_PRIMARY};
        }}
        QListWidget::item {{
            padding: 4px;
        }}
        QListWidget::item:selected {{
            background-color: {colors.ACCENT_PRIMARY};
            color: white;
        }}
        QTreeWidget {{
            background-color: {colors.ELEVATED_BG};
            border: 1px solid {colors.BORDER};
            color: {colors.TEXT_PRIMARY};
        }}
        QTreeWidget::item {{
            padding: 4px;
        }}
        QTreeWidget::item:selected {{
            background-color: {colors.ACCENT_PRIMARY};
            color: white;
        }}
        QLabel {{
            color: {colors.TEXT_PRIMARY};
        }}
        QToolTip {{
            background-color: {colors.ELEVATED_BG};
            color: {colors.TEXT_PRIMARY};
            border: 1px solid {colors.BORDER};
            padding: 4px;
        }}
    """


def get_table_stylesheet() -> str:
    """Get stylesheet specifically for table widgets."""
    colors = get_theme_colors()
    return f"""
        QTableWidget {{
            background-color: {colors.ELEVATED_BG};
            gridline-color: {colors.DIVIDER};
            border: 1px solid {colors.BORDER};
            selection-background-color: {colors.ACCENT_PRIMARY};
            selection-color: white;
            color: {colors.TEXT_PRIMARY};
        }}
        QHeaderView::section {{
            background-color: {colors.CARD_BG};
            color: {colors.TEXT_SECONDARY};
            padding: 6px;
            border: 1px solid {colors.BORDER};
            font-weight: bold;
        }}
        QTableWidget::item {{
            padding: 4px;
        }}
        QTableWidget::item:selected {{
            background-color: {colors.ACCENT_PRIMARY};
            color: white;
        }}
    """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_modern_style(widget: QWidget, style_type: str = "panel") -> None:
    """
    Apply modern styling to a widget.
    
    Args:
        widget: The widget to style
        style_type: Type of styling to apply
    """
    style_map = {
        "panel": get_panel_stylesheet,
        "button_primary": lambda: get_button_stylesheet("primary"),
        "button_secondary": lambda: get_button_stylesheet("secondary"),
        "button_icon": lambda: get_button_stylesheet("icon"),
        "button_toggle": lambda: get_button_stylesheet("toggle"),
        "card": get_card_stylesheet,
        "groupbox": get_group_box_stylesheet,
        "collapsible": get_collapsible_group_stylesheet,
        "combobox": get_combo_box_stylesheet,
        "slider": get_slider_stylesheet,
        "spinbox": get_spin_box_stylesheet,
        "checkbox": get_checkbox_stylesheet,
        "lineedit": get_line_edit_stylesheet,
        "progressbar": get_progress_bar_stylesheet,
    }
    
    if style_type in style_map:
        widget.setStyleSheet(style_map[style_type]())


def get_complete_panel_stylesheet() -> str:
    """Get complete stylesheet for modern panels with all components."""
    return "\n\n".join([
        get_panel_stylesheet(),
        get_button_stylesheet("primary"),
        get_button_stylesheet("secondary"),
        get_button_stylesheet("toggle"),
        get_card_stylesheet(),
        get_group_box_stylesheet(),
        get_collapsible_group_stylesheet(),
        get_combo_box_stylesheet(),
        get_slider_stylesheet(),
        get_spin_box_stylesheet(),
        get_checkbox_stylesheet(),
        get_line_edit_stylesheet(),
        get_progress_bar_stylesheet(),
    ])


# ============================================================================
# ANIMATION UTILITIES
# ============================================================================

class AnimationHelper:
    """Helper class for creating smooth animations."""
    
    @staticmethod
    def fade_in(widget: QWidget, duration: int = 200):
        """Fade in animation for widget."""
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
        """Fade out animation for widget."""
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
        """
        Slide in animation for widget.
        
        Args:
            widget: Widget to animate
            direction: Direction - 'up', 'down', 'left', 'right'
            duration: Animation duration in ms
        """
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

