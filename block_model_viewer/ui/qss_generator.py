"""
QSS Generator — builds complete theme stylesheets from design tokens.

Run directly to regenerate QSS files:
    python -m block_model_viewer.ui.qss_generator

Or call programmatically:
    from .qss_generator import generate_qss
    dark_qss = generate_qss("dark")
    light_qss = generate_qss("light")
"""

from __future__ import annotations

from .design_tokens import tokens, ColorPalette, DARK, LIGHT
from .keyboard_nav import FOCUS_QSS as _FOCUS_QSS
from .panel_toolkit import PANEL_QSS as _PANEL_QSS
from .panel_micro_widgets import MICRO_WIDGET_QSS as _MICRO_WIDGET_QSS
from .design_tokens import (
    FONT_FAMILY, FONT_FAMILY_MONO,
    FONT_SIZE_XS, FONT_SIZE_SM, FONT_SIZE_BASE, FONT_SIZE_MD, FONT_SIZE_LG,
    FONT_WEIGHT_MEDIUM, FONT_WEIGHT_SEMIBOLD, FONT_WEIGHT_BOLD,
    SPACING_XS, SPACING_SM, SPACING_MD, SPACING_LG,
    HEIGHT_INPUT, HEIGHT_BUTTON, HEIGHT_BUTTON_SM,
    BORDER_WIDTH, BORDER_RADIUS_SM, BORDER_RADIUS_MD, BORDER_RADIUS_LG,
    SCROLLBAR_WIDTH, SCROLLBAR_MIN_HANDLE, SCROLLBAR_RADIUS,
    HEIGHT_COLLAPSIBLE_TITLE,
)


def generate_qss(theme: str = "dark") -> str:
    """
    Generate a complete QSS stylesheet for the given theme.

    This single function produces ALL styling. Panels should NOT
    apply additional stylesheets — the application-level QSS handles everything.

    Args:
        theme: 'dark' or 'light'

    Returns:
        Complete QSS stylesheet string
    """
    c = LIGHT if theme == "light" else DARK
    sections = [
        _header(theme),
        _global(c),
        _main_window(c),
        _menubar(c),
        _toolbar(c),
        _dock_widget(c),
        _tabs(c),
        _statusbar(c),
        _buttons(c),
        _inputs(c),
        _combo_box(c),
        _spin_box(c),
        _checkbox_radio(c),
        _slider(c),
        _progress_bar(c),
        _scrollbar(c),
        _group_box(c),
        _lists_and_trees(c),
        _tables(c),
        _text_edit(c),
        _tooltip(c),
        _dialog(c),
        _collapsible_group(c),
        _cards(c),
        _custom_properties(c),
        _panel_header(c),
        _block_model_panels(c),
        _loopstructural_panel(c),
        _sgsim_widgets(c),
        _PANEL_QSS,          # Panel toolkit — consistent dock panel styling
        _MICRO_WIDGET_QSS,   # Micro-widgets — ConfirmButton armed state, etc.
        _FOCUS_QSS,          # Unified focus ring (keyboard_nav.py)
    ]
    return "\n\n".join(sections)


# =============================================================================
# QSS SECTION BUILDERS
# =============================================================================

def _header(theme: str) -> str:
    return f"/* GeoX {theme.title()} Theme — Auto-generated from design_tokens.py */\n/* Do not edit manually. Regenerate with qss_generator.py */\n"


def _global(c: ColorPalette) -> str:
    return f"""/* === Global === */
QWidget {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
    outline: none;
    selection-background-color: {c.ACCENT};
    selection-color: {c.TEXT_ON_ACCENT};
}}"""


def _main_window(c: ColorPalette) -> str:
    return f"""/* === Main Window === */
QMainWindow {{
    background-color: {c.BG_BASE};
}}
QMainWindow::separator {{
    background-color: {c.BORDER_SUBTLE};
    width: 3px;
    height: 3px;
}}
QMainWindow::separator:hover {{
    background-color: {c.ACCENT};
}}"""


def _menubar(c: ColorPalette) -> str:
    return f"""/* === Menu Bar === */
QMenuBar {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
    border-bottom: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    padding: 2px {SPACING_SM}px;
    font-size: {FONT_SIZE_BASE}px;
    spacing: 2px;
}}
QMenuBar::item {{
    background-color: transparent;
    padding: 6px {SPACING_MD}px;
    border-radius: {BORDER_RADIUS_SM}px;
    margin: 1px 0;
}}
QMenuBar::item:selected {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
}}
QMenuBar::item:pressed {{
    background-color: {c.ACCENT};
    color: {c.TEXT_ON_ACCENT};
}}
/* --- Dropdown Menus --- */
QMenu {{
    background-color: {c.BG_OVERLAY};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_LG}px;
    padding: {SPACING_XS}px;
}}
QMenu::item {{
    padding: 8px {SPACING_LG}px 8px {SPACING_MD}px;
    border-radius: {BORDER_RADIUS_SM}px;
    margin: 1px {SPACING_XS}px;
}}
QMenu::item:selected {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
}}
QMenu::item:disabled {{
    color: {c.TEXT_DISABLED};
}}
QMenu::separator {{
    height: {BORDER_WIDTH}px;
    background-color: {c.BORDER_SUBTLE};
    margin: {SPACING_XS}px {SPACING_MD}px;
}}
QMenu::icon {{
    padding-left: {SPACING_SM}px;
}}"""


def _toolbar(c: ColorPalette) -> str:
    return f"""/* === Toolbar === */
QToolBar {{
    background-color: {c.BG_ELEVATED};
    border: none;
    border-bottom: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    spacing: {SPACING_XS}px;
    padding: {SPACING_XS}px;
}}
QToolBar::separator {{
    background-color: {c.BORDER_DEFAULT};
    width: {BORDER_WIDTH}px;
    margin: {SPACING_XS}px 2px;
}}
QToolButton {{
    background-color: transparent;
    border: {BORDER_WIDTH}px solid transparent;
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px;
    color: {c.TEXT_PRIMARY};
}}
QToolButton:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.BORDER_DEFAULT};
}}
QToolButton:pressed {{
    background-color: {c.BG_ELEVATED};
}}
QToolButton:checked {{
    background-color: {c.ACCENT_SUBTLE};
    border-color: {c.ACCENT};
    color: {c.ACCENT};
}}"""


def _dock_widget(c: ColorPalette) -> str:
    return f"""/* === Dock Widgets === */
QDockWidget {{
    color: {c.TEXT_PRIMARY};
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    border: none;
}}
QDockWidget::title {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_SECONDARY};
    padding: 10px {SPACING_MD}px;
    border-bottom: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    font-size: {FONT_SIZE_MD}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background-color: transparent;
    border: none;
    padding: 4px;
    border-radius: {BORDER_RADIUS_SM}px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {c.BG_ELEVATED};
}}"""


def _tabs(c: ColorPalette) -> str:
    return f"""/* === Tabs === */
QTabWidget::pane {{
    border: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    background-color: {c.BG_SURFACE};
    top: -1px;
    border-radius: 0 0 {BORDER_RADIUS_MD}px {BORDER_RADIUS_MD}px;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {c.TEXT_TERTIARY};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px {SPACING_LG}px;
    margin-right: 0;
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_MEDIUM};
}}
QTabBar::tab:selected {{
    background-color: transparent;
    color: {c.ACCENT};
    border-bottom: 2px solid {c.ACCENT};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
}}
QTabBar::tab:hover:!selected {{
    color: {c.TEXT_PRIMARY};
    border-bottom: 2px solid {c.BORDER_STRONG};
}}"""


def _statusbar(c: ColorPalette) -> str:
    return f"""/* === Status Bar === */
QStatusBar {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_SECONDARY};
    border-top: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    font-size: {FONT_SIZE_SM}px;
}}
QStatusBar::item {{
    border: none;
}}"""


def _buttons(c: ColorPalette) -> str:
    return f"""/* === Buttons === */
QPushButton {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: 6px {SPACING_LG}px;
    min-height: {HEIGHT_BUTTON_SM}px;
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_MEDIUM};
}}
QPushButton:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.ACCENT};
    color: {c.ACCENT};
}}
QPushButton:pressed {{
    background-color: {c.ACCENT_SUBTLE};
    border-color: {c.ACCENT};
}}
QPushButton:disabled {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_DISABLED};
    border-color: {c.BORDER_SUBTLE};
}}
/* Primary action button */
QPushButton[objectName="PrimaryButton"],
QPushButton[primary="true"] {{
    background-color: {c.ACCENT};
    color: {c.TEXT_ON_ACCENT};
    border: none;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    padding: 8px {SPACING_LG}px;
}}
QPushButton[objectName="PrimaryButton"]:hover,
QPushButton[primary="true"]:hover {{
    background-color: {c.ACCENT_HOVER};
}}
QPushButton[objectName="PrimaryButton"]:pressed,
QPushButton[primary="true"]:pressed {{
    background-color: {c.ACCENT_PRESSED};
}}
QPushButton[objectName="PrimaryButton"]:disabled,
QPushButton[primary="true"]:disabled {{
    background-color: {c.BORDER_DEFAULT};
    color: {c.TEXT_DISABLED};
}}
/* Flat / icon-only button */
QPushButton[flat="true"] {{
    background-color: transparent;
    border: none;
}}
QPushButton[flat="true"]:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-radius: {BORDER_RADIUS_MD}px;
}}
/* Panel section title label */
QLabel#PanelSectionTitle {{
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
    color: {c.TEXT_PRIMARY};
    padding-bottom: {SPACING_XS}px;
}}"""


def _inputs(c: ColorPalette) -> str:
    return f"""/* === Line Edit === */
QLineEdit {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: {SPACING_SM}px {SPACING_MD}px;
    min-height: {HEIGHT_INPUT - 8}px;
    selection-background-color: {c.ACCENT};
    selection-color: {c.TEXT_ON_ACCENT};
    font-size: {FONT_SIZE_BASE}px;
}}
QLineEdit:hover {{
    border-color: {c.ACCENT};
}}
QLineEdit:focus {{
    border-color: {c.ACCENT};
    background-color: {c.BG_SURFACE};
}}
QLineEdit:disabled {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_DISABLED};
    border-color: {c.BORDER_SUBTLE};
}}
QLineEdit[readOnly="true"] {{
    background-color: {c.BG_BASE};
    border-color: {c.BORDER_SUBTLE};
}}"""


def _combo_box(c: ColorPalette) -> str:
    return f"""/* === Combo Box === */
QComboBox {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: {SPACING_SM}px {SPACING_MD}px;
    min-height: {HEIGHT_INPUT - 8}px;
    font-size: {FONT_SIZE_BASE}px;
}}
QComboBox:hover {{
    border-color: {c.ACCENT};
}}
QComboBox:focus {{
    border-color: {c.ACCENT};
    background-color: {c.BG_SURFACE};
}}
QComboBox:disabled {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_DISABLED};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
    padding-right: {SPACING_SM}px;
}}
QComboBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {c.TEXT_SECONDARY};
}}
QComboBox::down-arrow:hover {{
    border-top-color: {c.ACCENT};
}}
QComboBox::down-arrow:disabled {{
    border-top-color: {c.TEXT_DISABLED};
}}
QComboBox QAbstractItemView {{
    background-color: {c.BG_OVERLAY};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: {SPACING_XS}px;
    selection-background-color: {c.ACCENT_SUBTLE};
    selection-color: {c.ACCENT};
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    padding: 8px {SPACING_MD}px;
    min-height: {HEIGHT_INPUT - 8}px;
    border-radius: {BORDER_RADIUS_SM}px;
    margin: 1px {SPACING_XS}px;
}}
QComboBox QAbstractItemView::item:hover {{
    background-color: {c.ACCENT_SUBTLE};
}}"""


def _spin_box(c: ColorPalette) -> str:
    return f"""/* === Spin Boxes === */
QSpinBox, QDoubleSpinBox {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px {SPACING_SM}px;
    min-height: {HEIGHT_INPUT - 8}px;
    font-size: {FONT_SIZE_BASE}px;
}}
QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {c.BORDER_STRONG};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {c.ACCENT};
}}
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {c.BG_BASE};
    color: {c.TEXT_DISABLED};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: transparent;
    border: none;
    width: 20px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {c.BG_SURFACE_HOVER};
}}"""


def _checkbox_radio(c: ColorPalette) -> str:
    return f"""/* === Checkbox & Radio === */
QCheckBox, QRadioButton {{
    color: {c.TEXT_PRIMARY};
    spacing: 10px;
    font-size: {FONT_SIZE_BASE}px;
    min-height: 26px;
    padding: 2px 0;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {c.BORDER_STRONG};
    border-radius: {BORDER_RADIUS_SM}px;
    background-color: {c.BG_ELEVATED};
}}
QCheckBox::indicator:hover {{
    border-color: {c.ACCENT};
    background-color: {c.ACCENT_SUBTLE};
}}
QCheckBox::indicator:checked {{
    background-color: {c.ACCENT};
    border-color: {c.ACCENT};
}}
QCheckBox::indicator:checked:hover {{
    background-color: {c.ACCENT_HOVER};
    border-color: {c.ACCENT_HOVER};
}}
QCheckBox::indicator:disabled {{
    background-color: {c.BG_BASE};
    border-color: {c.BORDER_SUBTLE};
}}
QRadioButton::indicator {{
    width: 20px;
    height: 20px;
    border: 2px solid {c.BORDER_STRONG};
    border-radius: 10px;
    background-color: {c.BG_ELEVATED};
}}
QRadioButton::indicator:hover {{
    border-color: {c.ACCENT};
    background-color: {c.ACCENT_SUBTLE};
}}
QRadioButton::indicator:checked {{
    background-color: {c.ACCENT};
    border-color: {c.ACCENT};
}}
QRadioButton::indicator:disabled {{
    background-color: {c.BG_BASE};
    border-color: {c.BORDER_SUBTLE};
}}"""


def _slider(c: ColorPalette) -> str:
    return f"""/* === Slider === */
QSlider::groove:horizontal {{
    background: {c.BORDER_DEFAULT};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {c.ACCENT};
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
    border: 2px solid {c.BG_BASE};
}}
QSlider::handle:horizontal:hover {{
    background: {c.ACCENT_HOVER};
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
}}
QSlider::handle:horizontal:pressed {{
    background: {c.ACCENT_PRESSED};
}}
QSlider::sub-page:horizontal {{
    background: {c.ACCENT};
    border-radius: 3px;
}}
QSlider::groove:vertical {{
    background: {c.BORDER_DEFAULT};
    width: 6px;
    border-radius: 3px;
}}
QSlider::handle:vertical {{
    background: {c.ACCENT};
    width: 18px;
    height: 18px;
    margin: 0 -6px;
    border-radius: 9px;
    border: 2px solid {c.BG_BASE};
}}"""


def _progress_bar(c: ColorPalette) -> str:
    return f"""/* === Progress Bar === */
QProgressBar {{
    background-color: {c.BG_ELEVATED};
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: {c.TEXT_PRIMARY};
    font-size: {FONT_SIZE_XS}px;
}}
QProgressBar::chunk {{
    background-color: {c.ACCENT};
    border-radius: 5px;
}}"""


def _scrollbar(c: ColorPalette) -> str:
    return f"""/* === Scrollbar — Sleek thin bar === */
QScrollBar:vertical {{
    background-color: transparent;
    width: {SCROLLBAR_WIDTH}px;
    border: none;
    margin: 4px 1px;
}}
QScrollBar::handle:vertical {{
    background-color: {c.BORDER_DEFAULT};
    border-radius: {SCROLLBAR_RADIUS}px;
    min-height: {SCROLLBAR_MIN_HANDLE}px;
    margin: 0 1px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {c.ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    background-color: transparent;
    height: {SCROLLBAR_WIDTH}px;
    border: none;
    margin: 1px 4px;
}}
QScrollBar::handle:horizontal {{
    background-color: {c.BORDER_DEFAULT};
    border-radius: {SCROLLBAR_RADIUS}px;
    min-width: {SCROLLBAR_MIN_HANDLE}px;
    margin: 1px 0;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {c.ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}"""


def _group_box(c: ColorPalette) -> str:
    return f"""/* === Group Box === */
QGroupBox {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    border-radius: {BORDER_RADIUS_LG}px;
    margin-top: {SPACING_LG}px;
    padding: {SPACING_LG + 4}px {SPACING_MD}px {SPACING_MD}px {SPACING_MD}px;
    font-size: {FONT_SIZE_MD}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    color: {c.TEXT_PRIMARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {SPACING_MD}px;
    top: {SPACING_XS}px;
    padding: 2px {SPACING_SM}px;
    background-color: {c.BG_SURFACE};
    color: {c.ACCENT};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    font-size: {FONT_SIZE_SM}px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}"""


def _lists_and_trees(c: ColorPalette) -> str:
    return f"""/* === Lists and Trees === */
QListWidget, QListView {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
    alternate-background-color: {c.BG_ELEVATED};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    outline: none;
}}
QListWidget::item, QListView::item {{
    padding: {SPACING_XS}px {SPACING_SM}px;
    border-radius: {BORDER_RADIUS_SM}px;
}}
QListWidget::item:alternate, QListView::item:alternate {{
    background-color: {c.BG_ELEVATED};
}}
QListWidget::item:selected, QListView::item:selected {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
}}
QListWidget::item:hover, QListView::item:hover {{
    background-color: {c.BG_SURFACE_HOVER};
}}
QTreeWidget, QTreeView {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
    alternate-background-color: {c.BG_ELEVATED};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    outline: none;
}}
QTreeWidget::item, QTreeView::item {{
    padding: {SPACING_XS}px;
    border-radius: {BORDER_RADIUS_SM}px;
}}
QTreeWidget::item:alternate, QTreeView::item:alternate {{
    background-color: {c.BG_ELEVATED};
}}
QTreeWidget::item:selected, QTreeView::item:selected {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
}}
QTreeWidget::item:hover, QTreeView::item:hover {{
    background-color: {c.BG_SURFACE_HOVER};
}}"""


def _tables(c: ColorPalette) -> str:
    return f"""/* === Tables === */
QTableWidget, QTableView {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
    alternate-background-color: {c.BG_ELEVATED};
    border: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    border-radius: {BORDER_RADIUS_MD}px;
    gridline-color: {c.BORDER_SUBTLE};
    selection-background-color: {c.ACCENT_SUBTLE};
    selection-color: {c.ACCENT};
    outline: none;
}}
QTableWidget::item, QTableView::item {{
    padding: 6px {SPACING_MD}px;
}}
QTableWidget::item:alternate, QTableView::item:alternate {{
    background-color: {c.BG_ELEVATED};
}}
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
}}
QHeaderView::section {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_SECONDARY};
    padding: 8px {SPACING_MD}px;
    border: none;
    border-right: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    border-bottom: 2px solid {c.BORDER_DEFAULT};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    font-size: {FONT_SIZE_SM}px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QHeaderView::section:hover {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
}}"""


def _text_edit(c: ColorPalette) -> str:
    return f"""/* === Text Edit === */
QTextEdit, QPlainTextEdit {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_SM}px;
    selection-background-color: {c.ACCENT};
    selection-color: {c.TEXT_ON_ACCENT};
    font-family: {FONT_FAMILY_MONO};
    font-size: {FONT_SIZE_BASE}px;
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {c.ACCENT};
}}"""


def _tooltip(c: ColorPalette) -> str:
    return f"""/* === Tooltip === */
QToolTip {{
    background-color: {c.BG_OVERLAY};
    color: {c.TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: {SPACING_SM}px {SPACING_MD}px;
    font-size: {FONT_SIZE_SM}px;
}}"""


def _dialog(c: ColorPalette) -> str:
    return f"""/* === Dialogs === */
QDialog {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
}}
QMessageBox {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_PRIMARY};
}}
QMessageBox QLabel {{
    color: {c.TEXT_PRIMARY};
    background-color: transparent;
    font-size: {FONT_SIZE_BASE}px;
}}
QMessageBox QPushButton {{
    min-width: 70px;
    padding: {SPACING_SM}px {SPACING_LG}px;
}}"""


def _collapsible_group(c: ColorPalette) -> str:
    # Use 42px title bar height — more spacious than the 34px token value.
    # Left border accent strip gives a professional Leapfrog / Micromine feel.
    TITLE_H = 42
    return f"""/* === Collapsible Group — Professional Mining Software Style === */
QFrame[objectName="CollapsibleTitle"] {{
    background-color: {c.BG_ELEVATED};
    border: none;
    border-bottom: 1px solid {c.BORDER_SUBTLE};
    border-left: 3px solid {c.ACCENT};
    border-radius: 0px;
    min-height: {TITLE_H}px;
    max-height: {TITLE_H}px;
}}
QFrame[objectName="CollapsibleTitle"]:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-left-color: {c.ACCENT_HOVER};
}}
QLabel[objectName="CollapsibleTitleLabel"] {{
    color: {c.TEXT_PRIMARY};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    font-size: {FONT_SIZE_MD}px;
    background: transparent;
    border: none;
    letter-spacing: 0.2px;
}}
QPushButton[objectName="CollapsibleToggle"] {{
    background-color: transparent;
    border: none;
    color: {c.TEXT_SECONDARY};
    font-size: 11px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 0px;
}}
QPushButton[objectName="CollapsibleToggle"]:hover {{
    color: {c.TEXT_PRIMARY};
    background-color: transparent;
}}
QWidget[objectName="CollapsibleContent"] {{
    background-color: {c.BG_SURFACE};
    border: none;
    border-bottom: 1px solid {c.BORDER_SUBTLE};
    border-left: 3px solid transparent;
    border-radius: 0px;
}}"""


def _cards(c: ColorPalette) -> str:
    return f"""/* === Cards === */
QFrame[objectName="Card"] {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    border-radius: {BORDER_RADIUS_LG}px;
    padding: {SPACING_MD}px;
}}
QFrame[objectName="Card"]:hover {{
    border-color: {c.ACCENT};
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
/* === Status Bar — refined bottom strip === */
QStatusBar {{
    background-color: {c.BG_SURFACE};
    color: {c.TEXT_TERTIARY};
    border-top: {BORDER_WIDTH}px solid {c.BORDER_SUBTLE};
    font-size: {FONT_SIZE_SM}px;
    padding: 2px {SPACING_SM}px;
}}
QStatusBar::item {{
    border: none;
}}"""


def _custom_properties(c: ColorPalette) -> str:
    """Custom Qt property-based styling for domain-specific widgets."""
    return f"""/* === Custom Properties === */
/* Mapping rows (column mapping dialog) */
*[mappingRow="true"] {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
}}
*[mappingRow="true"]:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.BORDER_STRONG};
}}
/* Badges */
QLabel[badge="required"] {{
    background-color: {c.STATUS_ERROR_SUBTLE};
    color: {c.STATUS_ERROR};
    border-radius: 3px;
}}
QLabel[badge="optional"] {{
    background-color: {c.BG_ELEVATED};
    color: {c.TEXT_SECONDARY};
    border-radius: 3px;
}}
/* Hints and labels */
QLabel[hint="true"], QLabel[subtitle="true"] {{
    color: {c.TEXT_SECONDARY};
}}
QLabel[statLabel="true"] {{
    color: {c.TEXT_TERTIARY};
}}
QLabel[nullLabel="true"], QLabel[nullValue="true"] {{
    color: {c.STATUS_WARNING};
}}
/* Status icons */
QLabel[statusIcon="unmapped"] {{
    color: {c.TEXT_TERTIARY};
}}
QLabel[statusIcon="required-unmapped"] {{
    color: {c.STATUS_ERROR};
}}
QLabel[statusIcon="optional-unmapped"] {{
    color: {c.TEXT_TERTIARY};
}}
QLabel[statusIcon="mapped"] {{
    color: {c.STATUS_SUCCESS};
}}
/* Status messages */
QLabel[statusMessage="true"] {{
    color: {c.TEXT_SECONDARY};
}}
QLabel[statusMessage="success"] {{
    color: {c.STATUS_SUCCESS};
}}
QLabel[statusMessage="error"] {{
    color: {c.STATUS_ERROR};
}}
/* Selection badges */
QLabel[selectionBadge="all"] {{
    background-color: {c.STATUS_SUCCESS_SUBTLE};
    color: {c.STATUS_SUCCESS};
    border-radius: 12px;
}}
QLabel[selectionBadge="partial"] {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.ACCENT};
    border-radius: 12px;
}}
QLabel[selectionBadge="none"] {{
    background-color: {c.STATUS_ERROR_SUBTLE};
    color: {c.STATUS_ERROR};
    border-radius: 12px;
}}
/* Grade cards */
*[gradeCard="true"] {{
    background-color: {c.BG_SURFACE};
    border: 2px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_MD}px;
}}
*[gradeCard="true"]:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.BORDER_STRONG};
}}
*[gradeCard="true"][selected="true"] {{
    background-color: {c.ACCENT_SUBTLE};
    border-color: {c.ACCENT};
}}
/* Selection indicator */
QFrame[selectionIndicator="true"] {{
    background-color: {c.ACCENT};
    border-radius: 8px;
}}
*[selected="false"] QFrame[selectionIndicator="true"] {{
    background-color: {c.BORDER_DEFAULT};
}}
/* Dialog header */
QFrame[dialogHeader="true"] {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_LG}px;
}}
/* Warning frame */
QFrame[warning="true"] {{
    background-color: {c.STATUS_WARNING_SUBTLE};
    border: {BORDER_WIDTH}px solid {c.STATUS_WARNING};
    border-radius: {BORDER_RADIUS_MD}px;
}}
/* Validation error state */
*[validationError="true"] {{
    border-color: {c.STATUS_ERROR};
}}
QLabel[objectName="ValidationError"] {{
    color: {c.STATUS_ERROR};
    font-size: {FONT_SIZE_SM}px;
    background: transparent;
    border: none;
}}"""


def _panel_header(c) -> str:
    """QSS for PanelHeaderBar (from panel_header.py)."""
    return f"""/* ===== Panel Header Bar ===== */
#PanelHeaderBar {{
    background: {c.BG_SURFACE};
    border-bottom: 1px solid {c.BORDER_SUBTLE};
    padding: 2px;
}}

#PanelHeaderTitle {{
    font-weight: 600;
    font-size: {FONT_SIZE_MD}px;
    color: {c.TEXT_PRIMARY};
}}

#PanelHeaderStage {{
    font-size: {FONT_SIZE_SM}px;
    font-style: italic;
    color: {c.TEXT_TERTIARY};
}}

#PanelHeaderBtn {{
    border: 1px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 2px 8px;
    color: {c.TEXT_SECONDARY};
    background: {c.BG_BASE};
    min-height: {HEIGHT_BUTTON_SM}px;
}}

#PanelHeaderBtn:hover {{
    border-color: {c.ACCENT};
    color: {c.TEXT_PRIMARY};
    background: {c.BG_SURFACE_HOVER};
}}

#PanelHeaderBtn:pressed {{
    background: {c.ACCENT_PRESSED};
    color: {c.TEXT_ON_ACCENT};
}}"""


def _block_model_panels(c: ColorPalette) -> str:
    """QSS for block model panel widgets (Batch 2 migration)."""
    return f"""/* === Block Model Panel Widgets (Phase 4 Batch 2) === */

/* Stats label in BlockInfoPanel */
QLabel#StatsLabel {{
    font-size: {FONT_SIZE_XS}px;
    color: {c.TEXT_SECONDARY};
}}

/* Property cards in BlockModelColumnMappingDialog */
QFrame#PropertyCard {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 8px;
}}
QFrame#PropertyCard:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.BORDER_STRONG};
}}
QFrame#PropertyCardSelected {{
    background-color: {c.STATUS_SUCCESS_SUBTLE};
    border: {BORDER_WIDTH}px solid {c.STATUS_SUCCESS};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 8px;
}}
QFrame#PropertyCardSelected:hover {{
    background-color: {c.STATUS_SUCCESS_SUBTLE};
    border-color: {c.STATUS_SUCCESS};
}}

/* Mapping rows in BlockModelColumnMappingDialog */
QFrame#BlockMappingRow {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
}}
QFrame#BlockMappingRow:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.BORDER_STRONG};
}}

/* Grade cards in ColumnMappingDialog (Batch 3) */
QFrame#GradeCard {{
    background-color: {c.BG_SURFACE};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 8px;
}}
QFrame#GradeCard:hover {{
    background-color: {c.BG_SURFACE_HOVER};
    border-color: {c.BORDER_STRONG};
}}
QFrame#GradeCardSelected {{
    background-color: {c.STATUS_SUCCESS_SUBTLE};
    border: {BORDER_WIDTH}px solid {c.STATUS_SUCCESS};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 8px;
}}

/* Count labels in ColumnMappingDialog (Batch 3) */
QLabel#CountLabelEmpty {{
    color: {c.TEXT_DISABLED};
}}
QLabel#CountLabelFull {{
    color: {c.STATUS_SUCCESS};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
}}
QLabel#CountLabelPartial {{
    color: {c.ACCENT};
}}

/* Selection count labels in comparison_utils (Batch 4) */
QLabel#SelectionLabelNone {{
    color: {c.TEXT_DISABLED};
    font-size: {FONT_SIZE_SM}px;
    padding: 4px;
}}
QLabel#SelectionLabelWarning {{
    color: {c.STATUS_WARNING};
    font-size: {FONT_SIZE_SM}px;
    padding: 4px;
}}
QLabel#SelectionLabelOk {{
    color: {c.STATUS_SUCCESS};
    font-size: {FONT_SIZE_SM}px;
    padding: 4px;
}}

/* Data source selector — lineage banner (Batch 5) */
QFrame#LineageBanner {{
    background-color: {c.BG_ELEVATED};
    border: {BORDER_WIDTH}px solid {c.BORDER_DEFAULT};
    border-radius: {BORDER_RADIUS_LG}px;
    padding: {SPACING_SM}px;
}}
QLabel#LineageStep {{
    color: {c.TEXT_SECONDARY};
    font-size: {FONT_SIZE_SM}px;
}}
QLabel#LineageStepCurrent {{
    color: {c.STATUS_SUCCESS};
    font-weight: {FONT_WEIGHT_BOLD};
    font-size: {FONT_SIZE_SM}px;
}}
QLabel#LineageArrow {{
    color: {c.TEXT_TERTIARY};
    font-size: {FONT_SIZE_SM}px;
}}

/* Warning banner (Batch 5) */
QFrame#WarningBanner {{
    background-color: {c.STATUS_WARNING_SUBTLE};
    border: {BORDER_WIDTH}px solid {c.STATUS_WARNING};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: {SPACING_SM}px {SPACING_MD}px;
}}
QLabel#WarningText {{
    color: {c.STATUS_WARNING};
    font-size: {FONT_SIZE_SM}px;
}}"""


def _loopstructural_panel(c: ColorPalette) -> str:
    """QSS for LoopStructural geological modeling panel — clean light/white theme.

    This panel uses a hardcoded light color palette (Apple-like clean UI)
    that contrasts with the dark application theme.  All selectors are
    scoped under ``QWidget#LoopPanel`` so only the Loop panel receives the
    light treatment.

    The *c* palette argument is intentionally ignored; every color is
    taken from the inline constants below.
    """

    # -- Hardcoded light palette (ignore *c*) --------------------------------
    BG          = "#F7F8FA"
    SURFACE     = "#FFFFFF"
    BORDER      = "#E2E5EA"
    BORDER_HOVER = "#D1D5DB"
    TEXT_DARK   = "#111827"
    TEXT_MED    = "#374151"
    TEXT_MUTED  = "#6B7280"
    TEXT_LIGHT  = "#9CA3AF"
    ACCENT      = "#6366F1"
    ACCENT_HVR  = "#4F46E5"
    ACCENT_PRS  = "#4338CA"
    ACCENT_SUB  = "#EEF2FF"
    ACCENT_ON   = "#FFFFFF"
    SUCCESS     = "#10B981"
    SUCCESS_SUB = "#D1FAE5"
    WARNING     = "#F59E0B"
    WARNING_SUB = "#FEF3C7"
    ERROR       = "#EF4444"
    ERROR_SUB   = "#FEE2E2"
    DIVIDER     = "#E5E7EB"
    INPUT_BG    = "#FFFFFF"
    INPUT_BRD   = "#D1D5DB"
    INPUT_FOCUS = "#6366F1"
    DISABLED_TXT = "#9CA3AF"

    return f"""/* === LoopStructural Panel — Light / White Theme === */

/* ------------------------------------------------------------------ */
/*  1. SCOPED BASE WIDGETS                                            */
/* ------------------------------------------------------------------ */

/* 1. Root panel background */
QWidget#LoopPanel {{
    background-color: {BG};
    color: {TEXT_DARK};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
}}

/* 2. Labels — dark text on light background */
QWidget#LoopPanel QLabel {{
    color: {TEXT_DARK};
    background: transparent;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
}}

/* 3. Spin boxes — white inputs, subtle border */
QWidget#LoopPanel QSpinBox,
QWidget#LoopPanel QDoubleSpinBox {{
    background-color: {INPUT_BG};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {INPUT_BRD};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px {SPACING_SM}px;
    min-height: {HEIGHT_INPUT}px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
    selection-background-color: {ACCENT_SUB};
    selection-color: {ACCENT};
}}
QWidget#LoopPanel QSpinBox:focus,
QWidget#LoopPanel QDoubleSpinBox:focus {{
    border-color: {INPUT_FOCUS};
}}
QWidget#LoopPanel QSpinBox:hover,
QWidget#LoopPanel QDoubleSpinBox:hover {{
    border-color: {BORDER_HOVER};
}}
QWidget#LoopPanel QSpinBox::up-button,
QWidget#LoopPanel QDoubleSpinBox::up-button,
QWidget#LoopPanel QSpinBox::down-button,
QWidget#LoopPanel QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 18px;
}}
QWidget#LoopPanel QSpinBox::up-arrow,
QWidget#LoopPanel QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_MUTED};
    width: 0; height: 0;
}}
QWidget#LoopPanel QSpinBox::down-arrow,
QWidget#LoopPanel QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    width: 0; height: 0;
}}
QWidget#LoopPanel QSpinBox:disabled,
QWidget#LoopPanel QDoubleSpinBox:disabled {{
    background-color: {BG};
    color: {DISABLED_TXT};
    border-color: {BORDER};
}}

/* 4. Combo boxes */
QWidget#LoopPanel QComboBox {{
    background-color: {INPUT_BG};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {INPUT_BRD};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px {SPACING_SM}px;
    min-height: {HEIGHT_INPUT}px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
}}
QWidget#LoopPanel QComboBox:hover {{
    border-color: {BORDER_HOVER};
}}
QWidget#LoopPanel QComboBox:focus {{
    border-color: {INPUT_FOCUS};
}}
QWidget#LoopPanel QComboBox::drop-down {{
    border: none;
    background: transparent;
    width: 24px;
}}
QWidget#LoopPanel QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    width: 0; height: 0;
}}
QWidget#LoopPanel QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
    selection-background-color: {ACCENT_SUB};
    selection-color: {ACCENT};
    outline: none;
    padding: {SPACING_XS}px 0;
}}
QWidget#LoopPanel QComboBox QAbstractItemView::item {{
    padding: {SPACING_XS}px {SPACING_SM}px;
    min-height: 28px;
}}
QWidget#LoopPanel QComboBox QAbstractItemView::item:hover {{
    background-color: {ACCENT_SUB};
    color: {ACCENT};
}}
QWidget#LoopPanel QComboBox:disabled {{
    background-color: {BG};
    color: {DISABLED_TXT};
    border-color: {BORDER};
}}

/* 5. Checkboxes — dark text, indigo indicator */
QWidget#LoopPanel QCheckBox {{
    color: {TEXT_DARK};
    font-size: {FONT_SIZE_BASE}px;
    spacing: {SPACING_SM}px;
    background: transparent;
}}
QWidget#LoopPanel QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: {BORDER_WIDTH}px solid {INPUT_BRD};
    border-radius: 3px;
    background-color: {INPUT_BG};
}}
QWidget#LoopPanel QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QWidget#LoopPanel QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}
QWidget#LoopPanel QCheckBox::indicator:checked:hover {{
    background-color: {ACCENT_HVR};
    border-color: {ACCENT_HVR};
}}
QWidget#LoopPanel QCheckBox:disabled {{
    color: {DISABLED_TXT};
}}
QWidget#LoopPanel QCheckBox::indicator:disabled {{
    background-color: {BG};
    border-color: {BORDER};
}}

/* 6. List widgets — white with subtle borders */
QWidget#LoopPanel QListWidget {{
    background-color: {SURFACE};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
    outline: none;
    font-size: {FONT_SIZE_BASE}px;
}}
QWidget#LoopPanel QListWidget::item {{
    padding: {SPACING_SM}px {SPACING_MD}px;
    color: {TEXT_DARK};
    border-bottom: {BORDER_WIDTH}px solid {DIVIDER};
}}
QWidget#LoopPanel QListWidget::item:selected {{
    background-color: {ACCENT_SUB};
    color: {ACCENT};
}}
QWidget#LoopPanel QListWidget::item:hover:!selected {{
    background-color: {BG};
}}

/* 7. Table widgets — clean light table */
QWidget#LoopPanel QTableWidget {{
    background-color: {SURFACE};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
    gridline-color: {DIVIDER};
    font-size: {FONT_SIZE_BASE}px;
    selection-background-color: {ACCENT_SUB};
    selection-color: {ACCENT};
}}
QWidget#LoopPanel QTableWidget::item {{
    padding: {SPACING_SM}px {SPACING_MD}px;
    color: {TEXT_DARK};
}}
QWidget#LoopPanel QTableWidget::item:selected {{
    background-color: {ACCENT_SUB};
    color: {ACCENT};
}}

/* 8. Text edits — white, subtle border */
QWidget#LoopPanel QTextEdit {{
    background-color: {SURFACE};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_SM}px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
    selection-background-color: {ACCENT_SUB};
    selection-color: {ACCENT};
}}
QWidget#LoopPanel QTextEdit:focus {{
    border-color: {INPUT_FOCUS};
}}

/* 9. Push buttons — clean light default */
QWidget#LoopPanel QPushButton {{
    background-color: {SURFACE};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px {SPACING_MD}px;
    min-height: {HEIGHT_BUTTON_SM}px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_MEDIUM};
}}
QWidget#LoopPanel QPushButton:hover {{
    background-color: {BG};
    border-color: {BORDER_HOVER};
}}
QWidget#LoopPanel QPushButton:pressed {{
    background-color: {DIVIDER};
}}
QWidget#LoopPanel QPushButton:disabled {{
    color: {DISABLED_TXT};
    background-color: {BG};
    border-color: {BORDER};
}}

/* 10. Scroll area — transparent */
QWidget#LoopPanel QScrollArea {{
    background-color: transparent;
    border: none;
}}
QWidget#LoopPanel QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* 11. Scroll bars — thin, subtle */
QWidget#LoopPanel QScrollBar:vertical {{
    background: transparent;
    width: {SCROLLBAR_WIDTH}px;
    margin: 0;
    border: none;
}}
QWidget#LoopPanel QScrollBar::handle:vertical {{
    background-color: {BORDER_HOVER};
    min-height: {SCROLLBAR_MIN_HANDLE}px;
    border-radius: {SCROLLBAR_RADIUS}px;
}}
QWidget#LoopPanel QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_LIGHT};
}}
QWidget#LoopPanel QScrollBar::add-line:vertical,
QWidget#LoopPanel QScrollBar::sub-line:vertical {{
    height: 0;
    border: none;
    background: none;
}}
QWidget#LoopPanel QScrollBar::add-page:vertical,
QWidget#LoopPanel QScrollBar::sub-page:vertical {{
    background: none;
}}
QWidget#LoopPanel QScrollBar:horizontal {{
    background: transparent;
    height: {SCROLLBAR_WIDTH}px;
    margin: 0;
    border: none;
}}
QWidget#LoopPanel QScrollBar::handle:horizontal {{
    background-color: {BORDER_HOVER};
    min-width: {SCROLLBAR_MIN_HANDLE}px;
    border-radius: {SCROLLBAR_RADIUS}px;
}}
QWidget#LoopPanel QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_LIGHT};
}}
QWidget#LoopPanel QScrollBar::add-line:horizontal,
QWidget#LoopPanel QScrollBar::sub-line:horizontal {{
    width: 0;
    border: none;
    background: none;
}}
QWidget#LoopPanel QScrollBar::add-page:horizontal,
QWidget#LoopPanel QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* 12. Frames — transparent default */
QWidget#LoopPanel QFrame {{
    background: transparent;
    border: none;
}}

/* 13. Header view sections — light header */
QWidget#LoopPanel QHeaderView::section {{
    background-color: {BG};
    color: {TEXT_MED};
    padding: {SPACING_SM}px {SPACING_MD}px;
    border: none;
    border-bottom: {BORDER_WIDTH}px solid {BORDER};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    font-size: {FONT_SIZE_SM}px;
}}

/* Line edits inside the panel */
QWidget#LoopPanel QLineEdit {{
    background-color: {INPUT_BG};
    color: {TEXT_DARK};
    border: {BORDER_WIDTH}px solid {INPUT_BRD};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px {SPACING_SM}px;
    min-height: {HEIGHT_INPUT}px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
    selection-background-color: {ACCENT_SUB};
    selection-color: {ACCENT};
}}
QWidget#LoopPanel QLineEdit:focus {{
    border-color: {INPUT_FOCUS};
}}
QWidget#LoopPanel QLineEdit:hover {{
    border-color: {BORDER_HOVER};
}}

/* Radio buttons */
QWidget#LoopPanel QRadioButton {{
    color: {TEXT_DARK};
    font-size: {FONT_SIZE_BASE}px;
    spacing: {SPACING_SM}px;
    background: transparent;
}}
QWidget#LoopPanel QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: {BORDER_WIDTH}px solid {INPUT_BRD};
    border-radius: 8px;
    background-color: {INPUT_BG};
}}
QWidget#LoopPanel QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QWidget#LoopPanel QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* Group boxes */
QWidget#LoopPanel QGroupBox {{
    background: transparent;
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
    margin-top: {SPACING_MD}px;
    padding-top: {SPACING_LG}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    color: {TEXT_MED};
}}
QWidget#LoopPanel QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 {SPACING_SM}px;
    color: {TEXT_MED};
    font-size: {FONT_SIZE_SM}px;
}}

/* Tab widgets inside the panel */
QWidget#LoopPanel QTabWidget::pane {{
    background-color: {SURFACE};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_SM}px;
}}
QWidget#LoopPanel QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_MUTED};
    padding: {SPACING_SM}px {SPACING_MD}px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_MEDIUM};
}}
QWidget#LoopPanel QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom-color: {ACCENT};
}}
QWidget#LoopPanel QTabBar::tab:hover:!selected {{
    color: {TEXT_DARK};
}}

/* Progress bars */
QWidget#LoopPanel QProgressBar {{
    background-color: {DIVIDER};
    border: none;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
    text-align: center;
    font-size: 0;
}}
QWidget#LoopPanel QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* Tooltips */
QWidget#LoopPanel QToolTip {{
    background-color: {TEXT_DARK};
    color: {SURFACE};
    border: none;
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {SPACING_XS}px {SPACING_SM}px;
    font-size: {FONT_SIZE_SM}px;
}}

/* ------------------------------------------------------------------ */
/*  2. SCOPED COLLAPSIBLE GROUP (light variant)                       */
/* ------------------------------------------------------------------ */

QWidget#LoopPanel QFrame[objectName="CollapsibleTitle"] {{
    background-color: {SURFACE};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {BORDER_RADIUS_MD}px;
    min-height: {HEIGHT_COLLAPSIBLE_TITLE}px;
    max-height: {HEIGHT_COLLAPSIBLE_TITLE}px;
}}
QWidget#LoopPanel QFrame[objectName="CollapsibleTitle"]:hover {{
    background-color: {BG};
    border-color: {BORDER_HOVER};
}}
QWidget#LoopPanel QLabel[objectName="CollapsibleTitleLabel"] {{
    color: {TEXT_DARK};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    font-size: {FONT_SIZE_MD}px;
    background: transparent;
    border: none;
}}
QWidget#LoopPanel QPushButton[objectName="CollapsibleToggle"] {{
    background-color: transparent;
    border: none;
    color: {TEXT_MUTED};
    font-size: 12px;
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
}}
QWidget#LoopPanel QPushButton[objectName="CollapsibleToggle"]:hover {{
    color: {ACCENT};
    background-color: {ACCENT_SUB};
    border-radius: {BORDER_RADIUS_SM}px;
}}
QWidget#LoopPanel QWidget[objectName="CollapsibleContent"] {{
    background-color: {SURFACE};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-top: none;
    border-bottom-left-radius: {BORDER_RADIUS_MD}px;
    border-bottom-right-radius: {BORDER_RADIUS_MD}px;
}}

/* ------------------------------------------------------------------ */
/*  3. LOOP-SPECIFIC OBJECT-NAME SELECTORS (light theme)              */
/* ------------------------------------------------------------------ */

/* ---- Sidebar navigation ---- */
QFrame#LoopSidebar {{
    background-color: {SURFACE};
    border-right: {BORDER_WIDTH}px solid {BORDER};
    min-width: 200px;
    max-width: 200px;
}}

QLabel#LoopSidebarTitle {{
    color: {TEXT_DARK};
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
    background: transparent;
    padding: 0;
}}

QLabel#LoopSidebarSubtitle {{
    color: {TEXT_LIGHT};
    font-size: {FONT_SIZE_XS}px;
    background: transparent;
    padding: 0;
}}

QPushButton#LoopNavButton {{
    background: transparent;
    border: none;
    text-align: left;
    padding: {SPACING_SM}px {SPACING_LG}px;
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_MEDIUM};
    border-radius: {BORDER_RADIUS_SM}px;
    margin: 1px {SPACING_SM}px;
}}
QPushButton#LoopNavButton:hover {{
    background-color: {BG};
    color: {TEXT_DARK};
}}
QPushButton#LoopNavButton[active="true"] {{
    background-color: {ACCENT_SUB};
    color: {ACCENT};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    border-left: 3px solid {ACCENT};
}}

/* Content area background */
QStackedWidget#LoopContent {{
    background-color: {BG};
}}

/* ---- Cards ---- */
QWidget#LoopPanel QFrame#Card {{
    background-color: {SURFACE};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: 12px;
}}

/* ---- Section header — NOT uppercase, muted dark gray ---- */
QLabel#LoopSectionHeader {{
    color: {TEXT_MED};
    font-size: {FONT_SIZE_SM}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    letter-spacing: 0;
    text-transform: none;
    background: transparent;
    padding: 0;
}}

/* 1px horizontal divider */
QFrame#LoopDivider {{
    background-color: {DIVIDER};
    min-height: 1px;
    max-height: 1px;
    border: none;
}}

/* Action bar (bottom of sections) */
QFrame#LoopActionBar {{
    background-color: {SURFACE};
    border-top: {BORDER_WIDTH}px solid {DIVIDER};
}}

/* Action button — indigo text, no bg */
QPushButton#LoopActionButton {{
    background-color: transparent;
    color: {ACCENT};
    border: none;
    padding: {SPACING_XS}px {SPACING_MD}px;
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    border-radius: {BORDER_RADIUS_SM}px;
}}
QPushButton#LoopActionButton:hover {{
    background-color: {ACCENT_SUB};
}}
QPushButton#LoopActionButton:pressed {{
    background-color: {ACCENT_SUB};
    color: {ACCENT_HVR};
}}
QPushButton#LoopActionButton:disabled {{
    color: {DISABLED_TXT};
}}

/* Danger action button */
QPushButton#LoopActionButtonDanger {{
    background-color: transparent;
    color: {ERROR};
    border: none;
    padding: {SPACING_XS}px {SPACING_MD}px;
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    border-radius: {BORDER_RADIUS_SM}px;
}}
QPushButton#LoopActionButtonDanger:hover {{
    background-color: {ERROR_SUB};
}}
QPushButton#LoopActionButtonDanger:pressed {{
    background-color: {ERROR_SUB};
    color: #DC2626;
}}

/* Flat list — white bg, no outer border, subtle item separators */
QListWidget#LoopFlatList {{
    background-color: {SURFACE};
    border: none;
    outline: none;
    font-size: {FONT_SIZE_BASE}px;
}}
QListWidget#LoopFlatList::item {{
    background-color: {SURFACE};
    color: {TEXT_DARK};
    padding: {SPACING_SM}px {SPACING_LG}px;
    border-bottom: {BORDER_WIDTH}px solid {DIVIDER};
}}
QListWidget#LoopFlatList::item:selected {{
    background-color: {ACCENT_SUB};
    color: {ACCENT};
    border-left: 3px solid {ACCENT};
}}
QListWidget#LoopFlatList::item:hover:!selected {{
    background-color: {BG};
}}

/* Flat table — white, clean headers */
QTableWidget#LoopFlatTable {{
    background-color: {SURFACE};
    border: none;
    gridline-color: transparent;
    font-size: {FONT_SIZE_BASE}px;
    selection-background-color: {ACCENT_SUB};
    selection-color: {ACCENT};
}}
QTableWidget#LoopFlatTable::item {{
    padding: {SPACING_SM}px {SPACING_LG}px;
    color: {TEXT_DARK};
    border-bottom: {BORDER_WIDTH}px solid {DIVIDER};
}}
QTableWidget#LoopFlatTable::item:selected {{
    background-color: {ACCENT_SUB};
    color: {ACCENT};
}}
QTableWidget#LoopFlatTable QHeaderView::section {{
    background-color: {BG};
    color: {TEXT_MED};
    padding: {SPACING_SM}px;
    border: none;
    border-bottom: {BORDER_WIDTH}px solid {BORDER};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    font-size: {FONT_SIZE_SM}px;
}}

/* Validation status pill badges */
QLabel#LoopStatusPill {{
    font-size: {FONT_SIZE_XS}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    padding: 3px 10px;
    border-radius: 10px;
    background-color: #F3F4F6;
    color: {TEXT_MED};
}}
QLabel#LoopStatusPill[pillStatus="pass"] {{
    background-color: {SUCCESS_SUB};
    color: {SUCCESS};
}}
QLabel#LoopStatusPill[pillStatus="warn"] {{
    background-color: {WARNING_SUB};
    color: {WARNING};
}}
QLabel#LoopStatusPill[pillStatus="fail"] {{
    background-color: {ERROR_SUB};
    color: {ERROR};
}}

/* Audit summary banner */
QFrame#LoopAuditBanner {{
    background-color: {SURFACE};
    border: none;
}}
QFrame#LoopAuditBanner QLabel {{
    color: {TEXT_DARK};
}}
QFrame#LoopAuditBanner[auditStatus="pass"] {{
    background-color: {SUCCESS_SUB};
    border: 2px solid {SUCCESS};
    border-radius: 12px;
}}
QFrame#LoopAuditBanner[auditStatus="warn"] {{
    background-color: {WARNING_SUB};
    border: 2px solid {WARNING};
    border-radius: 12px;
}}
QFrame#LoopAuditBanner[auditStatus="fail"] {{
    background-color: {ERROR_SUB};
    border: 2px solid {ERROR};
    border-radius: 12px;
}}

/* Build phase indicator row */
QFrame#LoopPhaseRow {{
    background-color: {SURFACE};
    border-bottom: {BORDER_WIDTH}px solid {DIVIDER};
    padding: {SPACING_SM}px {SPACING_LG}px;
}}

/* Form rows — transparent bg */
QWidget#LoopFormRow,
QFrame#LoopFormRow {{
    background: transparent;
    border: none;
}}

/* Workflow breadcrumb step labels */
QLabel#LoopWorkflowStep {{
    font-size: {FONT_SIZE_XS}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    background: transparent;
    padding: 2px 6px;
    color: {TEXT_LIGHT};
}}
QLabel#LoopWorkflowStep[stepState="done"] {{
    color: {SUCCESS};
}}
QLabel#LoopWorkflowStep[stepState="active"] {{
    color: {ACCENT};
    font-weight: {FONT_WEIGHT_BOLD};
}}

/* Muted icon buttons */
QPushButton#LoopMutedButton {{
    background-color: transparent;
    color: {TEXT_MUTED};
    border: none;
    font-size: 18px;
    border-radius: {BORDER_RADIUS_SM}px;
}}
QPushButton#LoopMutedButton:hover {{
    background-color: {BG};
    color: {TEXT_DARK};
}}

/* Parameter hint label — indigo on very light indigo */
QLabel#LoopParamHint {{
    color: {ACCENT};
    font-size: {FONT_SIZE_XS}px;
    background-color: {ACCENT_SUB};
    padding: {SPACING_SM}px {SPACING_MD}px;
    border-radius: {BORDER_RADIUS_SM}px;
    border-left: 3px solid {ACCENT};
}}

/* Monospace data summary */
QTextEdit#LoopDataSummary {{
    background-color: #F9FAFB;
    border: none;
    padding: {SPACING_XS}px {SPACING_LG}px;
    font-family: {FONT_FAMILY_MONO};
    font-size: {FONT_SIZE_SM}px;
    color: {TEXT_MED};
}}

/* ---- Welcome card — white, indigo left border ---- */
QFrame#LoopWelcomeCard {{
    background-color: {SURFACE};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 12px;
}}

QLabel#LoopWelcomeTitle {{
    color: {TEXT_DARK};
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
    background: transparent;
    padding: 0;
}}

QLabel#LoopInfoText {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_BASE}px;
    background: transparent;
    padding: 0;
}}

/* Clickable option card — white, border, hover indigo */
QPushButton#LoopOptionCard {{
    background-color: {SURFACE};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: 12px;
    padding: {SPACING_MD}px;
    text-align: left;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}px;
    color: {TEXT_DARK};
}}
QPushButton#LoopOptionCard:hover {{
    border-color: {ACCENT};
    background-color: {ACCENT_SUB};
}}
QPushButton#LoopOptionCard:pressed {{
    background-color: {ACCENT_SUB};
    border-color: {ACCENT_HVR};
}}

/* Primary button (scoped to Loop panel) — solid indigo */
QWidget#LoopPanel QPushButton#PrimaryButton {{
    background-color: {ACCENT};
    color: {ACCENT_ON};
    border: none;
    border-radius: 8px;
    padding: {SPACING_SM}px {SPACING_LG}px;
    min-height: {HEIGHT_BUTTON}px;
    font-size: {FONT_SIZE_BASE}px;
    font-weight: {FONT_WEIGHT_SEMIBOLD};
}}
QWidget#LoopPanel QPushButton#PrimaryButton:hover {{
    background-color: {ACCENT_HVR};
}}
QWidget#LoopPanel QPushButton#PrimaryButton:pressed {{
    background-color: {ACCENT_PRS};
}}
QWidget#LoopPanel QPushButton#PrimaryButton:disabled {{
    background-color: {BORDER};
    color: {DISABLED_TXT};
}}"""


def _sgsim_widgets(t: "DesignTokens") -> str:
    """QSS for SGSIM workbench-specific widgets.

    Redesigned for spacious, professional mining-software feel.
    Key fixes vs previous:
    - Nav items: taller (16px v/padding) and wider
    - Section headers: bigger font, more top/bottom breathing room
    - AccentButton: padding was 2px 8px (ridiculous) → 10px 24px
    """
    return f"""
/* ── SGSIM Navigation Sidebar ──────────────────────────────── */
QListWidget#SGSIMNavList {{
    background-color: {t.BG_SURFACE};
    outline: none;
    padding: 6px 0;
    font-size: 13px;
}}
QListWidget#SGSIMNavList::item {{
    padding: 14px 16px;
    color: {t.TEXT_SECONDARY};
    font-weight: 500;
    border-left: 3px solid transparent;
    min-height: 20px;
}}
QListWidget#SGSIMNavList::item:hover {{
    background-color: {t.BG_ELEVATED};
    color: {t.TEXT_PRIMARY};
}}
QListWidget#SGSIMNavList::item:selected {{
    background-color: {t.BG_ELEVATED};
    color: {t.ACCENT};
    border-left: 3px solid {t.ACCENT};
    font-weight: 600;
}}

/* ── SGSIM Section Headers ──────────────────────────────────── */
QLabel#SGSIMSectionHeader {{
    color: {t.ACCENT};
    font-size: 10pt;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding-top: 6px;
    padding-bottom: 2px;
}}

/* ── SGSIM RUN / Accent button ──────────────────────────────── */
/* padding was 2px 8px (criminally tight) — fixed to 10px 24px  */
QPushButton#AccentButton {{
    background-color: {t.ACCENT};
    color: {t.TEXT_ON_ACCENT};
    font-weight: 700;
    font-size: 12pt;
    border: none;
    border-radius: {BORDER_RADIUS_MD}px;
    padding: 10px 24px;
    min-height: 44px;
    letter-spacing: 0.5px;
}}
QPushButton#AccentButton:hover {{
    background-color: {t.ACCENT_HOVER};
}}
QPushButton#AccentButton:pressed {{
    background-color: {t.ACCENT_PRESSED};
}}
QPushButton#AccentButton:disabled {{
    background-color: {t.BG_ELEVATED};
    color: {t.TEXT_DISABLED};
    border: 1px solid {t.BORDER_SUBTLE};
}}

/* ── Panel divider lines ────────────────────────────────────── */
QFrame#PanelDivider {{
    color: {t.BORDER_SUBTLE};
    background-color: {t.BORDER_SUBTLE};
    max-height: 1px;
}}
"""


# =============================================================================
# CLI — Regenerate QSS files
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import sys

    out_dir = Path(__file__).parent.parent / "assets" / "themes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for theme_name in ("dark", "light"):
        qss = generate_qss(theme_name)
        out_path = out_dir / f"{theme_name}.qss"
        out_path.write_text(qss, encoding="utf-8")
        print(f"Generated {out_path} ({len(qss):,} chars)")

    print("Done.")
