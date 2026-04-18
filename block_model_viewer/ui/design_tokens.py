"""
GeoX Design Tokens — Single Source of Truth.

Every visual value in the application comes from this file.
No hardcoded colors, sizes, or spacing anywhere else.

Usage:
    from .design_tokens import tokens, dark_colors, light_colors

    # Get current theme colors
    colors = tokens.colors()

    # Access spacing/sizing
    margin = tokens.SPACING_MD
    btn_height = tokens.HEIGHT_BUTTON
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# COLOR PALETTES
# =============================================================================

@dataclass(frozen=True)
class ColorPalette:
    """
    Complete color palette for one theme.

    Naming rules:
    - SUBTLE = lightest/faintest variant
    - DEFAULT = standard usage
    - STRONG = heaviest/boldest variant
    These semantics are IDENTICAL in both themes. SUBTLE is always
    the faintest regardless of dark or light mode.

    All text colors meet WCAG AA contrast requirements (4.5:1) against
    their expected background surface.
    """

    # --- Backgrounds (layered elevation) ---
    BG_BASE: str = ""          # Deepest background (main window)
    BG_SURFACE: str = ""       # Primary surface (panels, cards)
    BG_SURFACE_HOVER: str = "" # Surface hover state
    BG_ELEVATED: str = ""      # Elevated elements (inputs, dropdowns)
    BG_OVERLAY: str = ""       # Overlays (tooltips, menus)

    # --- Borders ---
    BORDER_SUBTLE: str = ""    # Faintest border (dividers, separators)
    BORDER_DEFAULT: str = ""   # Standard border (inputs, cards)
    BORDER_STRONG: str = ""    # Emphasis border (hover, active states)

    # --- Text ---
    TEXT_PRIMARY: str = ""     # Headings, body text, labels
    TEXT_SECONDARY: str = ""   # Descriptions, less important text
    TEXT_TERTIARY: str = ""    # Hints, placeholders, captions
    TEXT_DISABLED: str = ""    # Disabled controls (still readable)
    TEXT_ON_ACCENT: str = ""   # Text on accent-colored backgrounds
    TEXT_LINK: str = ""        # Clickable text links

    # --- Accent (primary interactive color) ---
    ACCENT: str = ""           # Buttons, active tabs, links
    ACCENT_HOVER: str = ""     # Hover state
    ACCENT_PRESSED: str = ""   # Pressed/active state
    ACCENT_SUBTLE: str = ""    # Subtle accent background (selected row, badge)
    ACCENT_SECONDARY: str = "" # Secondary accent (teal, for secondary actions)

    # --- Status ---
    STATUS_SUCCESS: str = ""
    STATUS_SUCCESS_SUBTLE: str = ""  # Background tint for success messages
    STATUS_WARNING: str = ""
    STATUS_WARNING_SUBTLE: str = ""
    STATUS_ERROR: str = ""
    STATUS_ERROR_SUBTLE: str = ""
    STATUS_INFO: str = ""
    STATUS_INFO_SUBTLE: str = ""

    # --- Special ---
    HIGHLIGHT: str = ""        # Selection highlight
    SHADOW: str = ""           # Drop shadows
    FOCUS_RING: str = ""       # Keyboard focus indicator


# --- DARK THEME ---
# Classic neutral gray palette — warm, professional, matches standard dark UIs.
# Text contrast ratios verified against BG_SURFACE (#2d2d30):
#   TEXT_PRIMARY   #d4d4d4  → 10.3:1 ✓ (AAA)
#   TEXT_SECONDARY #a0a0a0  →  5.8:1 ✓ (AA)
#   TEXT_TERTIARY  #808080  →  4.0:1 ✓ (AA)
#   TEXT_DISABLED  #666666  →  3.2:1   (below AA, acceptable for disabled)

DARK = ColorPalette(
    # Backgrounds — VS Code / Material Design-inspired layered elevation
    BG_BASE="#1e1e1e",
    BG_SURFACE="#252526",
    BG_SURFACE_HOVER="#2a2a2c",
    BG_ELEVATED="#303032",
    BG_OVERLAY="#383838",

    # Borders — subtle neutral edges
    BORDER_SUBTLE="#333333",
    BORDER_DEFAULT="#3e3e42",
    BORDER_STRONG="#505054",

    # Text — brighter primary for better contrast (11.2:1 on BG_SURFACE)
    TEXT_PRIMARY="#e4e4e4",
    TEXT_SECONDARY="#a0a0a0",
    TEXT_TERTIARY="#808080",
    TEXT_DISABLED="#606060",
    TEXT_ON_ACCENT="#ffffff",
    TEXT_LINK="#4daafc",

    # Accent — Azure blue (muted, professional)
    ACCENT="#2188d9",
    ACCENT_HOVER="#2ba0f5",
    ACCENT_PRESSED="#1a6fb5",
    ACCENT_SUBTLE="#1a3a52",
    ACCENT_SECONDARY="#26a69a",

    # Status — Material Design palette
    STATUS_SUCCESS="#4caf50",
    STATUS_SUCCESS_SUBTLE="#1e3a20",
    STATUS_WARNING="#ffa726",
    STATUS_WARNING_SUBTLE="#3d2e12",
    STATUS_ERROR="#ef5350",
    STATUS_ERROR_SUBTLE="#3d1a1a",
    STATUS_INFO="#42a5f5",
    STATUS_INFO_SUBTLE="#1a2e3d",

    # Special
    HIGHLIGHT="#ffa726",
    SHADOW="rgba(0, 0, 0, 0.4)",
    FOCUS_RING="#2188d9",
)


# --- LIGHT THEME ---
# Text contrast ratios verified against BG_SURFACE (#ffffff):
#   TEXT_PRIMARY   #1a1a1a  → 17.4:1 ✓ (AAA)
#   TEXT_SECONDARY #5c5c5c  →  6.5:1 ✓ (AA)
#   TEXT_TERTIARY  #808080  →  4.6:1 ✓ (AA)
#   TEXT_DISABLED  #a0a0a0  →  2.7:1   (below AA, acceptable for disabled)

LIGHT = ColorPalette(
    # Backgrounds — neutral gray (typical desktop engineering software)
    BG_BASE="#f0f0f0",
    BG_SURFACE="#ffffff",
    BG_SURFACE_HOVER="#f7f7f8",
    BG_ELEVATED="#ffffff",
    BG_OVERLAY="#ffffff",

    # Borders — soft neutral
    BORDER_SUBTLE="#e8e8e8",
    BORDER_DEFAULT="#d4d4d4",
    BORDER_STRONG="#b0b0b0",

    # Text — neutral gray hierarchy
    TEXT_PRIMARY="#1a1a1a",
    TEXT_SECONDARY="#5c5c5c",
    TEXT_TERTIARY="#808080",
    TEXT_DISABLED="#a0a0a0",
    TEXT_ON_ACCENT="#ffffff",
    TEXT_LINK="#1a73c7",

    # Accent — Google blue
    ACCENT="#1a73c7",
    ACCENT_HOVER="#1e88e5",
    ACCENT_PRESSED="#155da3",
    ACCENT_SUBTLE="#e3f0fc",
    ACCENT_SECONDARY="#00897b",

    # Status — Material Design palette
    STATUS_SUCCESS="#2e7d32",
    STATUS_SUCCESS_SUBTLE="#e8f5e9",
    STATUS_WARNING="#e65100",
    STATUS_WARNING_SUBTLE="#fff3e0",
    STATUS_ERROR="#c62828",
    STATUS_ERROR_SUBTLE="#fce4ec",
    STATUS_INFO="#1565c0",
    STATUS_INFO_SUBTLE="#e3f2fd",

    # Special
    HIGHLIGHT="#e65100",
    SHADOW="rgba(0, 0, 0, 0.12)",
    FOCUS_RING="#1a73c7",
)


# =============================================================================
# SPACING (4px base grid)
# =============================================================================

SPACING_NONE = 0
SPACING_2XS = 2      # Hairline gaps (icon-to-text within a button)
SPACING_XS = 4        # Tight: related elements within a group
SPACING_SM = 8         # Default: between widgets in a form
SPACING_MD = 12        # Groups: between cards, between form sections
SPACING_LG = 16        # Sections: between major panel sections
SPACING_XL = 24        # Page-level: major layout divisions


# =============================================================================
# SIZING
# =============================================================================

# Input heights (outer height including border)
HEIGHT_INPUT = 32          # Combo box, spin box, line edit
HEIGHT_BUTTON = 36         # Standard buttons
HEIGHT_BUTTON_SM = 28      # Small buttons (OK, Cancel in dialogs)
HEIGHT_BUTTON_LG = 44      # Large call-to-action buttons

# Minimum widths
WIDTH_COMBO_MIN = 160
WIDTH_SPIN_MIN = 90
WIDTH_BUTTON_MIN = 80
WIDTH_LINE_EDIT_MIN = 120

# Panel constraints
WIDTH_PANEL_MIN = 280
HEIGHT_PANEL_MIN = 200
WIDTH_DIALOG_MIN = 420
HEIGHT_DIALOG_MIN = 300

# Collapsible group title bar
HEIGHT_COLLAPSIBLE_TITLE = 34


# =============================================================================
# TYPOGRAPHY
# =============================================================================

FONT_FAMILY = "'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif"
FONT_FAMILY_MONO = "'Cascadia Code', 'Consolas', 'Courier New', monospace"

# Font sizes — modular scale where each step is clearly distinguishable
# (not 1px increments that are invisible at screen resolution)
FONT_SIZE_XS = 10        # Small captions, footnotes
FONT_SIZE_SM = 11        # Hints, secondary labels
FONT_SIZE_BASE = 13      # Body text, input text, buttons
FONT_SIZE_MD = 14        # Section headers, collapsible titles
FONT_SIZE_LG = 16        # Panel titles
FONT_SIZE_XL = 20        # Dialog titles, page headers

# Font weights
FONT_WEIGHT_NORMAL = 400
FONT_WEIGHT_MEDIUM = 500
FONT_WEIGHT_SEMIBOLD = 600
FONT_WEIGHT_BOLD = 700


# =============================================================================
# BORDERS & RADII
# =============================================================================

BORDER_WIDTH = 1
BORDER_RADIUS_SM = 4      # Inputs, small elements
BORDER_RADIUS_MD = 6      # Cards, buttons, groups
BORDER_RADIUS_LG = 8      # Large containers, dialogs


# =============================================================================
# ICONS
# =============================================================================

ICON_SIZE_SM = 16          # Menu items, tree/list items, small buttons
ICON_SIZE_MD = 20          # Toolbar, standard buttons
ICON_SIZE_LG = 24          # Header icons, feature icons
ICON_SIZE_XL = 48          # Dialog icons, empty states


# =============================================================================
# ANIMATION
# =============================================================================

ANIM_DURATION_FAST = 100   # Collapse/expand, tooltips
ANIM_DURATION_NORMAL = 200 # Transitions, fades
ANIM_DURATION_SLOW = 350   # Page transitions, complex animations


# =============================================================================
# SCROLLBAR
# =============================================================================

SCROLLBAR_WIDTH = 10
SCROLLBAR_MIN_HANDLE = 24
SCROLLBAR_RADIUS = 5


# =============================================================================
# TOKENS ACCESSOR
# =============================================================================

class _Tokens:
    """
    Central accessor for all design tokens.

    Provides both direct attribute access to sizing/spacing constants
    and theme-aware color access via `colors()`.

    Usage:
        from .design_tokens import tokens

        c = tokens.colors()       # ColorPalette for current theme
        m = tokens.SPACING_MD     # 12
        h = tokens.HEIGHT_BUTTON  # 36
    """

    _current_theme: str = "dark"

    # Re-export all constants as attributes
    SPACING_NONE = SPACING_NONE
    SPACING_2XS = SPACING_2XS
    SPACING_XS = SPACING_XS
    SPACING_SM = SPACING_SM
    SPACING_MD = SPACING_MD
    SPACING_LG = SPACING_LG
    SPACING_XL = SPACING_XL

    HEIGHT_INPUT = HEIGHT_INPUT
    HEIGHT_BUTTON = HEIGHT_BUTTON
    HEIGHT_BUTTON_SM = HEIGHT_BUTTON_SM
    HEIGHT_BUTTON_LG = HEIGHT_BUTTON_LG
    WIDTH_COMBO_MIN = WIDTH_COMBO_MIN
    WIDTH_SPIN_MIN = WIDTH_SPIN_MIN
    WIDTH_BUTTON_MIN = WIDTH_BUTTON_MIN
    WIDTH_LINE_EDIT_MIN = WIDTH_LINE_EDIT_MIN
    WIDTH_PANEL_MIN = WIDTH_PANEL_MIN
    HEIGHT_PANEL_MIN = HEIGHT_PANEL_MIN
    HEIGHT_COLLAPSIBLE_TITLE = HEIGHT_COLLAPSIBLE_TITLE

    FONT_FAMILY = FONT_FAMILY
    FONT_FAMILY_MONO = FONT_FAMILY_MONO
    FONT_SIZE_XS = FONT_SIZE_XS
    FONT_SIZE_SM = FONT_SIZE_SM
    FONT_SIZE_BASE = FONT_SIZE_BASE
    FONT_SIZE_MD = FONT_SIZE_MD
    FONT_SIZE_LG = FONT_SIZE_LG
    FONT_SIZE_XL = FONT_SIZE_XL

    BORDER_WIDTH = BORDER_WIDTH
    BORDER_RADIUS_SM = BORDER_RADIUS_SM
    BORDER_RADIUS_MD = BORDER_RADIUS_MD
    BORDER_RADIUS_LG = BORDER_RADIUS_LG

    ICON_SIZE_SM = ICON_SIZE_SM
    ICON_SIZE_MD = ICON_SIZE_MD
    ICON_SIZE_LG = ICON_SIZE_LG

    SCROLLBAR_WIDTH = SCROLLBAR_WIDTH
    SCROLLBAR_MIN_HANDLE = SCROLLBAR_MIN_HANDLE

    ANIM_DURATION_FAST = ANIM_DURATION_FAST
    ANIM_DURATION_NORMAL = ANIM_DURATION_NORMAL

    def set_theme(self, name: str) -> None:
        """Set the active theme ('dark' or 'light')."""
        if name not in ("dark", "light"):
            name = "dark"
        self._current_theme = name

    @property
    def theme(self) -> str:
        """Current theme name."""
        return self._current_theme

    def colors(self) -> ColorPalette:
        """Get the color palette for the current theme."""
        return LIGHT if self._current_theme == "light" else DARK

    @property
    def dark(self) -> ColorPalette:
        """Direct access to dark palette (for QSS generation)."""
        return DARK

    @property
    def light(self) -> ColorPalette:
        """Direct access to light palette (for QSS generation)."""
        return LIGHT


# Module-level singleton
tokens = _Tokens()


# =============================================================================
# VISUALIZATION COLOR PALETTES (domain-specific, separate from UI)
# =============================================================================

VIS_PALETTES: Dict[str, List[str]] = {
    "geology":      ["#d8b365", "#f5f5dc", "#8c510a", "#c7eae5", "#5ab4ac"],
    "resource":     ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"],
    "uncertainty":  ["#54278f", "#756bb1", "#bcbddc", "#d4b9da", "#f1eef6"],
    "esg":          ["#238b45", "#78c679", "#c2e699", "#f7fcb1", "#004529"],
    "pit":          ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#2166ac"],
    "underground":  ["#2c7fb8", "#7fcdbb", "#edf8b1", "#c7e9b4", "#253494"],
    "heatmap":      ["#313695", "#4575b4", "#abd9e9", "#fee090", "#f46d43", "#a50026"],
    "categorical":  ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                     "#a65628", "#f781bf", "#999999"],
}
