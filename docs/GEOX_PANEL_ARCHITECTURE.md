# GeoX Panel Architecture & UI/UX Design Documentation

**Document Version:** 1.0  
**Date:** February 21, 2026  
**Application:** GeoX Block Model Viewer  

---

## Table of Contents

1. [Overview](#overview)
2. [Panel Architecture](#panel-architecture)
3. [Panel Hierarchy](#panel-hierarchy)
4. [Panel Lifecycle Management](#panel-lifecycle-management)
5. [UI/UX Design Patterns](#uiux-design-patterns)
6. [Menu System Architecture](#menu-system-architecture)
7. [User Input Handling](#user-input-handling)
8. [Typography & Sizing Standards](#typography--sizing-standards)
9. [Theme System (Light/Dark)](#theme-system-lightdark)
10. [Widget Styling Reference](#widget-styling-reference)
11. [Panel Categories](#panel-categories)
12. [Best Practices](#best-practices)

---

## Overview

GeoX is a professional mining and geological analysis application built with PyQt6. The application follows a **modular panel architecture** where each functional area is encapsulated in a specialized panel class. This document describes the architecture, design patterns, and styling systems used throughout the application.

### Key Principles

- **Modularity**: Each panel is self-contained and reusable
- **Consistency**: Standardized base classes ensure uniform behavior
- **Maintainability**: Clear separation of concerns and template methods
- **Professional Appearance**: Modern, clean UI with dark and light themes
- **Accessibility**: High contrast, appropriate sizing, clear visual hierarchy

---

## Panel Architecture

### Core Components

The panel system consists of several layers:

1. **Base Classes** - Abstract templates defining panel structure
2. **Panel Manager** - Centralized lifecycle and state management
3. **Panel Registry** - Automatic discovery and registration
4. **Theme Manager** - Styling and appearance control
5. **Menu System** - Organized, modular menu construction

### File Structure

```
block_model_viewer/ui/
├── base_panel.py                 # Base panel class
├── base_analysis_panel.py        # Analysis panel template
├── base_display_panel.py         # Display panel template
├── base_dock_panel.py            # Dockable panel template
├── panel_manager.py              # Panel lifecycle manager
├── panel_registration.py         # Auto-registration system
├── theme_manager.py              # Theme loading and switching
├── modern_styles.py              # Stylesheet templates
├── menus/                        # Modular menu system
│   ├── __init__.py
│   ├── file_menu.py
│   ├── view_menu.py
│   ├── estimations_menu.py
│   └── ... (21 menu modules)
├── [panel_name]_panel.py        # Individual panel implementations
└── ...
```

---

## Panel Hierarchy

### 1. BasePanel (base_panel.py)

**Purpose**: Foundation for all UI panels in the application

**Key Features**:
- Common attributes (controller, block_model, panel_id)
- Template method pattern (setup_ui, connect_signals, refresh)
- Block model management
- Registry access pattern
- Error handling and status messages
- Clear/reset functionality

**Template Methods (Must Override)**:
```python
def setup_ui(self):
    """Build widget layout. Subclasses MUST override."""
    raise NotImplementedError()

def connect_signals(self):
    """Wire Qt signals. Optional override."""
    pass

def refresh(self):
    """Refresh panel state. Optional override."""
    pass
```

**Key Signals**:
- `status_message` - Emit messages to status bar
- `error_occurred` - Emit error notifications

**Usage Example**:
```python
class MyPanel(BasePanel):
    PANEL_ID = "my_panel"
    
    def setup_ui(self):
        self.main_layout.addWidget(QLabel("My Content"))
    
    def connect_signals(self):
        self.some_button.clicked.connect(self._on_clicked)
```

### 2. BaseDockPanel (derived from BasePanel)

**Purpose**: Panels that can be docked in MainWindow

**Additional Features**:
- Dock widget integration
- Window flags for proper taskbar behavior
- Minimum/maximum size constraints
- Floating window support

**Sizing Defaults**:
- Minimum width: 250px
- Minimum height: 200px
- Customizable per panel via `panel_registration.py`

### 3. BaseAnalysisPanel (base_analysis_panel.py)

**Purpose**: Template for geostatistical and analytical workflows

**Key Features**:
- Asynchronous task execution via AppController
- Progress dialog management
- Parameter validation framework
- Result processing hooks
- Scrollable content area (auto-wrapped)

**Template Methods**:
```python
def gather_parameters(self) -> Dict[str, Any]:
    """Collect all UI parameters. Must override."""
    raise NotImplementedError()

def validate_inputs(self) -> Tuple[bool, str]:
    """Validate parameters. Must override."""
    return (True, "")

def on_results(self, results: Dict[str, Any]):
    """Process controller results. Must override."""
    pass
```

**Workflow Pattern**:
1. User clicks "Run" button
2. Panel calls `gather_parameters()`
3. Panel calls `validate_inputs()`
4. If valid, submit to AppController via `run_task()`
5. AppController executes in background thread
6. Results returned via `on_results()` callback

**Used By**: Kriging, SGSIM, Variogram, Resource Classification, etc.

---

## Panel Lifecycle Management

### PanelManager (panel_manager.py)

The **PanelManager** provides centralized control over all panels:

#### Key Responsibilities

1. **Registration**: Track all available panel classes
2. **Instantiation**: Create panel instances on-demand (lazy loading)
3. **Visibility Control**: Show/hide panels (never destroy)
4. **State Persistence**: Save/restore visibility and geometry
5. **UI Controls**: Generate menus, toolbars, shortcuts
6. **Dock Management**: Handle docking, floating, tabification

#### Panel Metadata (PanelInfo)

Each panel is registered with metadata:

```python
@dataclass
class PanelInfo:
    panel_id: str                    # Unique identifier
    name: str                        # Display name
    category: PanelCategory          # Organization category
    panel_class: Type[BasePanel]     # Class to instantiate
    icon_name: Optional[str]         # Icon file name
    shortcut: Optional[str]          # Keyboard shortcut (e.g., "Ctrl+K")
    default_dock_area: DockArea      # LEFT, RIGHT, TOP, BOTTOM, FLOATING
    default_visible: bool            # Show on startup?
    minimum_width: int               # Minimum width (px)
    minimum_height: int              # Minimum height (px)
    tooltip: Optional[str]           # Hover tooltip
```

#### Hide-on-Close Behavior

**Critical Design Pattern**: Panels are **never destroyed** when closed. Instead:

1. User clicks X → Panel is hidden (not deleted)
2. Panel state is preserved
3. User reopens panel → Same instance is shown (state intact)
4. Benefits: Fast reopening, no data loss, smooth UX

Implementation:
```python
def _on_dock_close_event(self, panel_id: str):
    """Handle dock widget close event - HIDE only, never destroy."""
    panel_info = self._panels.get(panel_id)
    if panel_info and panel_info.dock_widget:
        panel_info.dock_widget.hide()  # Hide, don't close
        panel_info.is_visible = False
        self._save_workspace_state()
```

#### State Persistence (panel_states.json)

Panel states are saved to `panel_states.json`:

```json
{
  "variogram_panel": {
    "visible": true,
    "floating": false,
    "geometry": "AdnQywADAAAAA...",
    "area": "LeftDockWidgetArea"
  },
  "property_panel": {
    "visible": true,
    "floating": false,
    "area": "LeftDockWidgetArea"
  }
}
```

This ensures the workspace layout persists between sessions.

---

## UI/UX Design Patterns

### 1. Card-Based Layout

Modern panels use **card metaphor** for visual grouping:

```
┌─────────────────────────────────────┐
│ Panel Title                          │  ← Header bar
├─────────────────────────────────────┤
│ ┌─────────────────────────────────┐ │
│ │ Section 1 (Collapsible)         │ │  ← Card
│ │   • Content                      │ │
│ │   • Controls                     │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Section 2                        │ │  ← Card
│ │   • More content                 │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Implementation**: Use `CollapsibleGroup` widget or `QGroupBox` with card styling.

### 2. Collapsible Groups

The `CollapsibleGroup` widget provides expandable/collapsible sections:

**Features**:
- Animated collapse/expand (250ms duration)
- Click-anywhere-to-toggle (title bar, icon, hint text)
- Visual feedback (hover effects, cursor changes)
- Customizable title and icon
- State persistence

**Usage**:
```python
from .collapsible_group import CollapsibleGroup

group = CollapsibleGroup("Data Input", collapsed=False)
group.content_layout.addWidget(my_controls)
self.main_layout.addWidget(group)
```

**Styling**:
- Title bar: 42px height
- Font: 14px bold (title), 11px (hint)
- Colors: Theme-aware (adapts to light/dark)
- Borders: Subtle dividers between sections

### 3. Form Layout Pattern

For input forms, use consistent layout:

```python
form = QFormLayout()
form.setSpacing(8)
form.setContentsMargins(12, 12, 12, 12)
form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

form.addRow("Property:", property_combo)
form.addRow("Min Value:", min_spinbox)
form.addRow("Max Value:", max_spinbox)
```

**Benefits**:
- Aligned labels (right-aligned)
- Consistent spacing
- Professional appearance
- Easy to scan

### 4. Action Button Layout

Buttons follow a standard pattern:

**Primary Actions** (most important):
- Position: Bottom-right or top-right
- Style: Filled background (accent color)
- Size: Height 32-44px, padding 8-16px
- Font: 13px, semi-bold (600 weight)

**Secondary Actions**:
- Style: Outlined (border, no fill)
- Same size as primary

**Buttons Layout**:
```python
btn_layout = QHBoxLayout()
btn_layout.addStretch()
btn_layout.addWidget(cancel_btn)  # Secondary
btn_layout.addWidget(run_btn)     # Primary
```

### 5. Progressive Disclosure

Complex panels use **tab widgets** to organize content:

**Example**: LoopStructural Panel
- Tab 1: Data Input (validation)
- Tab 2: Domain Setup
- Tab 3: Faults Definition
- Tab 4: Foliation/Folding
- Tab 5: Results & Export

**Benefits**:
- Reduces cognitive load
- Logical workflow progression
- Clear visual hierarchy

### 6. Workflow Guidance

Analysis panels often include **workflow banners**:

```
┌─────────────────────────────────────────────────────┐
│  Step 1: Load Data  →  Step 2: Configure  →  Step 3: Run  │
│     ✓                      •                    ○         │
└─────────────────────────────────────────────────────┘
```

**Implementation**:
- Visual progress indicators
- Clickable steps (navigate to tabs)
- State indicators (✓ = complete, • = current, ○ = pending)
- Auto-update as user progresses

---

## Menu System Architecture

### Modular Menu Pattern

The menu system was refactored from a monolithic 1000+ line section in MainWindow to **21 separate modules** in `ui/menus/`.

### Menu Structure

```python
# ui/menus/__init__.py
from .file_menu import build_file_menu
from .view_menu import build_view_menu
from .estimations_menu import build_estimations_menu
# ... all 21 menus

# MainWindow._setup_menus()
menubar = self.menuBar()
build_file_menu(self, menubar)
build_view_menu(self, menubar)
build_estimations_menu(self, menubar)
# ... etc
```

### Menu Modules

Each menu module follows this pattern:

```python
# ui/menus/file_menu.py
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence

def build_file_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the File menu."""
    file_menu = menubar.addMenu("&File")
    
    # Add actions
    new_action = QAction("&New Project", main_window)
    new_action.setShortcut(QKeySequence.StandardKey.New)
    new_action.triggered.connect(main_window._new_project)
    file_menu.addAction(new_action)
    
    # ... more actions
    
    return file_menu
```

### Current Menus (21 Total)

1. **File** - Project, open, save, export
2. **Layout** - Reports and layout composer
3. **Search** - Find tools and data
4. **Edit** - Copy, paste, undo
5. **View** - Themes, lighting, workspace layouts
6. **Scan** - 3D scanning operations
7. **Remote Sensing** - InSAR and satellite data
8. **Survey** - Survey data import
9. **Tools** - Slice, filter, cross-sections
10. **Panels** - Show/hide panels
11. **Data** - Import, export, registry
12. **Mouse** - Interaction modes (select, pan, rotate)
13. **Drillholes** - Drillhole data and visualization
14. **Geology** - LoopStructural modeling
15. **Resources** - Resource estimation
16. **Estimations** - Kriging, simulation, variography
17. **Geotech** - Geotechnical analysis
18. **Mine Planning** - Pit optimization, scheduling
19. **ML** - Machine learning tools
20. **Dashboards** - ESG, production, research dashboards
21. **Workbench** - Custom workflows
22. **Workflows** - Predefined analysis sequences
23. **Help** - Documentation, about

### Hover-to-Open Behavior

Menus support modern hover behavior:

1. Click any menu → Opens normally
2. Hover to adjacent menu → Instantly switches
3. After 200ms delay → Menu opens on hover (no click needed)
4. Mouse leaves menu bar → Hover-open stops

**Implementation**: `MainWindow._setup_menubar_hover()` installs event filter.

### Keyboard Shortcuts

Shortcuts are defined in menu actions:

```python
action.setShortcut(QKeySequence("Ctrl+K"))         # Custom
action.setShortcut(QKeySequence.StandardKey.Save)  # Standard
```

**Common Shortcuts**:
- `Ctrl+O` - Open
- `Ctrl+S` - Save
- `Ctrl+N` - New Project
- `Ctrl+K` - Kriging Panel
- `Ctrl+X` - Cross-Section Tool
- `Ctrl+1, 2, 3` - Panel shortcuts

---

## User Input Handling

### Input Widget Standards

#### QComboBox (Dropdowns)

**Sizing**:
- Minimum width: 120-200px (content-dependent)
- Height: 28-32px
- Font: 12px

**Usage**:
```python
combo = QComboBox()
combo.setMinimumWidth(160)
combo.addItems(["Option 1", "Option 2"])
combo.currentTextChanged.connect(self._on_selection_changed)
```

**Styling**: Automatically styled by theme system (rounded corners, hover effects).

#### QSpinBox / QDoubleSpinBox

**Sizing**:
- Width: 80-120px
- Height: 28-32px
- Font: 12px

**Features**:
- Suffix support (e.g., "m", "kg", "%")
- Step buttons (increase/decrease)
- Keyboard input with validation
- Min/max range enforcement

**Usage**:
```python
spinbox = QDoubleSpinBox()
spinbox.setRange(0.0, 100.0)
spinbox.setSingleStep(0.1)
spinbox.setDecimals(2)
spinbox.setSuffix(" m")
spinbox.setValue(10.0)
```

#### QSlider

**Sizing**:
- Height: 20-24px (horizontal)
- Width: 200-400px

**Usage**: Typically paired with spinbox for visual + precise input.

```python
slider = QSlider(Qt.Orientation.Horizontal)
slider.setRange(0, 100)
slider.setValue(50)
slider.valueChanged.connect(spinbox.setValue)
spinbox.valueChanged.connect(slider.setValue)  # Bidirectional sync
```

#### QLineEdit

**Sizing**:
- Height: 28-32px
- Width: Depends on content (100-400px typical)
- Font: 12px

**Features**:
- Placeholder text
- Input validation
- Clear button (optional)
- Password mode

**Usage**:
```python
line_edit = QLineEdit()
line_edit.setPlaceholderText("Enter value...")
line_edit.textChanged.connect(self._on_text_changed)
```

#### QPushButton

See "Action Button Layout" section above.

**Size Standards**:
- Small: 80x28px (e.g., "OK", "Cancel")
- Medium: 120x32px (e.g., "Run Analysis")
- Large: 140x44px (e.g., "Generate Model")

### Signal Blocking Pattern

When programmatically updating widgets, block signals to prevent recursive updates:

```python
from PyQt6.QtCore import QSignalBlocker

# Method 1: Context manager
with QSignalBlocker(self.combo):
    self.combo.setCurrentText("New Value")

# Method 2: Manual blocking
self.combo.blockSignals(True)
self.combo.setCurrentText("New Value")
self.combo.blockSignals(False)
```

### Validation Pattern

Input validation follows this pattern:

```python
def validate_inputs(self) -> Tuple[bool, str]:
    """Validate all inputs. Returns (is_valid, error_message)."""
    
    # Check required fields
    if not self.property_combo.currentText():
        return (False, "Please select a property")
    
    # Check numeric ranges
    if self.min_spin.value() >= self.max_spin.value():
        return (False, "Min must be less than Max")
    
    # Check data availability
    registry = self.get_registry()
    if not registry.has_block_model():
        return (False, "No block model loaded")
    
    return (True, "")

def _on_run_clicked(self):
    """Run button handler."""
    is_valid, error_msg = self.validate_inputs()
    if not is_valid:
        QMessageBox.warning(self, "Invalid Input", error_msg)
        return
    
    # Proceed with operation
    self._execute_task()
```

---

## Typography & Sizing Standards

### Font Families

**Primary Font Stack**:
```
'Segoe UI', 'Roboto', 'Arial', sans-serif
```

**Fallback Order**:
1. Segoe UI (Windows native)
2. Roboto (cross-platform)
3. Arial (universal fallback)
4. System sans-serif

### Font Sizes

**Text Hierarchy**:
- **Panel Title**: 16px, bold
- **Section Header**: 14px, semi-bold (600)
- **Body Text**: 12px, regular
- **Input Fields**: 12px, regular
- **Button Text**: 13px, medium (500-600)
- **Hint/Caption**: 11px, regular
- **Small Text**: 10px, regular

**Usage in Code**:
```python
# Stylesheet approach
label.setStyleSheet("font-size: 14px; font-weight: 600;")

# QFont approach
font = QFont("Segoe UI", 14)
font.setWeight(QFont.Weight.DemiBold)
label.setFont(font)
```

### Spacing Standards

**Layout Margins**:
- Main Panel: 5-12px all sides
- Card/Group: 12px all sides
- Form Rows: 8px vertical spacing
- Button Groups: 10px between buttons

**Layout Spacing (between widgets)**:
- Section Spacing: 10-16px
- Widget Spacing: 8px
- Tight Spacing: 4px

**Implementation**:
```python
layout = QVBoxLayout()
layout.setContentsMargins(12, 12, 12, 12)  # L, T, R, B
layout.setSpacing(10)
```

### Widget Sizing

**Minimum Sizes**:
- Panel: 250 x 200px
- Dialog: 400 x 300px
- Combo Box: 120px width
- Spin Box: 80px width
- Button: 80px width (minimum)
- Input Field: 100px width (minimum)

**Maximum Sizes**:
- Avoid hard maximums (allow resize)
- Exception: Fixed-size buttons for consistency

**Responsive Sizing**:
```python
widget.setSizePolicy(
    QSizePolicy.Policy.Expanding,  # Horizontal
    QSizePolicy.Policy.Preferred   # Vertical
)
```

### Icon Sizes

- **Menu Icons**: 16x16px
- **Toolbar Icons**: 24x24px
- **Button Icons**: 16x16px (small), 24x24px (medium)
- **Tree/List Icons**: 16x16px
- **Dialog Icons**: 48x48px

---

## Theme System (Light/Dark)

### Overview

GeoX supports two professionally designed themes:

1. **Dark Theme** (default) - Optimized for long viewing sessions
2. **Light Theme** - Bright, high-contrast for well-lit environments

### Color Palettes

#### Dark Theme Colors (`DarkColors` class)

```python
# Backgrounds
PANEL_BG = "#1e1e1e"        # Main panel background
CARD_BG = "#252525"         # Card/section background
CARD_HOVER = "#2a2a2a"      # Hover state
ELEVATED_BG = "#2d2d2d"     # Elevated elements

# Borders
BORDER = "#3d3d3d"          # Default border
BORDER_LIGHT = "#4d4d4d"    # Light border
DIVIDER = "#333333"         # Section divider

# Text
TEXT_PRIMARY = "#e8e8e8"    # Primary text
TEXT_SECONDARY = "#b0b0b0"  # Secondary text
TEXT_DISABLED = "#6d6d6d"   # Disabled text
TEXT_HINT = "#8a8a8a"       # Placeholder/hint

# Accents
ACCENT_PRIMARY = "#0e7aca"  # Primary blue
ACCENT_HOVER = "#1a8cd8"    # Lighter blue
ACCENT_PRESSED = "#0c6ab5"  # Darker blue
ACCENT_SECONDARY = "#26a69a" # Teal

# Status
SUCCESS = "#4caf50"         # Green
WARNING = "#ff9800"         # Orange
ERROR = "#f44336"           # Red
INFO = "#2196f3"            # Blue

# Special
HIGHLIGHT = "#ffa726"       # Orange highlight
SHADOW = "rgba(0,0,0,0.3)"  # Drop shadow
```

#### Light Theme Colors (`LightColors` class)

```python
# Backgrounds
PANEL_BG = "#f5f5f5"        # Main panel (light gray)
CARD_BG = "#ffffff"         # Card (white)
CARD_HOVER = "#fafafa"      # Hover
ELEVATED_BG = "#ffffff"     # Elevated

# Borders
BORDER = "#e0e0e0"          # Default
BORDER_LIGHT = "#bdbdbd"    # Darker in light theme
DIVIDER = "#eeeeee"         # Light gray

# Text
TEXT_PRIMARY = "#212121"    # Near black
TEXT_SECONDARY = "#757575"  # Medium gray
TEXT_DISABLED = "#bdbdbd"   # Light gray
TEXT_HINT = "#9e9e9e"       # Gray

# Accents
ACCENT_PRIMARY = "#1976d2"  # Darker blue (better contrast)
ACCENT_HOVER = "#1e88e5"    
ACCENT_PRESSED = "#1565c0"  
ACCENT_SECONDARY = "#00897b" # Darker teal

# Status (Darker for contrast)
SUCCESS = "#388e3c"         
WARNING = "#f57c00"         
ERROR = "#d32f2f"           
INFO = "#1976d2"            

# Special
HIGHLIGHT = "#ff9800"       
SHADOW = "rgba(0,0,0,0.1)"  # Lighter shadow
```

### Theme Manager Architecture

**File**: `block_model_viewer/ui/theme_manager.py`

**Key Responsibilities**:
1. Load QSS stylesheets from `assets/themes/`
2. Apply theme to QApplication
3. Emit `theme_changed` signal
4. Sync `modern_styles.py` module state
5. Manage color palettes for visualizations

**Theme Files**:
- `assets/themes/dark.qss` - Dark theme stylesheet
- `assets/themes/light.qss` - Light theme stylesheet
- `assets/themes/color_palette.json` - Visualization colors

**Usage**:
```python
# Set theme
theme_manager.load_theme("dark")
theme_manager.apply_theme(app)

# Connect to theme changes
theme_manager.theme_changed.connect(self._on_theme_changed)

# Get current theme
current = theme_manager.current_theme()  # "dark" or "light"
```

### Theme Switching Flow

**User Action**: View → Theme → Light/Dark

**Propagation**:
1. User clicks theme menu action
2. `MainWindow.set_theme("light")` called
3. `ThemeManager.load_theme("light")` loads QSS
4. `ThemeManager.apply_theme(app)` applies to QApplication
5. `modern_styles.set_current_theme("light")` syncs module state
6. `theme_changed` signal emitted
7. `MainWindow._on_theme_changed()` triggered
8. `MainWindow._refresh_all_themed_widgets()` called
9. **Each panel's `refresh_theme()` method called**
10. Panels rebuild stylesheets with new theme colors
11. UI updates instantly

### Panel Theme Support

**Every panel must implement**:
```python
def refresh_theme(self):
    """Update colors when theme changes."""
    # Rebuild stylesheet with current theme colors
    colors = get_theme_colors()  # Returns DarkColors or LightColors
    
    stylesheet = f"""
        QWidget {{
            background-color: {colors.PANEL_BG};
            color: {colors.TEXT_PRIMARY};
        }}
        QPushButton {{
            background-color: {colors.ACCENT_PRIMARY};
            color: white;
        }}
    """
    self.setStyleSheet(stylesheet)
    
    # Recursively refresh child widgets
    for child in self.findChildren(QWidget):
        if hasattr(child, 'refresh_theme'):
            child.refresh_theme()
```

### Dynamic Color Access

**ModernColors Metaclass**: Provides dynamic access to current theme colors.

```python
from .modern_styles import ModernColors

# Access current theme colors
bg_color = ModernColors.PANEL_BG      # "#1e1e1e" or "#f5f5f5"
text_color = ModernColors.TEXT_PRIMARY  # "#e8e8e8" or "#212121"
accent = ModernColors.ACCENT_PRIMARY    # "#0e7aca" or "#1976d2"

# Always returns colors from active theme
# No need to check which theme is active
```

**Implementation**:
```python
class _ModernColorsMeta(type):
    """Metaclass enables class attribute access to theme colors."""
    def __getattr__(cls, name: str) -> str:
        colors = get_theme_colors()  # DarkColors or LightColors
        if hasattr(colors, name):
            return getattr(colors, name)
        raise AttributeError(f"Color '{name}' not found")

class ModernColors(metaclass=_ModernColorsMeta):
    """Dynamic color palette - adapts to current theme."""
    pass
```

### Stylesheet Templates

**File**: `block_model_viewer/ui/modern_styles.py`

**Available Functions**:
- `get_panel_stylesheet()` - Main panel styling
- `get_button_stylesheet(style)` - Button styles (primary, secondary, icon, toggle)
- `get_input_stylesheet()` - Input field styling
- `get_group_stylesheet()` - QGroupBox styling
- `get_table_stylesheet()` - QTableWidget styling
- `get_list_stylesheet()` - QListWidget styling
- `get_tree_stylesheet()` - QTreeWidget styling
- `get_complete_panel_stylesheet()` - Comprehensive panel styling

**Usage**:
```python
from .modern_styles import get_complete_panel_stylesheet

def refresh_theme(self):
    """Apply theme-aware stylesheet."""
    self.setStyleSheet(get_complete_panel_stylesheet())
```

### Theme Persistence

**Storage**: Config file (`user_settings.json` or QSettings)

```python
# Save theme preference
config.set('ui.theme', 'dark')
config.save_config()

# Load theme on startup
theme_name = config.get('ui.theme', 'dark')
theme_manager.load_theme(theme_name)
```

---

## Widget Styling Reference

### QGroupBox

```python
# Themed styling
colors = get_theme_colors()
stylesheet = f"""
    QGroupBox {{
        background-color: {colors.CARD_BG};
        border: 1px solid {colors.BORDER};
        border-radius: 8px;
        margin-top: 12px;
        padding: 16px;
        font-size: 14px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: {colors.TEXT_PRIMARY};
    }}
"""
group.setStyleSheet(stylesheet)
```

### QScrollArea

```python
# Styled scrollbars
stylesheet = f"""
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
        background: {colors.ACCENT_HOVER};
    }}
"""
scroll.setStyleSheet(stylesheet)
```

### QTableWidget

```python
# Professional table styling
stylesheet = f"""
    QTableWidget {{
        background-color: {colors.CARD_BG};
        alternate-background-color: {colors.CARD_HOVER};
        border: 1px solid {colors.BORDER};
        gridline-color: {colors.DIVIDER};
        selection-background-color: {colors.ACCENT_PRIMARY};
        selection-color: white;
        font-size: 11px;
    }}
    QTableWidget::item {{
        padding: 4px 8px;
    }}
    QHeaderView::section {{
        background-color: {colors.ELEVATED_BG};
        color: {colors.TEXT_PRIMARY};
        border: 1px solid {colors.BORDER};
        padding: 6px;
        font-weight: 600;
        font-size: 11px;
    }}
"""
table.setStyleSheet(stylesheet)
```

### QTabWidget

```python
# Modern tabs
stylesheet = f"""
    QTabWidget::pane {{
        border: 1px solid {colors.BORDER};
        border-radius: 4px;
        background-color: {colors.CARD_BG};
    }}
    QTabBar::tab {{
        background-color: {colors.ELEVATED_BG};
        color: {colors.TEXT_SECONDARY};
        border: 1px solid {colors.BORDER};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 16px;
        margin-right: 2px;
        font-size: 12px;
    }}
    QTabBar::tab:selected {{
        background-color: {colors.CARD_BG};
        color: {colors.TEXT_PRIMARY};
        border-bottom: 2px solid {colors.ACCENT_PRIMARY};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {colors.CARD_HOVER};
    }}
"""
tabs.setStyleSheet(stylesheet)
```

---

## Panel Categories

Panels are organized into categories for menu and organizational purposes:

### PanelCategory Enum

```python
class PanelCategory(Enum):
    VIEWER = "viewer"               # 3D viewer controls
    PROPERTY = "property"            # Property selection/filtering
    SCENE = "scene"                  # Scene settings, layers
    LAYER = "layer"                  # Layer management
    INFO = "info"                    # Information display
    ANALYSIS = "analysis"            # General analysis
    SELECTION = "selection"          # Block selection tools
    CROSS_SECTION = "cross_section"  # Cross-section tools
    DISPLAY = "display"              # Display settings
    RESOURCE = "resource"            # Resource estimation
    PLANNING = "planning"            # Mine planning
    GEOSTATS = "geostats"            # Geostatistics (kriging, simulation)
    DRILLHOLE = "drillhole"          # Drillhole operations
    ESG = "esg"                      # Environmental/social/governance
    GEOTECH = "geotech"              # Geotechnical analysis
    OPTIMIZATION = "optimization"    # Optimization engines
    CHART = "chart"                  # Charting/visualization
    REPORT = "report"                # Report generation
    CONFIG = "config"                # Configuration panels
    OTHER = "other"                  # Miscellaneous
```

### Panel Organization in Menus

Panels can be accessed via:

1. **Panels Menu** - Organized by category
   ```
   Panels
   ├── Viewer
   ├── Property
   ├── Scene
   ├── Geostats
   │   ├── Kriging Panel
   │   ├── Variogram Panel
   │   └── SGSIM Panel
   ├── Drillholes
   └── ...
   ```

2. **Keyboard Shortcuts** - Direct access
   - `Ctrl+K` - Kriging Panel
   - `Ctrl+V` - Variogram Panel
   - `Ctrl+D` - Display Settings

3. **Context Menus** - Right-click in 3D view, data tables, etc.

---

## Best Practices

### 1. Panel Development

**Do**:
- ✅ Inherit from appropriate base class (`BasePanel`, `BaseAnalysisPanel`)
- ✅ Implement all required template methods
- ✅ Use `get_registry()` for data access
- ✅ Implement `refresh_theme()` for theme switching
- ✅ Use `CollapsibleGroup` for section organization
- ✅ Add docstrings to all public methods
- ✅ Use consistent naming: `_on_button_clicked()` for slots
- ✅ Block signals when programmatically updating widgets
- ✅ Provide clear error messages with validation

**Don't**:
- ❌ Access MainWindow directly (use controller/registry)
- ❌ Hardcode colors (use `ModernColors` or theme functions)
- ❌ Hardcode font sizes (use standard sizes: 12, 14, 16)
- ❌ Forget to call `super().__init__()` in constructors
- ❌ Create panels in __init__ methods (use lazy loading)
- ❌ Perform long operations in UI thread (use controller tasks)

### 2. Styling

**Do**:
- ✅ Use `modern_styles.py` template functions
- ✅ Access colors via `ModernColors` or `get_theme_colors()`
- ✅ Implement `refresh_theme()` in all custom widgets
- ✅ Use consistent border-radius (4-8px)
- ✅ Apply hover effects for interactive elements
- ✅ Test in both light and dark themes

**Don't**:
- ❌ Hardcode hex colors in stylesheets
- ❌ Use inline styles without theme awareness
- ❌ Forget to update child widgets in `refresh_theme()`
- ❌ Mix QSS and QPalette approaches (stick to QSS)

### 3. Responsiveness

**Do**:
- ✅ Use size policies for responsive layouts
- ✅ Set minimum sizes to prevent squashing
- ✅ Use scroll areas for content that might overflow
- ✅ Test at different window sizes (1280x720 to 4K)
- ✅ Support both docked and floating modes

**Don't**:
- ❌ Use fixed sizes except for buttons/icons
- ❌ Assume large screen (design for 1280x720 minimum)
- ❌ Nest too many scroll areas (1-2 levels max)

### 4. User Experience

**Do**:
- ✅ Provide immediate feedback (progress bars, status messages)
- ✅ Show validation errors inline (near the field)
- ✅ Preserve state when panel is closed (hide-on-close)
- ✅ Auto-save user preferences
- ✅ Support keyboard navigation (tab order, shortcuts)
- ✅ Use tooltips for non-obvious controls

**Don't**:
- ❌ Block UI during long operations (use threading)
- ❌ Show cryptic error messages ("Error 42")
- ❌ Reset panel state on every open
- ❌ Require multi-step actions without guidance

### 5. Performance

**Do**:
- ✅ Lazy-load panels (create on first show)
- ✅ Cache expensive computations
- ✅ Use QTimer.singleShot for deferred updates
- ✅ Batch signal updates (block signals during bulk changes)
- ✅ Profile performance for data-heavy operations

**Don't**:
- ❌ Create all panels at startup (100+ panels = slow)
- ❌ Update UI on every data point (batch updates)
- ❌ Perform file I/O in UI thread
- ❌ Rebuild entire UI on every theme change (update colors only)

### 6. Testing

**Do**:
- ✅ Test with no data loaded (EMPTY state)
- ✅ Test with real-world data sizes (millions of blocks)
- ✅ Test theme switching while panel is open
- ✅ Test panel close/reopen (state preservation)
- ✅ Test keyboard shortcuts
- ✅ Test on different screen sizes

**Don't**:
- ❌ Test only with sample data
- ❌ Assume UI thread operations complete instantly
- ❌ Skip edge cases (empty lists, null values, huge numbers)

---

## Conclusion

The GeoX panel architecture provides a solid foundation for building professional, maintainable geological analysis tools. By following the patterns and standards outlined in this document, developers can create panels that:

- **Look Professional**: Consistent styling, modern appearance
- **Feel Responsive**: Smooth animations, immediate feedback
- **Work Reliably**: Error handling, validation, state management
- **Integrate Seamlessly**: Standard interfaces, signal/slot patterns
- **Scale Well**: Modular design, lazy loading, performance optimization

For specific implementation examples, refer to existing panels:
- **Simple Panel**: `property_panel.py`, `block_info_panel.py`
- **Analysis Panel**: `kriging_panel.py`, `variogram_panel.py`
- **Complex Panel**: `loopstructural_panel.py`, `esg_dashboard_panel.py`
- **Dashboard Panel**: `production_dashboard_panel.py`

For theme and styling questions, see:
- `modern_styles.py` - Stylesheet templates
- `theme_manager.py` - Theme loading/switching
- `collapsible_group.py` - Reusable UI component

---

**Document maintained by**: GeoX Development Team  
**Last updated**: February 21, 2026  
**Questions or suggestions**: Contact technical lead
