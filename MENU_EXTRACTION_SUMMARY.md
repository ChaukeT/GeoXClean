# Menu Extraction Summary

## ✅ Completed Extractions

**Date:** 2026-01-24  
**MainWindow Before:** ~15,500 lines  
**MainWindow After:** ~15,137 lines  
**Reduction:** ~363 lines (2.3%)

### Extracted Menus (9 menus)

1. ✅ **File Menu** (`ui/menus/file_menu.py`) - ~115 lines extracted
2. ✅ **Search Menu** (`ui/menus/search_menu.py`) - ~10 lines extracted
3. ✅ **Edit Menu** (`ui/menus/edit_menu.py`) - ~25 lines extracted
4. ✅ **Survey Menu** (`ui/menus/survey_menu.py`) - ~46 lines extracted
5. ✅ **Scan Menu** (`ui/menus/scan_menu.py`) - ~38 lines extracted
6. ✅ **Tools Menu** (`ui/menus/tools_menu.py`) - ~53 lines extracted
7. ✅ **Data Menu** (`ui/menus/data_menu.py`) - ~40 lines extracted
8. ✅ **Drillholes Menu** (`ui/menus/drillholes_menu.py`) - ~61 lines extracted
9. ✅ **Workbench Menu** (`ui/menus/workbench_menu.py`) - ~29 lines extracted
10. ✅ **Workflows Menu** (`ui/menus/workflows_menu.py`) - ~34 lines extracted

**Total Lines Extracted:** ~451 lines

## ⏳ Remaining Menus to Extract

The following menus are still in MainWindow and should be extracted:

1. **View Menu** (~195 lines) - Large menu with view presets, panels, lighting, themes
2. **Mouse Menu** (~75 lines) - Mouse interaction modes
3. **Panels Menu** (~20 lines) - Dock widget toggles
4. **Geology Menu** (~20 lines) - LoopStructural modeling
5. **Resources Menu** (~30 lines) - Resource calculations
6. **Estimations Menu** (~192 lines) - **LARGE** - Variograms, kriging, simulations
7. **Geotech Menu** (~10 lines) - Geotechnical dashboard
8. **Mine Planning Menu** (~80 lines) - Pit optimization, scheduling
9. **ML Menu** (~10 lines) - Machine learning
10. **Dashboards Menu** (~30 lines) - Various dashboards
11. **Help Menu** (~20 lines) - Documentation, about

**Estimated Remaining:** ~682 lines

## Pattern Established

All extracted menus follow this pattern:

```python
# ui/menus/<menu_name>_menu.py
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

def build_<menu>_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the <Menu> menu."""
    menu = menubar.addMenu("&<Menu>")
    # ... add actions ...
    return menu
```

## MainWindow Integration

MainWindow now calls menu builders:

```python
from .menus import (
    build_file_menu, build_search_menu, build_edit_menu,
    build_survey_menu, build_scan_menu, build_tools_menu,
    build_data_menu, build_drillholes_menu, build_workbench_menu,
    build_workflows_menu
)

def _setup_menus(self):
    menubar = self.menuBar()
    build_file_menu(self, menubar)
    build_search_menu(self, menubar)
    # ... etc
```

## Next Steps

### Priority 1: Large Menus
1. **Estimations Menu** (~192 lines) - Highest priority
2. **View Menu** (~195 lines) - Second priority

### Priority 2: Medium Menus
3. **Mine Planning Menu** (~80 lines)
4. **Mouse Menu** (~75 lines)

### Priority 3: Small Menus
5. Remaining small menus (~150 lines total)

## Target

**Goal:** Reduce MainWindow to 600-900 lines  
**Current:** 15,137 lines  
**Remaining:** ~14,237 lines to extract

**Note:** This is just menu extraction. Additional refactoring needed for:
- Dock setup (~300 lines)
- Action handlers (~2,000 lines)
- Other responsibilities

## Files Created

```
block_model_viewer/ui/menus/
├── __init__.py
├── file_menu.py
├── search_menu.py
├── edit_menu.py
├── survey_menu.py
├── scan_menu.py
├── tools_menu.py
├── data_menu.py
├── drillholes_menu.py
├── workbench_menu.py
└── workflows_menu.py
```

## Verification

- ✅ No linting errors
- ✅ All imports resolved
- ✅ Pattern consistent across all menus
- ✅ MainWindow calls menu builders correctly
- ⚠️ Some menus still need extraction (marked with TODO comments)

