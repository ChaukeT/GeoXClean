# MainWindow Refactoring Pattern

## ✅ Proof of Concept: Survey Menu Extraction

**Status:** COMPLETE  
**Date:** 2026-01-24  
**Lines Removed from MainWindow:** ~46 lines  
**New File Created:** `block_model_viewer/ui/menus/survey_menu.py` (95 lines)

## What Was Done

### 1. Created Menu Module Structure

```
block_model_viewer/ui/menus/
├── __init__.py          # Exports menu builders
└── survey_menu.py       # Survey menu implementation
```

### 2. Extracted Survey Menu

**Before:** Survey menu code lived in `MainWindow._setup_menus()` (lines 1089-1134)

**After:** 
- Menu construction moved to `ui/menus/survey_menu.py`
- Exports single function: `build_survey_menu(main_window, menubar)`
- MainWindow now calls: `build_survey_menu(self, menubar)`

### 3. Pattern Established

**Menu Module Contract:**
- Each menu file exports ONE function: `build_<menu>_menu(main_window, menubar)`
- Function creates menu, adds actions, returns QMenu
- Actions connect to methods on `main_window` (keeps handlers in MainWindow for now)
- No business logic in menu files - only UI construction

## File Size Impact

**MainWindow Before:** ~15,500 lines  
**MainWindow After:** ~15,454 lines  
**Reduction:** 46 lines (0.3%)

**Note:** This is just the first menu. Full refactor will remove 1,500+ lines.

## Pattern for Future Menus

### Step 1: Create Menu File

```python
# ui/menus/file_menu.py
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

def build_file_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the File menu."""
    file_menu = menubar.addMenu("&File")
    
    # Add actions...
    open_action = QAction("Open...", main_window)
    open_action.triggered.connect(main_window.open_file)
    file_menu.addAction(open_action)
    
    return file_menu
```

### Step 2: Update MainWindow

```python
# In MainWindow._setup_menus()
from .menus.file_menu import build_file_menu
build_file_menu(self, menubar)
```

### Step 3: Export in __init__.py

```python
# ui/menus/__init__.py
from .file_menu import build_file_menu
from .survey_menu import build_survey_menu

__all__ = ['build_file_menu', 'build_survey_menu']
```

## Next Steps (Recommended Order)

### Phase 1: Extract All Menus (Highest ROI)
1. ✅ Survey menu (DONE)
2. ⏭️ File menu (~200 lines)
3. ⏭️ View menu (~150 lines)
4. ⏭️ Data menu (~180 lines)
5. ⏭️ Drillholes menu (~250 lines)
6. ⏭️ Estimations menu (~200 lines)
7. ⏭️ Scan menu (~100 lines)
8. ⏭️ Tools menu (~150 lines)
9. ⏭️ Workbench menu (~100 lines)
10. ⏭️ Workflows menu (~100 lines)

**Estimated Reduction:** ~1,500 lines

### Phase 2: Extract Dock Setup
- Create `ui/layout/dock_setup.py`
- Move `_setup_docks()` logic
- Move `_setup_toolbar()` logic

**Estimated Reduction:** ~300 lines

### Phase 3: Extract Action Handlers (Optional)
- Create `ui/actions/` directory
- Move handler methods to action classes
- MainWindow delegates to action handlers

**Estimated Reduction:** ~2,000 lines

## Target Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| MainWindow lines | ~15,500 | 600-900 | 🔴 |
| Menu files | 0 | 10 | 🟡 (1/10) |
| Max menu file size | N/A | 150-300 | ✅ |
| Responsibilities per file | 8+ | 1-2 | 🔴 |

## Rules Going Forward

### ✅ DO:
- Extract menus to `ui/menus/`
- Keep menu files under 300 lines
- One function per menu file
- Connect actions to MainWindow methods (for now)

### ❌ DON'T:
- Add new menus directly in MainWindow
- Put business logic in menu files
- Create menus > 300 lines
- Break existing functionality

## Verification

- ✅ No linting errors
- ✅ Imports resolved correctly
- ✅ Survey menu functionality preserved
- ✅ Pattern documented and repeatable

## Notes

- Action handlers remain in MainWindow for now (can extract later)
- Menu files are pure UI construction (no logic)
- Pattern scales to all menus
- Each extraction is independent and safe

