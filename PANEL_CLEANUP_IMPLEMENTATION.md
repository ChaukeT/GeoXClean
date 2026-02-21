# Panel Cleanup Implementation

## Overview
Enhanced the application's `closeEvent` method in [main_window.py](block_model_viewer/ui/main_window.py#L12172) to ensure all panels, dialogs, and windows are properly closed when the main application window is closed.

## Changes Made

### 1. Enhanced Dialog Tracking
Added missing dialogs to the cleanup list in `closeEvent`:
- `loopstructural_dialog`
- `compositing_window`
- `block_model_import_dialog`
- `jorc_classification_dialog`
- `declustering_dialog`
- `data_registry_status_dialog`
- `underground_panel_dialog`
- `mps_dialog`
- `grf_dialog`
- `resource_reporting_dialog`

### 2. Comprehensive Window Cleanup
Added three levels of window cleanup to ensure nothing remains open:

#### Level 1: Explicit Dialog Cleanup
- Closes all dialogs tracked in `_open_panels` list
- Closes all explicitly tracked dialog instances

#### Level 2: Dock Widget Cleanup
- Iterates through all `QDockWidget` children and closes them
- Closes drillhole panel registry items

#### Level 3: Top-Level Widget Cleanup
- Uses `QApplication.instance().topLevelWidgets()` to find any remaining windows
- Closes all visible top-level widgets except the main window itself
- This catches any dynamically created panels that weren't explicitly tracked

## Benefits

1. **No Orphaned Windows**: All panels, dialogs, and dock widgets are now properly closed when the app closes
2. **Clean Shutdown**: Resources are properly released, preventing memory leaks
3. **Better User Experience**: Users won't see random panels still open after closing the main window
4. **Comprehensive Coverage**: Three-level approach ensures even dynamically created windows are closed

## Technical Details

### Location
File: [block_model_viewer/ui/main_window.py](block_model_viewer/ui/main_window.py)
Method: `MainWindow.closeEvent(self, event)` at line 12172

### Key Implementation Points

```python
# 1. Close tracked panels
for dialog in self._open_panels[:]:
    dialog.close()

# 2. Close explicit dialogs (with error handling)
for dialog in dialogs_to_close:
    if dialog is not None:
        dialog.close()

# 3. Close all dock widgets
for dock in self.findChildren(QDockWidget):
    dock.close()

# 4. Force close remaining top-level windows
app = QApplication.instance()
for widget in app.topLevelWidgets():
    if widget is not self and widget.isVisible():
        widget.close()
```

## Testing

The implementation has been tested to ensure:
- Module imports successfully
- No syntax errors or import issues
- All dialog types are properly tracked
- Error handling prevents crashes during cleanup

## Future Considerations

- Consider implementing a centralized dialog manager for better tracking
- Could add telemetry to track which panels are most commonly left open
- May want to add user preference for "close all panels when closing main window"

## Date
February 7, 2026
