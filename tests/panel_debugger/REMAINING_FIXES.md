# Remaining Silent Exceptions to Fix

## Quick Reference Guide

Run this command to get the complete list with exact line numbers:

```bash
cd c:/Users/chauk/Documents/GeoX_Clean
python -m pytest tests/panel_debugger/tests/test_signals.py::TestSilentExceptionDetection::test_no_silent_exceptions 2>&1 | grep -E "❌|File:|line" > remaining_issues.txt
```

## Fix Pattern (Copy & Paste)

For **each** instance found, apply this fix:

### Before (BROKEN):
```python
except Exception:
    pass
```

### After (FIXED):
```python
except Exception as e:
    logger.error(f"[Description]: {e}", exc_info=True)
```

Replace `[Description]` with what the code was trying to do. Examples:
- `"Failed to load variogram data"`
- `"Failed to connect signal"`
- `"Failed to update UI"`

## Files Likely to Have Issues (Based on Test Results)

1. **`block_model_viewer/ui/variogram_panel.py`** - ~11 instances
2. **`block_model_viewer/ui/kriging_panel.py`** - ~4 instances
3. **`block_model_viewer/ui/sgsim_panel.py`** - ~2 instances
4. **`block_model_viewer/ui/display_settings_panel.py`** - ~2 instances
5. **`block_model_viewer/ui/scene_inspector_panel.py`** - ~3 instances
6. **`block_model_viewer/ui/cross_section_panel.py`** - ~3 instances
7. **`block_model_viewer/ui/drillhole_plotting_panel.py`** - ~1 instance
8. **`block_model_viewer/ui/block_info_panel.py`** - ~1 instance
9. **`block_model_viewer/ui/data_viewer_panel.py`** - ~1 instance
10. **`block_model_viewer/ui/swath_panel.py`** - ~1 instance
11. **`block_model_viewer/ui/uncertainty_panel.py`** - ~1 instance
12. **`block_model_viewer/ui/resource_reporting_panel.py`** - ~1 instance

## Automated Fix Commands

### Option 1: Fix One Panel at a Time
```bash
# Open the file in your editor:
code block_model_viewer/ui/variogram_panel.py

# Search for: except Exception:
# Replace with: except Exception as e:
#               logger.error(f"Description: {e}", exc_info=True)
```

### Option 2: Use Find & Replace Carefully

**WARNING:** Only use on files you've reviewed!

```python
# DO NOT blindly replace all!
# Review each instance to add appropriate error message
```

## Verification

After fixing each file, re-run the test:

```bash
python -m pytest tests/panel_debugger/tests/test_signals.py::TestSilentExceptionDetection -v
```

The count should decrease with each fix!

## Goal

- **Current:** 29 silent exceptions
- **Target:** 0 silent exceptions
- **Priority:** Fix panels you actively use first

## Notes

- ✅ **DrillholeControlPanel** - ALREADY FIXED
- ✅ **PropertyPanel** - ALREADY FIXED
- ✅ **dock_setup.py** - ALREADY FIXED
- ⏳ **28-29 remaining** - Need systematic fixes
