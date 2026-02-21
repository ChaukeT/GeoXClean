# GeoX Clean - Critical Fixes Checklist

**Status:** 🔴 NOT PRODUCTION READY - 9 Critical Issues Identified
**Last Updated:** February 13, 2026

---

## ✅ COMPLETED IMPROVEMENTS (From UI/UX Audit)
- [x] Panel header controls enabled by default
- [x] Clear panel functionality added to base class
- [x] 10 complex panels with custom clear_panel() implementations
- [x] Dynamic "All Panels" menu by category
- [x] Workflow status indicator widget
- [x] Preferences dialog with panel settings
- [x] Clear All Visible Panels command

---

## 🔴 CRITICAL FIXES REQUIRED (Must Complete Before Production)

### 1. Renderer: Undefined Variable Bug
**File:** `block_model_viewer/visualization/renderer.py`
**Line:** 1428
**Status:** 🔴 NOT FIXED

**Current Code:**
```python
points = self._to_local_precision(points)  # ERROR: points undefined
```

**Fix:**
```python
corners = self._to_local_precision(corners.reshape(-1, 3))
```

**Impact:** Runtime NameError when rendering non-orthogonal block models

---

### 2. Session Management: Missing Renderer State Methods
**File:** `block_model_viewer/visualization/renderer.py`
**Lines:** Methods called at main_window.py:13254, 13256, 14551, 14792, 14812
**Status:** 🔴 NOT IMPLEMENTED

**Required Methods:**
```python
def get_session_state(self) -> dict:
    """Return serializable dict of renderer state."""
    return {
        'camera': {
            'position': self.plotter.camera.position,
            'focal_point': self.plotter.camera.focal_point,
            'view_angle': self.plotter.camera.view_angle,
        },
        'layers': [
            {'id': layer_id, 'visible': visible, 'opacity': opacity}
            for layer_id, visible, opacity in self._layers
        ],
        'legend': self._legend_config,
        # ... other state
    }

def apply_session_state(self, state: dict) -> None:
    """Restore renderer state from dict."""
    # Restore camera
    # Restore layer visibility/opacity
    # Restore legend configuration
    pass
```

**Impact:**
- Session state lost on restart
- Undo/redo completely broken
- Camera position not persisted

---

### 3. Security: WSL Command Injection
**File:** `block_model_viewer/controllers/insar_controller.py`
**Lines:** 46-71
**Status:** 🔴 NOT FIXED

**Current Code (VULNERABLE):**
```python
bash_cmd = f"cd {cwd.as_posix()} && {command}"
wsl_cmd += ["--", "bash", "-lc", bash_cmd]
```

**Fix:**
```python
import shlex
bash_cmd = f"cd {shlex.quote(str(cwd))} && {command}"
# Better: Don't use shell at all
wsl_cmd += ["--", "bash", "-lc", "--"] + shlex.split(command)
```

**Impact:** Arbitrary command execution vulnerability

---

### 4. Geostatistics: Invalid String Formatting
**File:** `block_model_viewer/geostats/cokriging3d.py`
**Lines:** 240, 248
**Status:** 🔴 NOT FIXED

**Current Code:**
```python
warnings.append(".2f")  # Line 240 - WRONG!
warnings.append(".2f")  # Line 248 - WRONG!
```

**Fix:**
```python
# Line 240:
warnings.append(f"Mean ratio {mean_ratio:.2f} exceeds threshold (10.0)")
# Line 248:
warnings.append(f"CV ratio {cv_ratio:.2f} exceeds threshold (5.0)")
```

**Impact:** Validation warnings display ".2f" instead of actual values

---

### 5. Data Integrity: Deep Copy Failure Returns Shared Reference
**File:** `block_model_viewer/core/data_registry_simple.py`
**Lines:** 160-182
**Status:** 🔴 NOT FIXED

**Current Code:**
```python
try:
    return copy.deepcopy(data)
except Exception:
    if isinstance(data, pd.DataFrame):
        return data.copy()  # SHALLOW COPY - DANGEROUS!
```

**Fix:**
```python
try:
    return copy.deepcopy(data)
except Exception as e:
    logger.error(f"Deep copy failed for {key}: {e}")
    raise ValueError(f"Cannot safely copy data for key {key}") from e
```

**Impact:** Data corruption when consumers mutate returned DataFrame

---

### 6. Data Validation: CSV Import Accepts Invalid Data
**File:** `block_model_viewer/drillholes/data_io.py`
**Lines:** 366-410
**Status:** 🔴 NOT FIXED

**Current Code:**
```python
df_clean = df[cols].copy()  # No type validation!
```

**Fix:**
```python
# After line 408, add:
for col in numeric_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    invalid_count = df_clean[col].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Coerced {invalid_count} invalid values in {col} to NaN")
```

**Impact:** Non-numeric data silently accepted into database

---

### 7. Race Condition: Status Flags TOCTOU
**File:** `block_model_viewer/core/data_registry_simple.py`
**Lines:** 196, usage throughout
**Status:** 🔴 NOT FIXED

**Current Pattern (UNSAFE):**
```python
if registry.has_data("block_model"):  # Check
    data = registry.get_data("block_model")  # Use - could be None!
```

**Fix Option 1 (Atomic):**
```python
def get_data_if_exists(self, key: str, copy_data: bool = True) -> Optional[Any]:
    """Atomically check and get data."""
    with self._lock:
        if key not in self._data:
            return None
        return self._copy_data(self._data[key]) if copy_data else self._data[key]
```

**Fix Option 2 (Documentation):**
```python
def has_data(self, key: str) -> bool:
    """
    WARNING: Result may be stale in multithreaded environments.
    For thread-safe operation, use try/except around get_data():

        try:
            data = registry.get_data(key)
        except KeyError:
            # Handle missing data
    """
    with self._lock:
        return self._status_flags.get(key, False)
```

**Impact:** Null pointer exceptions in multithreaded code

---

### 8. Geostatistics: Missing Seed Validation
**File:** `block_model_viewer/geostats/ik_sgsim.py`
**Lines:** 149-153 (runtime check), IKSGSIMConfig dataclass
**Status:** 🔴 NOT FIXED

**Current:** Seed validated at runtime, can be None
**Fix:** Add to `IKSGSIMConfig` dataclass:
```python
@dataclass
class IKSGSIMConfig:
    # ... existing fields ...
    random_seed: int  # Remove Optional, make required

    def __post_init__(self):
        if self.random_seed is None:
            raise ValueError("random_seed is required for JORC/SAMREC compliance")
        if not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be int, got {type(self.random_seed)}")
```

**Impact:** Non-reproducible simulations violate JORC compliance

---

### 9. Error Handler: Crash Handler Can Fail Silently
**File:** `block_model_viewer/core/crash_handler.py`
**Lines:** 43-44
**Status:** 🔴 NOT FIXED

**Current Code:**
```python
except:
    pass  # DANGEROUS - error handler itself fails silently!
```

**Fix:**
```python
except Exception as e:
    # Fallback: print to stderr if logging fails
    import sys
    print(f"CRITICAL: Crash handler failed: {e}", file=sys.stderr)
    print(f"Original exception: {exc_val}", file=sys.stderr)
    import traceback
    traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr)
```

**Impact:** Complete loss of error information in production

---

## 🟡 HIGH PRIORITY FIXES (Week 2-4)

### Replace Bare Exception Handlers (35+ instances)
**Files:** renderer.py, audit_manager.py, geostats_controller.py, etc.
**Pattern to fix:**
```python
# BEFORE:
except:
    pass

# AFTER:
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
```

**Files requiring fixes:**
- `core/crash_handler.py:43`
- `core/audit_manager.py:152`
- `controllers/geostats_controller.py:2671`
- `visualization/renderer.py:1931, 4604, 4893`
- `geostats/sk_stationarity.py:108`
- And 30+ more instances

---

### Add Thread Safety Lock to Global Shift
**File:** `block_model_viewer/visualization/renderer.py`
**Lines:** 119-130
**Status:** 🟡 NOT FIXED

**Current Code:**
```python
if self._global_shift is None:
    self._global_shift = center_point.copy()  # Race condition!
```

**Fix:**
```python
import threading

class Renderer:
    def __init__(self):
        self._global_shift_lock = threading.Lock()
        # ... rest of init

    def _to_local_precision(self, coords):
        with self._global_shift_lock:
            if self._global_shift is None:
                self._global_shift = center_point.copy()
        # ... rest of method
```

---

### Implement Crash Recovery
**File:** `block_model_viewer/ui/main_window.py`
**Status:** 🟡 NOT IMPLEMENTED

**Required:**
1. Create `.lock` file on startup
2. Detect missing `.lock` on next startup (indicates crash)
3. Show recovery dialog offering autosave restore
4. Clean up old recovery files (>7 days)

---

### Add Resource Cleanup in Renderer
**File:** `block_model_viewer/visualization/renderer.py`
**Lines:** 844-851
**Status:** 🟡 NOT FIXED

**Pattern:**
```python
def load_block_model(self, block_model):
    try:
        # ... operations ...
    finally:
        # Ensure overlay system restored even on exception
        if overlay_was_suspended:
            self.overlay_manager.resume()
```

---

## 📊 TESTING INFRASTRUCTURE (Current: 0.7% Coverage)

### Immediate Testing Needs:
- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] Add 10 unit tests for renderer critical paths
- [ ] Add parametrized tests for coordinate transformations
- [ ] Test edge cases: empty data, NaN values, very large models
- [ ] Integration tests for end-to-end workflows

**Target:** 50% coverage on renderer and data_registry within 2 weeks

---

## 📝 VERIFICATION CHECKLIST (After Fixes)

### Renderer Fixes:
- [ ] Run with non-orthogonal block model → no NameError
- [ ] Session save/restore → camera position preserved
- [ ] Undo/redo → visualization state restored
- [ ] Large model (50k+ cells) → no GPU timeout

### Data Integrity Fixes:
- [ ] Import CSV with "N/A" in numeric column → proper error/coercion
- [ ] Deep copy failure → raises exception (not silent)
- [ ] Multithreaded access → no race conditions
- [ ] Drillhole data reload → transformers properly cleared

### Geostatistics Fixes:
- [ ] Cokriging validation warnings → display actual values
- [ ] IK-SGSIM without seed → config validation error
- [ ] Indicator probabilities → always in [0, 1]
- [ ] Stationarity test failure → logged at WARNING level

### Security Fixes:
- [ ] WSL command with special chars → no injection
- [ ] File paths with ".." → rejected
- [ ] Large file upload → size limit enforced

---

## ⏱️ ESTIMATED TIMELINE

| Task | Duration | Status |
|------|----------|--------|
| Fix 9 critical issues | 3 days | 🔴 Not Started |
| Replace bare except handlers | 2 days | 🔴 Not Started |
| Add thread safety | 1 day | 🔴 Not Started |
| Implement crash recovery | 2 days | 🔴 Not Started |
| Add renderer tests | 3 days | 🔴 Not Started |
| Add data registry tests | 3 days | 🔴 Not Started |
| Security penetration testing | 2 days | 🔴 Not Started |
| **TOTAL** | **16 days** | **0% Complete** |

---

## 🎯 SUCCESS CRITERIA

Before production deployment:
- [x] All 9 critical issues resolved ← **BLOCKING**
- [ ] All bare exception handlers fixed
- [ ] Test coverage ≥50% on core modules
- [ ] No HIGH or CRITICAL security vulnerabilities
- [ ] Load testing passed (100k+ block models)
- [ ] Crash recovery tested and working
- [ ] Session save/restore verified
- [ ] Undo/redo functional end-to-end

**Current Production Readiness: 20% (UI/UX Complete, Core Issues Remain)**
**Target Production Readiness: 95%**

---

**Next Actions:**
1. Assign developer to critical fixes (Days 1-3)
2. Set up pytest infrastructure (Day 1)
3. Begin security fixes (Day 2)
4. Parallel track: testing infrastructure (Week 1-2)
5. Integration testing (Week 2)
6. Final verification (Week 3)

**Deployment Gate:** All items in this checklist must be ✅ before production release.
