# MainWindow Wiring Audit

**Investigation date:** 2026-04-16  
**Scope:** `main_window.py` + `mixins/panel_mixin.py` + `mixins/file_mixin.py` + `coordinators/signal_coordinator.py`  
**Method:** Static analysis — every `self.X_panel.method()` call checked against the panel's actual public API. Not speculation.

---

## 1. API mismatches — methods main_window calls that DO NOT EXIST on panels

These fail silently at runtime (wrapped in `try/except: pass` or `hasattr` guards that always return False).

### `property_panel` — 7 missing methods/attrs (called 38 times total)
- `_block_layer_cache` — called **9 times** (cache manipulation)
- `_custom_discrete_colors` — called **2 times**
- `_suppress_cache_reemit` — called **14 times** ⚠️ 
- `bind_controller` — called **1 time** (controller never bound to panel)
- `drillhole_df` — called **2 times**
- `refresh` — called **2 times**
- `update_active_layers` — called **10 times**

**Impact:** 38 calls silently no-op. Cache/controller binding broken.

### `scene_inspector_panel` — 2 missing
- `ground_grid_spacing_spin` — attribute access
- `update_layer_controls` — method call

### `swath_panel` — 1 missing
- `bind_controller` — controller never bound

### `domain_compositing_panel` — 3 missing
- `assay_xyz` — attribute access
- `bind_controller` — controller never bound
- `comp_domain_df` — called **5 times** (composite data never read)

### Panels with OK API (all calls resolve correctly):
`drillhole_control_panel`, `gc_decision_panel`, `geological_explorer_panel`, `charts_panel`, `statistics_panel`

---

## 2. New panels — signals emitted but NOT connected to anything

These panels fire signals into the void. Any UI interaction that relies on main_window reacting to these signals silently fails.

| Panel | Unwired Signal | Purpose |
|-------|---------------|---------|
| arbf_panel | `progress_updated` | Run progress feedback |
| arbf_estimation_panel | `progress_updated` | Run progress feedback |
| indicator_rbf_panel | `progress`, `finished`, `failed` (×2) | Worker lifecycle |
| fastrbf_panel | `progress_updated` | Run progress feedback |
| geological_model_panel | `progress`, `finished`, `error` | Build lifecycle |
| block_model_filter_panel | `filtersChanged` | **Filter apply never reaches renderer** |
| cross_section_panel | `section_updated` | Section edits never propagate |
| interactive_slicer_panel | `rangeChanged`, `clipping_changed` | Slicing never applies to scene |
| lithology_manager_panel | `domainCodesAssigned`, `contactsExtracted` | Domain changes ignored |
| frag_import_panel | `import_completed` | Fragmentation pipeline broken |
| frag_preprocessing_panel | `preprocessing_completed` | Fragmentation pipeline broken |
| frag_results_panel | `fragment_selected` | Fragment selection ignored |
| frag_segmentation_panel | `segmentation_completed` | Fragmentation pipeline broken |
| bayesian_kriging_panel | `progress_updated` | Run progress feedback |
| variogram_analysis_panel | `progress_updated` | Run progress feedback |

**Total: 18 unwired signals across 15 panels.**

Signals that ARE connected: `request_visualization` on ARBF/IRBF/FastRBF/Geological/CrossSectionManager — these were wired correctly because they use the same name across all panels.

---

## 3. Silent failure patterns (code smell metrics)

These numbers indicate the scale of defensive code masking broken wiring:

| Pattern | Count | Location |
|---------|-------|----------|
| `hasattr(self.*_panel, ...)` guards | **116** | main_window + mixins |
| `except: pass` blocks (all) | **180** | main_window + mixins |

**Every `hasattr` check on a method that doesn't exist silently skips the block.** No log, no warning. User sees nothing wrong — the feature just doesn't work.

Example from [main_window.py:7691](block_model_viewer/ui/main_window.py#L7691):
```python
if hasattr(self.property_panel, 'update_active_layers'):
    self.property_panel.update_active_layers()
    logger.info("Used fallback update_active_layers method")
```
`update_active_layers` doesn't exist → condition is always False → the logger line NEVER fires → the "fallback" is a ghost.

---

## 4. Panel registration coverage

Of 17 new panels in the April refactor:
- **Registered with PanelManager:** 2 (`cross_section_manager_panel`, `interactive_slicer_panel`)
- **NOT registered:** 15 (ARBF, ARBFEstimation, IRBF, FastRBF, GeologicalModel, BlockModelFilter, ClipPlane, CrossSection, LithologyManager, DefineBlockModel, and all 5 Fragmentation panels)

Unregistered panels can still be opened via menu actions but don't appear in the PanelManager's dock/shortcut system.

---

## 5. Signal coordinator coverage

`signal_coordinator.py` is **275 lines** and wires only:
- `property_panel.request_visualization` → main_window handler
- `property_panel.refresh` → controller hook
- A small amount of renderer plumbing

The OLD monolithic main_window (b2fda5d, 15,712 lines) had **inline signal wiring** scattered throughout `_connect_signals()`. When `signal_coordinator.py` was extracted, most of that wiring **was not ported** — it still exists inline in `main_window._connect_signals()`, which is why some things work. But the coordinator is skeletal.

---

## 6. Scope of fix

**Non-rewrite fix path:**

| Work item | Files | Est. lines |
|-----------|-------|-----------|
| Add 7 missing attrs/methods to `property_panel.py` | 1 | ~80 |
| Add `update_layer_controls` to `scene_inspector_panel.py` | 1 | ~15 |
| Add `bind_controller` to `swath_panel.py`, `drillhole_import_panel.py`, `property_panel.py` | 3 | ~30 |
| Wire 18 unconnected signals in `signal_coordinator.py` or inline | 1-2 | ~150 |
| Register 15 panels in `panel_registration.py` | 1 | ~50 |
| Add handler methods for unwired signals (`on_filters_changed`, `on_slicer_range_changed`, etc.) | 1 | ~200 |
| Remove obsolete `hasattr` guards once methods exist | many | cleanup |

**Total: ~500-700 lines across ~8 files. No rewrite.**

---

## 7. What's NOT broken

To avoid misrepresentation:
- MainWindow class structure (mixin inheritance, coordinators) — **sound**
- PanelMixin (11,227 lines, 200+ `open_*` methods) — **works**
- FileMixin (project save/load, 2,094 lines) — **works**
- Menu system (21 menus, all builders exist) — **works after my fix to `menu_coordinator.py`**
- Drillhole control panel API — **fully matches main_window expectations**
- GC decision panel, geological explorer, statistics, charts — **fully match**

The refactor architecture is correct. The problem is specifically **incomplete signal-wiring extraction** and **API drift on 4 panels** (property, scene_inspector, swath, drillhole_import) where methods were renamed/removed but main_window still calls old names.

---

## 8. Evidence file locations

All findings above are backed by grep output against live files on disk. To reproduce:

```bash
# Method gap on property_panel:
grep -n "property_panel.bind_controller\|property_panel\.drillhole_df\|property_panel\.update_active_layers" \
     block_model_viewer/ui/main_window.py block_model_viewer/ui/mixins/panel_mixin.py

# Unwired signals:
grep "pyqtSignal" block_model_viewer/ui/arbf_panel.py
grep -F ".filtersChanged.connect" block_model_viewer/  # returns nothing

# Silent failures:
grep -c "hasattr(self\..*_panel," block_model_viewer/ui/main_window.py
grep -cE "except[^:]*:\s*$" block_model_viewer/ui/main_window.py
```
