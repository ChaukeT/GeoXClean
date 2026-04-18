"""
Panel Utilities - Common functionality for all panels.

Provides:
- Close protection (confirmation dialog)
- Taskbar persistence (stay in taskbar when minimized)
- Window state management
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QWidget

logger = logging.getLogger(__name__)


def setup_panel_window_flags(widget: QWidget):
    """
    Setup window flags for a panel to behave like a software panel.
    
    - Stays in taskbar when minimized
    - Can be minimized/maximized
    - Non-modal behavior
    - Doesn't get deleted when closed
    
    Args:
        widget: QWidget, QDialog, or QMainWindow to configure
    """
    # Set window flags for proper minimize behavior (stay in taskbar)
    widget.setWindowFlags(
        Qt.WindowType.Window |
        Qt.WindowType.WindowMinimizeButtonHint |
        Qt.WindowType.WindowMaximizeButtonHint |
        Qt.WindowType.WindowCloseButtonHint
    )
    
    # Ensure non-modal behavior
    widget.setWindowModality(Qt.WindowModality.NonModal)
    
    # Prevent window from being deleted when closed or minimized
    widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)


def add_close_protection(
    widget: QWidget,
    has_unsaved_changes: Optional[Callable[[], bool]] = None,
    get_change_summary: Optional[Callable[[], str]] = None,
    save_callback: Optional[Callable[[], bool]] = None,
    panel_name: str = "Panel"
):
    """
    Add close protection to a panel widget.
    
    Shows a confirmation dialog if there are unsaved changes when the user
    tries to close the window.
    
    Args:
        widget: The widget to protect
        has_unsaved_changes: Optional function that returns True if there are unsaved changes
        get_change_summary: Optional function that returns a summary of changes
        save_callback: Optional function to save changes (returns True on success)
        panel_name: Name of the panel for dialog messages
    """
    original_close = getattr(widget, 'closeEvent', None)
    
    def protected_close_event(event):
        """Protected close event that checks for unsaved changes."""
        # Check if there are unsaved changes
        has_changes = False
        change_summary = ""
        
        if has_unsaved_changes:
            try:
                has_changes = has_unsaved_changes()
            except Exception as e:
                logger.warning(f"Error checking for unsaved changes: {e}")
                has_changes = False
        
        if get_change_summary:
            try:
                change_summary = get_change_summary()
            except Exception as e:
                logger.warning(f"Error getting change summary: {e}")
                change_summary = ""
        
        if has_changes:
            # Show confirmation dialog
            msg = QMessageBox(widget)
            msg.setWindowTitle(f"Close {panel_name}?")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(f"You have unsaved changes in the {panel_name}.")
            
            if change_summary:
                msg.setInformativeText(
                    f"Changes made:\n{change_summary}\n\n"
                    "What would you like to do?"
                )
            else:
                msg.setInformativeText("What would you like to do?")
            
            # Add buttons
            if save_callback:
                save_btn = msg.addButton("Save Changes", QMessageBox.ButtonRole.AcceptRole)
            hide_btn = msg.addButton("Hide Window", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = msg.addButton("Discard Changes", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            msg.setDefaultButton(cancel_btn)
            msg.exec()
            
            clicked = msg.clickedButton()
            
            if clicked == cancel_btn:
                # Cancel close
                event.ignore()
                return
            elif save_callback and clicked == save_btn:
                # Save changes first
                try:
                    if save_callback():
                        # After saving, hide the window (don't destroy it)
                        widget.hide()
                        event.ignore()
                    else:
                        # Save failed - ask again
                        retry = QMessageBox.question(
                            widget, "Save Failed",
                            f"Failed to save changes.\n\n"
                            "Do you want to close anyway?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                        if retry == QMessageBox.StandardButton.Yes:
                            widget.hide()
                            event.ignore()
                        else:
                            event.ignore()
                except Exception as e:
                    # If save fails, ask again
                    retry = QMessageBox.question(
                        widget, "Save Failed",
                        f"Failed to save changes:\n{str(e)}\n\n"
                        "Do you want to close anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if retry == QMessageBox.StandardButton.Yes:
                        widget.hide()
                        event.ignore()
                    else:
                        event.ignore()
                return
            elif clicked == hide_btn:
                # Hide window but keep state
                widget.hide()
                event.ignore()
                return
            elif clicked == discard_btn:
                # User confirmed they want to discard changes
                # Hide window (state is preserved in memory)
                widget.hide()
                event.ignore()
                return
        else:
            # No changes - just hide the window
            widget.hide()
            event.ignore()
    
    # Override closeEvent
    widget.closeEvent = protected_close_event


def resolve_variogram_for_variable(registry, variable_name: str, parent_widget: Optional[QWidget] = None):
    """Resolve variogram model/results for a given variable name from the registry.

    Used by ARBF, FastRBF, Bayesian kriging panels to fetch the appropriate
    variogram for the currently selected variable.

    Args:
        registry: DataRegistry instance.
        variable_name: Name of the variable (e.g. "Cu_NS", "FE").
        parent_widget: Optional parent for any error dialogs (unused currently).

    Returns:
        The variogram results dict/object matching the variable, or None if
        no matching variogram is found. When multiple variograms exist, the
        most recent one matching the variable is preferred; otherwise falls
        back to the most recent variogram regardless of variable.
    """
    if registry is None:
        return None

    # Try specific getter first
    try:
        if hasattr(registry, 'get_variogram_for_variable'):
            result = registry.get_variogram_for_variable(variable_name)
            if result is not None:
                return result
    except Exception as e:
        logger.debug(f"get_variogram_for_variable failed: {e}")

    # Fall back to generic variogram results and filter by variable name
    try:
        results = registry.get_variogram_results(copy_data=False) if hasattr(registry, 'get_variogram_results') else None
        if results is None:
            return None
        # Results may be a dict keyed by variable, or a single result dict
        if isinstance(results, dict):
            # If keyed by variable name
            for key in (variable_name, variable_name.upper(), variable_name.lower()):
                if key in results:
                    return results[key]
            # Or check if this is a single result for this variable
            if results.get('variable') == variable_name:
                return results
        # Otherwise return as-is — caller can inspect
        return results
    except Exception as e:
        logger.debug(f"Fallback variogram lookup failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# IRBF DOMAIN MASK — shared helper for sim / est panels
# ═══════════════════════════════════════════════════════════════════

def apply_irbf_domain_mask(
    values,
    xyz,
    registry,
    iso_value: Optional[float] = None,
    fill=None,
):
    """NaN-out cells outside the registered IRBF domain.

    The IRBF panel stores a probability volume + axes on the registry via
    :py:meth:`DataRegistry.register_indicator_rbf_domain`.  This helper reads
    that payload, nearest-neighbour-samples the probability volume at each
    ``xyz`` point, and writes ``fill`` (defaults to ``NaN``) wherever the
    probability is below ``iso_value`` (defaults to the iso_value stored with
    the domain, or 0.5).

    Args:
        values: ``(N,)`` numpy array of estimation/simulation values — modified
            in-place **and** returned.
        xyz: ``(N, 3)`` array of cell centroids in the same coordinate system
            the IRBF grid was built in.
        registry: :class:`DataRegistry` instance.
        iso_value: Probability threshold.  ``None`` → use the one in the
            registered payload, falling back to 0.5.
        fill: Value to write outside the domain.  ``None`` → ``np.nan``.

    Returns:
        The (possibly mutated) ``values`` array.  If no IRBF domain is
        registered, ``values`` is returned unchanged.
    """
    import numpy as np

    if registry is None:
        return values
    try:
        payload = registry.get_indicator_rbf_domain()
    except Exception as exc:
        logger.debug("apply_irbf_domain_mask: registry lookup failed: %s", exc)
        return values
    if not payload:
        return values

    prob = payload.get("probability_field")
    xs = payload.get("x")
    ys = payload.get("y")
    zs = payload.get("z")
    if prob is None or xs is None or ys is None or zs is None:
        logger.debug("apply_irbf_domain_mask: payload missing prob/axes")
        return values

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    prob = np.asarray(prob)

    # Probability volume axis order: indicator_rbf_engine stores shape (nx, ny, nz).
    if prob.shape != (xs.size, ys.size, zs.size):
        logger.warning(
            "apply_irbf_domain_mask: prob shape %s does not match axes (%d,%d,%d) — skipping",
            prob.shape, xs.size, ys.size, zs.size,
        )
        return values

    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        logger.debug("apply_irbf_domain_mask: xyz must be (N,3)")
        return values

    ix = np.clip(np.searchsorted(xs, xyz[:, 0]) - 0, 0, xs.size - 1)
    iy = np.clip(np.searchsorted(ys, xyz[:, 1]) - 0, 0, ys.size - 1)
    iz = np.clip(np.searchsorted(zs, xyz[:, 2]) - 0, 0, zs.size - 1)
    # Nearest neighbour correction (searchsorted rounds up; prefer the closer side)
    for axis_vals, idx, coord in ((xs, ix, xyz[:, 0]), (ys, iy, xyz[:, 1]), (zs, iz, xyz[:, 2])):
        prev = np.clip(idx - 1, 0, axis_vals.size - 1)
        use_prev = (idx > 0) & (np.abs(coord - axis_vals[prev]) < np.abs(coord - axis_vals[idx]))
        idx[use_prev] = prev[use_prev]

    if iso_value is None:
        iso_value = float(payload.get("iso_value", 0.5))

    p_at_cell = prob[ix, iy, iz]
    outside = p_at_cell < float(iso_value)

    values = np.asarray(values, dtype=float)
    if fill is None:
        fill = np.nan
    values[outside] = fill
    return values


def registry_has_irbf_domain(registry) -> bool:
    """Return True iff an IRBF domain is registered and has the arrays we need."""
    if registry is None:
        return False
    try:
        payload = registry.get_indicator_rbf_domain()
    except Exception:
        return False
    if not payload:
        return False
    return all(payload.get(k) is not None for k in ("probability_field", "x", "y", "z"))
