"""
pyvista_suppression.py — Quarantine module for PyVista monkey-patching.

GeoX replaces PyVista's built-in axes/bounds/bounding-box widgets with its own
unified HUD/overlay system (OverlayManager + FloatingAxes + ScaleBar3D).
To prevent PyVista from auto-creating its widgets, we patch the plotter instance.

WHY A SEPARATE FILE?
When PyVista changes its internals, modify THIS file only.
Patching code must not spread through geological rendering logic.
"""

import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def install_pyvista_suppression(plotter) -> Optional[Callable]:
    """
    Monkey-patch a PyVista plotter to suppress automatic creation of axes
    widgets, bounding boxes, and scalar bars.

    IMPORTANT: Call this AFTER saving the original ``show_bounds`` method
    if you need floating axes (see return value).

    Returns:
        The original ``show_bounds`` callable before patching, or ``None``.
        Store this as ``_original_show_bounds`` so the floating-axes system
        can call it directly (the patched instance method is a no-op).
    """
    # Save original BEFORE any patching so floating axes can call it later
    original_show_bounds = getattr(plotter, 'show_bounds', None)

    def _noop(*args, **kwargs):
        return None

    # Patch instance methods to no-ops so PyVista never auto-creates its widgets
    plotter.add_axes = _noop
    plotter.show_bounds = _noop
    plotter.add_bounding_box = _noop

    # Attempt module-level patching for completeness (version-tolerant)
    try:
        import pyvista as pv
        if hasattr(pv, 'plotting') and hasattr(pv.plotting, 'widgets'):
            widgets = pv.plotting.widgets
            if hasattr(widgets, 'WidgetHelper'):
                widgets.WidgetHelper.add_bounds_axes = lambda *a, **kw: None
    except Exception as exc:
        logger.debug(f"[pyvista_suppression] Module-level patch skipped: {exc}")

    logger.debug("[pyvista_suppression] Installed no-op patches: add_axes, show_bounds, add_bounding_box")
    return original_show_bounds


def remove_all_axes_actors(plotter) -> None:
    """
    Remove all VTK/PyVista axes actors from the renderer.

    This covers vtkCubeAxesActor, vtkCubeAxesActor2D, and PyVista's internal
    cached references.  Call this after any operation that might create axes
    (e.g., ``add_mesh``).
    """
    if plotter is None:
        return

    try:
        renderer = plotter.renderer

        # Remove vtkCubeAxesActor / vtkCubeAxesActor2D from VTK actor list
        actors_to_remove = []
        col = renderer.GetActors()
        col.InitTraversal()
        actor = col.GetNextItem()
        while actor is not None:
            if actor.IsA("vtkCubeAxesActor") or actor.IsA("vtkCubeAxesActor2D"):
                actors_to_remove.append(actor)
            actor = col.GetNextItem()

        for a in actors_to_remove:
            try:
                renderer.RemoveActor(a)
            except Exception:
                pass

        # Clear PyVista's axes_actor reference
        if hasattr(renderer, 'axes_actor') and renderer.axes_actor is not None:
            renderer.RemoveActor(renderer.axes_actor)
            renderer.axes_actor = None

        # Reset PyVista internal flags
        plotter._show_bounds = False
        if hasattr(plotter, '_cube_axes_actor'):
            plotter._cube_axes_actor = None
        if hasattr(plotter, '_cube_axes_actor2d'):
            plotter._cube_axes_actor2d = None

        plotter.show_axes = False
        if hasattr(plotter, 'hide_axes'):
            plotter.hide_axes()
        plotter.show_grid(False)

        try:
            plotter.remove_bounds_axes()
        except Exception:
            pass
        try:
            plotter.remove_bounding_box()
        except Exception:
            pass

    except Exception:
        pass  # Best-effort — never crash due to axes cleanup
