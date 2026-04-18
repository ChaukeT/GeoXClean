"""
renderer package — bridges the monolithic ``renderer.py`` sibling with
the modular renderer sub-packages.

Because this directory shadows ``visualization/renderer.py`` in Python's
import system, we load the sibling file explicitly via ``importlib`` so
that ``from block_model_viewer.visualization.renderer import Renderer``
still works.

Sub-packages (renderers/) contain extracted, modular renderers that can
be used alongside the monolithic Renderer class.
"""
import importlib
import importlib.util
from pathlib import Path

# Load the monolithic renderer.py sibling
_sibling = Path(__file__).parent.parent / "renderer.py"
if _sibling.exists():
    _spec = importlib.util.spec_from_file_location(
        "block_model_viewer.visualization._renderer_monolithic",
        str(_sibling),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    Renderer = getattr(_mod, "Renderer", None)
else:
    Renderer = None

__all__ = ["Renderer"]
