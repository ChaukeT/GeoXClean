"""
Layout Composer subsystem for GeoX.

Provides a document model, rendering engine, and export pipeline
for creating publication-ready print layouts with viewports, legends,
scale bars, north arrows, text annotations, and metadata blocks.
"""

from .layout_document import (
    PageSize,
    PageOrientation,
    PageSpec,
    LayoutItem,
    ViewportItem,
    LegendItem,
    ScaleBarItem,
    NorthArrowItem,
    TextItem,
    ImageItem,
    MetadataItem,
    LayoutDocument,
)
from .layout_io import (
    save_layout,
    load_layout,
    LAYOUT_FILE_EXTENSION,
    CURRENT_SCHEMA_VERSION,
)
from .layout_renderer import LayoutRenderer, DPI_PRESETS
from .layout_export import (
    export_pdf,
    export_png,
    export_tiff,
    get_export_options,
    verify_export_integrity,
)

__all__ = [
    # Document model
    "PageSize",
    "PageOrientation",
    "PageSpec",
    "LayoutItem",
    "ViewportItem",
    "LegendItem",
    "ScaleBarItem",
    "NorthArrowItem",
    "TextItem",
    "ImageItem",
    "MetadataItem",
    "LayoutDocument",
    # I/O
    "save_layout",
    "load_layout",
    "LAYOUT_FILE_EXTENSION",
    "CURRENT_SCHEMA_VERSION",
    # Renderer
    "LayoutRenderer",
    "DPI_PRESETS",
    # Export
    "export_pdf",
    "export_png",
    "export_tiff",
    "get_export_options",
    "verify_export_integrity",
]
