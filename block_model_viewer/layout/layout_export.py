"""
Layout Export for GeoX Layout Composer.

Handles export to PDF, PNG, and TIFF formats with audit record generation.
Every export creates a companion .audit.json file with complete provenance.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt6.QtCore import QBuffer, QIODevice

from .layout_document import LayoutDocument
from .layout_renderer import LayoutRenderer, DPI_PRESETS

if TYPE_CHECKING:
    from ..ui.viewer_widget import ViewerWidget

logger = logging.getLogger(__name__)

# Check for ReportLab availability (used for PDF export)
try:
    from reportlab.lib.pagesizes import A4, A3, letter
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available - PDF export will be limited")

# Check for PIL/Pillow availability (used for TIFF export)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available - TIFF export will be limited")


def get_software_version() -> str:
    """Get GeoX software version string."""
    try:
        from ..core.audit_manager import _get_software_version
        return _get_software_version()
    except ImportError:
        pass

    # Fallback: try to get version directly
    try:
        from .. import __version__
        return f"GeoX_v{__version__}"
    except ImportError:
        return "GeoX_v1.0.0"


def export_pdf(
    document: LayoutDocument,
    filepath: Path,
    dpi: int = 300,
    viewer_widget: Optional["ViewerWidget"] = None,
    metadata_values: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Export layout to PDF.

    The PDF contains the layout rendered as a high-resolution image.
    Text and vector elements are rasterized at the target DPI for
    consistent output across viewers.

    Args:
        document: Layout document to export
        filepath: Output file path
        dpi: Resolution for rendering (default 300)
        viewer_widget: Optional viewer widget for viewport capture
        metadata_values: Optional metadata values for dynamic fields

    Returns:
        Audit record dictionary

    Raises:
        ImportError: If ReportLab is not available
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "ReportLab is required for PDF export. "
            "Install with: pip install reportlab"
        )

    filepath = Path(filepath)
    if filepath.suffix.lower() != '.pdf':
        filepath = filepath.with_suffix('.pdf')

    # Render layout to image
    renderer = LayoutRenderer(document, viewer_widget)
    if metadata_values:
        renderer.set_metadata_values(metadata_values)

    layout_image = renderer.render_to_image(dpi, transparent=False)

    # Convert QImage to PIL Image for ReportLab
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    layout_image.save(buffer, "PNG")
    buffer.close()

    pil_image = Image.open(BytesIO(buffer.data()))

    # Calculate PDF page size in points (1 point = 1/72 inch)
    page = document.page
    pdf_width = page.width_mm * mm
    pdf_height = page.height_mm * mm

    # Create PDF
    c = canvas.Canvas(str(filepath), pagesize=(pdf_width, pdf_height))

    # Add metadata
    c.setTitle(document.name)
    c.setAuthor(getpass.getuser())
    c.setSubject(f"GeoX Layout Export - {document.name}")
    c.setCreator(get_software_version())

    # Draw image to fill page
    img_reader = ImageReader(pil_image)
    c.drawImage(img_reader, 0, 0, width=pdf_width, height=pdf_height)

    c.save()
    logger.info(f"PDF exported to {filepath}")

    # Generate and write audit record
    audit_record = _create_audit_record(
        document, filepath, "pdf", dpi, viewer_widget, metadata_values
    )

    return audit_record


def export_png(
    document: LayoutDocument,
    filepath: Path,
    dpi: int = 300,
    viewer_widget: Optional["ViewerWidget"] = None,
    metadata_values: Optional[Dict[str, str]] = None,
    transparent: bool = False,
) -> Dict[str, Any]:
    """
    Export layout to PNG image.

    Args:
        document: Layout document to export
        filepath: Output file path
        dpi: Resolution for rendering (default 300)
        viewer_widget: Optional viewer widget for viewport capture
        metadata_values: Optional metadata values for dynamic fields
        transparent: If True, use transparent background

    Returns:
        Audit record dictionary
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != '.png':
        filepath = filepath.with_suffix('.png')

    # Render layout to image
    renderer = LayoutRenderer(document, viewer_widget)
    if metadata_values:
        renderer.set_metadata_values(metadata_values)

    layout_image = renderer.render_to_image(dpi, transparent=transparent)

    # Save with DPI metadata
    layout_image.save(str(filepath), "PNG")
    logger.info(f"PNG exported to {filepath}")

    # Generate and write audit record
    audit_record = _create_audit_record(
        document, filepath, "png", dpi, viewer_widget, metadata_values
    )

    return audit_record


def export_tiff(
    document: LayoutDocument,
    filepath: Path,
    dpi: int = 300,
    viewer_widget: Optional["ViewerWidget"] = None,
    metadata_values: Optional[Dict[str, str]] = None,
    compression: str = "lzw",
) -> Dict[str, Any]:
    """
    Export layout to TIFF image.

    Args:
        document: Layout document to export
        filepath: Output file path
        dpi: Resolution for rendering (default 300)
        viewer_widget: Optional viewer widget for viewport capture
        metadata_values: Optional metadata values for dynamic fields
        compression: TIFF compression method (lzw, none, zip)

    Returns:
        Audit record dictionary

    Raises:
        ImportError: If PIL/Pillow is not available
    """
    if not PIL_AVAILABLE:
        raise ImportError(
            "PIL/Pillow is required for TIFF export. "
            "Install with: pip install Pillow"
        )

    filepath = Path(filepath)
    if filepath.suffix.lower() not in ('.tif', '.tiff'):
        filepath = filepath.with_suffix('.tiff')

    # Render layout to image
    renderer = LayoutRenderer(document, viewer_widget)
    if metadata_values:
        renderer.set_metadata_values(metadata_values)

    layout_image = renderer.render_to_image(dpi, transparent=False)

    # Convert QImage to PIL Image
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    layout_image.save(buffer, "PNG")
    buffer.close()

    pil_image = Image.open(BytesIO(buffer.data()))

    # Save as TIFF with DPI metadata
    pil_image.save(
        str(filepath),
        "TIFF",
        dpi=(dpi, dpi),
        compression=compression if compression != "none" else None,
    )
    logger.info(f"TIFF exported to {filepath}")

    # Generate and write audit record
    audit_record = _create_audit_record(
        document, filepath, "tiff", dpi, viewer_widget, metadata_values
    )

    return audit_record


def _create_audit_record(
    document: LayoutDocument,
    filepath: Path,
    export_format: str,
    dpi: int,
    viewer_widget: Optional["ViewerWidget"],
    metadata_values: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Create and write audit record for an export operation.

    The audit record provides complete provenance for the export,
    enabling reproducibility and compliance with regulatory requirements.

    Args:
        document: The exported layout document
        filepath: Path to the exported file
        export_format: Export format (pdf, png, tiff)
        dpi: Resolution used for export
        viewer_widget: Viewer widget used for viewport capture
        metadata_values: Metadata values used in export

    Returns:
        Audit record dictionary
    """
    # Compute hash of output file
    output_hash = ""
    if filepath.exists():
        with open(filepath, 'rb') as f:
            output_hash = hashlib.sha256(f.read()).hexdigest()

    # Capture camera state if viewer available
    camera_state = {}
    if viewer_widget:
        try:
            renderer = getattr(viewer_widget, 'renderer', None)
            if renderer and hasattr(renderer, 'get_camera_info'):
                camera_state = renderer.get_camera_info() or {}
        except Exception as e:
            logger.warning(f"Failed to capture camera state: {e}")

    # Capture legend state for viewport items
    legend_states = {}
    for item in document.items:
        if hasattr(item, 'legend_state') and item.legend_state:
            legend_states[item.id] = item.legend_state

    # Build audit record
    record = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "layout_export",
        "software_version": get_software_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "user": getpass.getuser(),
        "machine": platform.node(),
        "layout": {
            "name": document.name,
            "version": document.version,
            "created": document.created,
            "modified": document.modified,
            "item_count": len(document.items),
        },
        "export": {
            "format": export_format,
            "filepath": str(filepath.absolute()),
            "dpi": dpi,
            "page_width_mm": document.page.width_mm,
            "page_height_mm": document.page.height_mm,
            "page_size": document.page.size.value,
            "page_orientation": document.page.orientation.value,
        },
        "items": [
            {
                "id": item.id,
                "type": item.item_type,
                "name": item.name,
                "visible": item.visible,
                "x_mm": item.x_mm,
                "y_mm": item.y_mm,
                "width_mm": item.width_mm,
                "height_mm": item.height_mm,
            }
            for item in document.items
        ],
        "camera_state": camera_state,
        "legend_states": legend_states,
        "metadata_values": metadata_values or {},
        "output_hash": f"sha256:{output_hash}",
    }

    # Write audit JSON alongside export
    audit_path = filepath.with_suffix(filepath.suffix + ".audit.json")
    try:
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, default=str)
        logger.info(f"Audit record saved to {audit_path}")
    except Exception as e:
        logger.error(f"Failed to write audit record: {e}")

    return record


def verify_export_integrity(export_path: Path) -> tuple[bool, str]:
    """
    Verify the integrity of an exported file against its audit record.

    Args:
        export_path: Path to the exported file

    Returns:
        Tuple of (is_valid, message)
    """
    audit_path = export_path.with_suffix(export_path.suffix + ".audit.json")

    if not export_path.exists():
        return False, f"Export file not found: {export_path}"

    if not audit_path.exists():
        return False, f"Audit record not found: {audit_path}"

    try:
        with open(audit_path, 'r') as f:
            audit = json.load(f)

        recorded_hash = audit.get("output_hash", "")
        if recorded_hash.startswith("sha256:"):
            recorded_hash = recorded_hash[7:]

        with open(export_path, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        if current_hash == recorded_hash:
            return True, "Export integrity verified"
        else:
            return False, "Export hash mismatch - file may have been modified"

    except Exception as e:
        return False, f"Verification failed: {e}"


def get_export_options() -> Dict[str, Dict[str, Any]]:
    """
    Get available export formats and their options.

    Returns:
        Dictionary of format -> options
    """
    return {
        "pdf": {
            "available": REPORTLAB_AVAILABLE,
            "extension": ".pdf",
            "description": "PDF Document",
            "dpi_presets": DPI_PRESETS,
            "options": {},
        },
        "png": {
            "available": True,
            "extension": ".png",
            "description": "PNG Image",
            "dpi_presets": DPI_PRESETS,
            "options": {
                "transparent": {
                    "type": "bool",
                    "default": False,
                    "description": "Transparent background",
                },
            },
        },
        "tiff": {
            "available": PIL_AVAILABLE,
            "extension": ".tiff",
            "description": "TIFF Image",
            "dpi_presets": DPI_PRESETS,
            "options": {
                "compression": {
                    "type": "choice",
                    "choices": ["lzw", "zip", "none"],
                    "default": "lzw",
                    "description": "Compression method",
                },
            },
        },
    }
