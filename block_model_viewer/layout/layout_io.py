"""
Layout I/O for GeoX Layout Composer.

Handles saving and loading layout documents to/from JSON files
with schema versioning for forward compatibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .layout_document import LayoutDocument

logger = logging.getLogger(__name__)

LAYOUT_FILE_EXTENSION = ".geoxlayout.json"
CURRENT_SCHEMA_VERSION = "1.0.0"


def save_layout(document: LayoutDocument, filepath: Path) -> bool:
    """
    Save layout document to JSON file.

    Args:
        document: The layout document to save
        filepath: Destination file path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure proper extension
        if not str(filepath).endswith(LAYOUT_FILE_EXTENSION):
            filepath = Path(str(filepath) + LAYOUT_FILE_EXTENSION)

        # Update modification timestamp
        document.modified = datetime.now().isoformat()

        # Build save data with schema version
        data = document.to_dict()
        data["schema_version"] = CURRENT_SCHEMA_VERSION

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write with pretty formatting for readability
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Layout saved to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save layout to {filepath}: {e}")
        return False


def load_layout(filepath: Path) -> Optional[LayoutDocument]:
    """
    Load layout document from JSON file.

    Args:
        filepath: Source file path

    Returns:
        LayoutDocument if successful, None otherwise
    """
    try:
        if not filepath.exists():
            logger.error(f"Layout file not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check schema version for compatibility
        schema_version = data.get("schema_version", "1.0.0")
        if schema_version != CURRENT_SCHEMA_VERSION:
            logger.warning(
                f"Layout schema version mismatch: file={schema_version}, "
                f"current={CURRENT_SCHEMA_VERSION}. Attempting migration."
            )
            data = _migrate_schema(data, schema_version)

        document = LayoutDocument.from_dict(data)
        logger.info(f"Layout loaded from {filepath}")
        return document

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in layout file {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load layout from {filepath}: {e}")
        return None


def _migrate_schema(data: dict, from_version: str) -> dict:
    """
    Migrate layout data from older schema versions.

    Args:
        data: Raw layout data dictionary
        from_version: Source schema version

    Returns:
        Migrated data dictionary
    """
    # Version migration chain
    migrations = [
        # ("1.0.0", "1.1.0", _migrate_1_0_to_1_1),
        # Add future migrations here
    ]

    current = from_version
    for src, dst, migrate_fn in migrations:
        if current == src:
            data = migrate_fn(data)
            current = dst
            logger.info(f"Migrated layout schema from {src} to {dst}")

    if current != CURRENT_SCHEMA_VERSION:
        logger.warning(
            f"Could not fully migrate schema from {from_version} to "
            f"{CURRENT_SCHEMA_VERSION}. Some features may not work correctly."
        )

    return data


def get_layout_file_filter() -> str:
    """Get file filter string for file dialogs."""
    return f"GeoX Layout Files (*{LAYOUT_FILE_EXTENSION});;All Files (*)"


def suggest_layout_filename(base_name: str = "layout") -> str:
    """
    Suggest a filename for a new layout.

    Args:
        base_name: Base name for the file

    Returns:
        Suggested filename with extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{LAYOUT_FILE_EXTENSION}"


def validate_layout_file(filepath: Path) -> tuple[bool, str]:
    """
    Validate a layout file without fully loading it.

    Args:
        filepath: File to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not filepath.exists():
            return False, f"File not found: {filepath}"

        if not filepath.suffix.endswith('.json'):
            return False, "File must be a JSON file"

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check for required fields
        required_fields = ['version', 'page', 'items']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"

        # Check page specification
        if not isinstance(data.get('page'), dict):
            return False, "Invalid page specification"

        # Check items array
        if not isinstance(data.get('items'), list):
            return False, "Invalid items array"

        return True, ""

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def get_layout_info(filepath: Path) -> Optional[dict]:
    """
    Get basic information about a layout file without fully loading it.

    Args:
        filepath: File to inspect

    Returns:
        Dictionary with layout info or None if invalid
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "name": data.get("name", "Unknown"),
            "version": data.get("version", "Unknown"),
            "schema_version": data.get("schema_version", "Unknown"),
            "created": data.get("created"),
            "modified": data.get("modified"),
            "item_count": len(data.get("items", [])),
            "page_size": data.get("page", {}).get("size", "Unknown"),
            "page_orientation": data.get("page", {}).get("orientation", "Unknown"),
        }

    except Exception:
        return None
