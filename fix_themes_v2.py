"""
Enhanced Theme Fixing Script v2 for GeoX Clean

Handles:
- Proper f-string formatting for stylesheets
- Correct method placement
- Child widget detection
- Subpanel handling
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Color mapping from hex to ModernColors constants
COLOR_MAP = {
    # Panels/Cards
    '#1e1e1e': 'ModernColors.PANEL_BG',
    '#0f172a': 'ModernColors.PANEL_BG',
    '#1e293b': 'ModernColors.PANEL_BG',
    '#2d2d2d': 'ModernColors.CARD_BG',
    '#2d3748': 'ModernColors.CARD_BG',
    '#374151': 'ModernColors.CARD_HOVER',
    '#475569': 'ModernColors.CARD_HOVER',
    '#526278': 'ModernColors.CARD_HOVER',

    # Borders
    '#3d3d3d': 'ModernColors.BORDER',
    '#64748b': 'ModernColors.BORDER',
    '#334155': 'ModernColors.BORDER_SUBTLE',

    # Text colors
    '#ffffff': 'ModernColors.TEXT_PRIMARY',
    '#e0e0e0': 'ModernColors.TEXT_PRIMARY',
    '#f8fafc': 'ModernColors.TEXT_PRIMARY',
    '#f1f5f9': 'ModernColors.TEXT_PRIMARY',
    '#cbd5e1': 'ModernColors.TEXT_SECONDARY',
    '#b0b0b0': 'ModernColors.TEXT_SECONDARY',
    '#94a3b8': 'ModernColors.TEXT_SECONDARY',
    '#64748b': 'ModernColors.TEXT_HINT',
    '#5f5f5f': 'ModernColors.TEXT_DISABLED',

    # Primary/Accent
    '#1a73e8': 'ModernColors.ACCENT_PRIMARY',
    '#4285f4': 'ModernColors.ACCENT_PRIMARY',
    '#3b82f6': 'ModernColors.ACCENT_PRIMARY',
    '#1557b0': 'ModernColors.ACCENT_PRIMARY_DARK',

    # Success
    '#34a853': 'ModernColors.SUCCESS',
    '#22c55e': 'ModernColors.SUCCESS',
    '#10b981': 'ModernColors.SUCCESS',
    '#86efac': 'ModernColors.SUCCESS_LIGHT',

    # Warning
    '#fbbc04': 'ModernColors.WARNING',
    '#ffa000': 'ModernColors.WARNING',
    '#f59e0b': 'ModernColors.WARNING',

    # Error
    '#ea4335': 'ModernColors.ERROR',
    '#ef4444': 'ModernColors.ERROR',
    '#dc2626': 'ModernColors.ERROR',
    '#fca5a5': 'ModernColors.ERROR_LIGHT',

    # Info
    '#8b5cf6': 'ModernColors.INFO',
    '#6366f1': 'ModernColors.INFO',

    # Selection
    '#dbeafe': 'ModernColors.SELECTION_BG',
    '#bfdbfe': 'ModernColors.SELECTION_HOVER',
    '#1e3a8a': 'ModernColors.SELECTION_TEXT',
    '#1e3a5f': 'ModernColors.HOVER',
}

def fix_single_file(file_path: Path) -> bool:
    """Fix a single file with proper theme support"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Check if file has setStyleSheet calls
        if '.setStyleSheet(' not in content:
            return False

        # 1. Ensure imports
        content = ensure_imports(content)

        # 2. Convert stylesheets to use theme colors
        content = convert_stylesheets(content)

        # 3. Add refresh_theme methods if missing
        content = add_refresh_theme_methods(content)

        # Write if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def ensure_imports(content: str) -> str:
    """Ensure modern_styles is imported"""
    if 'from .modern_styles import' in content or 'from ..modern_styles import' in content:
        # Check what's imported
        if 'get_theme_colors' not in content:
            content = re.sub(
                r'from (\.+)modern_styles import ([^\n]+)',
                lambda m: f"from {m.group(1)}modern_styles import {m.group(2)}, get_theme_colors, ModernColors",
                content,
                count=1
            )
        return content

    # Add import after PyQt6 imports
    pyqt_match = re.search(r'(from PyQt6\.[^\n]+\n)+', content)
    if pyqt_match:
        pos = pyqt_match.end()
        content = content[:pos] + 'from .modern_styles import get_theme_colors, ModernColors\n' + content[pos:]

    return content

def convert_stylesheets(content: str) -> str:
    """Convert setStyleSheet calls to use f-strings with theme colors"""

    # Find all setStyleSheet calls
    pattern = r'\.setStyleSheet\(("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')\)'

    def convert_stylesheet_call(match):
        stylesheet = match.group(1)

        # Check if already an f-string
        if match.group(0).startswith('.setStyleSheet(f'):
            return match.group(0)

        # Replace colors
        converted = stylesheet
        has_replacements = False

        for hex_color, const in COLOR_MAP.items():
            if hex_color.lower() in converted.lower():
                converted = re.sub(
                    re.escape(hex_color),
                    f'{{{const}}}',
                    converted,
                    flags=re.IGNORECASE
                )
                has_replacements = True

        # If we made replacements, make it an f-string
        if has_replacements:
            return f'.setStyleSheet(f{converted})'

        return match.group(0)

    content = re.sub(pattern, convert_stylesheet_call, content)

    return content

def add_refresh_theme_methods(content: str) -> str:
    """Add refresh_theme methods to classes that use setStyleSheet"""

    # Find all class definitions
    class_pattern = r'class\s+(\w+)\([^)]+\):(.*?)(?=\nclass\s+\w+\(|\Z)'

    def add_refresh_to_class(match):
        class_name = match.group(1)
        class_body = match.group(2)

        # Skip if refresh_theme already exists
        if 'def refresh_theme(self' in class_body:
            return match.group(0)

        # Check if class uses setStyleSheet
        if '.setStyleSheet(' not in class_body:
            return match.group(0)

        # Find __init__ method
        init_match = re.search(r'(\n    def __init__\(self.*?\):.*?(?=\n    def |\Z))', class_body, re.DOTALL)
        if not init_match:
            return match.group(0)

        # Find end of __init__
        init_end = init_match.end()

        # Create refresh_theme method
        refresh_method = '''
    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Rebuild UI with new colors
        if hasattr(self, '_setup_ui'):
            # Re-run setup if available
            pass
        # Update all stylesheets
        self.style().unpolish(self)
        self.style().polish(self)
        # Refresh children
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
'''

        # Insert after __init__
        new_class_body = class_body[:init_end] + refresh_method + class_body[init_end:]

        return f'class {class_name}({match.group(1).split("(", 1)[1]}\n' + new_class_body

    content = re.sub(class_pattern, add_refresh_to_class, content, flags=re.DOTALL)

    return content

def main():
    ui_dir = Path(r"c:\Users\chauk\Documents\GeoX_Clean\block_model_viewer\ui")

    # High priority files from user's list
    priority_files = [
        "column_mapping_dialog.py",
        "loopstructural_compliance_panel.py",
        "loopstructural_advisory_panel.py",
        "structural_import_panel.py",
        "grade_tonnage_basic_panel.py",
        "drillhole_status_bar.py",
        "data_source_mixin.py",
        "cokriging_panel.py",
        "cosgsim_panel.py",
        "swath_analysis_3d_panel.py",
        "declustering_panel.py",
        "grade_transformation_panel.py",
        "layout/layout_property_panel.py",
        "dialogs/legend_add_dialog.py",
        "dbs_panel.py",
        "mps_panel.py",
        "blockmodel_builder_panel.py",
        "ik_sgsim_panel.py",
        "sis_panel.py",
        "turning_bands_panel.py",
        "grf_panel.py",
        "legend_element_widget.py",
        "layout/layout_window.py",
        "north_arrow_widget.py",
        "project_loading_dialog.py",
        "scale_bar_widget.py",
        "simple_kriging_panel.py",
        "multi_legend_widget.py",
        "axes_scalebar_panel.py",
        "universal_kriging_panel.py",
        "mouse_panel.py",
        "fault_definition_panel.py",
        "fold_definition_panel.py",
        "indicator_kriging_panel.py",
        "vein_definition_panel.py",
    ]

    logger.info("=" * 80)
    logger.info("ENHANCED THEME FIX v2")
    logger.info("=" * 80)

    fixed = 0

    for file_name in priority_files:
        file_path = ui_dir / file_name
        if not file_path.exists():
            logger.warning(f"❌ Not found: {file_name}")
            continue

        logger.info(f"\nFixing: {file_name}")
        if fix_single_file(file_path):
            logger.info(f"  ✅ Fixed")
            fixed += 1
        else:
            logger.info(f"  ⏭️  No changes needed")

    logger.info(f"\n\n✅ Fixed {fixed}/{len(priority_files)} priority files")

if __name__ == "__main__":
    main()
