"""
Automated Theme Fixing Script for GeoX Clean

This script systematically updates all UI panels to support theme switching by:
1. Adding modern_styles imports if missing
2. Replacing hardcoded colors with ModernColors constants
3. Adding/updating refresh_theme() methods
4. Handling subpanels and child widgets
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Color mapping from hex/RGB to ModernColors constants
COLOR_MAPPINGS = {
    # Backgrounds
    '#1e1e1e': 'ModernColors.PANEL_BG',
    '#0f172a': 'ModernColors.PANEL_BG',
    '#1e293b': 'ModernColors.PANEL_BG',
    '#2d2d2d': 'ModernColors.CARD_BG',
    '#2d3748': 'ModernColors.CARD_BG',
    '#374151': 'ModernColors.CARD_HOVER',
    '#475569': 'ModernColors.CARD_HOVER',

    # Borders
    '#3d3d3d': 'ModernColors.BORDER',
    '#64748b': 'ModernColors.BORDER',
    '#475569': 'ModernColors.BORDER',
    '#334155': 'ModernColors.BORDER_SUBTLE',

    # Text
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

    # Status colors
    '#34a853': 'ModernColors.SUCCESS',
    '#22c55e': 'ModernColors.SUCCESS',
    '#10b981': 'ModernColors.SUCCESS',
    '#86efac': 'ModernColors.SUCCESS_LIGHT',

    '#fbbc04': 'ModernColors.WARNING',
    '#ffa000': 'ModernColors.WARNING',
    '#f59e0b': 'ModernColors.WARNING',

    '#ea4335': 'ModernColors.ERROR',
    '#ef4444': 'ModernColors.ERROR',
    '#dc2626': 'ModernColors.ERROR',
    '#fca5a5': 'ModernColors.ERROR_LIGHT',

    '#8b5cf6': 'ModernColors.INFO',
    '#6366f1': 'ModernColors.INFO',

    # Special UI
    '#dbeafe': 'ModernColors.SELECTION_BG',
    '#bfdbfe': 'ModernColors.SELECTION_HOVER',
    '#1e3a8a': 'ModernColors.SELECTION_TEXT',
    '#1e3a5f': 'ModernColors.HOVER',
}

# RGB color mappings
RGB_MAPPINGS = {
    'QColor(30, 30, 30)': 'QColor(ModernColors.PANEL_BG)',
    'QColor(45, 45, 45)': 'QColor(ModernColors.CARD_BG)',
    'QColor(61, 61, 61)': 'QColor(ModernColors.BORDER)',
    'QColor(224, 224, 224)': 'QColor(ModernColors.TEXT_PRIMARY)',
    'QColor(26, 115, 232)': 'QColor(ModernColors.ACCENT_PRIMARY)',
    'QColor(52, 168, 83)': 'QColor(ModernColors.SUCCESS)',
    'QColor(251, 188, 4)': 'QColor(ModernColors.WARNING)',
    'QColor(234, 67, 53)': 'QColor(ModernColors.ERROR)',
}

class ThemeFixer:
    def __init__(self, ui_dir: Path):
        self.ui_dir = ui_dir
        self.fixed_files = []
        self.skipped_files = []
        self.errors = []

    def fix_all_files(self):
        """Fix all Python files in UI directory"""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE THEME FIX")
        logger.info("=" * 80)

        # Get all .py files
        all_files = list(self.ui_dir.rglob("*.py"))
        panel_files = [f for f in all_files if self._is_ui_file(f)]

        logger.info(f"\nFound {len(panel_files)} UI files to process\n")

        for i, file_path in enumerate(sorted(panel_files), 1):
            logger.info(f"[{i}/{len(panel_files)}] Processing: {file_path.relative_to(self.ui_dir)}")
            try:
                self.fix_file(file_path)
            except Exception as e:
                self.errors.append((file_path, str(e)))
                logger.error(f"  ❌ ERROR: {e}")

        self._print_summary()

    def _is_ui_file(self, path: Path) -> bool:
        """Check if file is a UI file that needs theme fixing"""
        name = path.name.lower()
        # Skip backups, __init__, test files
        if any(x in name for x in ['backup', 'test_', '__init__', '_old', 'before_restore']):
            return False
        # Include panels, dialogs, widgets, windows
        return any(x in name for x in ['panel', 'dialog', 'window', 'widget', 'bar', 'menu'])

    def fix_file(self, file_path: Path):
        """Fix a single file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Check if file has hardcoded colors
        if not self._has_hardcoded_colors(content):
            logger.info("  ⏭️  No hardcoded colors found, skipping")
            self.skipped_files.append(file_path)
            return

        # 1. Add import if missing
        content = self._add_imports(content)

        # 2. Replace hex colors in stylesheets
        content = self._replace_hex_colors(content)

        # 3. Replace RGB QColor calls
        content = self._replace_rgb_colors(content)

        # 4. Add/update refresh_theme method
        content = self._add_refresh_theme(content, file_path)

        # Only write if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.fixed_files.append(file_path)
            logger.info("  ✅ Fixed successfully")
        else:
            logger.info("  ⏭️  No changes needed")
            self.skipped_files.append(file_path)

    def _has_hardcoded_colors(self, content: str) -> bool:
        """Check if content has hardcoded colors"""
        # Check for hex colors
        if re.search(r'#[0-9a-fA-F]{6}', content):
            return True
        # Check for RGB QColor
        if re.search(r'QColor\(\s*\d+\s*,\s*\d+\s*,\s*\d+', content):
            return True
        return False

    def _add_imports(self, content: str) -> str:
        """Add modern_styles import if missing"""
        if 'from .modern_styles import' in content or 'from ..modern_styles import' in content:
            # Already has import, check if get_theme_colors and ModernColors are there
            if 'get_theme_colors' not in content or 'ModernColors' not in content:
                # Update existing import
                content = re.sub(
                    r'from \.modern_styles import ([^\n]+)',
                    lambda m: f"from .modern_styles import {m.group(1)}, get_theme_colors, ModernColors" if 'get_theme_colors' not in m.group(1) else m.group(0),
                    content
                )
            return content

        # Find where to insert import
        # Look for other PyQt6 imports
        import_section = re.search(r'(from PyQt6\.[^\n]+\n)', content)
        if import_section:
            pos = import_section.end()
            # Check if it's in ui subdir or ui root
            if '\nfrom .modern_styles' not in content[:pos+100]:
                content = content[:pos] + 'from .modern_styles import get_theme_colors, ModernColors\n' + content[pos:]

        return content

    def _replace_hex_colors(self, content: str) -> str:
        """Replace hex colors with ModernColors constants"""
        for hex_color, modern_const in COLOR_MAPPINGS.items():
            # Replace in stylesheets
            content = re.sub(
                re.escape(hex_color),
                f'{{{modern_const}}}',
                content,
                flags=re.IGNORECASE
            )
        return content

    def _replace_rgb_colors(self, content: str) -> str:
        """Replace RGB QColor calls with themed versions"""
        for rgb_call, modern_call in RGB_MAPPINGS.items():
            content = content.replace(rgb_call, modern_call)
        return content

    def _add_refresh_theme(self, content: str, file_path: Path) -> str:
        """Add or update refresh_theme method"""
        # Check if refresh_theme already exists
        if 'def refresh_theme(self' in content:
            logger.info("  📝 refresh_theme() already exists")
            return content

        # Find the class definition
        class_match = re.search(r'class\s+(\w+)\([^)]+\):', content)
        if not class_match:
            logger.info("  ⚠️  No class definition found")
            return content

        class_name = class_match.group(1)

        # Find a good place to insert refresh_theme
        # Look for __init__ method
        init_match = re.search(r'(\n    def __init__\(.*?\n(?:.*?\n)*?        super\(\).__init__\(.*?\)\n)', content)
        if init_match:
            # Insert after __init__
            insert_pos = init_match.end()

            refresh_method = '''
    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
'''
            content = content[:insert_pos] + refresh_method + content[insert_pos:]
            logger.info(f"  ➕ Added refresh_theme() to {class_name}")

        return content

    def _print_summary(self):
        """Print summary of fixes"""
        logger.info("\n" + "=" * 80)
        logger.info("FIX SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\n✅ Fixed: {len(self.fixed_files)} files")
        logger.info(f"⏭️  Skipped: {len(self.skipped_files)} files")
        logger.info(f"❌ Errors: {len(self.errors)} files")

        if self.errors:
            logger.info("\nERRORS:")
            for path, error in self.errors:
                logger.info(f"  {path.name}: {error}")

        logger.info("\nFIXED FILES:")
        for path in self.fixed_files[:20]:  # Show first 20
            logger.info(f"  ✅ {path.relative_to(self.ui_dir)}")
        if len(self.fixed_files) > 20:
            logger.info(f"  ... and {len(self.fixed_files) - 20} more")


if __name__ == "__main__":
    ui_dir = Path(r"c:\Users\chauk\Documents\GeoX_Clean\block_model_viewer\ui")
    fixer = ThemeFixer(ui_dir)
    fixer.fix_all_files()
