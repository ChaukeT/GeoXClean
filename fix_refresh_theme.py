#!/usr/bin/env python3
"""
Fix refresh_theme() methods to properly rebuild stylesheets.

This script fixes panels that have the broken pattern:
    def refresh_theme(self):
        colors = get_theme_colors()
        self.setStyleSheet(self.styleSheet())  # Wrong! Just re-applies old stylesheet

The fix:
    1. Extract stylesheet from _build_ui() or __init__() into _get_stylesheet() method
    2. Update refresh_theme() to call self.setStyleSheet(self._get_stylesheet())
    3. Update original stylesheet location to call self._get_stylesheet()
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RefreshThemeFixer:
    """Fixes refresh_theme() methods in PyQt6 panels."""

    def __init__(self, ui_dir: Path):
        self.ui_dir = ui_dir
        self.files_fixed = 0
        self.files_failed = 0
        self.backup_created = False

    def fix_all_panels(self, dry_run: bool = True):
        """Fix all panel files with broken refresh_theme() methods."""

        # Find all Python files in ui directory
        panel_files = list(self.ui_dir.glob("*.py"))
        panel_files.extend(self.ui_dir.glob("**/*.py"))

        logger.info(f"Found {len(panel_files)} Python files to check")

        for filepath in panel_files:
            # Skip backup files
            if '.backup.' in str(filepath):
                continue

            try:
                self.fix_panel(filepath, dry_run=dry_run)
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")
                self.files_failed += 1

        logger.info(f"\n{'DRY RUN ' if dry_run else ''}SUMMARY:")
        logger.info(f"Files fixed: {self.files_fixed}")
        logger.info(f"Files failed: {self.files_failed}")

    def fix_panel(self, filepath: Path, dry_run: bool = True):
        """Fix a single panel file."""

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if file has broken refresh_theme() pattern
        if not self._has_broken_refresh_theme(content):
            return

        logger.info(f"\nProcessing: {filepath.name}")

        # Extract stylesheet from _build_ui() or __init__()
        stylesheet, start_line, end_line = self._extract_stylesheet(content)

        if not stylesheet:
            logger.warning(f"  Could not extract stylesheet from {filepath.name}")
            return

        # Generate the fix
        fixed_content = self._apply_fix(content, stylesheet, start_line, end_line)

        if dry_run:
            logger.info(f"  Would fix {filepath.name}")
            logger.info(f"    - Extracted stylesheet ({len(stylesheet)} chars)")
            logger.info(f"    - Created _get_stylesheet() method")
            logger.info(f"    - Updated refresh_theme()")
        else:
            # Create backup
            backup_path = filepath.with_suffix('.py.backup_refresh_theme')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Write fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            logger.info(f"  ✓ Fixed {filepath.name}")
            logger.info(f"    - Backup: {backup_path.name}")

        self.files_fixed += 1

    def _has_broken_refresh_theme(self, content: str) -> bool:
        """Check if file has broken refresh_theme() pattern."""
        # Look for: def refresh_theme(self): ... self.setStyleSheet(self.styleSheet())
        pattern = r'def refresh_theme\(self\):.*?self\.setStyleSheet\(self\.styleSheet\(\)\)'
        return bool(re.search(pattern, content, re.DOTALL))

    def _extract_stylesheet(self, content: str) -> Tuple[Optional[str], int, int]:
        """
        Extract stylesheet from _build_ui() or __init__().

        Returns:
            (stylesheet_string, start_line_number, end_line_number) or (None, 0, 0)
        """
        # Pattern to match: self.setStyleSheet(f"""...""") or self.setStyleSheet(f'''...''')
        pattern = r'self\.setStyleSheet\(f"""(.*?)"""\)'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            # Try triple single quotes
            pattern = r"self\.setStyleSheet\(f'''(.*?)'''\)"
            match = re.search(pattern, content, re.DOTALL)

        if not match:
            return None, 0, 0

        stylesheet = match.group(1)

        # Find line numbers
        start_pos = match.start()
        end_pos = match.end()
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1

        return stylesheet, start_line, end_line

    def _apply_fix(self, content: str, stylesheet: str, start_line: int, end_line: int) -> str:
        """Apply the fix to the content."""

        lines = content.split('\n')

        # Find the refresh_theme() method
        refresh_theme_line = None
        for i, line in enumerate(lines):
            if 'def refresh_theme(self):' in line:
                refresh_theme_line = i
                break

        if refresh_theme_line is None:
            logger.warning("Could not find refresh_theme() method")
            return content

        # Find indentation level of refresh_theme()
        indent = len(lines[refresh_theme_line]) - len(lines[refresh_theme_line].lstrip())
        indent_str = ' ' * indent

        # Create _get_stylesheet() method
        get_stylesheet_method = [
            f"{indent_str}def _get_stylesheet(self) -> str:",
            f"{indent_str}    \"\"\"Get the stylesheet for current theme.\"\"\"",
            f"{indent_str}    return f\"\"\"",
        ]

        # Add stylesheet lines with proper indentation
        stylesheet_lines = stylesheet.split('\n')
        for line in stylesheet_lines:
            get_stylesheet_method.append(f"{indent_str}    {line}")

        get_stylesheet_method.append(f"{indent_str}    \"\"\"")
        get_stylesheet_method.append("")

        # Insert _get_stylesheet() method before refresh_theme()
        for line in reversed(get_stylesheet_method):
            lines.insert(refresh_theme_line, line)

        # Update refresh_theme() to call self.setStyleSheet(self._get_stylesheet())
        # Find the line with self.setStyleSheet(self.styleSheet())
        for i in range(refresh_theme_line + len(get_stylesheet_method),
                      min(refresh_theme_line + len(get_stylesheet_method) + 20, len(lines))):
            if 'self.setStyleSheet(self.styleSheet())' in lines[i]:
                # Replace with proper call
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                lines[i] = ' ' * current_indent + '# Rebuild stylesheet with new theme colors'
                lines.insert(i + 1, ' ' * current_indent + 'self.setStyleSheet(self._get_stylesheet())')
                break

        # Update original setStyleSheet call to use _get_stylesheet()
        pattern = r'self\.setStyleSheet\(f""".*?"""\)'
        fixed_content = '\n'.join(lines)
        fixed_content = re.sub(
            pattern,
            'self.setStyleSheet(self._get_stylesheet())',
            fixed_content,
            count=1,
            flags=re.DOTALL
        )

        # Also handle triple single quotes
        pattern = r"self\.setStyleSheet\(f'''.*?'''\)"
        fixed_content = re.sub(
            pattern,
            'self.setStyleSheet(self._get_stylesheet())',
            fixed_content,
            count=1,
            flags=re.DOTALL
        )

        return fixed_content


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fix refresh_theme() methods in GeoX panels')
    parser.add_argument('--apply', action='store_true', help='Apply fixes (default is dry-run)')
    parser.add_argument('--ui-dir', type=str, default='block_model_viewer/ui',
                       help='Path to UI directory')

    args = parser.parse_args()

    ui_dir = Path(args.ui_dir)
    if not ui_dir.exists():
        logger.error(f"UI directory not found: {ui_dir}")
        sys.exit(1)

    fixer = RefreshThemeFixer(ui_dir)
    fixer.fix_all_panels(dry_run=not args.apply)

    if not args.apply:
        logger.info("\n💡 This was a DRY RUN. Use --apply to actually fix the files.")


if __name__ == '__main__':
    main()
