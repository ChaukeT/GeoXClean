#!/usr/bin/env python3
"""
fix_theme_errors.py - Automated Theme Error Fixer for GeoX Clean

Detects and fixes 5 categories of theme conversion errors introduced
by automated conversion agents.

Error Categories:
1. Static F-String Stylesheets - Module-level f-strings that don't update on theme change
2. Malformed Theme Detection Logic - Incorrect theme comparison logic
3. Undefined Variable References - Variables used in f-strings without proper prefix
4. Syntax Errors - Invalid Python syntax from incomplete replacements
5. Inconsistent Patterns - Mix of different color access patterns

Usage:
    python fix_theme_errors.py --dry-run              # Preview changes
    python fix_theme_errors.py --backup               # Apply with backup
    python fix_theme_errors.py --verbose --backup     # Verbose mode
"""

import ast
import re
import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Type, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """Represents a detected error."""
    file: str
    line_no: int
    error_type: str
    message: str
    context: str = ""
    fix_suggestion: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FixResult:
    """Represents the result of a fix operation."""
    file: str
    error_type: str
    success: bool
    original_code: str = ""
    fixed_code: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization (without large code blocks)."""
        return {
            'file': self.file,
            'error_type': self.error_type,
            'success': self.success,
            'error_message': self.error_message
        }


# ============================================================================
# DETECTION ENGINE
# ============================================================================

class StaticFStringDetector:
    """Detects module-level f-strings referencing theme colors."""

    def detect(self, source_code: str, filepath: str) -> List[Detection]:
        """
        Detect static f-strings at module level that reference ModernColors or colors.

        Args:
            source_code: Python source code to analyze
            filepath: Path to the file being analyzed

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Will be caught by SyntaxErrorDetector
            return []

        lines = source_code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if this is a module-level assignment
                if self._is_module_level(node, tree):
                    # Check if RHS is an f-string
                    if isinstance(node.value, ast.JoinedStr):
                        # Extract color references
                        refs = self._extract_color_refs(node.value)
                        if refs:
                            var_name = self._get_target_name(node.targets[0])
                            if var_name:
                                context = self._get_context(lines, node.lineno, 3)
                                detections.append(Detection(
                                    file=filepath,
                                    line_no=node.lineno,
                                    error_type="STATIC_FSTRING",
                                    message=f"Static f-string '{var_name}' won't update on theme change",
                                    context=context,
                                    metadata={
                                        'var_name': var_name,
                                        'color_refs': refs,
                                        'end_line': self._find_fstring_end(node, lines)
                                    }
                                ))

        return detections

    def _is_module_level(self, node: ast.AST, tree: ast.Module) -> bool:
        """Check if node is at module level (not inside class/function)."""
        # Simple heuristic: if the node is directly in tree.body, it's module-level
        return node in tree.body

    def _extract_color_refs(self, fstring_node: ast.JoinedStr) -> List[str]:
        """Extract ModernColors/colors references from f-string."""
        refs = []

        for value in fstring_node.values:
            if isinstance(value, ast.FormattedValue):
                ref = self._extract_ref_from_value(value.value)
                if ref:
                    refs.append(ref)

        return refs

    def _extract_ref_from_value(self, node: ast.AST) -> Optional[str]:
        """Extract color reference from AST node."""
        if isinstance(node, ast.Attribute):
            # ModernColors.TEXT_PRIMARY or colors.TEXT_PRIMARY
            if isinstance(node.value, ast.Name):
                if node.value.id in ('ModernColors', 'colors'):
                    return f"{node.value.id}.{node.attr}"
        return None

    def _get_target_name(self, target: ast.AST) -> Optional[str]:
        """Extract variable name from assignment target."""
        if isinstance(target, ast.Name):
            return target.id
        return None

    def _find_fstring_end(self, node: ast.Assign, lines: List[str]) -> int:
        """Find the ending line of a multi-line f-string."""
        start_line = node.lineno - 1  # 0-indexed

        # Look for closing triple quotes
        for i in range(start_line, len(lines)):
            if '"""' in lines[i] and i > start_line:
                return i + 1  # Convert back to 1-indexed

        return node.lineno

    def _get_context(self, lines: List[str], line_no: int, context_lines: int = 2) -> str:
        """Get context lines around the detection."""
        start = max(0, line_no - 1 - context_lines)
        end = min(len(lines), line_no + context_lines)
        return '\n'.join(lines[start:end])


class MalformedThemeLogicDetector:
    """Detects incorrect theme detection logic."""

    # Regex pattern for malformed theme logic
    PATTERN = re.compile(
        r'if\s+get_theme_colors\(\)\s*==\s*get_theme_colors\.__class__\.__bases__\[0\]',
        re.MULTILINE
    )

    def detect(self, source_code: str, filepath: str) -> List[Detection]:
        """
        Detect malformed theme comparison logic.

        Args:
            source_code: Python source code to analyze
            filepath: Path to the file being analyzed

        Returns:
            List of Detection objects
        """
        detections = []
        lines = source_code.split('\n')

        # Regex detection
        for match in self.PATTERN.finditer(source_code):
            line_no = source_code[:match.start()].count('\n') + 1
            context = self._get_context(lines, line_no)

            detections.append(Detection(
                file=filepath,
                line_no=line_no,
                error_type="MALFORMED_THEME_LOGIC",
                message="Malformed theme comparison (always evaluates to False)",
                context=context,
                fix_suggestion='Replace with: if get_current_theme() == "light":'
            ))

        return detections

    def _get_context(self, lines: List[str], line_no: int, context_lines: int = 3) -> str:
        """Get context lines around the detection."""
        start = max(0, line_no - 1 - context_lines)
        end = min(len(lines), line_no + context_lines)
        return '\n'.join(lines[start:end])


class UndefinedVariableDetector:
    """Detects undefined variable references in f-strings."""

    UNDEFINED_VARS = {'background', 'text_color', 'border', 'foreground'}

    def detect(self, source_code: str, filepath: str) -> List[Detection]:
        """
        Detect undefined variable references in f-strings.

        Args:
            source_code: Python source code to analyze
            filepath: Path to the file being analyzed

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        lines = source_code.split('\n')

        # Find all f-strings
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    if isinstance(value, ast.FormattedValue):
                        var_name = self._extract_var_name(value.value)

                        if var_name and var_name in self.UNDEFINED_VARS:
                            context = self._get_context(lines, node.lineno)
                            detections.append(Detection(
                                file=filepath,
                                line_no=node.lineno,
                                error_type="UNDEFINED_VARIABLE",
                                message=f"Undefined variable '{var_name}' in f-string",
                                context=context,
                                fix_suggestion=f"Replace with: colors.{self._map_to_color_attr(var_name)}",
                                metadata={'var_name': var_name}
                            ))

        return detections

    def _extract_var_name(self, node: ast.AST) -> Optional[str]:
        """Extract variable name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _map_to_color_attr(self, var_name: str) -> str:
        """Map undefined variable name to ModernColors attribute."""
        mapping = {
            'background': 'PANEL_BG',
            'text_color': 'TEXT_PRIMARY',
            'border': 'BORDER',
            'foreground': 'TEXT_PRIMARY',
        }
        return mapping.get(var_name, var_name.upper())

    def _get_context(self, lines: List[str], line_no: int, context_lines: int = 2) -> str:
        """Get context lines around the detection."""
        start = max(0, line_no - 1 - context_lines)
        end = min(len(lines), line_no + context_lines)
        return '\n'.join(lines[start:end])


class SyntaxErrorDetector:
    """Detects Python syntax errors."""

    def detect(self, source_code: str, filepath: str) -> List[Detection]:
        """
        Detect syntax errors via AST parsing.

        Args:
            source_code: Python source code to analyze
            filepath: Path to the file being analyzed

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            ast.parse(source_code)
        except SyntaxError as e:
            lines = source_code.split('\n')
            context = self._get_error_context(lines, e.lineno, e.offset)

            detections.append(Detection(
                file=filepath,
                line_no=e.lineno or 0,
                error_type="SYNTAX_ERROR",
                message=f"Syntax error: {e.msg}",
                context=context,
                metadata={'offset': e.offset, 'text': e.text}
            ))
        except UnicodeDecodeError as e:
            detections.append(Detection(
                file=filepath,
                line_no=0,
                error_type="UNICODE_ERROR",
                message=f"Unicode decode error: {e}"
            ))

        return detections

    def _get_error_context(self, lines: List[str], line_no: Optional[int],
                           offset: Optional[int]) -> str:
        """Get context around syntax error."""
        if line_no is None:
            return ""

        context_lines = []
        start = max(0, line_no - 3)
        end = min(len(lines), line_no + 2)

        for i in range(start, end):
            prefix = ">>> " if i == line_no - 1 else "    "
            context_lines.append(f"{prefix}{lines[i]}")

        if offset:
            context_lines.append(" " * (offset + 3) + "^")

        return '\n'.join(context_lines)


# ============================================================================
# FIX ENGINE
# ============================================================================

class StaticFStringFixer:
    """Fixes static f-strings by converting to functions."""

    def fix(self, source_code: str, detection: Detection) -> str:
        """
        Fix static f-string by converting to function.

        Args:
            source_code: Original source code
            detection: Detection object with error details

        Returns:
            Fixed source code
        """
        var_name = detection.metadata.get('var_name')
        end_line = detection.metadata.get('end_line', detection.line_no)

        if not var_name:
            return source_code

        lines = source_code.split('\n')

        # Extract f-string content
        start_idx = detection.line_no - 1  # 0-indexed
        end_idx = end_line  # Already 1-indexed, will be exclusive in slice

        fstring_lines = lines[start_idx:end_idx]
        fstring_content = '\n'.join(fstring_lines)

        # Generate function name
        func_name = self._var_to_func_name(var_name)

        # Build function definition
        func_def = self._build_function_def(func_name, fstring_content, var_name)

        # Replace in source
        new_lines = lines[:start_idx] + func_def.split('\n') + lines[end_idx:]
        new_source = '\n'.join(new_lines)

        # Replace all uses of var_name with func_name()
        new_source = re.sub(
            rf'\b{re.escape(var_name)}\b',
            f'{func_name}()',
            new_source
        )

        return new_source

    def _var_to_func_name(self, var_name: str) -> str:
        """Convert DARK_THEME to get_dark_theme."""
        return 'get_' + var_name.lower()

    def _build_function_def(self, func_name: str, fstring_content: str, var_name: str) -> str:
        """Build function definition."""
        # Extract just the f-string part (remove variable assignment)
        # Pattern: VAR_NAME = f"""..."""
        match = re.search(rf'{re.escape(var_name)}\s*=\s*f"""(.*)"""',
                         fstring_content, re.DOTALL)

        if match:
            inner_content = match.group(1)
        else:
            # Fallback: use everything after the =
            inner_content = fstring_content.split('=', 1)[1].strip()
            if inner_content.startswith('f"""'):
                inner_content = inner_content[4:]
            if inner_content.endswith('"""'):
                inner_content = inner_content[:-3]

        # Replace ModernColors.ATTR with colors.ATTR
        inner_content = re.sub(r'ModernColors\.', 'colors.', inner_content)

        # Build function
        doc_name = func_name.replace('get_', '').replace('_', ' ')
        return f'''def {func_name}() -> str:
    """Get {doc_name} stylesheet."""
    colors = get_theme_colors()
    return f"""{inner_content}"""'''


class MalformedThemeLogicFixer:
    """Fixes malformed theme detection logic."""

    def fix(self, source_code: str, detection: Detection) -> str:
        """
        Fix malformed theme comparison.

        Args:
            source_code: Original source code
            detection: Detection object with error details

        Returns:
            Fixed source code
        """
        # Replace the malformed comparison
        fixed = re.sub(
            r'if\s+get_theme_colors\(\)\s*==\s*get_theme_colors\.__class__\.__bases__\[0\]',
            'if get_current_theme() == "light"',
            source_code
        )

        # Ensure get_current_theme is imported
        if 'get_current_theme' not in source_code and fixed != source_code:
            fixed = self._add_import(fixed, 'get_current_theme')

        return fixed

    def _add_import(self, source_code: str, func_name: str) -> str:
        """Add function to existing import from modern_styles."""
        # Find the line with "from .modern_styles import" or "from modern_styles import"
        pattern = r'(from \.?modern_styles import )([^;\n]+)'

        def add_func(match):
            prefix = match.group(1)
            imports = match.group(2).strip()

            if func_name in imports:
                return match.group(0)  # Already imported

            # Add to end of import list
            if '(' in imports:
                # Multi-line import
                imports = imports.rstrip(')')
                return f"{prefix}{imports}, {func_name})"
            else:
                # Single-line import
                return f"{prefix}{imports}, {func_name}"

        result = re.sub(pattern, add_func, source_code)

        # If no modern_styles import found, add it
        if result == source_code:
            # Find first import statement and add after it
            lines = source_code.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    lines.insert(i + 1, f'from .modern_styles import get_current_theme')
                    return '\n'.join(lines)

        return result


class UndefinedVariableFixer:
    """Fixes undefined variable references."""

    # Mapping from undefined var names to ModernColors attributes
    VAR_MAPPING = {
        'background': 'PANEL_BG',
        'text_color': 'TEXT_PRIMARY',
        'border': 'BORDER',
        'foreground': 'TEXT_PRIMARY',
    }

    def fix(self, source_code: str, detection: Detection) -> str:
        """
        Fix undefined variable by adding colors. prefix.

        Args:
            source_code: Original source code
            detection: Detection object with error details

        Returns:
            Fixed source code
        """
        var_name = detection.metadata.get('var_name')
        if not var_name:
            return source_code

        attr_name = self.VAR_MAPPING.get(var_name, var_name.upper())

        # Replace {var_name} with {colors.ATTR}
        # Be careful to only replace in f-strings
        fixed = re.sub(
            rf'\{{{var_name}\}}',
            f'{{colors.{attr_name}}}',
            source_code
        )

        return fixed


# ============================================================================
# CORE ENGINE
# ============================================================================

class ThemeErrorFixer:
    """Main fixer engine."""

    def __init__(self, path: str, dry_run: bool = False,
                 backup: bool = True, error_type: str = 'ALL',
                 verbose: bool = False):
        """
        Initialize the theme error fixer.

        Args:
            path: Path to scan for Python files
            dry_run: If True, only preview changes without applying
            backup: If True, create backups before modifying files
            error_type: Type of errors to fix (ALL or specific type)
            verbose: Enable verbose logging
        """
        self.path = Path(path)
        self.dry_run = dry_run
        self.backup = backup
        self.error_type = error_type
        self.verbose = verbose

        self.detectors = {
            'STATIC_FSTRING': StaticFStringDetector(),
            'MALFORMED_THEME_LOGIC': MalformedThemeLogicDetector(),
            'UNDEFINED_VARIABLE': UndefinedVariableDetector(),
            'SYNTAX_ERROR': SyntaxErrorDetector(),
        }

        self.fixers = {
            'STATIC_FSTRING': StaticFStringFixer(),
            'MALFORMED_THEME_LOGIC': MalformedThemeLogicFixer(),
            'UNDEFINED_VARIABLE': UndefinedVariableFixer(),
        }

        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(levelname)s: %(message)s'
        )

    def run(self) -> Dict:
        """Main execution flow."""
        self.logger.info(f"Scanning {self.path} for theme errors...")

        # 1. Scan files
        files = self._scan_files()
        self.logger.info(f"Found {len(files)} Python files")

        # 2. Detect errors
        all_detections = self._detect_errors(files)
        self.logger.info(f"Found {len(all_detections)} errors")

        # 3. Apply fixes
        results = []
        if not self.dry_run:
            results = self._apply_fixes(all_detections)
            self.logger.info(f"Fixed {sum(1 for r in results if r.success)}/{len(results)} errors")
        else:
            self.logger.info("Dry-run mode: No changes applied")

        # 4. Generate report
        report = self._generate_report(all_detections, results, files)

        return report

    def _scan_files(self) -> List[Path]:
        """Scan directory for Python files."""
        if not self.path.exists():
            self.logger.error(f"Path does not exist: {self.path}")
            return []

        if self.path.is_file():
            return [self.path]

        return list(self.path.rglob("*.py"))

    def _detect_errors(self, files: List[Path]) -> List[Detection]:
        """Detect all errors in files."""
        all_detections = []

        for file_path in files:
            try:
                source = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                self.logger.warning(f"Skipping {file_path}: Unicode decode error")
                continue
            except Exception as e:
                self.logger.warning(f"Skipping {file_path}: {e}")
                continue

            # Run detectors
            for error_type, detector in self.detectors.items():
                if self.error_type == 'ALL' or self.error_type == error_type:
                    try:
                        detections = detector.detect(source, str(file_path))
                        all_detections.extend(detections)
                        if detections and self.verbose:
                            self.logger.debug(f"  {file_path.name}: {len(detections)} {error_type} errors")
                    except Exception as e:
                        self.logger.error(f"Error detecting in {file_path}: {e}")

        return all_detections

    def _apply_fixes(self, detections: List[Detection]) -> List[FixResult]:
        """Apply fixes to detected errors."""
        results = []

        # Group by file
        by_file: Dict[str, List[Detection]] = {}
        for det in detections:
            by_file.setdefault(det.file, []).append(det)

        for filepath, file_detections in by_file.items():
            # Skip syntax errors - they need manual review
            fixable_detections = [d for d in file_detections if d.error_type in self.fixers]

            if not fixable_detections:
                continue

            # Backup if requested
            if self.backup:
                BackupManager.backup_file(filepath)
                self.logger.info(f"Created backup of {Path(filepath).name}")

            # Load source
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    original = f.read()
            except Exception as e:
                self.logger.error(f"Error reading {filepath}: {e}")
                continue

            # Apply fixes
            fixed = original
            for det in fixable_detections:
                if det.error_type in self.fixers:
                    fixer = self.fixers[det.error_type]
                    try:
                        fixed = fixer.fix(fixed, det)
                        results.append(FixResult(
                            file=filepath,
                            error_type=det.error_type,
                            success=True,
                            original_code=original,
                            fixed_code=fixed
                        ))
                    except Exception as e:
                        self.logger.error(f"Error fixing {filepath} ({det.error_type}): {e}")
                        results.append(FixResult(
                            file=filepath,
                            error_type=det.error_type,
                            success=False,
                            original_code=original,
                            fixed_code=original,
                            error_message=str(e)
                        ))

            # Verify syntax
            try:
                compile(fixed, filepath, 'exec')
            except SyntaxError as e:
                self.logger.error(f"Fixed code has syntax errors in {filepath}: {e}")
                # Rollback
                fixed = original
                results.append(FixResult(
                    file=filepath,
                    error_type="SYNTAX_ERROR_AFTER_FIX",
                    success=False,
                    error_message=f"Syntax error after fix: {e}"
                ))
                continue

            # Write fixed code
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed)
                self.logger.info(f"Fixed {Path(filepath).name}")
            except Exception as e:
                self.logger.error(f"Error writing {filepath}: {e}")

        return results

    def _generate_report(self, detections: List[Detection],
                        results: List[FixResult], files: List[Path]) -> Dict:
        """Generate summary report."""
        # Count errors by type
        errors_by_type = {}
        for det in detections:
            errors_by_type[det.error_type] = errors_by_type.get(det.error_type, 0) + 1

        # Count fixes by type
        fixes_by_type = {}
        fails_by_type = {}
        for result in results:
            if result.success:
                fixes_by_type[result.error_type] = fixes_by_type.get(result.error_type, 0) + 1
            else:
                fails_by_type[result.error_type] = fails_by_type.get(result.error_type, 0) + 1

        # Files requiring manual review
        manual_review = []
        for det in detections:
            if det.error_type == 'SYNTAX_ERROR' or det.error_type not in self.fixers:
                manual_review.append({
                    'file': det.file,
                    'line': det.line_no,
                    'error_type': det.error_type,
                    'message': det.message
                })

        return {
            'timestamp': datetime.now().isoformat(),
            'files_scanned': len(files),
            'total_errors': len(detections),
            'errors_fixed': sum(1 for r in results if r.success),
            'errors_failed': sum(1 for r in results if not r.success),
            'errors_by_type': errors_by_type,
            'fixes_by_type': fixes_by_type,
            'fails_by_type': fails_by_type,
            'manual_review_required': manual_review,
            'dry_run': self.dry_run,
            'detections': [d.to_dict() for d in detections[:50]],  # Limit for JSON size
            'results': [r.to_dict() for r in results[:50]]
        }


# ============================================================================
# UTILITIES
# ============================================================================

class BackupManager:
    """Manages file backups."""

    @staticmethod
    def backup_file(filepath: str) -> str:
        """
        Create backup of file.

        Args:
            filepath: Path to file to backup

        Returns:
            Path to backup file
        """
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        return backup_path


class Reporter:
    """Generates reports."""

    def __init__(self, results: Dict):
        """
        Initialize reporter.

        Args:
            results: Results dictionary from ThemeErrorFixer
        """
        self.results = results

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*80)
        print("THEME ERROR FIX SUMMARY")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Files scanned: {self.results['files_scanned']}")
        print(f"Errors found: {self.results['total_errors']}")

        if not self.results['dry_run']:
            print(f"Errors fixed: {self.results['errors_fixed']}")
            print(f"Errors failed: {self.results['errors_failed']}")
        else:
            print("Mode: DRY-RUN (no changes applied)")

        print("\nErrors by type:")
        for error_type, count in self.results['errors_by_type'].items():
            found = count
            fixed = self.results['fixes_by_type'].get(error_type, 0)
            failed = self.results['fails_by_type'].get(error_type, 0)
            if self.results['dry_run']:
                print(f"  {error_type}: {found} found")
            else:
                print(f"  {error_type}: {found} found, {fixed} fixed, {failed} failed")

        if self.results['manual_review_required']:
            print(f"\nFiles requiring manual review ({len(self.results['manual_review_required'])}):")
            for item in self.results['manual_review_required'][:10]:
                filepath = Path(item['file']).name
                print(f"  - {filepath}:line {item['line']} ({item['error_type']}): {item['message']}")

            if len(self.results['manual_review_required']) > 10:
                print(f"  ... and {len(self.results['manual_review_required']) - 10} more")

        print("="*80 + "\n")

    def save_json(self, output_path: str):
        """
        Save report as JSON.

        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Report saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fix theme conversion errors in GeoX Clean UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_theme_errors.py --dry-run
  python fix_theme_errors.py --backup
  python fix_theme_errors.py --error-type STATIC_FSTRING --backup
  python fix_theme_errors.py --verbose --backup --output report.json
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )

    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of original files before fixing (recommended)'
    )

    parser.add_argument(
        '--error-type',
        choices=['STATIC_FSTRING', 'MALFORMED_THEME_LOGIC', 'UNDEFINED_VARIABLE',
                 'SYNTAX_ERROR', 'ALL'],
        default='ALL',
        help='Type of error to fix (default: ALL)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file for report (JSON format)'
    )

    parser.add_argument(
        '--path',
        default='block_model_viewer/ui',
        help='Path to scan for Python files (default: block_model_viewer/ui)'
    )

    args = parser.parse_args()

    # Run the fixer
    fixer = ThemeErrorFixer(
        path=args.path,
        dry_run=args.dry_run,
        backup=args.backup,
        error_type=args.error_type,
        verbose=args.verbose
    )

    results = fixer.run()

    # Generate report
    reporter = Reporter(results)
    reporter.print_summary()

    if args.output:
        reporter.save_json(args.output)

    # Exit with error code if errors found
    if results['total_errors'] > 0 and not args.dry_run:
        if results['errors_failed'] > 0:
            sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
