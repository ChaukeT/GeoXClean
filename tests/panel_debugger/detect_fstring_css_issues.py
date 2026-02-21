"""
Detect f-string CSS brace issues without modifying files.

Finds places where CSS braces are not escaped in f-strings.
"""

import re
from pathlib import Path


def check_file(file_path: Path) -> list:
    """
    Check a single file for f-string CSS issues.

    Returns:
        List of (line_number, issue_description) tuples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return []

    issues = []
    in_fstring = False
    fstring_start = 0

    for i, line in enumerate(lines, 1):
        # Detect start of f-string
        if re.search(r'return\s+f["\']{{3}}', line):
            in_fstring = True
            fstring_start = i

        if in_fstring:
            # Check for CSS selector with single braces
            # Pattern: word followed by single { (not {{)
            # Examples: "ModernStatusBar {", "QWidget {"
            if re.search(r'\w+\s+\{(?!\{)', line):
                issues.append((i, f"Unescaped CSS selector brace (should be {{{{)"))

            # Check for closing CSS brace
            if re.search(r'(?<!\})\}(?!\})', line) and 'return' not in line:
                # Single } that's not part of }}
                if not re.search(r'\{[^}]*\}', line):  # Not a Python var
                    issues.append((i, f"Unescaped CSS closing brace (should be }}}})"))

        # Detect end of f-string
        if in_fstring and '"""' in line and 'return' not in line:
            in_fstring = False

    return issues


def main():
    """Run detection across all UI files."""
    ui_dir = Path('block_model_viewer/ui')

    if not ui_dir.exists():
        print(f"Error: {ui_dir} not found. Run from project root.")
        return

    print("=" * 80)
    print("F-STRING CSS BRACE ISSUE DETECTION")
    print("=" * 80)
    print()

    files_with_issues = []
    total_issues = 0

    for py_file in sorted(ui_dir.glob('**/*.py')):
        if py_file.name.startswith('__'):
            continue

        issues = check_file(py_file)

        if issues:
            files_with_issues.append((py_file, issues))
            total_issues += len(issues)

            print(f"\n{py_file.name}")
            print("-" * 80)
            for line_no, description in issues:
                print(f"  Line {line_no:4d}: {description}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files with issues: {len(files_with_issues)}")
    print(f"Total issues:      {total_issues}")
    print()

    if files_with_issues:
        print("Files needing fixes:")
        for file, issues in files_with_issues:
            print(f"  - {file.name:50s} ({len(issues)} issues)")

    print()
    print("FIX: Replace { with {{ and } with }} in CSS blocks within f-strings")
    print("KEEP: Python variables like {ModernColors.PANEL_BG} stay as single braces")


if __name__ == '__main__':
    main()
