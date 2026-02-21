"""
Fix f-string CSS brace issues across the codebase.

This script finds f-strings containing CSS blocks and properly escapes
the CSS braces while preserving Python variable interpolations.

Issue: In f-strings, { and } are Python syntax, but CSS uses them for selectors.
Fix: CSS braces must be doubled: {{ and }}
Keep: Python variables stay single: {ModernColors.PANEL_BG}
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_css_braces_in_fstring(content: str, start_pos: int, end_pos: int) -> str:
    """
    Fix CSS braces within an f-string.

    Strategy:
    1. First, temporarily protect Python variable interpolations {var}
    2. Then double all remaining { and } (these are CSS braces)
    3. Finally, restore Python variable interpolations
    """
    fstring_content = content[start_pos:end_pos]

    # Step 1: Protect Python interpolations by replacing with placeholders
    # Pattern: {anything_that's_not_a_closing_brace}
    # This matches {ModernColors.PANEL_BG}, {Colors.TEXT}, etc.
    python_vars = []

    def save_python_var(match):
        python_vars.append(match.group(0))
        return f"<<<PYVAR_{len(python_vars)-1}>>>"

    # Match {something} where something is a Python expression (no spaces around CSS properties)
    # Python expressions typically have: ModernColors.X, Colors.X, self.X, variable_name
    protected = re.sub(
        r'\{([A-Za-z_][A-Za-z0-9_.()[\]"\'\s,]*)\}',
        save_python_var,
        fstring_content
    )

    # Step 2: Double all remaining { and } (these are CSS braces)
    protected = protected.replace('{', '{{').replace('}', '}}')

    # Step 3: Restore Python variables (un-double them)
    for i, var in enumerate(python_vars):
        protected = protected.replace(f"<<{{{{PYVAR_{i}}}}}>>", var)

    return content[:start_pos] + protected + content[end_pos:]


def fix_file(file_path: Path) -> Tuple[bool, str]:
    """
    Fix f-string CSS issues in a single file.

    Returns:
        (was_modified, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        return False, f"Error reading: {e}"

    content = original_content
    modified = False

    # Find all f-strings (both f""" and f''')
    # Pattern: return f""" ... """ or return f''' ... '''
    pattern = re.compile(
        r'(return\s+f""")(.*?)(""")',
        re.DOTALL
    )

    matches = list(pattern.finditer(content))

    # Process matches in reverse order to preserve positions
    for match in reversed(matches):
        fstring_body = match.group(2)

        # Check if this f-string contains CSS (has properties like background:, color:, etc.)
        if re.search(r'(?:background|color|border|margin|padding|font-|width|height):', fstring_body):
            # Check if it has unescaped CSS braces (selector braces not doubled)
            # Look for pattern: word followed by space and single {
            if re.search(r'\w+\s+\{[^{]', fstring_body):
                # This needs fixing
                start = match.start()
                end = match.end()
                content = fix_css_braces_in_fstring(content, start, end)
                modified = True

    if modified:
        # Write back
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed CSS braces in f-strings"
        except Exception as e:
            return False, f"Error writing: {e}"

    return False, "No changes needed"


def main():
    """Run the fix across all UI files."""
    ui_dir = Path('block_model_viewer/ui')

    if not ui_dir.exists():
        print(f"Error: {ui_dir} not found. Run from project root.")
        return

    print("=" * 80)
    print("F-STRING CSS BRACE FIX")
    print("=" * 80)
    print()

    fixed_files = []
    skipped_files = []
    error_files = []

    for py_file in sorted(ui_dir.glob('**/*.py')):
        if py_file.name.startswith('__'):
            continue

        was_modified, message = fix_file(py_file)

        if was_modified:
            fixed_files.append((py_file, message))
            print(f"✅ {py_file.name}: {message}")
        elif "Error" in message:
            error_files.append((py_file, message))
            print(f"❌ {py_file.name}: {message}")
        else:
            skipped_files.append(py_file)
            # Don't print skipped files (too verbose)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Fixed:   {len(fixed_files)} files")
    print(f"Skipped: {len(skipped_files)} files (no issues)")
    print(f"Errors:  {len(error_files)} files")
    print()

    if fixed_files:
        print("Fixed files:")
        for file, msg in fixed_files:
            print(f"  - {file.name}")

    if error_files:
        print()
        print("Files with errors:")
        for file, msg in error_files:
            print(f"  - {file.name}: {msg}")


if __name__ == '__main__':
    main()
