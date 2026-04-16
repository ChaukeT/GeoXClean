"""Regression guard: nothing in production should import the legacy
variogram engine modules.

The variogram engine consolidation (Option A) deleted
``block_model_viewer/models/variogram3d.py`` and
``block_model_viewer/models/variogram_functions.py``; every production
caller now routes through ``block_model_viewer/geostats/`` modules
instead. This test scans the entire codebase for any file that still
references those legacy modules and fails if one appears.

Allow-list:
  - ``block_model_viewer/models/variogram3d.py`` itself (it will be
    deleted in D3 but may briefly exist during staged commits).
  - ``block_model_viewer/models/variogram_functions.py`` itself (same).
  - ``block_model_viewer/geostats/variogram_recommender.py`` — the
    intentional facade whose body will be ported post-D3.
  - ``tests/**`` — tests are free to reference either path.

See ``C:/Users/chauk/.claude/plans/vectorized-foraging-pizza.md``
Phase D2 for the plan this test enforces.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = REPO_ROOT / "block_model_viewer"

LEGACY_PATTERNS = [
    # Full-path dotted imports
    re.compile(r"from\s+block_model_viewer\.models\.variogram3d\b"),
    re.compile(r"from\s+block_model_viewer\.models\.variogram_functions\b"),
    # Relative imports (dot-prefixed)
    re.compile(r"from\s+\.\.models\.variogram3d\b"),
    re.compile(r"from\s+\.\.models\.variogram_functions\b"),
    re.compile(r"from\s+\.variogram3d\b"),
    re.compile(r"from\s+\.variogram_functions\b"),
    # Aliased `import X as Y` forms
    re.compile(r"import\s+.*variogram3d\s+as\b"),
    re.compile(r"import\s+.*variogram_functions\s+as\b"),
    # Bare `import X` forms
    re.compile(r"^\s*import\s+block_model_viewer\.models\.variogram3d\b", re.M),
    re.compile(r"^\s*import\s+block_model_viewer\.models\.variogram_functions\b", re.M),
]

ALLOW_LIST = {
    # Legacy files themselves — delete in D3, but may transiently exist.
    (PRODUCTION_ROOT / "models" / "variogram3d.py").resolve(),
    (PRODUCTION_ROOT / "models" / "variogram_functions.py").resolve(),
    # Intentional facade. Will be emptied when the recommender body is
    # ported into geostats/ after D3.
    (PRODUCTION_ROOT / "geostats" / "variogram_recommender.py").resolve(),
}


def _iter_production_py_files():
    for path in PRODUCTION_ROOT.rglob("*.py"):
        if not path.is_file():
            continue
        # Skip bundled legacy test folder inside the package
        if "tests" in path.parts:
            continue
        # Skip the legacy files themselves and the facade
        if path.resolve() in ALLOW_LIST:
            continue
        yield path


def test_no_production_legacy_variogram_imports():
    """Fail if any production module imports the legacy variogram path."""
    offenders: list[tuple[str, int, str]] = []
    for path in _iter_production_py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"Could not read {path}: {exc}")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in LEGACY_PATTERNS:
                if pattern.search(line):
                    offenders.append((str(path.relative_to(REPO_ROOT)), lineno, line.strip()))
                    break

    if offenders:
        msg_lines = ["Legacy variogram imports found in production files:"]
        for rel, lineno, line in offenders:
            msg_lines.append(f"  {rel}:{lineno}  {line}")
        msg_lines.append("")
        msg_lines.append(
            "Production code must import from block_model_viewer.geostats "
            "(experimental_variogram / variogram_fitting / variogram_orientation "
            "/ variogram_bridge_v2 / variogram_recommender) instead."
        )
        pytest.fail("\n".join(msg_lines))
