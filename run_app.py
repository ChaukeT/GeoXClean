import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-flight check: verify critical dependencies before importing the app
_MISSING = []
for _mod in ("PyQt6", "numpy", "pandas", "pyvista"):
    try:
        __import__(_mod)
    except ImportError:
        _MISSING.append(_mod)
if _MISSING:
    print(
        f"ERROR: Missing required packages: {', '.join(_MISSING)}\n"
        f"Python interpreter: {sys.executable}\n"
        f"\nIf you are using Anaconda, activate the environment first:\n"
        f"  conda activate base\n"
        f"  python run_app.py\n"
        f"\nOr install the missing packages:\n"
        f"  pip install {' '.join(_MISSING)}",
        file=sys.stderr,
    )
    sys.exit(1)

from block_model_viewer.main import main

if __name__ == "__main__":
    main()
