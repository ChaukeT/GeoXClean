# GeoX - Clean Distribution

This is a clean distribution of the GeoX (Geostatistics) application containing only essential files needed to run and build the software.

## Contents

### Source Code
- **block_model_viewer/** - Main application source code
  - All Python modules and packages
  - UI components and panels
  - Controllers and models
  - Visualization and rendering
  - Geostatistics engines (including RBF interpolation)
  - All core functionality

- **geox/** - Additional GeoX modules

### Assets
- **block_model_viewer/assets/** - Application assets
  - Icons and branding files
  - Themes and stylesheets
  - UI resources

### Configuration Files
- **requirements.txt** - Python dependencies
- **requirements-dev.txt** - Development dependencies
- **pyproject.toml** - Project metadata
- **setup.py** - Package setup script
- **MANIFEST.in** - Package manifest
- **GeoX.spec** - PyInstaller specification
- **app.ico** - Application icon

### Installer Files
- **installer/** - Installer build scripts
  - Windows MSI installer (WiX)
  - macOS installer scripts
  - Build automation

### Entry Points
- **run_app.py** - Main application entry point
- **run_app_py313.bat** - Windows batch launcher
- **run_app_py313.ps1** - PowerShell launcher

## Excluded Files

The following were excluded from this clean distribution:

- **Test files** - All `*test*.py` files
- **Documentation** - Markdown files, reports, guides
- **Build artifacts** - `build/`, `dist/`, `__pycache__/`
- **Log files** - All `.log`, `.txt` log files
- **Temporary files** - `.diff`, `.tmp`, `.bak` files
- **Development scripts** - Analysis, cleanup, debug scripts
- **Old/legacy files** - Backup and deprecated code
- **Audit/review files** - Development audit reports

## Usage

### Running the Application

```bash
# Direct Python execution
python run_app.py

# Or use the batch file (Windows)
run_app_py313.bat
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Building with PyInstaller

```bash
pyinstaller GeoX.spec
```

### Building MSI Installer

```powershell
# Windows
.\installer\windows\build-msi.ps1
```

## Statistics

- **Files**: 510 source files
- **Directories**: 63 folders
- **Size**: ~8.76 MB (source code only, excluding dependencies)

## Requirements

- Python 3.12+
- See `requirements.txt` for full dependency list
- PyQt6, PyVista, NumPy, SciPy, Pandas, and other geostatistics libraries

## License

See LICENSE file (if included) or project documentation.

## Support

For issues or questions, refer to the main project documentation or repository.

