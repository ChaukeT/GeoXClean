# Building the GeoX Windows Installer

This guide explains how to build a Windows installer for GeoX.

## Quick Start

```batch
# One-click build (creates both .exe and installer)
build_installer.bat
```

## Prerequisites

### Required
1. **Python 3.10+** - https://www.python.org/downloads/
2. **PyInstaller** - Will be installed automatically by the build script

### Optional (for creating installer)
3. **Inno Setup 6.x** - https://jrsoftware.org/isdl.php
   - Download and install the latest version
   - Default installation path is fine

## Build Options

### Option 1: Full Build (Recommended)

Creates both the standalone executable and a Windows installer:

```batch
build_installer.bat
```

**Output:**
- `dist/GeoX/GeoX.exe` - Standalone application
- `dist/GeoX_Setup_1.0.0.exe` - Windows installer

### Option 2: Executable Only

Creates just the executable without an installer:

```batch
build_installer.bat --exe
```

**Output:**
- `dist/GeoX/` - Portable application folder

### Option 3: Clean Build

Removes all build artifacts before building:

```batch
build_installer.bat --clean
```

## Manual Build Steps

### Step 1: Build Executable with PyInstaller

```batch
# Install PyInstaller if not already installed
pip install pyinstaller

# Build the executable
pyinstaller GeoX.spec --noconfirm --clean
```

This creates `dist/GeoX/GeoX.exe` with all dependencies.

### Step 2: Build Installer with Inno Setup

1. Open Inno Setup Compiler
2. File -> Open -> `installer/BlockModelViewer.iss`
3. Build -> Compile (or press Ctrl+F9)

This creates `dist/GeoX_Setup_1.0.0.exe`.

## Distribution Options

### Option A: Windows Installer (.exe)
- File: `dist/GeoX_Setup_1.0.0.exe`
- Creates Start Menu shortcuts
- Adds uninstaller
- Registers in Windows Programs
- **Best for:** End users

### Option B: Portable Application (folder)
- Folder: `dist/GeoX/`
- Zip the entire folder for distribution
- No installation required
- **Best for:** USB drives, testing, users without admin rights

### Option C: MSI Installer (Enterprise)
- Use the PowerShell script: `installer/windows/build-msi.ps1`
- Requires WiX Toolset v4
- **Best for:** Enterprise deployment via Group Policy

## Build Files

| File | Purpose |
|------|---------|
| `GeoX.spec` | PyInstaller configuration |
| `version_info.txt` | Windows version resources |
| `build_installer.bat` | One-click build script |
| `installer/BlockModelViewer.iss` | Inno Setup script |
| `installer/windows/build-msi.ps1` | WiX MSI build script |

## Troubleshooting

### PyInstaller Errors

**Missing modules:**
Add them to `hiddenimports` in `GeoX.spec`.

**Missing data files:**
Add them to `datas` in `GeoX.spec`.

### Build takes too long
First build takes 10-15 minutes due to dependency analysis. Subsequent builds are faster.

### Antivirus blocks the executable
PyInstaller-generated executables are sometimes flagged as false positives. You may need to:
1. Sign the executable with a code signing certificate
2. Submit to antivirus vendors for whitelisting

### Application doesn't start
Run from command line to see error messages:
```batch
dist\GeoX\GeoX.exe
```

Or enable console mode in `GeoX.spec`:
```python
console=True,  # Change from False
```

## Updating Version

1. Update version in `pyproject.toml`
2. Update version in `version_info.txt`
3. Update version in `installer/BlockModelViewer.iss`
4. Rebuild

## Code Signing (Optional)

For professional distribution, sign the executable:

```batch
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\GeoX\GeoX.exe
```
