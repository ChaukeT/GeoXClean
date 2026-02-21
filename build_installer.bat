@echo off
REM =============================================================================
REM GeoX Windows Installer Build Script
REM =============================================================================
REM
REM This script builds a Windows installer for GeoX in two steps:
REM   1. PyInstaller: Creates dist/GeoX/GeoX.exe
REM   2. Inno Setup:  Creates dist/GeoX_Setup_X.X.X.exe
REM
REM Prerequisites:
REM   - Python 3.10+ with pip
REM   - PyInstaller (will be installed automatically)
REM   - Inno Setup 6.x (download from https://jrsoftware.org/isinfo.php)
REM
REM Usage:
REM   build_installer.bat           - Build everything
REM   build_installer.bat --exe     - Only build executable (skip installer)
REM   build_installer.bat --clean   - Clean build artifacts first
REM
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   GeoX Windows Installer Builder
echo ========================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Parse arguments
set "BUILD_INSTALLER=1"
set "CLEAN_FIRST=0"

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--exe" set "BUILD_INSTALLER=0"
if /i "%~1"=="--clean" set "CLEAN_FIRST=1"
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --exe      Only build executable (skip Inno Setup installer)
echo   --clean    Clean build artifacts before building
echo   -h, --help Show this help message
echo.
exit /b 0

:done_parsing

REM =============================================================================
REM Step 0: Clean (optional)
REM =============================================================================

if "%CLEAN_FIRST%"=="1" (
    echo [0/4] Cleaning build artifacts...
    if exist "build" rmdir /s /q "build"
    if exist "dist" rmdir /s /q "dist"
    echo       Done.
    echo.
)

REM =============================================================================
REM Step 1: Check Python
REM =============================================================================

echo [1/4] Checking prerequisites...

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.10+ and add it to PATH.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo       Python: %PYTHON_VERSION%

REM =============================================================================
REM Step 2: Install/Check PyInstaller
REM =============================================================================

where pyinstaller >nul 2>&1
if errorlevel 1 (
    echo       PyInstaller not found. Installing...
    python -m pip install pyinstaller --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller.
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('pyinstaller --version 2^>^&1') do set "PYINSTALLER_VERSION=%%i"
echo       PyInstaller: %PYINSTALLER_VERSION%
echo.

REM =============================================================================
REM Step 3: Build with PyInstaller
REM =============================================================================

echo [2/4] Building executable with PyInstaller...
echo       This may take 5-15 minutes...
echo.

pyinstaller GeoX.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    echo Check the output above for errors.
    exit /b 1
)

REM Verify output
if not exist "dist\GeoX\GeoX.exe" (
    echo ERROR: GeoX.exe was not created.
    exit /b 1
)

for %%A in ("dist\GeoX\GeoX.exe") do set "EXE_SIZE=%%~zA"
set /a "EXE_SIZE_MB=%EXE_SIZE% / 1048576"
echo       Created: dist\GeoX\GeoX.exe (%EXE_SIZE_MB% MB)

REM Verify llvmlite.dll is included (critical for numba)
echo       Verifying llvmlite.dll inclusion...
if exist "dist\GeoX\_internal\llvmlite\binding\llvmlite.dll" (
    echo       [OK] llvmlite.dll found in _internal\llvmlite\binding\
) else (
    echo       [WARNING] llvmlite.dll not in expected location
    echo       Searching for llvmlite.dll...
    dir /s /b "dist\GeoX\*llvmlite*.dll" 2>nul
    if errorlevel 1 (
        echo       [ERROR] llvmlite.dll not found! Numba may fail.
    )
)
echo.

REM Skip installer if --exe flag was used
if "%BUILD_INSTALLER%"=="0" (
    echo [3/4] Skipping installer (--exe flag)
    echo [4/4] Skipping installer
    goto :build_complete
)

REM =============================================================================
REM Step 4: Check Inno Setup
REM =============================================================================

echo [3/4] Checking Inno Setup...

set "ISCC_PATH="

REM Check common Inno Setup locations
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
) else (
    REM Try to find via PATH
    where ISCC.exe >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%i in ('where ISCC.exe') do set "ISCC_PATH=%%i"
    )
)

if "%ISCC_PATH%"=="" (
    echo.
    echo WARNING: Inno Setup not found.
    echo.
    echo To create an installer, install Inno Setup 6.x from:
    echo   https://jrsoftware.org/isdl.php
    echo.
    echo The executable has been created at:
    echo   dist\GeoX\GeoX.exe
    echo.
    echo You can distribute the entire dist\GeoX folder as a portable app.
    echo.
    exit /b 0
)

echo       Inno Setup: %ISCC_PATH%
echo.

REM =============================================================================
REM Step 5: Build Installer
REM =============================================================================

echo [4/4] Building installer with Inno Setup...

"%ISCC_PATH%" "installer\BlockModelViewer.iss"

if errorlevel 1 (
    echo.
    echo ERROR: Inno Setup build failed.
    echo Check the output above for errors.
    exit /b 1
)

REM Find the generated installer
for %%f in (dist\GeoX_Setup_*.exe) do set "INSTALLER_FILE=%%f"

if not exist "%INSTALLER_FILE%" (
    echo WARNING: Installer file not found.
) else (
    for %%A in ("%INSTALLER_FILE%") do set "INSTALLER_SIZE=%%~zA"
    set /a "INSTALLER_SIZE_MB=!INSTALLER_SIZE! / 1048576"
    echo       Created: %INSTALLER_FILE% (!INSTALLER_SIZE_MB! MB)
)

:build_complete

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo Output files:
echo   Executable: dist\GeoX\GeoX.exe
if "%BUILD_INSTALLER%"=="1" if exist "%INSTALLER_FILE%" (
    echo   Installer:  %INSTALLER_FILE%
)
echo.
echo To test the executable:
echo   dist\GeoX\GeoX.exe
echo.

exit /b 0
