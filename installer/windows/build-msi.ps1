#Requires -Version 5.1
<#
.SYNOPSIS
    Builds the GeoX MSI installer using PyInstaller and WiX Toolset v4.

.DESCRIPTION
    This script automates the complete build process:
    1. Reads version from pyproject.toml
    2. Runs PyInstaller to create the executable
    3. Uses WiX heat to harvest all files from the dist folder
    4. Compiles the MSI installer with WiX

.PARAMETER SkipPyInstaller
    Skip the PyInstaller step (use existing dist folder)

.PARAMETER Clean
    Clean build artifacts before building

.EXAMPLE
    .\build-msi.ps1
    
.EXAMPLE
    .\build-msi.ps1 -SkipPyInstaller
    
.EXAMPLE
    .\build-msi.ps1 -Clean
#>

param(
    [switch]$SkipPyInstaller,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName
$DistDir = Join-Path $ProjectRoot "dist"
$GeoXDistDir = Join-Path $DistDir "GeoX"
$InstallerDir = Join-Path $ProjectRoot "installer\windows"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GeoX MSI Installer Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to read version from pyproject.toml
function Get-ProjectVersion {
    $pyprojectPath = Join-Path $ProjectRoot "pyproject.toml"
    if (-not (Test-Path $pyprojectPath)) {
        Write-Error "pyproject.toml not found at: $pyprojectPath"
        exit 1
    }
    
    $content = Get-Content $pyprojectPath -Raw
    if ($content -match 'version\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }
    
    Write-Error "Could not find version in pyproject.toml"
    exit 1
}

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Clean build artifacts
if ($Clean) {
    Write-Host "[1/5] Cleaning build artifacts..." -ForegroundColor Yellow
    
    $artifactsToRemove = @(
        (Join-Path $DistDir "*"),
        (Join-Path $ProjectRoot "build"),
        (Join-Path $InstallerDir "*.wixobj"),
        (Join-Path $InstallerDir "InternalFiles.wxs"),
        (Join-Path $InstallerDir "*.wixpdb")
    )
    
    foreach ($artifact in $artifactsToRemove) {
        if (Test-Path $artifact) {
            Remove-Item $artifact -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  Removed: $artifact" -ForegroundColor Gray
        }
    }
    Write-Host "  Clean complete." -ForegroundColor Green
} else {
    Write-Host "[1/5] Skipping clean (use -Clean to clean first)" -ForegroundColor Gray
}

# Get version
$Version = Get-ProjectVersion
Write-Host ""
Write-Host "Building GeoX version: $Version" -ForegroundColor Magenta
Write-Host ""

# Check prerequisites
Write-Host "[2/5] Checking prerequisites..." -ForegroundColor Yellow

# Check Python
if (-not (Test-Command "python")) {
    Write-Error "Python is not installed or not in PATH"
    exit 1
}
Write-Host "  Python: $(python --version)" -ForegroundColor Green

# Check PyInstaller
if (-not (Test-Command "pyinstaller")) {
    Write-Host "  PyInstaller not found. Installing..." -ForegroundColor Yellow
    python -m pip install pyinstaller --quiet
}
Write-Host "  PyInstaller: $(pyinstaller --version)" -ForegroundColor Green

# Check WiX Toolset
if (-not (Test-Command "wix")) {
    Write-Host ""
    Write-Host "ERROR: WiX Toolset v4 is not installed or not in PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "To install WiX Toolset v4:" -ForegroundColor Yellow
    Write-Host "  Option 1 (dotnet tool - Recommended):" -ForegroundColor Cyan
    Write-Host "    dotnet tool install --global wix" -ForegroundColor White
    Write-Host ""
    Write-Host "  Option 2 (winget):" -ForegroundColor Cyan
    Write-Host "    winget install WixToolset.WixToolset" -ForegroundColor White
    Write-Host ""
    Write-Host "  Option 3 (Manual download):" -ForegroundColor Cyan
    Write-Host "    https://wixtoolset.org/docs/intro/" -ForegroundColor White
    Write-Host ""
    Write-Host "After installation, restart your terminal and run this script again." -ForegroundColor Yellow
    exit 1
}
Write-Host "  WiX Toolset: $(wix --version)" -ForegroundColor Green

# Add WiX UI extension if not present
Write-Host "  Adding WiX UI extension..." -ForegroundColor Gray
wix extension add WixToolset.UI.wixext -g 2>$null
Write-Host "  WiX extensions ready." -ForegroundColor Green

Write-Host ""

# Run PyInstaller
if (-not $SkipPyInstaller) {
    Write-Host "[3/5] Running PyInstaller..." -ForegroundColor Yellow
    
    Push-Location $ProjectRoot
    try {
        $specFile = Join-Path $InstallerDir "pyinstaller.spec"
        if (-not (Test-Path $specFile)) {
            Write-Error "PyInstaller spec file not found: $specFile"
            exit 1
        }
        
        # Run PyInstaller
        pyinstaller $specFile --noconfirm --clean
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "PyInstaller failed with exit code: $LASTEXITCODE"
            exit 1
        }
        
        # Verify output
        $exePath = Join-Path $GeoXDistDir "GeoX.exe"
        if (-not (Test-Path $exePath)) {
            Write-Error "PyInstaller did not create expected executable: $exePath"
            exit 1
        }
        
        Write-Host "  PyInstaller completed successfully." -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
} else {
    Write-Host "[3/5] Skipping PyInstaller (using existing dist folder)" -ForegroundColor Gray
    
    # Verify dist folder exists
    if (-not (Test-Path $GeoXDistDir)) {
        Write-Error "Dist folder not found: $GeoXDistDir. Run without -SkipPyInstaller first."
        exit 1
    }
}

Write-Host ""

# Harvest files with WiX heat
Write-Host "[4/5] Harvesting files with WiX heat..." -ForegroundColor Yellow

Push-Location $InstallerDir
try {
    $internalDir = Join-Path $GeoXDistDir "_internal"
    
    if (Test-Path $internalDir) {
        # Use heat to generate component group for _internal folder
        # -cg: Component Group name
        # -dr: Directory reference
        # -srd: Suppress root directory
        # -ag: Auto-generate GUIDs
        # -sfrag: Suppress fragment wrapper
        # -var: Variable for source directory
        
        Write-Host "  Harvesting _internal directory..." -ForegroundColor Gray
        
        wix heat dir $internalDir `
            -cg InternalComponents `
            -dr InternalFolder `
            -srd `
            -ag `
            -sfrag `
            -var var.InternalDir `
            -o InternalFiles.wxs
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "WiX heat failed. Using fallback method..."
            
            # Create a minimal fallback WiX file
            $fallbackContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">
  <Fragment>
    <DirectoryRef Id="InternalFolder">
      <Component Id="InternalPlaceholder" Guid="*">
        <CreateFolder/>
      </Component>
    </DirectoryRef>
    <ComponentGroup Id="InternalComponents">
      <ComponentRef Id="InternalPlaceholder"/>
    </ComponentGroup>
  </Fragment>
</Wix>
"@
            $fallbackContent | Out-File -FilePath "InternalFiles.wxs" -Encoding UTF8
            Write-Host "  Created fallback InternalFiles.wxs" -ForegroundColor Yellow
        } else {
            Write-Host "  File harvesting completed." -ForegroundColor Green
        }
    } else {
        Write-Warning "_internal directory not found. Creating placeholder..."
        
        # Create placeholder for builds without _internal
        $placeholderContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">
  <Fragment>
    <ComponentGroup Id="InternalComponents"/>
  </Fragment>
</Wix>
"@
        $placeholderContent | Out-File -FilePath "InternalFiles.wxs" -Encoding UTF8
    }
}
finally {
    Pop-Location
}

Write-Host ""

# Build MSI
Write-Host "[5/5] Building MSI installer..." -ForegroundColor Yellow

Push-Location $InstallerDir
try {
    $outputMsi = Join-Path $DistDir "GeoX_Setup_$Version.msi"
    $internalDir = Join-Path $GeoXDistDir "_internal"
    
    # Build command with version substitution
    # -d: Define preprocessor variable
    # -ext: Load extension
    # -o: Output file
    
    $buildArgs = @(
        "build"
        "GeoX.wxs"
        "InternalFiles.wxs"
        "-ext", "WixToolset.UI.wixext"
        "-d", "Version=$Version"
        "-d", "InternalDir=$internalDir"
        "-o", $outputMsi
    )
    
    Write-Host "  Running: wix $($buildArgs -join ' ')" -ForegroundColor Gray
    
    & wix @buildArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "WiX build failed with exit code: $LASTEXITCODE"
        exit 1
    }
    
    # Verify output
    if (-not (Test-Path $outputMsi)) {
        Write-Error "MSI was not created: $outputMsi"
        exit 1
    }
    
    $msiSize = (Get-Item $outputMsi).Length / 1MB
    Write-Host "  MSI created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Build Complete!" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Output: $outputMsi" -ForegroundColor White
    Write-Host "  Size:   $([math]::Round($msiSize, 2)) MB" -ForegroundColor White
    Write-Host "  Version: $Version" -ForegroundColor White
    Write-Host ""
}
finally {
    Pop-Location
}

