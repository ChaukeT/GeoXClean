# GeoX Application Icons

This directory contains the icon files for the GeoX (Geostatistics) application.

## Icon Files

### Main Icon Set
- **geox_icon.svg** - Scalable vector icon (source file)
- **geox_icon.ico** - Windows icon file (multi-size: 16x16 to 256x256)
- **geox_icon_*.png** - PNG icons at various sizes:
  - 16x16, 32x32, 48x48, 64x64, 128x128, 256x256, 512x512

### Legacy Icons
- **icon.ico** - Legacy Windows icon
- **icon.icns** - macOS icon (legacy)
- **app_logo.png** - Application logo
- **app_logo.svg** - Application logo (vector)
- **splash.png** - Splash screen image

## Icon Design

The GeoX icon features:
- **3D Block Model Representation**: Visualizes the core functionality of 3D geological modeling
- **Layered Blocks**: Shows multiple layers (base, middle, top) representing geological strata
- **Highlighted Ore Blocks**: Orange/gold blocks represent high-grade ore zones
- **Grid Overlay**: Blue grid lines represent the structured block model grid
- **Professional Color Scheme**: Dark blue background with orange/gold accents

## Usage

### Windows
Use `geox_icon.ico` for the application icon. This file contains multiple sizes and will automatically scale.

### macOS
To create an ICNS file, run on macOS:
```bash
python create_icons.py
```

### Linux/Other
Use `geox_icon_256x256.png` or `geox_icon_512x512.png` for application icons.

## Regenerating Icons

To regenerate all icon files from the SVG source:

```bash
# Simple method (uses PIL/Pillow)
python create_simple_icon.py

# Advanced method (requires cairosvg)
pip install cairosvg pillow
python create_icons.py
```

## Icon Specifications

- **Format**: PNG (raster), SVG (vector), ICO (Windows), ICNS (macOS)
- **Color Space**: RGBA with transparency support
- **Background**: Transparent (PNG) or dark blue circle (ICO)
- **Design**: Modern, professional, represents geostatistics/mining software

## Integration

To use the icon in the application:

```python
from pathlib import Path
from PyQt6.QtGui import QIcon

icon_path = Path(__file__).parent / "assets" / "branding" / "geox_icon.ico"
app.setWindowIcon(QIcon(str(icon_path)))
```

## License

Icons are part of the GeoX application and follow the same license terms.
