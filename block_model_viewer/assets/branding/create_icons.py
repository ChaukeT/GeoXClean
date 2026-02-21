"""
Script to generate icon files from SVG for GeoX application.

Generates:
- PNG files at various sizes (16x16, 32x32, 48x48, 64x64, 128x128, 256x256, 512x512)
- ICO file (Windows icon) with multiple sizes
- ICNS file (macOS icon) with multiple sizes

Requirements:
- cairosvg or svglib+reportlab for SVG to PNG conversion
- Pillow for image manipulation
- iconutil (macOS) for ICNS generation
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
    import cairosvg
except ImportError:
    print("Required packages not installed. Installing...")
    print("Please run: pip install cairosvg pillow")
    sys.exit(1)


def svg_to_png(svg_path: Path, png_path: Path, size: int):
    """Convert SVG to PNG at specified size."""
    try:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=size,
            output_height=size
        )
        print(f"✓ Created {png_path.name} ({size}x{size})")
        return True
    except Exception as e:
        print(f"✗ Failed to create {png_path.name}: {e}")
        return False


def create_ico(png_files: list, ico_path: Path):
    """Create ICO file from multiple PNG files."""
    try:
        images = []
        for png_file in png_files:
            if png_file.exists():
                img = Image.open(png_file)
                images.append(img)
        
        if images:
            images[0].save(
                str(ico_path),
                format='ICO',
                sizes=[(img.size[0], img.size[1]) for img in images]
            )
            print(f"✓ Created {ico_path.name}")
            return True
    except Exception as e:
        print(f"✗ Failed to create {ico_path.name}: {e}")
    return False


def create_icns(png_dir: Path, icns_path: Path):
    """Create ICNS file for macOS (requires iconutil)."""
    try:
        import subprocess
        
        # Create iconset directory
        iconset_dir = png_dir / "geox_icon.iconset"
        iconset_dir.mkdir(exist_ok=True)
        
        # Required sizes for ICNS
        sizes = {
            "icon_16x16.png": 16,
            "icon_16x16@2x.png": 32,
            "icon_32x32.png": 32,
            "icon_32x32@2x.png": 64,
            "icon_128x128.png": 128,
            "icon_128x128@2x.png": 256,
            "icon_256x256.png": 256,
            "icon_256x256@2x.png": 512,
            "icon_512x512.png": 512,
            "icon_512x512@2x.png": 1024
        }
        
        svg_path = png_dir.parent / "geox_icon.svg"
        
        # Generate all required PNG files
        for filename, size in sizes.items():
            png_path = iconset_dir / filename
            svg_to_png(svg_path, png_path, size)
        
        # Use iconutil to create ICNS (macOS only)
        # SECURITY: Validate paths before subprocess call
        if sys.platform == "darwin":
            from pathlib import Path
            from block_model_viewer.utils.security import validate_file_path
            
            # Validate paths to prevent command injection
            try:
                validated_iconset = validate_file_path(iconset_dir, must_exist=True)
                validated_icns = validate_file_path(icns_path.parent, must_exist=False)
                validated_icns = validated_icns / icns_path.name
            except Exception as e:
                print(f"Security validation failed: {e}")
                return False
            
            subprocess.run([
                "iconutil",
                "-c", "icns",
                str(validated_iconset),
                "-o", str(validated_icns)
            ], check=True)
            print(f"✓ Created {icns_path.name}")
            
            # Clean up iconset directory
            import shutil
            shutil.rmtree(iconset_dir)
            return True
        else:
            print("⚠ ICNS creation requires macOS (iconutil)")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create {icns_path.name}: {e}")
    return False


def main():
    """Generate all icon files."""
    script_dir = Path(__file__).parent
    svg_path = script_dir / "geox_icon.svg"
    
    if not svg_path.exists():
        print(f"✗ SVG file not found: {svg_path}")
        return
    
    print("Generating GeoX icon files...")
    print("=" * 50)
    
    # PNG sizes to generate
    png_sizes = [16, 32, 48, 64, 128, 256, 512]
    png_files = []
    
    # Generate PNG files
    for size in png_sizes:
        png_path = script_dir / f"geox_icon_{size}x{size}.png"
        if svg_to_png(svg_path, png_path, size):
            png_files.append(png_path)
    
    # Create ICO file (Windows)
    ico_path = script_dir / "geox_icon.ico"
    create_ico(png_files, ico_path)
    
    # Create ICNS file (macOS)
    icns_path = script_dir / "geox_icon.icns"
    create_icns(script_dir, icns_path)
    
    print("=" * 50)
    print("Icon generation complete!")
    print(f"\nGenerated files in: {script_dir}")
    print("\nPNG files:")
    for png_file in png_files:
        print(f"  - {png_file.name}")
    print(f"\nICO file: {ico_path.name if ico_path.exists() else 'Failed'}")
    print(f"ICNS file: {icns_path.name if icns_path.exists() else 'Failed (requires macOS)'}")


if __name__ == "__main__":
    main()
