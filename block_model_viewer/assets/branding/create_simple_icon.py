"""
Simple icon generator for GeoX - creates PNG icon programmatically.
Works without external SVG libraries.
"""

import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow required. Install with: pip install pillow")
    sys.exit(1)


def create_geox_icon(size=512):
    """Create GeoX icon programmatically."""
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background circle (dark blue gradient)
    center = size // 2
    radius = int(size * 0.47)
    
    # Draw gradient background (simplified - solid color)
    bg_color = (30, 60, 114)  # #1e3c72
    # Draw circle (width parameter may not be available in older Pillow)
    try:
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill=bg_color,
            outline=(255, 255, 255, 50),
            width=max(2, size // 128)
        )
    except TypeError:
        # Fallback for older Pillow versions (no width parameter)
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill=bg_color,
            outline=(255, 255, 255, 50)
        )
    
    # Draw 3D block model representation
    block_size = int(size * 0.15)
    spacing = int(size * 0.05)
    start_x = center - block_size * 1.5 - spacing
    start_y = center - block_size * 0.5
    
    # Base layer (darker, offset back)
    base_color = (52, 152, 219)  # Blue-gray
    base_offset = int(size * 0.08)
    
    for row in range(2):
        for col in range(3):
            x = start_x + col * (block_size + spacing) + base_offset
            y = start_y + row * (block_size + spacing) + base_offset
            # Use regular rectangle (rounded_rectangle may not be available in older Pillow)
            draw.rectangle(
                [x, y, x + block_size, y + block_size],
                fill=base_color + (100,)
            )
    
    # Middle layer
    mid_color = (52, 152, 219)  # Blue-gray
    mid_offset = int(size * 0.04)
    
    for row in range(2):
        for col in range(3):
            x = start_x + col * (block_size + spacing) + mid_offset
            y = start_y + row * (block_size + spacing) - block_size + mid_offset
            draw.rectangle(
                [x, y, x + int(block_size * 0.75), y + int(block_size * 0.75)],
                fill=mid_color + (150,)
            )
    
    # Top layer - highlighted blocks (orange/gold)
    top_color = (243, 156, 18)  # Orange
    highlight_color = (255, 255, 255, 80)
    
    blocks = [
        (0, 0), (1, 0), (2, 0),  # Top row
        (0, 1), (1, 1)  # Second row (partial)
    ]
    
    for col, row in blocks:
        x = start_x + col * (block_size + spacing)
        y = start_y + row * (block_size + spacing) - block_size * 2
        block_w = int(block_size * 0.5)
        block_h = int(block_size * 0.5)
        
        # Block
        draw.rectangle(
            [x, y, x + block_w, y + block_h],
            fill=top_color
        )
        
        # Highlight on top
        highlight_h = int(block_h * 0.3)
        draw.rectangle(
            [x, y, x + block_w, y + highlight_h],
            fill=highlight_color
        )
    
    # Grid lines
    grid_color = (255, 255, 255, 80)
    grid_width = max(1, size // 256)
    
    # Vertical lines
    for i in range(4):
        x = start_x + i * (block_size + spacing) - spacing // 2
        try:
            draw.line([x, start_y - block_size * 2, x, start_y + block_size + base_offset], 
                     fill=grid_color, width=grid_width)
        except TypeError:
            # Fallback for older Pillow versions (no width parameter)
            draw.line([x, start_y - block_size * 2, x, start_y + block_size + base_offset], 
                     fill=grid_color)
    
    # Horizontal lines
    for i in range(3):
        y = start_y + i * (block_size + spacing) - block_size * 2
        try:
            draw.line([start_x - spacing // 2, y, start_x + block_size * 3 + spacing, y], 
                     fill=grid_color, width=grid_width)
        except TypeError:
            # Fallback for older Pillow versions (no width parameter)
            draw.line([start_x - spacing // 2, y, start_x + block_size * 3 + spacing, y], 
                     fill=grid_color)
    
    # Text "GeoX"
    try:
        # Try to use a nice font
        font_size = int(size * 0.14)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = "GeoX"
    # Get text bounding box (textbbox is available in Pillow 8.0+, fallback to textsize)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)
    
    text_x = center - text_width // 2
    text_y = int(center + radius * 0.6)
    
    # Draw text with shadow
    shadow_offset = max(1, size // 256)
    draw.text((text_x + shadow_offset, text_y + shadow_offset), text, 
             font=font, fill=(0, 0, 0, 100))
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 240))
    
    return img


def main():
    """Generate icon files."""
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    print("Creating GeoX icon files...")
    
    # Generate different sizes
    sizes = {
        16: "geox_icon_16x16.png",
        32: "geox_icon_32x32.png",
        48: "geox_icon_48x48.png",
        64: "geox_icon_64x64.png",
        128: "geox_icon_128x128.png",
        256: "geox_icon_256x256.png",
        512: "geox_icon_512x512.png"
    }
    
    created_files = []
    
    for size, filename in sizes.items():
        try:
            img = create_geox_icon(size)
            output_path = script_dir / filename
            img.save(output_path, 'PNG')
            created_files.append(output_path)
            print(f"✓ Created {filename} ({size}x{size})")
        except Exception as e:
            print(f"✗ Failed to create {filename}: {e}")
    
    # Create ICO file from multiple sizes
    try:
        ico_sizes = [16, 32, 48, 64, 128, 256]
        ico_images = []
        
        for size in ico_sizes:
            png_path = script_dir / f"geox_icon_{size}x{size}.png"
            if png_path.exists():
                ico_images.append(Image.open(png_path))
        
        if ico_images:
            ico_path = script_dir / "geox_icon.ico"
            ico_images[0].save(
                str(ico_path),
                format='ICO',
                sizes=[(img.size[0], img.size[1]) for img in ico_images]
            )
            created_files.append(ico_path)
            print(f"✓ Created geox_icon.ico")
    except Exception as e:
        print(f"✗ Failed to create ICO file: {e}")
    
    print(f"\n✓ Generated {len(created_files)} icon files in {script_dir}")
    return created_files


if __name__ == "__main__":
    main()
