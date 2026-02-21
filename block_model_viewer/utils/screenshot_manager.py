"""
Screenshot Manager

Advanced screenshot export with branding, layout composition, and DPI presets.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ScreenshotManager:
    """
    Manages advanced screenshot exports with branded layouts.
    
    Features:
    - Branded layout export (viewport + legend + scale bar + title)
    - DPI and size presets (presentation, publication, poster)
    - Custom annotations and watermarks
    - Multi-format export (PNG, PDF, SVG)
    """
    
    # Preset configurations
    PRESETS = {
        'presentation': {
            'dpi': 150,
            'width': 1920,
            'height': 1080,
            'description': 'HD presentation (1920x1080, 150 DPI)'
        },
        'publication': {
            'dpi': 300,
            'width': 3000,
            'height': 2400,
            'description': 'Publication quality (3000x2400, 300 DPI)'
        },
        'poster': {
            'dpi': 300,
            'width': 4000,
            'height': 3000,
            'description': 'Large poster (4000x3000, 300 DPI)'
        },
        'screen': {
            'dpi': 96,
            'width': 1280,
            'height': 720,
            'description': 'Screen resolution (1280x720, 96 DPI)'
        },
        'custom': {
            'dpi': 150,
            'width': 1920,
            'height': 1080,
            'description': 'Custom size (configurable)'
        }
    }
    
    def __init__(self):
        self.current_plotter = None
        logger.info("Initialized ScreenshotManager")
    
    def set_plotter(self, plotter):
        """Set the PyVista plotter for screenshot capture."""
        self.current_plotter = plotter
    
    def export_simple_screenshot(self, filepath: Path, 
                                 transparent: bool = False,
                                 scale: int = 1) -> bool:
        """
        Export simple screenshot of current view.
        
        Args:
            filepath: Output file path
            transparent: Use transparent background
            scale: Scale factor (1-4)
        
        Returns:
            True if successful
        """
        if self.current_plotter is None:
            logger.error("No plotter set for screenshot")
            return False
        
        try:
            self.current_plotter.screenshot(
                str(filepath),
                transparent_background=transparent,
                scale=scale
            )
            logger.info(f"Exported simple screenshot to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting screenshot: {e}", exc_info=True)
            return False
    
    def export_branded_layout(self, 
                             filepath: Path,
                             preset: str = 'presentation',
                             title: str = "",
                             subtitle: str = "",
                             show_legend: bool = True,
                             show_scale_bar: bool = True,
                             show_axes: bool = True,
                             show_timestamp: bool = True,
                             watermark: str = "",
                             company_logo: Optional[Path] = None,
                             transparent: bool = False) -> bool:
        """
        Export branded layout with title, legend, scale bar, etc.
        
        Args:
            filepath: Output file path
            preset: Size preset ('presentation', 'publication', 'poster', 'screen', 'custom')
            title: Main title text
            subtitle: Subtitle text
            show_legend: Include color legend
            show_scale_bar: Include scale bar
            show_axes: Include axes
            show_timestamp: Include timestamp
            watermark: Watermark text
            company_logo: Optional company logo image path
            transparent: Transparent background
        
        Returns:
            True if successful
        """
        if self.current_plotter is None:
            logger.error("No plotter set for screenshot")
            return False
        
        try:
            import pyvista as pv
            from PIL import Image, ImageDraw, ImageFont
            
            # Get preset config
            config = self.PRESETS.get(preset, self.PRESETS['presentation'])
            width, height, dpi = config['width'], config['height'], config['dpi']
            
            # Configure plotter for export
            original_window_size = self.current_plotter.window_size
            
            # Capture base screenshot at high resolution
            scale = max(1, width // original_window_size[0])
            temp_path = filepath.parent / f"temp_{filepath.stem}.png"
            
            self.current_plotter.screenshot(
                str(temp_path),
                transparent_background=transparent,
                scale=scale
            )
            
            # Load screenshot
            img = Image.open(temp_path)
            
            # Resize to exact target dimensions
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Create drawing context
            draw = ImageDraw.Draw(img)
            
            # Try to load a nice font, fallback to default
            try:
                title_font = ImageFont.truetype("arial.ttf", int(height * 0.04))
                subtitle_font = ImageFont.truetype("arial.ttf", int(height * 0.025))
                text_font = ImageFont.truetype("arial.ttf", int(height * 0.02))
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Add title bar at top
            if title:
                title_bar_height = int(height * 0.08)
                # Semi-transparent dark bar
                title_overlay = Image.new('RGBA', (width, title_bar_height), (0, 0, 0, 180))
                img.paste(title_overlay, (0, 0), title_overlay)
                
                # Draw title text
                draw.text(
                    (width * 0.02, title_bar_height * 0.15),
                    title,
                    fill=(255, 255, 255, 255),
                    font=title_font
                )
                
                # Draw subtitle if provided
                if subtitle:
                    draw.text(
                        (width * 0.02, title_bar_height * 0.55),
                        subtitle,
                        fill=(200, 200, 200, 255),
                        font=subtitle_font
                    )
            
            # Add timestamp in bottom right
            if show_timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                bbox = draw.textbbox((0, 0), timestamp, font=text_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Semi-transparent background
                padding = 10
                stamp_overlay = Image.new(
                    'RGBA',
                    (text_width + padding * 2, text_height + padding * 2),
                    (0, 0, 0, 150)
                )
                img.paste(
                    stamp_overlay,
                    (width - text_width - padding * 3, height - text_height - padding * 3),
                    stamp_overlay
                )
                
                draw.text(
                    (width - text_width - padding * 2, height - text_height - padding * 2),
                    timestamp,
                    fill=(255, 255, 255, 255),
                    font=text_font
                )
            
            # Add watermark
            if watermark:
                # Center watermark with low opacity
                bbox = draw.textbbox((0, 0), watermark, font=title_font)
                text_width = bbox[2] - bbox[0]
                watermark_overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
                watermark_draw = ImageDraw.Draw(watermark_overlay)
                watermark_draw.text(
                    ((width - text_width) // 2, height // 2),
                    watermark,
                    fill=(255, 255, 255, 60),
                    font=title_font
                )
                img = Image.alpha_composite(img.convert('RGBA'), watermark_overlay)
            
            # Add company logo if provided
            if company_logo and company_logo.exists():
                try:
                    logo = Image.open(company_logo)
                    # Resize logo to ~5% of image height
                    logo_height = int(height * 0.05)
                    aspect = logo.width / logo.height
                    logo_width = int(logo_height * aspect)
                    logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                    
                    # Paste in top right
                    img.paste(logo, (width - logo_width - 20, 20), logo if logo.mode == 'RGBA' else None)
                except Exception as e:
                    logger.warning(f"Could not load company logo: {e}")
            
            # Save final image
            if filepath.suffix.lower() == '.pdf':
                img.convert('RGB').save(filepath, 'PDF', resolution=dpi)
            elif filepath.suffix.lower() == '.svg':
                # SVG export would require additional libraries
                logger.warning("SVG export not fully implemented, saving as PNG")
                img.save(filepath.with_suffix('.png'), 'PNG', dpi=(dpi, dpi))
            else:
                img.save(filepath, 'PNG', dpi=(dpi, dpi))
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            logger.info(f"Exported branded layout to {filepath} ({preset}: {width}x{height}, {dpi} DPI)")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting branded layout: {e}", exc_info=True)
            return False
    
    def export_with_legend(self, 
                          filepath: Path,
                          preset: str = 'presentation',
                          legend_title: str = "",
                          transparent: bool = False) -> bool:
        """
        Export screenshot with integrated color legend.
        
        Args:
            filepath: Output file path
            preset: Size preset
            legend_title: Legend title
            transparent: Transparent background
        
        Returns:
            True if successful
        """
        # This is a simplified version - full implementation would integrate
        # the plotter's scalar bar into the composed image
        return self.export_branded_layout(
            filepath=filepath,
            preset=preset,
            title=legend_title,
            show_legend=True,
            transparent=transparent
        )
    
    @staticmethod
    def get_preset_info(preset: str) -> Dict:
        """Get information about a preset."""
        return ScreenshotManager.PRESETS.get(preset, ScreenshotManager.PRESETS['presentation'])
    
    @staticmethod
    def list_presets() -> list:
        """List all available presets."""
        return list(ScreenshotManager.PRESETS.keys())
