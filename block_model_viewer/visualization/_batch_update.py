"""
Batch update context manager for Renderer.

Allows multiple actor/mesh changes with a single render call.
"""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class _BatchUpdateContext:
    """Context manager for batched renderer updates."""
    
    def __init__(self, renderer):
        self.renderer = renderer
    
    def __enter__(self):
        self.renderer._batch_update_active = True
        self.renderer._batch_update_count += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.renderer._batch_update_count -= 1
        if self.renderer._batch_update_count <= 0:
            self.renderer._batch_update_active = False
            self.renderer._batch_update_count = 0
            # Render once at the end
            if self.renderer.plotter is not None:
                try:
                    self.renderer.plotter.render()
                except Exception as e:
                    logger.debug(f"Error rendering after batch update: {e}")
        return False

