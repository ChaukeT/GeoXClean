"""
Renderer Integration Tests

Tests that panels correctly integrate with the mock renderer.
"""

import pytest
import numpy as np
from tests.panel_debugger.core.mock_factory import MockDataFactory

@pytest.mark.renderer
class TestRendererIntegration:
    """Test renderer integration"""
    
    def test_renderer_can_add_drillhole_layer(self, mock_renderer):
        """Test renderer can add drillhole layer"""
        mock_data = MockDataFactory.create_drillhole_data(n_holes=5)
        
        result = mock_renderer.add_drillhole_layer(mock_data)
        
        assert result is True
        assert mock_renderer.was_method_called('add_drillhole_layer')
        assert 'drillholes' in mock_renderer.active_layers
    
    def test_renderer_can_add_block_model_layer(self, mock_renderer):
        """Test renderer can add block model layer"""
        mock_model = MockDataFactory.create_block_model(nx=10, ny=10, nz=5)
        
        result = mock_renderer.add_block_model_layer(mock_model)
        
        assert result is True
        assert mock_renderer.was_method_called('add_block_model_layer')
        assert 'blocks' in mock_renderer.active_layers
    
    def test_renderer_tracks_large_models(self, mock_renderer):
        """Test renderer detects large models"""
        # Create large model (>30k cells)
        mock_model = MockDataFactory.create_block_model(nx=50, ny=50, nz=20)
        
        mock_renderer.add_block_model_layer(mock_model)
        
        assert mock_renderer._has_large_model is True
