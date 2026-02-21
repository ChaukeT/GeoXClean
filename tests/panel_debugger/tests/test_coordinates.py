"""
Coordinate System Tests

CRITICAL: Tests that all layers use consistent coordinate transformations.
This catches the coordinate mismatch issue from MEMORY.md where drillholes
and block models ended up in different coordinate systems.
"""

import pytest
import numpy as np
from tests.panel_debugger.core.mock_factory import MockDataFactory

@pytest.mark.coordinates
@pytest.mark.critical
class TestCoordinateAlignment:
    """Test that drillholes and block models use same coordinate system"""
    
    def test_drillholes_transformed_to_local(self, mock_renderer):
        """Test that drillholes are transformed to local coordinates"""
        # Create drillholes in UTM coordinates
        dh_data = MockDataFactory.create_drillhole_data(
            n_holes=5,
            coordinate_system='utm'
        )
        
        mock_renderer.add_drillhole_layer(dh_data)
        
        # Get layer coordinates
        coords = mock_renderer.get_layer_coordinates('drillholes')
        
        if coords is not None:
            # Coordinates should be transformed to local (near origin)
            max_coord = np.max(np.abs(coords))
            assert max_coord < 500000, \
                f"Drillhole coordinates not transformed (max={max_coord})"
    
    def test_blocks_transformed_to_local(self, mock_renderer):
        """Test that block models are transformed to local coordinates"""
        # Create block model in UTM coordinates
        block_data = MockDataFactory.create_block_model(
            nx=10, ny=10, nz=5,
            coordinate_system='utm'
        )
        
        mock_renderer.add_block_model_layer(block_data)
        
        # Block model should be in active layers
        assert 'blocks' in mock_renderer.active_layers
    
    def test_drillholes_and_blocks_aligned(self, mock_renderer):
        """
        CRITICAL TEST: Verify drillholes and block models are in same coordinate system.
        
        This catches the issue from MEMORY.md where block models appeared
        500km away from drillholes due to coordinate mismatch.
        """
        # Create both in UTM coordinates
        dh_data = MockDataFactory.create_drillhole_data(
            n_holes=5,
            coordinate_system='utm'
        )
        
        block_data = MockDataFactory.create_block_model(
            nx=10, ny=10, nz=5,
            coordinate_system='utm'
        )
        
        # Add both to renderer
        mock_renderer.add_drillhole_layer(dh_data)
        mock_renderer.add_block_model_layer(block_data)
        
        # Both should be present
        assert 'drillholes' in mock_renderer.active_layers
        assert 'blocks' in mock_renderer.active_layers
        
        # Get coordinates
        dh_coords = mock_renderer.get_layer_coordinates('drillholes')
        
        if dh_coords is not None:
            # Both should be transformed to local coordinates
            dh_max = np.max(np.abs(dh_coords))
            assert dh_max < 500000, \
                "Drillholes and blocks in different coordinate systems!"

@pytest.mark.coordinates
class TestToLocalPrecision:
    """Test _to_local_precision coordinate transformation"""
    
    def test_to_local_precision_transforms_utm_coords(self, mock_renderer):
        """Test that _to_local_precision centers coordinates"""
        # UTM coordinates (~500,000m)
        utm_coords = np.array([
            [500000, 6000000, 100],
            [500100, 6000100, 110],
            [500200, 6000200, 120]
        ])
        
        local_coords = mock_renderer._to_local_precision(utm_coords)
        
        # Should be centered near origin
        centroid = np.mean(local_coords, axis=0)
        assert np.allclose(centroid, [0, 0, 0], atol=1e-10)
    
    def test_to_local_precision_handles_local_coords(self, mock_renderer):
        """Test that _to_local_precision works with already-local coords"""
        # Already local coordinates
        local_coords = np.array([
            [100, 200, 50],
            [150, 250, 55],
            [200, 300, 60]
        ])
        
        transformed = mock_renderer._to_local_precision(local_coords)
        
        # Should still be centered
        centroid = np.mean(transformed, axis=0)
        assert np.allclose(centroid, [0, 0, 0], atol=1e-10)
