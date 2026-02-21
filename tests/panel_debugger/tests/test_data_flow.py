"""
Data Flow Tests - Registry → Panel Data Propagation

Tests that data flows correctly from DataRegistry to panels and that panels
update their UI appropriately.
"""

import pytest
from tests.panel_debugger.core.mock_factory import MockDataFactory
from tests.panel_debugger.core.signal_tester import SignalSpy

@pytest.mark.data_flow
class TestDrillholeDataFlow:
    """Test drillhole data propagation through the system"""
    
    def test_registry_stores_drillhole_data(self, mock_registry):
        """Test that registry stores drillhole data correctly"""
        mock_data = MockDataFactory.create_drillhole_data(n_holes=5)
        
        mock_registry.set_drillhole_data(mock_data)
        
        retrieved = mock_registry.get_drillhole_data()
        assert retrieved is not None
        assert retrieved == mock_data
    
    def test_drillhole_data_emits_signal(self, mock_registry):
        """Test that setting drillhole data emits signal"""
        spy = SignalSpy(mock_registry.signals.drillholeDataLoaded)
        
        mock_data = MockDataFactory.create_drillhole_data(n_holes=5)
        mock_registry.set_drillhole_data(mock_data)
        
        assert spy.count() == 1
        assert spy.get_emission(0) == mock_data

@pytest.mark.data_flow
class TestBlockModelDataFlow:
    """Test block model data propagation"""
    
    def test_registry_stores_block_model(self, mock_registry):
        """Test that registry stores block models correctly"""
        mock_model = MockDataFactory.create_block_model(nx=10, ny=10, nz=5)
        
        mock_registry.register_block_model(mock_model, model_id="test_model")
        
        retrieved = mock_registry.get_block_model("test_model")
        assert retrieved is not None
    
    def test_block_model_emits_signal(self, mock_registry):
        """Test that registering block model emits signals"""
        spy_loaded = SignalSpy(mock_registry.signals.blockModelLoaded)
        spy_loaded_ex = SignalSpy(mock_registry.signals.blockModelLoadedEx)
        
        mock_model = MockDataFactory.create_block_model(nx=10, ny=10, nz=5)
        mock_registry.register_block_model(mock_model, model_id="test_model")
        
        assert spy_loaded.count() == 1
        assert spy_loaded_ex.count() == 1

@pytest.mark.data_flow
class TestVariogramDataFlow:
    """Test variogram data propagation"""
    
    def test_registry_stores_variogram_results(self, mock_registry):
        """Test that registry stores variogram results"""
        mock_variogram = MockDataFactory.create_variogram_results()
        
        mock_registry.register_variogram_results(mock_variogram)
        
        retrieved = mock_registry.get_variogram_results()
        assert retrieved is not None
        assert 'variograms' in retrieved
