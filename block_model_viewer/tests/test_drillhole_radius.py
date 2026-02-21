
import unittest
import pyvista as pv
import numpy as np
from unittest.mock import MagicMock, patch



from block_model_viewer.visualization.renderer import Renderer
from block_model_viewer.drillholes.datamodel import DrillholeDatabase

class TestDrillholeRadius(unittest.TestCase):

    def setUp(self):
        """Set up a mock plotter and renderer for testing."""
        self.plotter = pv.Plotter(off_screen=True)
        self.renderer = Renderer()
        self.renderer.plotter = self.plotter

        # Create a mock drillhole database
        self.db = DrillholeDatabase()
        self.db.collars = MagicMock()
        self.db.surveys = MagicMock()
        self.db.assays = MagicMock()
        self.db.lithology = MagicMock()

        collars_data = np.array([('DH1', 0, 0, 100)], dtype=[('hole_id', 'U10'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        self.db.collars.df = pv.pyvista_ndarray(collars_data)
        surveys_data = np.array([('DH1', 0, 90, 0), ('DH1', 100, 90, 0)], dtype=[('hole_id', 'U10'), ('depth', 'f8'), ('dip', 'f8'), ('azimuth', 'f8')])
        self.db.surveys.df = pv.pyvista_ndarray(surveys_data)
        assays_data = np.array([('DH1', 0, 100, 1.0)], dtype=[('hole_id', 'U10'), ('from', 'f8'), ('to', 'f8'), ('value', 'f8')])
        self.db.assays.df = pv.pyvista_ndarray(assays_data)
        lithology_data = np.array([('DH1', 0, 100, 'rock')], dtype=[('hole_id', 'U10'), ('from', 'f8'), ('to', 'f8'), ('lith', 'U10')])
        self.db.lithology.df = pv.pyvista_ndarray(lithology_data)


    @patch('block_model_viewer.visualization.renderer.build_drillhole_polylines')
    def test_increase_drillhole_radius(self, mock_build_polylines):
        """Test that drillholes are still visible after increasing radius."""
        
        # Mock the output of build_drillhole_polylines
        poly = pv.PolyData()
        poly.points = np.array([[0, 0, 100], [0, 0, 0]])
        poly.lines = np.array([2, 0, 1])
        
        mock_build_polylines.return_value = {
            "hole_polys": {"DH1": poly},
            "hole_segment_lith": {"DH1": ['rock']},
            "hole_segment_assay": {"DH1": [1.0]},
            "lith_colors": {'rock': 'red'},
            "lith_to_index": {'rock': 0},
            "assay_field": "value",
            "assay_min": 0,
            "assay_max": 2,
            "hole_ids": ["DH1"],
            "collar_coords": {"DH1": [0,0,100]}
        }

        # 1. Add drillhole layer with initial radius
        initial_radius = 1.0
        self.renderer.add_drillhole_layer(self.db, radius=initial_radius)
        
        # Check that drillholes are present
        self.assertIn("_merged", self.renderer._drillhole_hole_actors)
        initial_actor = self.renderer._drillhole_hole_actors["_merged"]
        self.assertTrue(initial_actor.GetVisibility())
        
        # Get number of points in the initial mesh
        initial_n_points = initial_actor.GetMapper().GetInput().GetNumberOfPoints()
        self.assertGreater(initial_n_points, 0)
        
        # 2. Update drillhole radius
        new_radius = 5.0
        self.renderer.update_drillhole_radius(new_radius)
        
        # Check that drillholes are still present and visible
        self.assertIn("_merged", self.renderer._drillhole_hole_actors)
        updated_actor = self.renderer._drillhole_hole_actors["_merged"]
        self.assertTrue(updated_actor.GetVisibility())
        
        # Check that the actor is the same
        self.assertIs(initial_actor, updated_actor)
        
        # Get number of points in the updated mesh
        updated_n_points = updated_actor.GetMapper().GetInput().GetNumberOfPoints()
        self.assertGreater(updated_n_points, 0)

if __name__ == '__main__':
    unittest.main()
