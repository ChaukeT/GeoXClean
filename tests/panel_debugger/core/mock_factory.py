"""
Mock Data Factory for Panel Testing

This module generates synthetic datasets for testing GeoX panels.
All data structures match the actual data formats used in the application.

Features:
- Deterministic (seeded) random generation for reproducibility
- Configurable coordinate systems (local vs UTM)
- Fast in-memory generation (no file I/O)
- Realistic data structures matching production code
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class MockDataFactory:
    """Generate synthetic datasets for panel testing"""

    # Default random seed for reproducibility
    DEFAULT_SEED = 42

    @staticmethod
    def set_seed(seed: int = DEFAULT_SEED):
        """Set random seed for reproducibility"""
        np.random.seed(seed)

    @classmethod
    def create_drillhole_data(
        cls,
        n_holes: int = 10,
        n_assays_per_hole: int = 20,
        n_composites_per_hole: int = 10,
        coordinate_system: str = 'local',  # 'local' or 'utm'
        properties: Optional[List[str]] = None,
        has_lithology: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic drillhole dataset.

        Parameters:
            n_holes: Number of drillholes to generate
            n_assays_per_hole: Number of assay intervals per hole
            n_composites_per_hole: Number of composite intervals per hole
            coordinate_system: 'local' (near origin) or 'utm' (realistic UTM coords)
            properties: List of property names (default: ['AU_PPM', 'CU_PCT', 'FE_PCT'])
            has_lithology: Include lithology data
            seed: Random seed (default: DEFAULT_SEED)

        Returns:
            Dict with keys: collars, surveys, assays, composites, trajectories,
            hole_segment_lith (if has_lithology), lith_to_index (if has_lithology)
        """
        if seed is not None:
            cls.set_seed(seed)
        else:
            cls.set_seed(cls.DEFAULT_SEED)

        if properties is None:
            properties = ['AU_PPM', 'CU_PCT', 'FE_PCT']

        # Determine coordinate offset
        if coordinate_system == 'utm':
            easting_offset = 500000
            northing_offset = 6000000
        else:  # local
            easting_offset = 0
            northing_offset = 0

        # 1. Generate collars
        collar_data = []
        for i in range(n_holes):
            collar_data.append({
                'HOLEID': f'DH{i+1:03d}',
                'EAST': easting_offset + np.random.uniform(-1000, 1000),
                'NORTH': northing_offset + np.random.uniform(-1000, 1000),
                'ELEV': np.random.uniform(50, 150),
                'DEPTH': np.random.uniform(100, 300),
                'AZIMUTH': np.random.uniform(0, 360),
                'DIP': np.random.uniform(-90, -60)
            })
        collars_df = pd.DataFrame(collar_data)

        # 2. Generate surveys
        survey_data = []
        for i in range(n_holes):
            hole_id = f'DH{i+1:03d}'
            depth = collar_data[i]['DEPTH']
            n_surveys = max(3, int(depth / 30))

            for j in range(n_surveys):
                survey_depth = depth * j / (n_surveys - 1) if n_surveys > 1 else 0
                survey_data.append({
                    'HOLEID': hole_id,
                    'DEPTH': survey_depth,
                    'AZIMUTH': collar_data[i]['AZIMUTH'] + np.random.uniform(-5, 5),
                    'DIP': collar_data[i]['DIP'] + np.random.uniform(-2, 2)
                })
        surveys_df = pd.DataFrame(survey_data)

        # 3. Generate assays
        assay_data = []
        for i in range(n_holes):
            hole_id = f'DH{i+1:03d}'
            depth = collar_data[i]['DEPTH']

            for j in range(n_assays_per_hole):
                from_depth = depth * j / n_assays_per_hole
                to_depth = depth * (j + 1) / n_assays_per_hole

                assay_row = {
                    'HOLEID': hole_id,
                    'FROM': from_depth,
                    'TO': to_depth,
                    'INTERVAL_ID': f'{hole_id}_ASSAY_{j}'
                }

                # Add property values
                for prop in properties:
                    if 'AU' in prop or 'PPM' in prop:
                        assay_row[prop] = np.random.lognormal(0, 1) * 100
                    elif 'CU' in prop or 'PCT' in prop:
                        assay_row[prop] = np.random.lognormal(0, 0.5) * 0.5
                    elif 'FE' in prop:
                        assay_row[prop] = np.random.uniform(10, 60)
                    else:
                        assay_row[prop] = np.random.uniform(0, 100)

                assay_data.append(assay_row)
        assays_df = pd.DataFrame(assay_data)

        # 4. Generate composites (coarser intervals)
        composite_data = []
        for i in range(n_holes):
            hole_id = f'DH{i+1:03d}'
            depth = collar_data[i]['DEPTH']

            for j in range(n_composites_per_hole):
                from_depth = depth * j / n_composites_per_hole
                to_depth = depth * (j + 1) / n_composites_per_hole

                comp_row = {
                    'HOLEID': hole_id,
                    'FROM': from_depth,
                    'TO': to_depth,
                    'INTERVAL_ID': f'{hole_id}_COMP_{j}'
                }

                # Add property values (averaged/smoothed from assays)
                for prop in properties:
                    if 'AU' in prop or 'PPM' in prop:
                        comp_row[prop] = np.random.lognormal(0, 0.8) * 100
                    elif 'CU' in prop or 'PCT' in prop:
                        comp_row[prop] = np.random.lognormal(0, 0.4) * 0.5
                    elif 'FE' in prop:
                        comp_row[prop] = np.random.uniform(15, 55)
                    else:
                        comp_row[prop] = np.random.uniform(0, 100)

                composite_data.append(comp_row)
        composites_df = pd.DataFrame(composite_data)

        # 5. Generate 3D trajectories (minimum curvature desurveying)
        trajectory_points = []
        for i in range(n_holes):
            collar = collar_data[i]
            hole_surveys = surveys_df[surveys_df['HOLEID'] == collar['HOLEID']]

            # Simple linear interpolation (real code uses minimum curvature)
            for _, survey in hole_surveys.iterrows():
                depth = survey['DEPTH']
                azimuth = np.radians(survey['AZIMUTH'])
                dip = np.radians(survey['DIP'])

                # Calculate 3D position
                x = collar['EAST'] + depth * np.sin(azimuth) * np.cos(dip)
                y = collar['NORTH'] + depth * np.cos(azimuth) * np.cos(dip)
                z = collar['ELEV'] + depth * np.sin(dip)

                trajectory_points.append([x, y, z])

        trajectories_array = np.array(trajectory_points)

        # Create PyVista mesh if available
        if PYVISTA_AVAILABLE:
            trajectories = pv.PolyData(trajectories_array)
        else:
            trajectories = trajectories_array

        # 6. Generate lithology data (if requested)
        hole_segment_lith = None
        lith_to_index = None

        if has_lithology:
            lith_data = []
            lithology_codes = {
                'OXIDE': 0,
                'TRANSITION': 1,
                'SULFIDE': 2,
                'WASTE': 3,
                'BIF': 4
            }

            for i in range(n_holes):
                hole_id = f'DH{i+1:03d}'
                depth = collar_data[i]['DEPTH']

                # Generate 3-5 lithology intervals per hole
                n_lith = np.random.randint(3, 6)
                for j in range(n_lith):
                    from_depth = depth * j / n_lith
                    to_depth = depth * (j + 1) / n_lith

                    lith_code = np.random.choice(list(lithology_codes.keys()))

                    lith_data.append({
                        'HOLEID': hole_id,
                        'FROM': from_depth,
                        'TO': to_depth,
                        'LITHOLOGY': lith_code
                    })

            hole_segment_lith = pd.DataFrame(lith_data)
            lith_to_index = lithology_codes

        # 7. Assemble drillhole data dictionary
        drillhole_data = {
            'collars': collars_df,
            'surveys': surveys_df,
            'assays': assays_df,
            'composites': composites_df,
            'trajectories': trajectories,
            'properties': properties,
            'coordinate_system': coordinate_system,
            'n_holes': n_holes
        }

        if has_lithology:
            drillhole_data['hole_segment_lith'] = hole_segment_lith
            drillhole_data['lith_to_index'] = lith_to_index

        return drillhole_data

    @classmethod
    def create_block_model(
        cls,
        nx: int = 50,
        ny: int = 50,
        nz: int = 20,
        coordinate_system: str = 'local',  # 'local' or 'utm'
        properties: Optional[List[str]] = None,
        origin: Optional[Tuple[float, float, float]] = None,
        cell_size: Tuple[float, float, float] = (10.0, 10.0, 5.0),
        seed: Optional[int] = None
    ):
        """
        Generate synthetic block model.

        Parameters:
            nx, ny, nz: Number of blocks in each direction
            coordinate_system: 'local' or 'utm'
            properties: List of property names
            origin: Origin coordinates (auto-calculated if None)
            cell_size: Cell dimensions (dx, dy, dz)
            seed: Random seed

        Returns:
            PyVista UniformGrid or dict if PyVista not available
        """
        if seed is not None:
            cls.set_seed(seed)
        else:
            cls.set_seed(cls.DEFAULT_SEED)

        if properties is None:
            properties = ['AU_PPM', 'CU_PCT', 'DENSITY']

        # Determine origin based on coordinate system
        if origin is None:
            if coordinate_system == 'utm':
                origin = (499500, 5999500, 0)
            else:  # local
                origin = (-250, -250, -50)

        # Create grid
        if PYVISTA_AVAILABLE:
            # Create uniform grid
            grid = pv.UniformGrid()
            grid.dimensions = (nx + 1, ny + 1, nz + 1)
            grid.origin = origin
            grid.spacing = cell_size

            # Add properties
            n_cells = nx * ny * nz
            for prop in properties:
                if 'AU' in prop or 'PPM' in prop:
                    values = np.random.lognormal(0, 1, n_cells) * 100
                elif 'CU' in prop or 'PCT' in prop:
                    values = np.random.lognormal(0, 0.5, n_cells) * 0.5
                elif 'DENSITY' in prop:
                    values = np.random.uniform(2.5, 3.5, n_cells)
                else:
                    values = np.random.uniform(0, 100, n_cells)

                grid.cell_data[prop] = values

            return grid
        else:
            # Return dict representation if PyVista not available
            n_cells = nx * ny * nz
            block_data = {
                'dimensions': (nx, ny, nz),
                'origin': origin,
                'spacing': cell_size,
                'properties': {}
            }

            for prop in properties:
                if 'AU' in prop:
                    values = np.random.lognormal(0, 1, n_cells) * 100
                elif 'CU' in prop:
                    values = np.random.lognormal(0, 0.5, n_cells) * 0.5
                elif 'DENSITY' in prop:
                    values = np.random.uniform(2.5, 3.5, n_cells)
                else:
                    values = np.random.uniform(0, 100, n_cells)

                block_data['properties'][prop] = values

            return block_data

    @classmethod
    def create_variogram_results(
        cls,
        properties: Optional[List[str]] = None,
        variogram_type: str = 'spherical',  # 'spherical', 'exponential', 'gaussian'
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate mock variogram results.

        Parameters:
            properties: List of properties to generate variograms for
            variogram_type: Type of variogram model
            seed: Random seed

        Returns:
            Dict with fitted variogram parameters
        """
        if seed is not None:
            cls.set_seed(seed)
        else:
            cls.set_seed(cls.DEFAULT_SEED)

        if properties is None:
            properties = ['AU_PPM']

        variogram_results = {}

        for prop in properties:
            variogram_results[prop] = {
                'type': variogram_type,
                'nugget': np.random.uniform(0, 0.2),
                'sill': np.random.uniform(0.8, 1.2),
                'range': np.random.uniform(50, 200),
                'anisotropy': {
                    'major_range': np.random.uniform(100, 200),
                    'minor_range': np.random.uniform(50, 100),
                    'vertical_range': np.random.uniform(20, 50),
                    'azimuth': np.random.uniform(0, 180),
                    'dip': np.random.uniform(0, 90),
                    'rake': np.random.uniform(0, 90)
                }
            }

        return {
            'variograms': variogram_results,
            'timestamp': datetime.now().isoformat(),
            'source': 'MockDataFactory'
        }

    @classmethod
    def create_geology_package(
        cls,
        n_units: int = 3,
        use_unified_mesh: bool = True,
        coordinate_system: str = 'local',
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate mock geological model (surfaces and/or solids).

        Parameters:
            n_units: Number of geological units
            use_unified_mesh: Use unified mesh approach
            coordinate_system: 'local' or 'utm'
            seed: Random seed

        Returns:
            Dict with surfaces and solids
        """
        if seed is not None:
            cls.set_seed(seed)
        else:
            cls.set_seed(cls.DEFAULT_SEED)

        # Determine origin
        if coordinate_system == 'utm':
            x_center, y_center = 500000, 6000000
        else:
            x_center, y_center = 0, 0

        geology = {
            'surfaces': {},
            'solids': {},
            'coordinate_system': coordinate_system
        }

        unit_names = [f'Unit_{i+1}' for i in range(n_units)]

        # Generate surfaces if PyVista available
        if PYVISTA_AVAILABLE:
            for unit_name in unit_names:
                # Create simple surface (flat with noise)
                x = np.linspace(x_center - 500, x_center + 500, 50)
                y = np.linspace(y_center - 500, y_center + 500, 50)
                xx, yy = np.meshgrid(x, y)

                # Add noise for realistic surface
                z_base = np.random.uniform(0, 50)
                noise = np.random.normal(0, 10, xx.shape)
                zz = z_base + noise

                # Create mesh
                surface = pv.StructuredGrid(xx, yy, zz)

                geology['surfaces'][unit_name] = surface

        return geology
