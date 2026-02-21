"""
DIAGNOSTIC TEST: Renderer Unified Mesh
======================================
This script tests the renderer's handling of the unified mesh
to identify exactly where the rendering issue occurs.

Run with: python -m block_model_viewer.tests.test_renderer_unified_mesh
"""

import logging
import sys
import os
import numpy as np
import pandas as pd

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def print_separator(title: str):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def create_synthetic_geology_package():
    """
    Create a synthetic geology package with unified mesh.
    """
    print_separator("STEP 1: Creating Synthetic Geology Package")
    
    # Import the model runner
    from block_model_viewer.geology.model_runner import GeologicalModelRunner
    
    # Define extent
    extent = {
        'xmin': 0, 'xmax': 1000,
        'ymin': 0, 'ymax': 1000,
        'zmin': 0, 'zmax': 500
    }
    
    # Create synthetic drillhole data
    x_holes = np.linspace(100, 900, 5)
    y_holes = np.linspace(100, 900, 5)
    
    contacts = []
    for x in x_holes:
        for y in y_holes:
            noise = np.random.uniform(-20, 20)
            # Unit_A (bottom)
            contacts.append({'X': x, 'Y': y, 'Z': 150 + noise, 'val': 1.0, 'formation': 'Unit_A'})
            # Unit_B (middle)
            contacts.append({'X': x, 'Y': y, 'Z': 350 + noise, 'val': 2.0, 'formation': 'Unit_B'})
            # Unit_C (top)
            contacts.append({'X': x, 'Y': y, 'Z': 500 + noise, 'val': 3.0, 'formation': 'Unit_C'})
    
    df = pd.DataFrame(contacts)
    print(f"Created {len(df)} contact points")
    
    # Build geology package
    print("\nBuilding geological model...")
    runner = GeologicalModelRunner(
        extent=extent,
        resolution=30,
        cgw=0.1,
        smoothing_iterations=20,
        boundary_padding=0.1
    )
    
    package = runner.build_audit_package(
        drillhole_df=df,
        chronology=['Unit_A', 'Unit_B', 'Unit_C'],
        faults=None
    )
    
    print(f"\nPackage built:")
    print(f"  - surfaces: {len(package.get('surfaces', []))}")
    print(f"  - solids: {len(package.get('solids', []))}")
    print(f"  - unified_mesh: {'YES' if package.get('unified_mesh') else 'NO'}")
    
    unified_mesh = package.get('unified_mesh')
    if unified_mesh:
        print(f"\nUnified mesh details:")
        print(f"  - n_units: {unified_mesh.get('n_units')}")
        print(f"  - has _pyvista_grid: {'_pyvista_grid' in unified_mesh}")
        print(f"  - has _scaler: {'_scaler' in unified_mesh}")
        
        pv_grid = unified_mesh.get('_pyvista_grid')
        if pv_grid:
            print(f"  - grid n_points: {pv_grid.n_points}")
            print(f"  - grid n_cells: {pv_grid.n_cells}")
            print(f"  - grid bounds: {pv_grid.bounds}")
            print(f"  - point_data keys: {list(pv_grid.point_data.keys())}")
    
    return package


def test_renderer_unified_mesh_handling(package):
    """
    Test the renderer's _render_unified_geology_mesh method step by step.
    """
    print_separator("STEP 2: Testing Renderer Unified Mesh Handling")
    
    unified_mesh = package.get('unified_mesh')
    if not unified_mesh:
        print("[FAIL] No unified mesh in package!")
        return False
    
    print("[OK] unified_mesh exists in package")
    
    # Check required keys
    required_keys = ['_pyvista_grid', '_scaler', 'formation_names', 'n_units', 'formation_ids']
    missing = [k for k in required_keys if k not in unified_mesh]
    
    if missing:
        print(f"[FAIL] Missing required keys: {missing}")
        return False
    
    print(f"[OK] All required keys present")
    
    # Get the PyVista grid
    pv_grid = unified_mesh['_pyvista_grid']
    scaler = unified_mesh['_scaler']
    formation_names = unified_mesh['formation_names']
    n_units = unified_mesh['n_units']
    
    print(f"\nAnalyzing PyVista grid:")
    print(f"  - Type: {type(pv_grid).__name__}")
    print(f"  - n_points: {pv_grid.n_points}")
    print(f"  - n_cells: {pv_grid.n_cells}")
    print(f"  - bounds: {pv_grid.bounds}")
    
    # Check Formation_ID
    if 'Formation_ID' not in pv_grid.point_data:
        print(f"[FAIL] Formation_ID not in point_data!")
        return False
    
    formation_ids = pv_grid.point_data['Formation_ID']
    print(f"\nFormation_ID analysis:")
    print(f"  - Shape: {formation_ids.shape}")
    print(f"  - dtype: {formation_ids.dtype}")
    print(f"  - Unique values: {np.unique(formation_ids)}")
    
    for fid in np.unique(formation_ids):
        count = np.sum(formation_ids == fid)
        fname = formation_names.get(int(fid), f'Unit_{fid}')
        print(f"  - Formation {fid} ({fname}): {count} points")
    
    return True


def test_coordinate_transformation(package):
    """
    Test the coordinate transformation from scaled to world space.
    """
    print_separator("STEP 3: Testing Coordinate Transformation")
    
    unified_mesh = package.get('unified_mesh')
    pv_grid = unified_mesh['_pyvista_grid']
    scaler = unified_mesh['_scaler']
    
    # Original grid points (scaled space [0,1])
    scaled_points = np.asarray(pv_grid.points, dtype=np.float64)
    
    print(f"Scaled space (GPU internal):")
    print(f"  X range: [{scaled_points[:, 0].min():.4f}, {scaled_points[:, 0].max():.4f}]")
    print(f"  Y range: [{scaled_points[:, 1].min():.4f}, {scaled_points[:, 1].max():.4f}]")
    print(f"  Z range: [{scaled_points[:, 2].min():.4f}, {scaled_points[:, 2].max():.4f}]")
    
    # Transform to world coordinates
    world_points = scaler.inverse_transform(scaled_points)
    
    print(f"\nWorld space (UTM):")
    print(f"  X range: [{world_points[:, 0].min():.2f}, {world_points[:, 0].max():.2f}]")
    print(f"  Y range: [{world_points[:, 1].min():.2f}, {world_points[:, 1].max():.2f}]")
    print(f"  Z range: [{world_points[:, 2].min():.2f}, {world_points[:, 2].max():.2f}]")
    
    # Calculate local origin (center of model)
    local_origin = np.mean(world_points, axis=0)
    
    print(f"\nLocal origin (for GPU precision):")
    print(f"  [{local_origin[0]:.2f}, {local_origin[1]:.2f}, {local_origin[2]:.2f}]")
    
    # After shift
    local_points = world_points - local_origin
    
    print(f"\nAfter local origin shift (GPU coordinates):")
    print(f"  X range: [{local_points[:, 0].min():.2f}, {local_points[:, 0].max():.2f}]")
    print(f"  Y range: [{local_points[:, 1].min():.2f}, {local_points[:, 1].max():.2f}]")
    print(f"  Z range: [{local_points[:, 2].min():.2f}, {local_points[:, 2].max():.2f}]")
    
    return True


def test_pyvista_rendering(package):
    """
    Test actual PyVista rendering with the unified mesh.
    """
    print_separator("STEP 4: Testing PyVista Rendering (Headless)")
    
    try:
        import pyvista as pv
        from matplotlib.colors import ListedColormap
        
        # Enable off-screen rendering for testing
        pv.OFF_SCREEN = True
        
        unified_mesh = package.get('unified_mesh')
        pv_grid = unified_mesh['_pyvista_grid']
        scaler = unified_mesh['_scaler']
        formation_names = unified_mesh['formation_names']
        n_units = unified_mesh['n_units']
        
        # Transform to world and apply local origin shift
        scaled_points = np.asarray(pv_grid.points, dtype=np.float64)
        world_points = scaler.inverse_transform(scaled_points)
        local_origin = np.mean(world_points, axis=0)
        local_points = world_points - local_origin
        
        # Create shifted grid
        shifted_grid = pv_grid.copy()
        shifted_grid.points = local_points
        
        print(f"Shifted grid created:")
        print(f"  - n_points: {shifted_grid.n_points}")
        print(f"  - n_cells: {shifted_grid.n_cells}")
        print(f"  - Formation_ID in point_data: {'Formation_ID' in shifted_grid.point_data}")
        
        # Create discrete colormap
        geology_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        colors_for_units = [geology_colors[i % len(geology_colors)] for i in range(n_units)]
        discrete_cmap = ListedColormap(colors_for_units)
        
        print(f"\nCreated discrete colormap with {n_units} colors")
        
        # Create plotter
        print("\nCreating PyVista plotter...")
        plotter = pv.Plotter(off_screen=True)
        
        # Try to add the mesh
        print("Adding mesh to plotter...")
        try:
            actor = plotter.add_mesh(
                shifted_grid,
                scalars="Formation_ID",
                cmap=discrete_cmap,
                categories=True,
                smooth_shading=False,
                interpolate_before_map=False,
                opacity=1.0,
                lighting=True,
                show_edges=False,
            )
            
            if actor is not None:
                print("[OK] Mesh added successfully!")
                print(f"  - Actor type: {type(actor).__name__}")
                
                # Check actor properties
                prop = actor.GetProperty()
                if prop:
                    print(f"  - Opacity: {prop.GetOpacity()}")
                    print(f"  - Interpolation: {prop.GetInterpolation()}")
                
                # Check mapper
                mapper = actor.GetMapper()
                if mapper:
                    print(f"  - ScalarVisibility: {mapper.GetScalarVisibility()}")
                    
                    # Check scalar range
                    scalar_range = mapper.GetScalarRange()
                    print(f"  - Scalar range: {scalar_range}")
            else:
                print("[FAIL] Actor is None!")
                
        except Exception as e:
            print(f"[FAIL] Error adding mesh: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Try to render
        print("\nAttempting render...")
        try:
            plotter.show(auto_close=False)
            print("[OK] Render completed")
        except Exception as e:
            print(f"[WARN] Render issue (may be headless): {e}")
        
        plotter.close()
        print("[OK] Plotter closed")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] PyVista rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_renderer_load_geology_package(package):
    """
    Test the actual Renderer.load_geology_package method.
    """
    print_separator("STEP 5: Testing Renderer.load_geology_package()")
    
    try:
        import pyvista as pv
        from block_model_viewer.visualization.renderer import Renderer
        
        # Enable off-screen rendering
        pv.OFF_SCREEN = True
        
        print("Creating Renderer instance...")
        renderer = Renderer()
        
        print("Initializing plotter...")
        renderer.plotter = pv.Plotter(off_screen=True)
        renderer.active_layers = {}
        renderer.scene_layers = {}  # Required for layer registration
        renderer.default_opacity = {'default': 1.0, 'geology_unified': 1.0, 'geology_solid': 1.0}
        renderer.layer_change_callback = None
        
        # Check if unified_mesh is in package
        unified_mesh = package.get('unified_mesh')
        print(f"\nPackage contents:")
        print(f"  - unified_mesh: {'YES' if unified_mesh else 'NO'}")
        print(f"  - surfaces: {len(package.get('surfaces', []))}")
        print(f"  - solids: {len(package.get('solids', []))}")
        
        if unified_mesh:
            print(f"  - unified_mesh n_units: {unified_mesh.get('n_units')}")
            print(f"  - unified_mesh has _pyvista_grid: {'_pyvista_grid' in unified_mesh}")
        
        # Call load_geology_package with solids mode (to trigger unified mesh)
        print("\nCalling renderer.load_geology_package(package, render_mode='solids')...")
        
        try:
            renderer.load_geology_package(package, render_mode='solids')
            print("[OK] load_geology_package completed")
            
            # Check what layers were created
            print(f"\nActive layers after load:")
            for layer_name, layer_info in renderer.active_layers.items():
                layer_type = layer_info.get('layer_type', 'unknown')
                print(f"  - {layer_name} (type: {layer_type})")
            
            # Check if unified mesh was used
            unified_used = any('Unified' in name for name in renderer.active_layers.keys())
            if unified_used:
                print("\n[OK] UNIFIED MESH WAS USED!")
            else:
                print("\n[FAIL] UNIFIED MESH WAS NOT USED - fell back to multi-mesh!")
                
                # Debug: check why
                print("\nDEBUGGING: Why unified mesh wasn't used:")
                print(f"  - unified_mesh is not None: {unified_mesh is not None}")
                if unified_mesh:
                    print(f"  - _pyvista_grid exists: {'_pyvista_grid' in unified_mesh}")
                    print(f"  - _scaler exists: {'_scaler' in unified_mesh}")
            
        except Exception as e:
            print(f"[FAIL] load_geology_package raised exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        renderer.plotter.close()
        return True
        
    except Exception as e:
        print(f"[FAIL] Renderer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("  RENDERER UNIFIED MESH DIAGNOSTIC")
    print("  Testing why rendering behaves incorrectly")
    print("=" * 70)
    
    # Step 1: Create package
    package = create_synthetic_geology_package()
    
    # Step 2: Test unified mesh structure
    if not test_renderer_unified_mesh_handling(package):
        print("\n[FAIL] Unified mesh structure test failed!")
        return
    
    # Step 3: Test coordinate transformation
    test_coordinate_transformation(package)
    
    # Step 4: Test PyVista rendering directly
    test_pyvista_rendering(package)
    
    # Step 5: Test actual Renderer class
    test_renderer_load_geology_package(package)
    
    print_separator("DIAGNOSTIC COMPLETE")
    print("Check the logs above for any [FAIL] messages.")


if __name__ == "__main__":
    main()

