"""
DIAGNOSTIC TEST: Geological Modeling Pipeline
==============================================
This script tests every step of the geology pipeline with synthetic data
to identify exactly where issues occur.

Run with: python -m block_model_viewer.tests.test_geology_pipeline_diagnostic
"""

import logging
import sys
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


def create_synthetic_drillhole_data() -> pd.DataFrame:
    """
    Create synthetic drillhole data with 3 geological units:
    - Unit_A (bottom, scalar val=1.0)
    - Unit_B (middle, scalar val=2.0) 
    - Unit_C (top, scalar val=3.0)
    
    This simulates a simple layered stratigraphy that should
    produce clear, non-overlapping units.
    """
    print_separator("STEP 1: Creating Synthetic Drillhole Data")
    
    # Define bounding box (small scale for testing)
    xmin, xmax = 0, 1000
    ymin, ymax = 0, 1000
    zmin, zmax = 0, 500
    
    # Create regular grid of "drillholes"
    n_holes_x, n_holes_y = 5, 5
    x_holes = np.linspace(xmin + 100, xmax - 100, n_holes_x)
    y_holes = np.linspace(ymin + 100, ymax - 100, n_holes_y)
    
    contacts = []
    
    # Layer boundaries (Z values)
    z_base = zmin  # Bottom of Unit_A
    z_contact_ab = 150  # Contact between A and B
    z_contact_bc = 350  # Contact between B and C
    z_top = zmax  # Top of Unit_C
    
    for x in x_holes:
        for y in y_holes:
            # Add some variation to contacts
            noise = np.random.uniform(-20, 20)
            
            # Unit_A (bottom) - scalar value 1.0
            contacts.append({
                'X': x, 'Y': y, 'Z': z_contact_ab + noise,
                'val': 1.0, 'formation': 'Unit_A'
            })
            
            # Unit_B (middle) - scalar value 2.0
            contacts.append({
                'X': x, 'Y': y, 'Z': z_contact_bc + noise,
                'val': 2.0, 'formation': 'Unit_B'
            })
            
            # Unit_C (top) - scalar value 3.0
            contacts.append({
                'X': x, 'Y': y, 'Z': z_top + noise,
                'val': 3.0, 'formation': 'Unit_C'
            })
    
    df = pd.DataFrame(contacts)
    
    print(f"Created {len(df)} contact points:")
    print(f"  - {len(df[df['formation'] == 'Unit_A'])} points for Unit_A (val=1.0)")
    print(f"  - {len(df[df['formation'] == 'Unit_B'])} points for Unit_B (val=2.0)")
    print(f"  - {len(df[df['formation'] == 'Unit_C'])} points for Unit_C (val=3.0)")
    print(f"\nCoordinate ranges:")
    print(f"  X: [{df['X'].min():.1f}, {df['X'].max():.1f}]")
    print(f"  Y: [{df['Y'].min():.1f}, {df['Y'].max():.1f}]")
    print(f"  Z: [{df['Z'].min():.1f}, {df['Z'].max():.1f}]")
    print(f"\nScalar (val) range: [{df['val'].min()}, {df['val'].max()}]")
    
    return df, {
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'zmin': zmin, 'zmax': zmax
    }


def test_chronos_engine(df: pd.DataFrame, extent: dict):
    """Test the ChronosEngine directly."""
    print_separator("STEP 2: Testing ChronosEngine")
    
    try:
        from block_model_viewer.geology.chronos_engine import ChronosEngine
        print("[OK] ChronosEngine imported successfully")
    except ImportError as e:
        print(f"[FAIL] FAILED to import ChronosEngine: {e}")
        return None
    
    # Check if LoopStructural is available
    if not ChronosEngine.is_available():
        print("[FAIL] LoopStructural is NOT available - geological modeling disabled")
        return None
    
    print("[OK] LoopStructural is available")
    
    # Initialize engine
    try:
        engine = ChronosEngine(extent, resolution=30, boundary_padding=0.1)
        print(f"[OK] ChronosEngine initialized with resolution=30")
    except Exception as e:
        print(f"[FAIL] FAILED to initialize ChronosEngine: {e}")
        return None
    
    # Prepare data
    try:
        scaled_df = engine.prepare_data(df)
        print(f"[OK] Data prepared and scaled to [0,1]")
        print(f"  Scaled X range: [{scaled_df['X_s'].min():.4f}, {scaled_df['X_s'].max():.4f}]")
        print(f"  Scaled Y range: [{scaled_df['Y_s'].min():.4f}, {scaled_df['Y_s'].max():.4f}]")
        print(f"  Scaled Z range: [{scaled_df['Z_s'].min():.4f}, {scaled_df['Z_s'].max():.4f}]")
    except Exception as e:
        print(f"[FAIL] FAILED to prepare data: {e}")
        return None
    
    # Build model
    try:
        stratigraphy = ['Unit_A', 'Unit_B', 'Unit_C']
        
        # Create orientations (synthetic horizontal)
        orientations = scaled_df[['X_s', 'Y_s', 'Z_s']].copy()
        orientations['gx'] = 0.0
        orientations['gy'] = 0.0
        orientations['gz'] = 1.0
        
        engine.build_model(
            stratigraphy=stratigraphy,
            contacts=scaled_df,
            orientations=orientations,
            faults=None,
            cgw=0.1,
            interpolator_type="FDI"
        )
        print(f"[OK] Model built successfully with FDI interpolation")
    except Exception as e:
        print(f"[FAIL] FAILED to build model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return engine, stratigraphy


def test_surface_extraction(engine, stratigraphy):
    """Test surface extraction."""
    print_separator("STEP 3: Testing Surface Extraction")
    
    try:
        surfaces = engine.extract_meshes()
        print(f"[OK] Extracted {len(surfaces)} surfaces")
        
        for i, surf in enumerate(surfaces):
            verts = surf.get('vertices', [])
            faces = surf.get('faces', [])
            name = surf.get('name', f'Surface_{i}')
            val = surf.get('val', 'N/A')
            print(f"  [{i}] {name}: {len(verts)} verts, {len(faces)} faces, val={val}")
        
        return surfaces
    except Exception as e:
        print(f"[FAIL] FAILED to extract surfaces: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_solid_extraction(engine, stratigraphy):
    """Test solid extraction."""
    print_separator("STEP 4: Testing Solid Extraction")
    
    try:
        solids = engine.extract_solids(stratigraphy)
        print(f"[OK] Extracted {len(solids)} solids")
        
        for i, solid in enumerate(solids):
            verts = solid.get('vertices', [])
            faces = solid.get('faces', [])
            name = solid.get('unit_name', solid.get('name', f'Solid_{i}'))
            vol = solid.get('volume_m3', 0)
            print(f"  [{i}] {name}: {len(verts)} verts, {len(faces)} faces, volume={vol:,.0f} m³")
        
        return solids
    except Exception as e:
        print(f"[FAIL] FAILED to extract solids: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_unified_mesh_extraction(engine, stratigraphy):
    """Test unified mesh extraction - THE KEY TEST."""
    print_separator("STEP 5: Testing UNIFIED MESH Extraction (Industry-Standard)")
    
    try:
        unified = engine.extract_unified_geology_mesh(stratigraphy)
        
        if unified is None:
            print("[FAIL] Unified mesh extraction returned None!")
            return None
        
        print("[OK] Unified mesh extracted successfully!")
        print(f"  - n_units: {unified.get('n_units')}")
        print(f"  - grid_dimensions: {unified.get('grid_dimensions')}")
        print(f"  - field_range: {unified.get('field_range')}")
        print(f"  - boundaries: {unified.get('boundaries')}")
        print(f"  - formation_names: {unified.get('formation_names')}")
        
        # Check Formation_ID distribution
        formation_ids = unified.get('formation_ids')
        if formation_ids is not None:
            print(f"\n  Formation_ID distribution:")
            unique, counts = np.unique(formation_ids, return_counts=True)
            for fid, count in zip(unique, counts):
                fname = unified.get('formation_names', {}).get(fid, f'Unit_{fid}')
                pct = 100 * count / len(formation_ids)
                print(f"    Formation {fid} ({fname}): {count:,} voxels ({pct:.1f}%)")
        
        # Check PyVista grid
        pv_grid = unified.get('_pyvista_grid')
        if pv_grid is not None:
            print(f"\n  PyVista grid:")
            print(f"    - n_points: {pv_grid.n_points}")
            print(f"    - n_cells: {pv_grid.n_cells}")
            print(f"    - bounds: {pv_grid.bounds}")
            print(f"    - point_data keys: {list(pv_grid.point_data.keys())}")
        else:
            print("  [FAIL] PyVista grid is None!")
        
        return unified
    except Exception as e:
        print(f"[FAIL] FAILED to extract unified mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_runner_full_stack(df: pd.DataFrame, extent: dict):
    """Test the full GeologicalModelRunner pipeline."""
    print_separator("STEP 6: Testing GeologicalModelRunner Full Stack")
    
    try:
        from block_model_viewer.geology.model_runner import GeologicalModelRunner
        print("[OK] GeologicalModelRunner imported successfully")
    except ImportError as e:
        print(f"[FAIL] FAILED to import GeologicalModelRunner: {e}")
        return None
    
    try:
        runner = GeologicalModelRunner(
            extent=extent,
            resolution=30,
            cgw=0.1,
            smoothing_iterations=20,
            boundary_padding=0.1
        )
        print("[OK] GeologicalModelRunner initialized")
    except Exception as e:
        print(f"[FAIL] FAILED to initialize runner: {e}")
        return None
    
    try:
        stratigraphy = ['Unit_A', 'Unit_B', 'Unit_C']
        result = runner.run_full_stack(
            contacts_df=df,
            chronology=stratigraphy,
            orientations_df=None,
            faults=None,
            extract_solids=True
        )
        
        print(f"\n[OK] Full stack completed:")
        print(f"  - Surfaces: {len(result.surfaces)}")
        print(f"  - Solids: {len(result.solids)}")
        print(f"  - Unified mesh: {'YES' if result.unified_mesh else 'NO'}")
        print(f"  - JORC compliant: {result.is_jorc_compliant}")
        print(f"  - Total volume: {result.total_volume_m3:,.0f} m³")
        
        if result.unified_mesh:
            print(f"\n  Unified mesh details:")
            print(f"    - n_units: {result.unified_mesh.get('n_units')}")
            print(f"    - has _pyvista_grid: {'_pyvista_grid' in result.unified_mesh}")
        else:
            print("\n  [FAIL] WARNING: Unified mesh was NOT extracted!")
        
        return result
    except Exception as e:
        print(f"[FAIL] FAILED to run full stack: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_build_audit_package(df: pd.DataFrame, extent: dict):
    """Test build_audit_package (the package sent to renderer)."""
    print_separator("STEP 7: Testing build_audit_package (Package for Renderer)")
    
    try:
        from block_model_viewer.geology.model_runner import GeologicalModelRunner
        
        runner = GeologicalModelRunner(
            extent=extent,
            resolution=30,
            cgw=0.1,
            smoothing_iterations=20,
            boundary_padding=0.1
        )
        
        stratigraphy = ['Unit_A', 'Unit_B', 'Unit_C']
        package = runner.build_audit_package(
            drillhole_df=df,
            chronology=stratigraphy,
            faults=None
        )
        
        print(f"[OK] Audit package built:")
        print(f"  Package keys: {list(package.keys())}")
        print(f"  - surfaces: {len(package.get('surfaces', []))}")
        print(f"  - solids: {len(package.get('solids', []))}")
        print(f"  - unified_mesh: {'YES' if package.get('unified_mesh') else 'NO'}")
        print(f"  - is_jorc_compliant: {package.get('is_jorc_compliant')}")
        
        unified = package.get('unified_mesh')
        if unified:
            print(f"\n  Unified mesh in package:")
            print(f"    - n_units: {unified.get('n_units')}")
            print(f"    - has _pyvista_grid: {'_pyvista_grid' in unified}")
            print(f"    - has _scaler: {'_scaler' in unified}")
        else:
            print("\n  [FAIL] CRITICAL: unified_mesh NOT in package!")
            print("    The renderer will fall back to multi-mesh mode (Z-fighting possible)")
        
        return package
    except Exception as e:
        print(f"[FAIL] FAILED to build audit package: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_renderer_unified_mesh(package: dict):
    """Test if renderer can handle the unified mesh."""
    print_separator("STEP 8: Testing Renderer Unified Mesh Handling")
    
    unified = package.get('unified_mesh')
    if not unified:
        print("[FAIL] No unified mesh in package - cannot test renderer handling")
        return
    
    # Check required keys
    required_keys = ['_pyvista_grid', '_scaler', 'formation_names', 'n_units', 'formation_ids']
    missing = [k for k in required_keys if k not in unified]
    
    if missing:
        print(f"[FAIL] Unified mesh missing required keys: {missing}")
    else:
        print(f"[OK] Unified mesh has all required keys")
    
    # Verify PyVista grid has Formation_ID
    pv_grid = unified.get('_pyvista_grid')
    if pv_grid:
        if 'Formation_ID' in pv_grid.point_data:
            print(f"[OK] PyVista grid has 'Formation_ID' array")
            fids = pv_grid.point_data['Formation_ID']
            print(f"  - Shape: {fids.shape}")
            print(f"  - dtype: {fids.dtype}")
            print(f"  - Unique values: {np.unique(fids)}")
        else:
            print(f"[FAIL] PyVista grid MISSING 'Formation_ID' array!")
            print(f"  Available arrays: {list(pv_grid.point_data.keys())}")


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("  GEOLOGICAL MODELING PIPELINE DIAGNOSTIC")
    print("  Testing with Synthetic Data")
    print("=" * 70)
    
    # Step 1: Create synthetic data
    df, extent = create_synthetic_drillhole_data()
    
    # Step 2-5: Test ChronosEngine directly
    result = test_chronos_engine(df, extent)
    if result is None:
        print("\n[FAIL] ChronosEngine tests failed - stopping diagnostic")
        return
    
    engine, stratigraphy = result
    
    # Step 3: Test surface extraction
    surfaces = test_surface_extraction(engine, stratigraphy)
    
    # Step 4: Test solid extraction
    solids = test_solid_extraction(engine, stratigraphy)
    
    # Step 5: Test unified mesh extraction (THE KEY TEST)
    unified = test_unified_mesh_extraction(engine, stratigraphy)
    
    # Step 6: Test full ModelRunner stack
    model_result = test_model_runner_full_stack(df, extent)
    
    # Step 7: Test audit package (what renderer receives)
    package = test_build_audit_package(df, extent)
    
    # Step 8: Test renderer handling
    if package:
        test_renderer_unified_mesh(package)
    
    # Final Summary
    print_separator("DIAGNOSTIC SUMMARY")
    
    issues = []
    
    if not surfaces:
        issues.append("Surface extraction failed or returned empty")
    
    if not solids:
        issues.append("Solid extraction failed or returned empty")
    
    if not unified:
        issues.append("UNIFIED MESH extraction failed - Z-fighting will occur!")
    
    if package and not package.get('unified_mesh'):
        issues.append("unified_mesh NOT in package sent to renderer")
    
    if issues:
        print("ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. [FAIL] {issue}")
    else:
        print("[OK] All tests passed - pipeline working correctly")
    
    print("\nDone.")


if __name__ == "__main__":
    main()

