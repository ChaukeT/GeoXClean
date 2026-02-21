# Fault Integration Fix

## Problem Identified

When adding faults to the geological model, they had no effect on the model. The terminal logs showed:

```
WARNING: No data for Fault_1, skipping
```

## Root Cause

The fault integration code was calling `model.create_and_add_fault(name, displacement)` but **LoopStructural requires actual geometric data** (fault trace points and orientations) in `model.data` to interpolate the fault surface. Without this data, LoopStructural skips the fault entirely.

The workflow was:
1. User defines fault with geometric parameters (dip, azimuth, point, displacement)
2. Code adds fault to model with just name and displacement
3. LoopStructural has no point data to define where the fault is in 3D space
4. Fault is skipped → no effect on geology

## Solution Implemented

### 1. Created `block_model_viewer/geology/faults.py`

New module with `FaultPlane` class that:
- Converts fault parameters (dip, azimuth, point) into 3D geometry
- Generates fault trace points distributed across the fault plane
- Generates orientation vectors (normal to the fault plane)
- Provides proper coordinate transformations

Key methods:
- `get_normal_vector()`: Calculates fault plane normal from dip/azimuth
- `generate_fault_trace_points()`: Creates grid of points on fault plane
- `generate_fault_orientations()`: Creates orientation constraints
- `to_dict()` / `from_dict()`: Serialization support

### 2. Modified `block_model_viewer/geology/chronos_engine.py`

Updated the `build_model()` method to:
1. Check if fault has geometric data (dip, azimuth, point)
2. If yes, use `FaultPlane` to generate fault geometry
3. Scale fault geometry to normalized [0,1] space
4. **Add fault trace points and orientations to `model.data` BEFORE calling `create_and_add_fault()`**
5. Then create the fault feature with displacement

This ensures LoopStructural has the geometric data it needs to interpolate the fault surface.

## Technical Details

### Fault Geometry Generation

The fault plane is defined by:
- **Point**: A point on the fault plane (x, y, z)
- **Dip**: Angle from horizontal (0-90°)
- **Azimuth**: Dip direction (0-360°, clockwise from north)

From these parameters, we:
1. Calculate the normal vector using spherical coordinates
2. Create two orthogonal vectors in the fault plane
3. Generate a grid of points on the plane within model bounds
4. Add orientation data (gradient = normal vector)

### Coordinate System

- X-axis: East
- Y-axis: North  
- Z-axis: Up
- Azimuth: Measured clockwise from North
- Right-hand rule conventions

### Data Format for LoopStructural

Fault trace points:
```
X, Y, Z, feature_name, val
```

Fault orientations:
```
X, Y, Z, gx, gy, gz, feature_name
```

These are added to `model.data` before calling `create_and_add_fault()`.

## Testing

To test the fix:
1. Open the LoopStructural panel
2. Add a fault using the Fault Definition panel
3. Build the geological model
4. Check the logs - you should see:
   ```
   Added 20 fault trace points and 10 orientations for 'Fault_1'
   Added fault 'Fault_1' with displacement 50.0m
   ```
5. The fault should now visibly displace the geological layers

## Files Modified

- `block_model_viewer/geology/faults.py` (NEW)
- `block_model_viewer/geology/chronos_engine.py` (MODIFIED)

## Compliance

- ✅ Follows GeoX invariants (deterministic, auditable)
- ✅ Preserves coordinate system conventions
- ✅ No changes to data schemas
- ✅ No cross-boundary violations (UI ↔ engine)
- ✅ Surgical fix - minimal changes to existing code

