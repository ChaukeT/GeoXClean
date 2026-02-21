# How to Use Faults in Geological Models

## Current Status

The fault integration has been fixed at the engine level. Faults now work correctly **IF** they include geometric data (dip, azimuth, point).

## Problem

The LoopStructural panel's fault table only captures:
- Name
- Displacement
- Type

It does NOT capture the geometric parameters needed:
- Dip angle
- Azimuth (dip direction)
- Point on fault plane (X, Y, Z)

Without these parameters, the fault cannot be positioned in 3D space and will be skipped by LoopStructural.

## Workaround Solution

Until the UI is updated, you can manually add fault geometry by modifying the fault table data structure:

### Option 1: Use the Fault Definition Panel (Recommended)

1. Open the **Fault Definition** panel from the menu
2. Click "Add Fault"
3. Enter fault parameters:
   - Name: e.g., "Fault_1"
   - Point (X, Y, Z): A point on the fault plane in world coordinates
   - Dip: Angle from horizontal (0-90°)
   - Azimuth: Dip direction (0-360°, clockwise from North)
   - Throw: Displacement magnitude in meters
4. Click "Apply to Model" to save to registry

**Note:** The Fault Definition Panel is currently not connected to the LoopStructural panel's model builder. This needs to be implemented.

### Option 2: Manually Edit Fault Parameters in Code

If you have programming access, modify `_get_fault_params()` in `loopstructural_panel.py` to include geometry:

```python
def _get_fault_params(self) -> List[Dict[str, Any]]:
    """Get fault parameters from the table."""
    faults = []
    for row in range(self._fault_table.rowCount()):
        name_item = self._fault_table.item(row, 0)
        disp_item = self._fault_table.item(row, 1)
        type_widget = self._fault_table.cellWidget(row, 2)
        
        if name_item and disp_item:
            # Get model center as default fault location
            extent = {
                'xmin': self._xmin_spin.value(),
                'xmax': self._xmax_spin.value(),
                'ymin': self._ymin_spin.value(),
                'ymax': self._ymax_spin.value(),
                'zmin': self._zmin_spin.value(),
                'zmax': self._zmax_spin.value(),
            }
            center_x = (extent['xmin'] + extent['xmax']) / 2
            center_y = (extent['ymin'] + extent['ymax']) / 2
            center_z = (extent['zmin'] + extent['zmax']) / 2
            
            faults.append({
                'name': name_item.text(),
                'displacement': float(disp_item.text()),
                'type': type_widget.currentText() if type_widget else 'normal',
                # ADD THESE LINES:
                'point': [center_x, center_y, center_z],
                'dip': 60.0,  # Default dip angle
                'azimuth': 90.0,  # Default azimuth (East-dipping)
            })
    
    return faults
```

## What the Fix Does

When you provide fault geometry (dip, azimuth, point), the system now:

1. **Generates fault trace points**: Creates a grid of 20 points on the fault plane
2. **Generates fault orientations**: Creates 10 orientation vectors normal to the fault plane
3. **Scales to normalized space**: Converts from world coordinates to [0,1] space
4. **Adds to model.data**: Inserts fault geometry into LoopStructural's data structure
5. **Creates fault feature**: Registers the fault with its displacement

This allows LoopStructural to interpolate the fault surface and apply displacement to the geological layers.

## Expected Behavior

After the fix, when you build a model with properly defined faults, you should see in the logs:

```
Added 20 fault trace points and 10 orientations for 'Fault_1'
Added fault 'Fault_1' with displacement 50.0m
```

Instead of:

```
WARNING: No data for Fault_1, skipping
```

## Next Steps (TODO)

To fully integrate faults into the UI:

1. **Connect Fault Definition Panel to LoopStructural Panel**
   - Add a "Load Faults from Registry" button in LoopStructural panel
   - Query registry for faults defined in Fault Definition Panel
   - Populate fault table with full geometry

2. **Enhance Fault Table**
   - Add columns for Dip, Azimuth, Point (X, Y, Z)
   - Make table editable for all parameters
   - Add validation for geometric parameters

3. **Create geo_model_registry Module**
   - Implement registry storage for geological model components
   - Support fault_system, fold_system, etc.
   - Provide query methods for other panels

4. **Add Fault Visualization**
   - Show fault planes in 3D viewer
   - Allow interactive fault definition (click to place)
   - Display fault displacement vectors

## Testing

To test if faults are working:

1. Build a geological model with faults that have full geometry
2. Check the terminal logs for fault trace point messages
3. Examine the model - layers should be offset across the fault
4. Switch to wireframe view to see the fault surface

