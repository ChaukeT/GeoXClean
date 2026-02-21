"""
Fix SGSIM panel indentation by replacing broken methods with clean code.
"""

# The clean, properly indented replacement code
CLEAN_METHODS = '''    def _visualize_summary(self, stat):
        """Visualize SGSIM summary statistics."""
        import numpy as np
        import pyvista as pv
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QMessageBox

        self._log_event(f"Visualizing {stat.upper()}...", "progress")

        if not self.sgsim_results:
            QMessageBox.warning(self, "No Results", "Run SGSIM first.")
            return

        summary = self.sgsim_results.get('summary', {})
        stat_data = summary.get(stat)
        if stat_data is None:
            QMessageBox.warning(self, "Not Available", f"{stat.upper()} not found.")
            return

        metadata = self.sgsim_results.get('metadata', {})
        dims = metadata.get('grid_dims', (self.nx.value(), self.ny.value(), self.nz.value()))
        spacing = metadata.get('grid_spacing', (self.dx.value(), self.dy.value(), self.dz.value()))
        origin = metadata.get('grid_origin', (self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value()))

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
        element = metadata.get('element', 'VALUE')
        property_name = f"{element}_SGSIM_{stat.upper()}"
        grid.cell_data[property_name] = stat_data.flatten(order='C')

        if abs(origin[0]) < 1000:
            grid._coordinate_shifted = True

        QTimer.singleShot(100, lambda: self._emit_viz(grid, property_name, stat))

    def _visualize_probability(self, cut):
        """Visualize probability maps."""
        import pyvista as pv
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QMessageBox

        if not self.sgsim_results:
            return

        prob_data = self.sgsim_results.get('probability_maps', {}).get(cut)
        if prob_data is None:
            QMessageBox.warning(self, "Not Available", f"Prob>{cut} not found.")
            return

        metadata = self.sgsim_results.get('metadata', {})
        dims = metadata.get('grid_dims', (self.nx.value(), self.ny.value(), self.nz.value()))
        spacing = metadata.get('grid_spacing', (self.dx.value(), self.dy.value(), self.dz.value()))
        origin = metadata.get('grid_origin', (self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value()))

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
        element = metadata.get('element', 'VALUE')
        property_name = f"{element}_SGSIM_PROB_{cut}"
        grid.cell_data[property_name] = prob_data.flatten(order='C')

        if abs(origin[0]) < 1000:
            grid._coordinate_shifted = True

        QTimer.singleShot(100, lambda: self._emit_viz(grid, property_name, f"Prob>{cut}"))

    def _emit_viz(self, grid, property_name, description):
        """Safe GPU handoff."""
        try:
            self.request_visualization.emit(grid, property_name)
            self._log_event(f"SUCCESS: {description} sent to viewer", "success")
        except Exception as e:
            self._log_event(f"ERROR: {e}", "error")

'''

# Read file
with open('block_model_viewer/ui/sgsim_panel.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the broken section (starts at line 2292, goes until we find next method)
start_line = None
end_line = None

for i, line in enumerate(lines):
    if '    def _visualize_summary(self, stat):' in line:
        start_line = i
    elif start_line is not None and '    def _auto_back_transform(self):' in line:
        end_line = i
        break

if start_line is None or end_line is None:
    print(f"ERROR: Could not find method boundaries")
    print(f"  start_line: {start_line}, end_line: {end_line}")
    exit(1)

print(f"Replacing lines {start_line+1} to {end_line} ({end_line-start_line} lines)")

# Replace
new_content = (
    ''.join(lines[:start_line]) +
    CLEAN_METHODS +
    ''.join(lines[end_line:])
)

# Write
with open('block_model_viewer/ui/sgsim_panel.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("SUCCESS: Replaced broken methods with clean code")
