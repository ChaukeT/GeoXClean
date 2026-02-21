"""Mine planning algorithms package (underground stope optimization & scheduling).
Placeholder implementations added to satisfy UI imports; replace with full algorithms as needed.
"""

from .ug import optimize_stopes, generate_grid_stopes, quick_stope_grid, calculate_dilution_contact, UGCapacities
# Note: optimize_stopes is deprecated alias for generate_grid_stopes (rigid grid binning, NOT real optimization)
# For real stope optimization, use block_model_viewer.ug.stope_opt.optimizer.optimize_stopes
# quick_stope_grid is also a deprecated alias for generate_grid_stopes
