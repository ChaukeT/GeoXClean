from .core import generate_grid_stopes, quick_stope_grid, calculate_dilution_contact, Stope
# Backward compatibility aliases (deprecated)
generate_stope_grid = generate_grid_stopes  # Alias for clarity
optimize_stopes = generate_grid_stopes  # ⚠️ DEPRECATED: This is NOT a real optimizer - it's just grid binning. Use block_model_viewer.ug.stope_opt.optimizer.optimize_stopes for real optimization
from .cut_and_fill import schedule_caf, UGCapacities
