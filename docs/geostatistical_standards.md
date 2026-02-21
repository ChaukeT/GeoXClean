**Geostatistical Standards**

Purpose: State the required geostatistical practices and minimal checks employed by GeoX algorithms.

Core expectations
- Variogram fitting: Provide experimental variograms and fitted models; report nugget, sill, ranges and anisotropy. Keep fitting reproducible and parameterised.
- Search neighbourhoods: Default sensible neighbourhood sizes; expose parameters; document assumptions for anisotropy/rotation.
- Validation: Provide diagnostics (MAE, RMSE, R²), cross-validation or hold-out tests, and visual residuals.
- Declustering & weighting: Implement standard declustering strategies and document their use with downstream effects on estimates.
- Simulation: For stochastic outputs, provide multiple realizations, and report summary statistics (mean, variance) across realizations.

Data quality preconditions
- Algorithms should validate coordinate systems, detect duplicated coordinates, and check sample density vs grid resolution.

Reproducibility
- Capture algorithm version, parameters, RNG seeds and input dataset versions for every run.
