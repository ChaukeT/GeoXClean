**Classification Philosophy**

Purpose: Outline how resource classification (Measured, Indicated, Inferred, etc.) should be approached within GeoX workflows.

Principles
- Classification is a professional judgement: GeoX provides metrics and diagnostics, but final classification is by the qualified person.
- Data-driven metrics: Use data density, search pass counts, variogram behavior, and estimation variance to inform classification boundaries.
- Transparent criteria: Classification rules (thresholds, pass counts, spatial buffers) must be codified in configuration files and recorded in provenance.

Recommended approach
- Define spatial blocks of support and compute neighbourhood statistics per block (sample count, effective range fraction, average distance to samples).
- Establish thresholds for classification levels and test them on historical datasets.

Documentation
- Store classification parameters with the model and include them in reports and PRD acceptance criteria when relevant.
