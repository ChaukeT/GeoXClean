**No ML Unless Explicit**

Purpose: Policy governing the usage of machine learning methods in GeoX workflows.

Policy Statement
- Machine Learning (ML) methods may only be used when explicitly requested, configured, and documented by the user and approved by a project steward.
- ML models that influence reported resource estimates must be auditable, versioned, and deterministic (seeded) where appropriate.

Requirements for ML use
- Explicit opt-in: ML must be turned on per-project or per-job; default is OFF.
- Documentation: Record model type, training data versions, hyperparameters, training code version, and evaluation metrics in provenance.
- Reproducibility: Training pipelines must be reproducible; training seeds, data splits and environment must be captured.
- Review: Any ML-driven classification or estimate used in reporting must be reviewed by a qualified person and treated as advisory until signed off.

Operational guidance
- Isolate ML workloads: training should run in controlled environments; inference in production must check for data drift and record input versions.
- Provide fallback: For any ML-driven output, provide a deterministic fallback (e.g., rule-based estimate) for verification.
