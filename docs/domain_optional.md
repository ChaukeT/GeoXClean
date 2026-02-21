**Domain Optional**

Purpose: Clarify which domain-specific modules are optional and how to decouple them.

Policy
- Core GeoX must remain usable with a minimal installation: basic data IO, rendering and a lightweight estimation engine should work with minimal optional dependencies.
- Optional domain modules (e.g., `geomet_chain/`, heavy simulation engines, proprietary connectors) must be packaged as separate plugins or gated behind feature flags.

Implementation guidance
- Use plugin discovery for optional features and fallback behaviour when packages are not available.
- Provide clear error messages explaining missing optional dependencies and how to install them.
