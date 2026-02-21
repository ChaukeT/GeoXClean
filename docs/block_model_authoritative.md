**Block Model - Authoritative Source**

Purpose: Define what makes a block model 'authoritative' and how it should be managed.

Definition
- An authoritative block model is a dataset promoted by a steward that has passed validation, QA and sign-off. It is the canonical source for resource quantities and downstream reporting.

Requirements to become authoritative
- Built from authoritative input datasets (drillholes, wireframes, density tables) only.
- All transformations recorded with provenance and versioned.
- Acceptance tests passed: validation, diagnostics, and QA acceptance criteria.
- Steward sign-off recorded in DataRegistry.

Usage
- Only authoritative block models should be used for reporting exports (JORC/SAMREC), contractual decisions or published numbers.

Versioning and Deprecation
- New authoritative models supersede old ones with a deprecation entry explaining differences and rationale.
