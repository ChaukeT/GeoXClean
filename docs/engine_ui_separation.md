**Engine vs UI Separation**

Purpose: Define strict boundaries between computational engines and the user interface.

Rules
- No UI code in engines: Engine modules (`geostats/`, `geomet/`, etc.) must not import or reference UI packages (PyQt, Panel, etc.).
- Engines expose deterministic, side-effect-free functions or classes that accept inputs and return outputs or well-defined objects.
- UI orchestrates: The UI layer is responsible for user interactions, parameter gathering, job submission and visualization of returned results.
- Communication via APIs: Use function/class APIs, data models (DataFrames, NumPy arrays), or job-queue messages. Avoid global state.
- Long-running work: Engines should be callable from background workers; they must provide progress callbacks or emit status via controller-managed channels (not direct UI events).

Testing & Validation
- Engines contain their own unit tests and do not rely on UI test harnesses. Mock data is used for algorithm validation.
