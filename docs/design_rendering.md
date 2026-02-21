**GeoX — Rendering & Visualization Design**

Purpose: Document the 3D rendering architecture, UI interactions, performance strategies and integration points used by the block model viewer.

Overview
- Main UI package: `block_model_viewer/` with controllers in `block_model_viewer/controllers/`.
- Assets: `block_model_viewer/assets/` (icons, themes, styles).
- Visualization stack: likely PyQt6 + PyVista/VTK for 3D rendering (confirm in `requirements.txt`).

Responsibilities
- Render block models, drillholes and wireframes in 3D with interactive camera controls.
- Provide layer visibility, colour mapping by attribute, slicing and clipping, and LOD controls.

Architecture
- `vis_controller.py` orchestrates render updates and translates UI actions into scene updates.
- Rendering engine module (could be `rendering/` under `geox/` or integrated in `block_model_viewer/`) holds wrappers for PyVista/VTK or OpenGL contexts.
- Scene graph: maintain lightweight scene model describing visible objects and their attributes.

Key Components
- Scene Manager: tracks objects, visibility, materials.
- Renderer Adapter: abstracts underlying renderer (PyVista/VTK) and exposes methods: `add_mesh`, `update_mesh`, `remove_mesh`, `set_camera`, `render()`.
- Data decimator: down-samples dense meshes or pointclouds for fast preview.

Performance Strategies
- Level-of-Detail (LOD): provide multiple resolutions of block models and drillholes.
- Tiling/Chunking: stream geometry in tiles when viewing large datasets.
- GPU usage: rely on PyVista/VTK hardware acceleration; ensure transfer of numpy arrays is efficient.
- Async updates: run heavy geometry building in background threads/processes and push results to UI thread safely.

UX / Interaction
- Slicing plane controls, attribute colormap controls, selection/pick tools for querying block values.
- Export current view as image/scene snapshot.

Testing
- Visual regression: smoke test render loop doesn't crash and renders expected object counts.
- Performance tests: render a standard sample block model and record frame-times.

Integration Points
- Controllers call `RendererAdapter` to update geometry after geostats compute finishes.
- Data IO produces mesh/voxel representations consumable by renderer.

Notes
- Keep renderer adapter small and stable; do not place domain logic inside rendering modules.
