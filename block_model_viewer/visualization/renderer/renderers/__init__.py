"""
Sub-renderers for specific geometry types.
Each sub-renderer handles mesh building and actor management for one domain.
None of these call plotter.render() — only the orchestrator controls render timing.
"""
