# GeoX Panel Debugger

Comprehensive automated testing system for all 109 panels in GeoX Clean.

## Features

- **Panel Instantiation Tests**: Verify all panels can be created without errors
- **Signal Connection Tests**: Detect broken signal connections and **silent exceptions**
- **Data Flow Tests**: Verify data propagates from registry to panels
- **Renderer Integration Tests**: Test visualization without GPU
- **Coordinate System Tests**: **Catch coordinate mismatches** that cause layers to disappear
- **Integration Tests**: End-to-end workflow testing

## Critical Tests

The system includes tests specifically designed to catch known issues:

### 1. Silent Exception Detection (`test_signals.py`)
Detects bare `except: pass` statements at:
- `drillhole_control_panel.py:255`
- `dock_setup.py:129`

### 2. Coordinate System Alignment (`test_coordinates.py`)
Ensures drillholes and block models use same coordinate system, preventing the "block model 500km away" issue.

### 3. End-to-End Drillhole Loading (`test_integration.py`)
Tests the complete workflow: data load → registry → signal → renderer

## Installation

```bash
cd c:/Users/chauk/Documents/GeoX_Clean
pip install -r tests/requirements-test.txt
```

## Usage

### Run all tests
```bash
python -m panel_debugger
```

### Run specific category
```bash
python -m panel_debugger --category signals
```

### Run critical tests only
```bash
python -m panel_debugger -m critical
```

### Run with verbose output
```bash
python -m panel_debugger --verbose
```

### Generate HTML report
```bash
python -m panel_debugger --html reports/test_report.html
```

## Test Categories

- `panel_init` - Panel instantiation
- `signals` - Signal connections (includes silent exception detection)
- `data_flow` - Data propagation through registry
- `renderer` - Renderer integration
- `coordinates` - Coordinate system alignment
- `integration` - End-to-end workflows

## Expected Results

The system will identify:
- **Silent exceptions** (CRITICAL) - Currently 2 known issues
- **Missing signal connections** (ERROR)
- **Coordinate mismatches** (ERROR)
- **Missing imports** (ERROR)
- **Slow initialization** (WARNING)
- **Memory leaks** (WARNING)

## Extending

To add panels to testing:

1. Update `config/panel_manifest.json` with panel metadata
2. Tests will automatically include the new panel

## Architecture

```
panel_debugger/
├── core/              # Testing infrastructure
│   ├── fixtures.py    # Pytest fixtures (mock QApp, registry, renderer)
│   ├── mock_factory.py # Test data generation
│   └── signal_tester.py # Signal testing utilities
├── tests/             # Test modules
│   ├── test_panel_init.py
│   ├── test_signals.py (CRITICAL - detects silent exceptions)
│   ├── test_data_flow.py
│   ├── test_renderer.py
│   ├── test_coordinates.py (CRITICAL - detects coordinate mismatches)
│   └── test_integration.py (CRITICAL - end-to-end drillhole loading)
├── config/
│   └── panel_manifest.json # Panel metadata
└── cli.py             # Command-line interface
```

## Notes

- **GeologicalExplorerPanel**: Does NOT show drillholes by design (only geological models)
- All tests run in headless mode (no GUI required)
- Mock renderer uses PyVista off-screen mode
- Tests are deterministic (seeded random data)
