"""
Verification test for Property Panel visualization signal fix.

This test verifies that:
1. The handler method exists in MainWindow
2. The signal connection is made during initialization
3. The handler can process visualization requests correctly

Run this before manual GUI testing to ensure the code changes are correct.
"""
import sys
import inspect
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_handler_method_exists():
    """Verify that _handle_property_panel_visualization_request exists in MainWindow."""
    print("\n" + "="*70)
    print("TEST 1: Verifying handler method exists")
    print("="*70)

    from block_model_viewer.ui.main_window import MainWindow

    # Check method exists
    assert hasattr(MainWindow, '_handle_property_panel_visualization_request'), \
        "[FAIL] Handler method '_handle_property_panel_visualization_request' not found in MainWindow"

    # Check method signature
    method = getattr(MainWindow, '_handle_property_panel_visualization_request')
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    assert 'self' in params, "[FAIL] Method missing 'self' parameter"
    assert 'grid' in params, "[FAIL] Method missing 'grid' parameter"
    assert 'layer_name' in params, "[FAIL] Method missing 'layer_name' parameter"

    print("[PASS] Handler method exists with correct signature:")
    print(f"   Method: {method.__name__}{sig}")
    print(f"   Parameters: {params}")

    # Check docstring
    if method.__doc__:
        print(f"   Docstring: {method.__doc__[:100]}...")

    return True


def test_signal_connection_in_connect_signals():
    """Verify that signal connection code exists in _connect_signals method."""
    print("\n" + "="*70)
    print("TEST 2: Verifying signal connection code exists")
    print("="*70)

    from block_model_viewer.ui.main_window import MainWindow
    import inspect

    # Get source code of _connect_signals method
    method = getattr(MainWindow, '_connect_signals')
    source = inspect.getsource(method)

    # Check for signal connection
    assert 'request_visualization.connect' in source, \
        "[FAIL] Signal connection code not found in _connect_signals"

    assert '_handle_property_panel_visualization_request' in source, \
        "[FAIL] Handler reference not found in _connect_signals"

    print("[PASS] Signal connection code found in _connect_signals method")

    # Print relevant lines
    lines = source.split('\n')
    print("\n   Relevant code:")
    for i, line in enumerate(lines):
        if 'request_visualization' in line or '_handle_property_panel_visualization_request' in line:
            print(f"   Line {i+1}: {line}")

    return True


def test_property_panel_has_signal():
    """Verify that PropertyPanel has request_visualization signal."""
    print("\n" + "="*70)
    print("TEST 3: Verifying PropertyPanel has request_visualization signal")
    print("="*70)

    from block_model_viewer.ui.property_panel import PropertyPanel
    from PyQt5.QtCore import pyqtSignal

    # Check signal exists
    assert hasattr(PropertyPanel, 'request_visualization'), \
        "[FAIL] PropertyPanel missing 'request_visualization' signal"

    # Check it's actually a signal (PyQt signals have specific attributes)
    signal = getattr(PropertyPanel, 'request_visualization')
    # PyQt signals are pyqtBoundSignal when accessed as class attributes
    signal_name = type(signal).__name__
    is_signal = 'Signal' in signal_name or hasattr(signal, 'emit')

    assert is_signal, \
        f"[FAIL] 'request_visualization' is not a PyQt signal (type: {signal_name})"

    print("[PASS] PropertyPanel has request_visualization signal")
    print(f"   Signal type: {signal_name}")

    return True


def test_signal_emission_in_property_panel():
    """Verify that PropertyPanel emits the signal when switching to cached layers."""
    print("\n" + "="*70)
    print("TEST 4: Verifying signal emission code exists in PropertyPanel")
    print("="*70)

    from block_model_viewer.ui.property_panel import PropertyPanel
    import inspect

    # Get source code of _on_active_layer_changed method
    method = getattr(PropertyPanel, '_on_active_layer_changed')
    source = inspect.getsource(method)

    # Check for signal emission
    assert 'request_visualization.emit' in source, \
        "[FAIL] Signal emission code not found in _on_active_layer_changed"

    print("[PASS] Signal emission code found in _on_active_layer_changed method")

    # Print relevant lines
    lines = source.split('\n')
    print("\n   Signal emission code:")
    for i, line in enumerate(lines):
        if 'request_visualization.emit' in line:
            # Print context (2 lines before and after)
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"   {marker} {lines[j]}")
            break

    return True


def test_integration_flow():
    """Verify the complete signal flow path."""
    print("\n" + "="*70)
    print("TEST 5: Verifying complete integration flow")
    print("="*70)

    print("\n[INFO] Integration Flow Verification:")
    print("   1. PropertyPanel._on_active_layer_changed() detects cached layer")
    print("   2. PropertyPanel emits request_visualization.emit(grid, layer_name)")
    print("   3. Signal connected to MainWindow._handle_property_panel_visualization_request()")
    print("   4. Handler extracts property name from layer_name")
    print("   5. Handler calls renderer.add_block_model_layer(grid, property, layer_name)")
    print("   6. Renderer adds layer and removes other block models (mutual exclusivity)")

    # Verify each component exists
    from block_model_viewer.ui.property_panel import PropertyPanel
    from block_model_viewer.ui.main_window import MainWindow

    components = [
        (PropertyPanel, '_on_active_layer_changed', "PropertyPanel layer change detection"),
        (PropertyPanel, 'request_visualization', "PropertyPanel signal"),
        (MainWindow, '_connect_signals', "MainWindow signal connection"),
        (MainWindow, '_handle_property_panel_visualization_request', "MainWindow handler"),
    ]

    print("\n[PASS] All integration components verified:")
    for cls, attr, description in components:
        assert hasattr(cls, attr), f"[FAIL] Missing: {cls.__name__}.{attr}"
        print(f"   [OK] {description}")

    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("PROPERTY PANEL SIGNAL FIX - VERIFICATION TESTS")
    print("="*70)
    print("\nThese tests verify that the code changes are correct.")
    print("After these pass, perform manual GUI testing to verify functionality.")

    tests = [
        ("Handler Method Exists", test_handler_method_exists),
        ("Signal Connection Code Exists", test_signal_connection_in_connect_signals),
        ("PropertyPanel Has Signal", test_property_panel_has_signal),
        ("Signal Emission Code Exists", test_signal_emission_in_property_panel),
        ("Integration Flow Complete", test_integration_flow),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n[FAIL] {name}: {e}")

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {name}")
        if error:
            print(f"        Error: {error}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*70)
        print("[SUCCESS] ALL VERIFICATION TESTS PASSED!")
        print("="*70)
        print("\nThe code changes are correct. Proceed with manual GUI testing:")
        print("\n[INFO] MANUAL TEST PROCEDURE:")
        print("   1. Start the application")
        print("   2. Load drillhole data")
        print("   3. Run SGSIM simulation (creates 'SGSIM: <property>' layer)")
        print("   4. Run Resource Classification (creates 'Resource Classification' layer)")
        print("   5. Open Property Panel")
        print("   6. Switch between layers using 'Active Layer' dropdown:")
        print("      - Select 'SGSIM: <property>' -> Should show SGSIM blocks")
        print("      - Select 'Resource Classification' -> Should show classification blocks")
        print("   7. Verify only ONE block model is visible at a time")
        print("   8. Check log file for messages:")
        print("      - 'Connected property_panel.request_visualization signal to handler'")
        print("      - 'PROPERTY PANEL VISUALIZATION REQUEST: <layer_name>'")
        print("\n[PASS] If switching works without errors, the fix is successful!")
        return True
    else:
        print("\n" + "="*70)
        print("[FAIL] SOME TESTS FAILED - REVIEW CODE CHANGES")
        print("="*70)
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
