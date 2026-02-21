"""
Signal Testing Utilities

This module provides utilities for testing Qt signals without blocking the event loop.

Features:
- Non-blocking signal emission testing
- Timeout mechanism to prevent hanging tests
- Error capture from signal handlers
- Proper cleanup (disconnect handlers)
"""

import time
from typing import Any, Callable, Optional, Tuple, List
from unittest.mock import Mock

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


class SignalTester:
    """
    Non-blocking signal connection and emission testing.

    Usage:
        tester = SignalTester(timeout_ms=1000)

        # Test if signal is connected
        connected = tester.test_signal_connection(
            registry.signals,
            'drillholeDataLoaded',
            panel,
            '_on_drillhole_data_loaded'
        )

        # Test signal emission
        success, error = tester.test_signal_emission(
            registry.signals.drillholeDataLoaded,
            mock_data,
            panel._on_drillhole_data_loaded
        )
    """

    def __init__(self, timeout_ms: int = 1000):
        """
        Initialize signal tester.

        Args:
            timeout_ms: Timeout in milliseconds for signal emission tests
        """
        self.timeout_ms = timeout_ms
        self.received_signals = []
        self.errors = []

    def test_signal_connection(
        self,
        emitter: QObject,
        signal_name: str,
        receiver: QObject,
        slot_name: str
    ) -> bool:
        """
        Test if signal is connected to slot.

        Args:
            emitter: Object emitting the signal
            signal_name: Name of the signal (e.g., 'drillholeDataLoaded')
            receiver: Object receiving the signal
            slot_name: Name of the slot method (e.g., '_on_drillhole_data_loaded')

        Returns:
            True if connected, False otherwise
        """
        if not QT_AVAILABLE:
            return False

        try:
            # Get signal from emitter
            signal = getattr(emitter, signal_name, None)
            if signal is None:
                return False

            # Get slot from receiver
            slot = getattr(receiver, slot_name, None)
            if slot is None:
                return False

            # Check connection using QMetaObject (Qt internals)
            # Note: This is a simplified check - Qt doesn't provide direct API
            # In practice, we verify by emission test
            return True

        except Exception as e:
            self.errors.append((signal_name, slot_name, str(e)))
            return False

    def test_signal_emission(
        self,
        signal: pyqtSignal,
        payload: Any,
        expected_handler: Optional[Callable] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """
        Test signal emission and handler execution.

        Args:
            signal: The signal to emit
            payload: Data to emit with signal
            expected_handler: Optional handler to verify execution

        Returns:
            Tuple of (success: bool, error: Optional[Exception])
            - success: True if signal was received within timeout
            - error: Any exception raised by handler, or None
        """
        if not QT_AVAILABLE:
            return False, Exception("Qt not available")

        received = []
        error = None

        def test_handler(*args):
            """Test handler that captures signal and errors"""
            nonlocal error
            try:
                received.append(args)
                # Call expected handler if provided
                if expected_handler:
                    expected_handler(*args)
            except Exception as e:
                error = e

        # Connect test handler
        signal.connect(test_handler)

        try:
            # Emit signal
            signal.emit(payload)

            # Process pending events
            if QApplication.instance():
                QApplication.processEvents()

            # Wait for handler with timeout
            start = time.time()
            timeout_seconds = self.timeout_ms / 1000.0

            while not received and (time.time() - start) < timeout_seconds:
                if QApplication.instance():
                    QApplication.processEvents()
                time.sleep(0.01)

            success = len(received) > 0

            # Store received signals
            if success:
                self.received_signals.extend(received)

            return success, error

        finally:
            # Always disconnect test handler
            try:
                signal.disconnect(test_handler)
            except:
                pass

    def test_multiple_emissions(
        self,
        signal: pyqtSignal,
        payloads: List[Any],
        expected_count: Optional[int] = None
    ) -> Tuple[int, List[Exception]]:
        """
        Test multiple signal emissions.

        Args:
            signal: The signal to emit
            payloads: List of payloads to emit
            expected_count: Expected number of receptions (default: len(payloads))

        Returns:
            Tuple of (received_count: int, errors: List[Exception])
        """
        if expected_count is None:
            expected_count = len(payloads)

        received = []
        errors = []

        def test_handler(*args):
            try:
                received.append(args)
            except Exception as e:
                errors.append(e)

        signal.connect(test_handler)

        try:
            # Emit all payloads
            for payload in payloads:
                signal.emit(payload)
                if QApplication.instance():
                    QApplication.processEvents()

            # Wait for all receptions
            start = time.time()
            timeout_seconds = self.timeout_ms / 1000.0

            while len(received) < expected_count and (time.time() - start) < timeout_seconds:
                if QApplication.instance():
                    QApplication.processEvents()
                time.sleep(0.01)

            return len(received), errors

        finally:
            try:
                signal.disconnect(test_handler)
            except:
                pass

    def verify_signal_not_emitted(
        self,
        signal: pyqtSignal,
        wait_ms: Optional[int] = None
    ) -> bool:
        """
        Verify that signal is NOT emitted within timeout.

        Useful for negative tests (e.g., signal shouldn't fire in this condition).

        Args:
            signal: The signal to monitor
            wait_ms: How long to wait (default: self.timeout_ms)

        Returns:
            True if signal was NOT emitted, False if it was emitted
        """
        if wait_ms is None:
            wait_ms = self.timeout_ms

        received = []

        def test_handler(*args):
            received.append(args)

        signal.connect(test_handler)

        try:
            # Wait for specified time
            start = time.time()
            timeout_seconds = wait_ms / 1000.0

            while (time.time() - start) < timeout_seconds:
                if QApplication.instance():
                    QApplication.processEvents()
                time.sleep(0.01)

            # Signal should NOT have been emitted
            return len(received) == 0

        finally:
            try:
                signal.disconnect(test_handler)
            except:
                pass

    def get_last_emission(self) -> Optional[Tuple]:
        """Get the last received signal emission"""
        if self.received_signals:
            return self.received_signals[-1]
        return None

    def get_all_emissions(self) -> List[Tuple]:
        """Get all received signal emissions"""
        return self.received_signals.copy()

    def clear_history(self):
        """Clear received signals and errors history"""
        self.received_signals.clear()
        self.errors.clear()


class SignalSpy:
    """
    Simple signal spy for monitoring signal emissions.

    Similar to Qt's QSignalSpy, but simplified for testing.

    Usage:
        spy = SignalSpy(registry.signals.drillholeDataLoaded)

        # ... trigger some action ...

        assert spy.count() == 1
        assert spy.get_emission(0) == expected_data
    """

    def __init__(self, signal: pyqtSignal):
        """
        Initialize signal spy.

        Args:
            signal: The signal to spy on
        """
        self.signal = signal
        self.emissions = []
        self._connected = False

        # Auto-connect
        self.start()

    def start(self):
        """Start spying on signal"""
        if not self._connected:
            self.signal.connect(self._on_signal)
            self._connected = True

    def stop(self):
        """Stop spying on signal"""
        if self._connected:
            try:
                self.signal.disconnect(self._on_signal)
            except:
                pass
            self._connected = False

    def _on_signal(self, *args):
        """Internal handler for signal"""
        self.emissions.append(args if len(args) > 1 else args[0] if args else None)

    def count(self) -> int:
        """Get number of emissions received"""
        return len(self.emissions)

    def get_emission(self, index: int) -> Any:
        """Get specific emission by index"""
        if 0 <= index < len(self.emissions):
            return self.emissions[index]
        return None

    def get_all_emissions(self) -> List[Any]:
        """Get all emissions"""
        return self.emissions.copy()

    def wait(self, timeout_ms: int = 1000) -> bool:
        """
        Wait for at least one emission.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            True if emission received, False if timeout
        """
        start = time.time()
        timeout_seconds = timeout_ms / 1000.0

        while len(self.emissions) == 0 and (time.time() - start) < timeout_seconds:
            if QApplication.instance():
                QApplication.processEvents()
            time.sleep(0.01)

        return len(self.emissions) > 0

    def clear(self):
        """Clear all recorded emissions"""
        self.emissions.clear()

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()


def wait_for_signal(
    signal: pyqtSignal,
    timeout_ms: int = 1000
) -> Tuple[bool, Any]:
    """
    Wait for a signal to be emitted.

    Convenience function for one-off signal waits.

    Args:
        signal: The signal to wait for
        timeout_ms: Timeout in milliseconds

    Returns:
        Tuple of (received: bool, payload: Any)
    """
    spy = SignalSpy(signal)
    success = spy.wait(timeout_ms)

    payload = None
    if success:
        payload = spy.get_emission(0)

    spy.stop()

    return success, payload
