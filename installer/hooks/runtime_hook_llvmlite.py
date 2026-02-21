# -*- coding: utf-8 -*-
"""
Runtime hook for llvmlite - fixes DLL loading in frozen executables

This hook runs BEFORE the main application and ensures llvmlite can find its DLL.
The LLVMPY_AddSymbol KeyError occurs when llvmlite.dll is not in the expected path
or when the DLL fails to initialize due to missing dependencies.
"""
import os
import sys
import ctypes

def _fix_llvmlite_path():
    """Add llvmlite binding directory to DLL search path and preload DLL."""
    if not getattr(sys, 'frozen', False):
        return

    # Running as frozen executable
    base_path = sys._MEIPASS

    # Possible locations for llvmlite.dll (in order of preference)
    llvmlite_paths = [
        base_path,  # Root _internal folder (most reliable)
        os.path.join(base_path, 'llvmlite', 'binding'),
        os.path.join(base_path, 'llvmlite'),
    ]

    # Add all paths to DLL search directories (Windows 3.8+)
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        for path in llvmlite_paths:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                except (OSError, AttributeError):
                    pass

    # Also add to PATH environment variable (fallback)
    current_path = os.environ.get('PATH', '')
    for path in llvmlite_paths:
        if os.path.exists(path) and path not in current_path:
            os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')

    # Try to preload llvmlite.dll explicitly
    dll_loaded = False
    for path in llvmlite_paths:
        dll_path = os.path.join(path, 'llvmlite.dll')
        if os.path.exists(dll_path):
            try:
                # Use LOAD_WITH_ALTERED_SEARCH_PATH to load DLL with its dependencies
                ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
                dll_loaded = True
                break
            except OSError:
                # Try LoadLibraryEx with altered search path
                try:
                    kernel32 = ctypes.windll.kernel32
                    LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
                    handle = kernel32.LoadLibraryExW(dll_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
                    if handle:
                        dll_loaded = True
                        break
                except Exception:
                    pass

# Execute the fix immediately
_fix_llvmlite_path()
