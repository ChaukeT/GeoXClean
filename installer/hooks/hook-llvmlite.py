# -*- coding: utf-8 -*-
"""
PyInstaller hook for llvmlite
Ensures llvmlite DLL is properly collected
"""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os
import sys

# Collect all llvmlite data and DLLs
datas = collect_data_files('llvmlite')
binaries = collect_dynamic_libs('llvmlite')

# Explicitly find and add llvmlite.dll
try:
    import llvmlite
    llvmlite_dir = os.path.dirname(llvmlite.__file__)
    binding_dir = os.path.join(llvmlite_dir, 'binding')

    # Add all DLLs from the binding directory
    if os.path.exists(binding_dir):
        for filename in os.listdir(binding_dir):
            if filename.endswith('.dll') or filename.endswith('.pyd'):
                src_path = os.path.join(binding_dir, filename)
                binaries.append((src_path, 'llvmlite/binding'))
except ImportError:
    pass

hiddenimports = [
    'llvmlite',
    'llvmlite.binding',
    'llvmlite.binding.ffi',
    'llvmlite.ir',
    'llvmlite.llvmpy',
    'llvmlite.llvmpy.core',
]
