from setuptools import setup, find_packages

setup(
    name="block-model-viewer",
    version="1.0.0",
    description="Desktop application for uploading, parsing, and visualizing 3D block models",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.5.0",
        "pyvista>=0.42.0",
        "pyvistaqt>=0.10.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "trimesh>=3.23.0",
        "matplotlib>=3.7.0",
        "vtk>=9.2.0",
    ],
    entry_points={
        "console_scripts": [
            "block-model-viewer=block_model_viewer.main:main",
        ],
    },
    python_requires=">=3.10",
)
