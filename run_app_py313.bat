@echo off
REM Launcher script for Block Model Viewer using Python 3.13
REM This is required because VTK doesn't yet support Python 3.14

echo Starting Block Model Viewer with Python 3.13...
py -3.13 run_app.py
pause

