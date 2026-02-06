@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
python scripts/verify_pipeline.py
pause
