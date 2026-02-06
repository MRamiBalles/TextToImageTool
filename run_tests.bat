@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
python tests/test_vae.py
pause
