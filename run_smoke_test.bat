@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
echo Cleaning Pip Cache...
pip cache purge
echo Custom Cache Dir set to D:\TextToImageTool\cache
echo Starting Smoke Test with DistillT5 (Dummy Mode)...
python scripts/run_inference.py --prompt "A beautiful mountain landscape" --steps 4 --distill --dummy

