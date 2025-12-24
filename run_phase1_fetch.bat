@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Running Phase 1 data collection...
python fetch_espn_data_phase1.py
pause

