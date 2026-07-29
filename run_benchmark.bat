@echo off
REM  SwingScope - engineering assessment
REM  Efficiency, precision, continuity, load and regression checks.
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo  [X] No virtual environment. Run run.bat first.
    pause & exit /b 1
)
call venv\Scripts\activate.bat
echo.
echo  Running engineering assessment ^(2-4 minutes^)...
echo.
python benchmark.py
echo.
pause
