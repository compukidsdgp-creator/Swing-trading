@echo off
REM ===================================================================
REM  SwingScope - invariant test suite
REM
REM  Property-based tests. Each property is checked against hundreds of
REM  randomised and adversarial inputs. Run this after ANY code change.
REM ===================================================================

setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo  [X] No virtual environment. Run run.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

set ITERS=%1
if "%ITERS%"=="" set ITERS=50

echo.
echo  Running invariant suite with %ITERS% iterations per property...
echo  (higher numbers find rarer bugs; 200+ takes several minutes)
echo.

python test_invariants.py %ITERS%

echo.
pause
