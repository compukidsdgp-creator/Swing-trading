@echo off
REM ===================================================================
REM  SwingScope - full statistical validation suite
REM
REM  Runs the tests that exist but have never been executed:
REM    - Out-of-sample split (chronological, with evaluation ledger)
REM    - Monte Carlo trade-sequence scrambling
REM    - Sharpe / Sortino / Calmar / VaR / CVaR
REM
REM  These are the gap in "Statistical Validation" - the tooling was
REM  built and left idle.
REM ===================================================================
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo  [X] No virtual environment. Run run.bat first.
    pause & exit /b 1
)
call venv\Scripts\activate.bat
echo.
echo  Running full validation suite ^(5-15 minutes^)...
echo.
python run_validation.py
echo.
pause
