@echo off
REM ===================================================================
REM  SwingScope - run the pipeline manually
REM
REM  Same job the GitHub Action does each Monday: fetch universe, score,
REM  build the bucket, log picks, evaluate matured ones, write a report.
REM
REM  Telegram delivery requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in
REM  the environment. Without them it runs and writes reports locally.
REM ===================================================================

setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo  [X] No virtual environment. Run run.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo  Running SwingScope pipeline ^(momentum model^)...
echo.

python pipeline.py --universe "Nifty 500" --size 10 --horizon 15 --no-pdf --backtest-ic 0.0479

echo.
echo  Done. Reports written to the reports\ folder.
echo.
pause
