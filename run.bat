@echo off
REM ===================================================================
REM  SwingScope - local launcher
REM
REM  Creates a virtual environment on first run, installs dependencies,
REM  and starts the app. Subsequent runs skip straight to launch.
REM
REM  Usage: double-click, or run  run.bat  from a command prompt.
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ==========================================
echo   SwingScope - local launcher
echo  ==========================================
echo.

REM --- Locate Python -------------------------------------------------
set "PY="
py --version >nul 2>&1 && set "PY=py"
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )

if not defined PY (
    echo  [X] Python not found.
    echo.
    echo      Install Python 3.11 or later from https://python.org
    echo      During install, tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo  [+] Python !PYVER! found

REM --- Sanity check: are we in the right folder? ----------------------
if not exist "app.py" (
    echo.
    echo  [X] app.py not found in this folder.
    echo      Put run.bat in the same folder as app.py.
    echo.
    pause
    exit /b 1
)

REM --- Virtual environment -------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo  [+] Creating virtual environment ^(one-off, ~30s^)...
    %PY% -m venv venv
    if errorlevel 1 (
        echo  [X] Could not create the virtual environment.
        pause
        exit /b 1
    )
    set "FIRSTRUN=1"
) else (
    echo  [+] Virtual environment found
)

call venv\Scripts\activate.bat

REM --- Dependencies ---------------------------------------------------
if defined FIRSTRUN (
    echo  [+] Installing dependencies ^(2-3 minutes on first run^)...
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  [X] Dependency installation failed. Check the messages above.
        pause
        exit /b 1
    )
    echo  [+] Dependencies installed
) else (
    python -c "import streamlit, yfinance, pandas, plotly" >nul 2>&1
    if errorlevel 1 (
        echo  [!] Some packages are missing - reinstalling...
        python -m pip install -r requirements.txt
    )
)

REM --- Optional: broker integration -----------------------------------
if exist ".streamlit\secrets.toml" (
    findstr /i "paytm" ".streamlit\secrets.toml" >nul 2>&1
    if not errorlevel 1 (
        python -c "import pmclient" >nul 2>&1
        if errorlevel 1 (
            echo  [+] Paytm credentials detected - installing broker SDK from GitHub...
            python -m pip install git+https://github.com/paytmmoney/pyPMClient.git --quiet
            if errorlevel 1 (
                echo  [!] SDK install failed ^(Git may not be installed^).
                echo      The Broker tab will show setup instructions.
            )
        )
    )
) else (
    echo  [i] No .streamlit\secrets.toml - Broker tab will be inactive.
)

REM --- Launch ---------------------------------------------------------
echo.
echo  ==========================================
echo   Starting SwingScope
echo   Opens at http://localhost:8501
echo   Press Ctrl+C in this window to stop
echo  ==========================================
echo.

python -m streamlit run app.py

echo.
echo  SwingScope stopped.
pause
