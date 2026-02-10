@echo off
REM Quick Start Script for YOLO Training GUI
REM Windows Batch File

echo ========================================
echo YOLO Training GUI - Quick Start
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python detected.
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo PySide6 not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK.
)

echo.
echo ========================================
echo Launching YOLO Training GUI...
echo ========================================
echo.

REM Launch the application
python yolo_trainer_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    pause
)
