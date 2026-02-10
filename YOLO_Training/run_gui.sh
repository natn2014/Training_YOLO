#!/bin/bash
# Quick Start Script for YOLO Training GUI
# Linux/Jetson Bash Script

echo "========================================"
echo "YOLO Training GUI - Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ using your package manager"
    exit 1
fi

echo "Python detected: $(python3 --version)"
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import PySide6" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PySide6 not found. Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
else
    echo "Dependencies OK."
fi

echo ""
echo "========================================"
echo "Launching YOLO Training GUI..."
echo "========================================"
echo ""

# Launch the application
python3 yolo_trainer_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Application failed to start"
    exit 1
fi
