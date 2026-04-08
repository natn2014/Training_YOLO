#!/bin/bash

# Universal AI Inspection - Setup Script
# This script sets up the complete environment for the YOLO-based AI inspection system

set -e  # Exit on any error

echo "🚀 Setting up Universal AI Inspection Environment"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu/Debian
check_system() {
    print_status "Checking system compatibility..."
    
    if command -v apt &> /dev/null; then
        print_success "Debian/Ubuntu system detected"
    else
        print_error "This script is designed for Ubuntu/Debian systems with apt package manager"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package list
    print_status "Updating package list..."
    sudo apt update
    
    # Install Qt/XCB dependencies for PySide6
    print_status "Installing Qt/XCB libraries..."
    sudo apt install -y \
        libxcb-cursor0 \
        libxcb-cursor-dev \
        libxcb1-dev \
        libxkbcommon-x11-0 \
        libxkbcommon-x11-dev \
        python3-venv \
        python3-pip
    
    print_success "System dependencies installed successfully"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    # Check if venv already exists
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf venv
        else
            print_status "Using existing virtual environment"
        fi
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating new virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Python virtual environment ready"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Make sure we're in the virtual environment
    source venv/bin/activate
    
    # Install dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        pip install -r requirements.txt
        print_success "Python dependencies installed successfully"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    source venv/bin/activate
    
    # Check if key packages are installed
    python -c "
import sys
try:
    import PySide6
    print('✓ PySide6 imported successfully')
except ImportError:
    print('✗ PySide6 import failed')
    sys.exit(1)

try:
    import cv2
    print('✓ OpenCV imported successfully')
except ImportError:
    print('✗ OpenCV import failed')
    sys.exit(1)

try:
    import ultralytics
    print('✓ Ultralytics imported successfully')
except ImportError:
    print('✗ Ultralytics import failed')
    sys.exit(1)

try:
    import numpy
    print('✓ NumPy imported successfully')
except ImportError:
    print('✗ NumPy import failed')
    sys.exit(1)

print('All key dependencies verified!')
"
    
    print_success "Installation verification completed"
}

# Main setup function
main() {
    echo
    print_status "Starting setup process..."
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    check_system
    echo
    
    install_system_deps
    echo
    
    setup_python_env
    echo
    
    install_python_deps
    echo
    
    verify_installation
    echo
    
    print_success "Setup completed successfully!"
    echo
    echo "🎉 Environment is ready!"
    echo "To activate the environment in the future, run:"
    echo "   source venv/bin/activate"
    echo
    echo "To run the application:"
    echo "   ./venv/bin/python main.py"
    echo "   or activate the environment first, then: python main.py"
}

# Run main function
main "$@"