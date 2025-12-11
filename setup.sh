#!/bin/bash
# TMHNA Financial Intelligence Portal - Setup Script

echo "=============================================="
echo "  TMHNA Financial Intelligence Portal Setup"
echo "=============================================="
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python -m venv venv
else
    echo "[1/4] Virtual environment already exists"
fi

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "[4/4] Creating directories..."
mkdir -p data models static/css static/js static/img templates

echo
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo
echo "To start the application:"
echo "  python run.py"
echo
echo "Then open: http://localhost:5000"
echo

