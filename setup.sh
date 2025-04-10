#!/bin/bash
echo "🔧 Setting up Runway_Position_Estimation environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download model weights
python scripts/download_weights.py

echo "✅ Setup complete. To activate, run: source venv/bin/activate"
