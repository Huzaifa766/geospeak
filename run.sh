#!/bin/bash

# GeoSpeak Setup and Run Script

echo "🌍 GeoSpeak - AI Language Translation Setup 🌍"
echo "================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your Google Gemini API key!"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Press Enter to continue after setting up your API key..."
fi

# Run the application
echo "Starting GeoSpeak application..."
echo "🚀 Application will be available at: http://localhost:5000"
echo ""
python app.py
