#!/bin/bash
# start.sh — Start the attendance backend
# Run: bash start.sh

set -e
cd "$(dirname "$0")"

echo "=== Attendance Backend Startup ==="

# Create virtualenv if not present
if [ ! -d "venv" ]; then
    echo "[Setup] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "[Setup] Installing / verifying requirements..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[Start] Launching FastAPI server on 0.0.0.0:8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
