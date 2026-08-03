#!/bin/bash
# Double-click this file to launch the private, offline adjudication app.

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "First run: Setting up virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

if ! command -v streamlit &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
streamlit run adjudication_app.py \
    --server.address 127.0.0.1 \
    --server.port 8502 \
    --browser.gatherUsageStats false
