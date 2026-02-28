#!/usr/bin/env bash
# Run a script using the project venv. Creates/updates venv if needed.
# Usage: ./run.sh scripts/control_panel.py [args...]
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script.py> [args...]"
    echo "Examples:"
    echo "  ./run.sh scripts/test_robot.py"
    echo "  ./run.sh scripts/control_panel.py"
    echo "  ./run.sh scripts/debug_robot.py"
    echo "  ./run.sh src/main.py"
    exit 1
fi

# Create or update venv if needed
if [ ! -d .venv ]; then
    echo "Creating Python venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install -r requirements.txt --quiet
    echo "venv ready."
elif [ requirements.txt -nt .venv/.deps_installed ]; then
    echo "Updating dependencies..."
    .venv/bin/pip install -r requirements.txt --quiet
    touch .venv/.deps_installed
    echo "Dependencies updated."
fi

# Ensure ROS2 dobot driver is running (provides motion port 30003)
if command -v docker &>/dev/null; then
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^dobot-driver$'; then
        echo "Starting dobot driver (docker compose)..."
        if [ ! -f .env ]; then
            cp .env.example .env 2>/dev/null || echo "DOBOT_IP=192.168.5.1" > .env
        fi
        docker compose --profile dobot up -d 2>/dev/null || echo "WARNING: Failed to start dobot driver. MovL/MovJ may not work."
    fi
fi

exec .venv/bin/python "$@"
