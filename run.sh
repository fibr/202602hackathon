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

# Ensure .env exists for docker compose
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# Start ROS2 driver container
if ! command -v docker &>/dev/null; then
    echo "WARNING: Docker not found. MovJ/MovL will not work (jog fallback only)."
    echo "  Install Docker: https://docs.docker.com/engine/install/"
elif ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^dobot-driver$'; then
    echo "Starting ROS2 driver (docker compose --profile dobot up -d)..."
    if ! docker compose --profile dobot up -d 2>&1; then
        echo "WARNING: Failed to start ROS2 driver. MovJ/MovL will not work (jog fallback only)."
        echo "  Try: docker compose --profile dobot build"
    fi
fi

exec .venv/bin/python "$@"
