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

# Ensure ROS2 dobot driver is running (only for nova5, not arm101)
# Config: prefer ~/.config/202602hackathon, fall back to local config/
_CONFIG_DIR="${HOME}/.config/202602hackathon"
[ -d "$_CONFIG_DIR" ] || _CONFIG_DIR="config"
ROBOT_TYPE=$(grep -m1 '^robot_type:' "$_CONFIG_DIR/robot_config.yaml" 2>/dev/null | awk '{print $2}' || echo "nova5")
if [ "$ROBOT_TYPE" = "nova5" ]; then
    if command -v docker &>/dev/null; then
        if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^dobot-driver$'; then
            echo "Starting dobot driver (docker compose)..."
            if [ ! -f .env ]; then
                cp .env.example .env 2>/dev/null || echo "DOBOT_IP=192.168.5.1" > .env
            fi
            docker compose --profile dobot up -d 2>/dev/null || echo "WARNING: Failed to start dobot driver. MovL/MovJ may not work."
        fi
    fi
fi

# Suppress OpenCV Qt font warnings by symlinking system fonts into cv2's expected
# location (QT_QPA_FONTDIR alone doesn't suppress the warning from OpenCV's
# embedded Qt — it needs the directory to actually exist).
CV2_QT_FONTS=$(.venv/bin/python -c "import cv2, os; print(os.path.join(os.path.dirname(cv2.__file__), 'qt', 'fonts'))" 2>/dev/null || true)
if [ -n "$CV2_QT_FONTS" ] && [ ! -e "$CV2_QT_FONTS" ]; then
    for d in /usr/share/fonts/truetype/dejavu /usr/share/fonts/truetype /usr/share/fonts; do
        if [ -d "$d" ]; then
            mkdir -p "$(dirname "$CV2_QT_FONTS")"
            ln -s "$d" "$CV2_QT_FONTS" 2>/dev/null || true
            break
        fi
    done
fi

# Shell scripts run directly; Python scripts run in venv
if [[ "$1" == *.sh ]]; then
    exec bash "$@"
else
    exec .venv/bin/python "$@"
fi
