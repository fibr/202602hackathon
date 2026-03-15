#!/usr/bin/env bash
# One-time setup: create venv and install dependencies.
# Re-running is safe — pip will skip already-installed packages.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Creating Python venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip --quiet
# Uninstall opencv-python (non-headless) if present — its bundled Qt plugins
# conflict with PyQt5.  opencv-python-headless provides the same API without Qt.
if pip show opencv-python &>/dev/null; then
    echo "Replacing opencv-python with opencv-python-headless (fixes Qt conflict)..."
    pip uninstall opencv-python -y --quiet
fi
pip install -r requirements.txt --quiet

# Ensure current user can access serial ports (needed for arm101 USB servo bus).
# /dev/ttyACM* is owned by group 'dialout' on most Linux distros.
if [ -e /dev/ttyACM0 ] || [ -e /dev/ttyUSB0 ]; then
    if ! id -nG | grep -qw dialout; then
        echo ""
        echo "Adding $USER to 'dialout' group for serial port access..."
        sudo usermod -aG dialout "$USER"
        echo "✓ Added to dialout — log out and back in (or run: newgrp dialout)"
    fi
fi

# Install servo SDK into Isaac Lab's Python if Isaac Lab is available.
# Isaac Lab uses its own Python, so deps like feetech-servo-sdk must be
# installed there separately for the digital twin mirror mode to work.
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/src/IsaacLab}"
ISAACLAB_PY="$ISAACLAB_DIR/_isaac_sim/python.sh"
if [ -x "$ISAACLAB_PY" ]; then
    echo "Installing servo deps into Isaac Lab Python..."
    "$ISAACLAB_PY" -m pip install feetech-servo-sdk pyserial pyyaml --quiet 2>/dev/null || true
fi

echo "✓ venv ready — run:  source .venv/bin/activate"
