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
pip install -r requirements.txt --quiet

# Fix OpenCV Qt font warnings: symlink system fonts into cv2's expected location.
# Without this, Qt spams "QFontDatabase: Cannot find font directory .../cv2/qt/fonts"
CV2_QT_FONTS=$(python -c "import cv2, os; print(os.path.join(os.path.dirname(cv2.__file__), 'qt', 'fonts'))" 2>/dev/null || true)
if [ -n "$CV2_QT_FONTS" ] && [ ! -e "$CV2_QT_FONTS" ]; then
    SYSTEM_FONTS=""
    for d in /usr/share/fonts/truetype/dejavu /usr/share/fonts/truetype /usr/share/fonts; do
        if [ -d "$d" ]; then
            SYSTEM_FONTS="$d"
            break
        fi
    done
    if [ -n "$SYSTEM_FONTS" ]; then
        ln -s "$SYSTEM_FONTS" "$CV2_QT_FONTS"
        echo "✓ Linked system fonts → cv2/qt/fonts (suppresses Qt font warnings)"
    else
        mkdir -p "$CV2_QT_FONTS"
        echo "⚠ Created empty cv2/qt/fonts (no system fonts found)"
    fi
fi

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

echo "✓ venv ready — run:  source .venv/bin/activate"
