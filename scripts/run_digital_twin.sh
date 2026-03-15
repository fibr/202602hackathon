#!/bin/bash
# Launch the ARM101 digital twin in Isaac Sim via Isaac Lab.
#
# Usage:
#   ./scripts/run_digital_twin.sh              # Interactive GUI mode
#   ./scripts/run_digital_twin.sh --headless   # Headless (no GUI)
#   ./scripts/run_digital_twin.sh --enable_cameras              # Enable camera rendering
#   ./scripts/run_digital_twin.sh --enable_cameras --save_images # Render + save PNGs
#   ./scripts/run_digital_twin.sh --mirror     # Mirror real arm joints (requires HW)
#   ./scripts/run_digital_twin.sh --mirror --enable_cameras     # Full digital twin
#
# Prerequisites:
#   - Isaac Sim 5.0+ installed at ~/isaac-sim
#   - Isaac Lab 2.2+ installed at ~/src/IsaacLab

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/src/IsaacLab}"

if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "ERROR: Isaac Lab not found at $ISAACLAB_DIR"
    echo "Set ISAACLAB_DIR to your Isaac Lab installation path"
    exit 1
fi

if [ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo "ERROR: isaaclab.sh not found or not executable at $ISAACLAB_DIR/isaaclab.sh"
    exit 1
fi

# ── Check runtime dependencies for mirror mode ──────────────────────────
# Isaac Lab uses its own Python; feetech-servo-sdk and pyserial must be
# installed there (setup.sh does this, but warn if they're missing).
ISAACLAB_PY="$ISAACLAB_DIR/_isaac_sim/python.sh"
if [ -x "$ISAACLAB_PY" ]; then
    _missing=()
    "$ISAACLAB_PY" -c "import serial" 2>/dev/null || _missing+=("pyserial")
    "$ISAACLAB_PY" -c "import scservo_sdk" 2>/dev/null || _missing+=("feetech-servo-sdk")
    if [ ${#_missing[@]} -gt 0 ]; then
        echo ""
        echo "WARNING: Missing deps in Isaac Lab Python: ${_missing[*]}"
        echo "  Mirror mode (--mirror) will NOT work without these."
        echo "  Fix: run ./setup.sh, or manually:"
        echo "    $ISAACLAB_PY -m pip install ${_missing[*]}"
        echo ""
    fi
fi

echo "=== SO-ARM101 Digital Twin ==="
echo "Isaac Lab: $ISAACLAB_DIR"
echo "Script:    $SCRIPT_DIR/digital_twin.py"
echo "Args:      $*"
echo "=============================="

cd "$ISAACLAB_DIR"
exec ./isaaclab.sh -p "$SCRIPT_DIR/digital_twin.py" "$@"
