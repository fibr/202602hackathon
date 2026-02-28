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
echo "✓ venv ready — run:  source .venv/bin/activate"
