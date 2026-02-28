#!/usr/bin/env bash
# Stop the dobot ROS2 driver container.
# Usage: ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^dobot-driver$'; then
    echo "Stopping dobot driver..."
    docker compose --profile dobot down
    echo "Stopped."
else
    echo "dobot driver is not running."
fi
