#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /dobot_ws/install/setup.bash

export DOBOT_TYPE=nova5
# The bringup node reads the robot IP from this env var
export IP_address="${DOBOT_IP:?Set DOBOT_IP in .env}"

ros2 launch cr_robot_ros2 dobot_bringup_ros2.launch.py
