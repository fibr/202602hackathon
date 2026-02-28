FROM ros:humble

RUN apt-get update && apt-get install -y \
    git python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Build DOBOT 6Axis ROS2 V4 from source (no apt package available)
RUN mkdir -p /dobot_ws/src && \
    cd /dobot_ws/src && \
    git clone --depth 1 https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4.git && \
    cd /dobot_ws && \
    . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

WORKDIR /dobot_ws

COPY scripts/ ./scripts/
RUN sed -i 's/\r$//' scripts/*.sh && chmod +x scripts/*.sh

CMD ["bash"]
