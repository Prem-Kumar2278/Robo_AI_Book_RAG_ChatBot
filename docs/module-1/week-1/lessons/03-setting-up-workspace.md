---
sidebar_position: 3
title: "Lesson 3: Setting Up Your First ROS 2 Workspace"
---

# Setting Up Your First ROS 2 Workspace

Let's create a basic ROS 2 workspace and run our first example.

### Creating a Workspace

```bash
# Create a workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source the ROS 2 installation
source /opt/ros/humble/setup.bash  # Adjust for your ROS 2 distribution

# Build the workspace
colcon build
```

### Sourcing the Workspace

After building, you need to source the workspace:

```bash
source ~/ros2_ws/install/setup.bash
```