---
sidebar_position: 4
title: "Lesson 4: Creating Your First ROS 2 Package"
---

# Creating Your First ROS 2 Package

Let's create a simple package that will serve as our first ROS 2 application:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_first_robot --dependencies rclpy std_msgs
```

This creates a Python-based ROS 2 package named `my_first_robot`.

### Package Structure

After creating the package, you'll see the following structure:

```
my_first_robot/
├── my_first_robot/
│   ├── __init__.py
│   └── my_first_robot.py
├── package.xml
├── setup.cfg
├── setup.py
└── test/
    ├── __init__.py
    └── test_copyright.py
    └── test_flake8.py
    └── test_pep257.py
```