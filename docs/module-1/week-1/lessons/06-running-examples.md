---
sidebar_position: 6
title: "Lesson 6: Running Examples"
---

# Running Examples

To run the publisher and subscriber nodes:

```bash
# Terminal 1: Run the publisher
source ~/ros2_ws/install/setup.bash
ros2 run my_first_robot publisher_node

# Terminal 2: Run the subscriber
source ~/ros2_ws/install/setup.bash
ros2 run my_first_robot subscriber_node
```