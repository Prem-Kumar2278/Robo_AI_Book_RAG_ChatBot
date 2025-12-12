---
sidebar_position: 7
title: "Lesson 7: Quality of Service (QoS)"
---

# Quality of Service (QoS)

ROS 2 provides Quality of Service profiles that allow you to configure communication behavior:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)
```