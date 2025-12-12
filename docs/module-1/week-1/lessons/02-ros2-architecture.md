---
sidebar_position: 2
title: "Lesson 2: ROS 2 Architecture"
---

# ROS 2 Architecture

ROS 2 uses a decentralized architecture based on the DDS (Data Distribution Service) middleware. This eliminates the single point of failure that existed in ROS 1 (the roscore master process).

### DDS Middleware

DDS (Data Distribution Service) is a middleware standard that provides a publish-subscribe communication model. It handles the discovery, data delivery, and quality of service aspects of communication between ROS 2 nodes.

### Client Libraries

ROS 2 provides client libraries for different programming languages:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rclrs**: Rust client library (experimental)
- **rclc**: C client library (micro-ROS)