---
sidebar_position: 1
title: "Lesson 1: Introduction to Robot Operating System (ROS 2)"
---

# Introduction to Robot Operating System (ROS 2)

The Robot Operating System (ROS) is not actually an operating system, but rather a flexible framework for writing robot software. ROS 2 is the next generation of ROS, designed to address the limitations of ROS 1 and to provide a more robust, secure, and scalable platform for robotics development.

### What is ROS 2?

ROS 2 is a collection of software frameworks, libraries, and tools that help developers create robot applications. It provides hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. ROS 2 is designed to be used by researchers, engineers, and developers working on robotics applications.

### Key Features of ROS 2

- **Real-time support**: ROS 2 provides real-time capabilities that were not available in ROS 1
- **Security**: Built-in security features including authentication, authorization, and encryption
- **Distributed system**: ROS 2 is designed to work across multiple machines and networks
- **Quality of Service (QoS)**: Configurable communication policies for different types of data
- **Cross-platform**: Runs on Linux, macOS, Windows, and embedded systems
- **Standard middleware**: Uses DDS (Data Distribution Service) as the default middleware

### ROS 1 vs ROS 2

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Architecture | Centralized (roscore) | Decentralized |
| Middleware | Custom | DDS-based |
| Real-time support | Limited | Full support |
| Security | None | Built-in security |
| Multi-machine | Challenging | Native support |
| Quality of Service | Basic | Advanced QoS policies |