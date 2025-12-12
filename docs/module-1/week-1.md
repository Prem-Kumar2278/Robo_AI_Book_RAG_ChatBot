---
sidebar_position: 2
title: "Module 1 - Week 1: Introduction to ROS 2"
---

# Module 1 - Week 1: Introduction to ROS 2

## Learning Objectives

By the end of this week, you will be able to:
- Explain the fundamental concepts of ROS 2 and its architecture
- Understand the differences between ROS 1 and ROS 2
- Set up a basic ROS 2 workspace
- Create and run a simple ROS 2 package
- Describe the core concepts of nodes, topics, and services

## Introduction to Robot Operating System (ROS 2)

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

## ROS 2 Architecture

ROS 2 uses a decentralized architecture based on the DDS (Data Distribution Service) middleware. This eliminates the single point of failure that existed in ROS 1 (the roscore master process).

### DDS Middleware

DDS (Data Distribution Service) is a middleware standard that provides a publish-subscribe communication model. It handles the discovery, data delivery, and quality of service aspects of communication between ROS 2 nodes.

### Client Libraries

ROS 2 provides client libraries for different programming languages:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rclrs**: Rust client library (experimental)
- **rclc**: C client library (micro-ROS)

## Setting Up Your First ROS 2 Workspace

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

## Creating Your First ROS 2 Package

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

## Basic ROS 2 Concepts

### Nodes

A node is a process that performs computation. In ROS 2, nodes are written using client libraries (like rclpy for Python or rclcpp for C++). Nodes communicate with each other through topics, services, and actions.

Here's a simple node example:

```python
# my_first_robot/my_first_robot/simple_node.py
import rclpy
from rclpy.node import Node

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.get_logger().info('Simple Node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Publishers/Subscribers

Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic.

```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisher_node = PublisherNode()
    rclpy.spin(publisher_node)
    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# Subscriber example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    subscriber_node = SubscriberNode()
    rclpy.spin(subscriber_node)
    subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Services

Services provide a request/response communication pattern. A service client sends a request to a service server, which processes the request and returns a response.

## Running Examples

To run the publisher and subscriber nodes:

```bash
# Terminal 1: Run the publisher
source ~/ros2_ws/install/setup.bash
ros2 run my_first_robot publisher_node

# Terminal 2: Run the subscriber
source ~/ros2_ws/install/setup.bash
ros2 run my_first_robot subscriber_node
```

## Quality of Service (QoS)

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

## Exercises

1. **Workspace Setup**: Create your own ROS 2 workspace and build it successfully
2. **Package Creation**: Create a new package called `robot_basics` with dependencies on `rclpy` and `std_msgs`
3. **Node Implementation**: Implement a node that publishes the current time every second
4. **Topic Communication**: Create a publisher and subscriber that communicate a custom message type
5. **Service Implementation**: Create a simple service that adds two numbers

## Code Example: Simple Publisher/Subscriber

Here's a complete example that demonstrates a publisher and subscriber:

```python
# publisher_subscriber_example.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def run_publisher():
    rclpy.init()
    talker = TalkerNode()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

def run_subscriber():
    rclpy.init()
    listener = ListenerNode()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()
```

## Ethical Considerations

As we develop robotics applications, it's important to consider the ethical implications:

- **Safety**: Ensure that robots operate safely in human environments
- **Privacy**: Protect user data collected by robots
- **Transparency**: Make robot behavior understandable to users
- **Accountability**: Establish clear responsibility for robot actions

## Summary

In this week, we've covered:
- The fundamentals of ROS 2 and its advantages over ROS 1
- The architecture of ROS 2 based on DDS middleware
- How to create and build a ROS 2 workspace
- Basic ROS 2 concepts: nodes, topics, and services
- Quality of Service configurations
- Practical examples of publisher and subscriber patterns

## References

1. ROS 2 Documentation. (2023). Retrieved from https://docs.ros.org/en/humble/
2. Faconti, P., et al. (2018). ROS 2 Design Overview. Open Robotics.
3. Object Management Group. (2015). Data Distribution Service (DDS) specification.

---

**Next Week**: [ROS 2 Nodes and Topics](./week-2.md)