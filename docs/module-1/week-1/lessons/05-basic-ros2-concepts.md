---
sidebar_position: 5
title: "Lesson 5: Basic ROS 2 Concepts"
---

# Basic ROS 2 Concepts

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