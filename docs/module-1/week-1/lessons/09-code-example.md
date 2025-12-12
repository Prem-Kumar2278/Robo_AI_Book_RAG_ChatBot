---
sidebar_position: 9
title: "Lesson 9: Code Example: Simple Publisher/Subscriber"
---

# Code Example: Simple Publisher/Subscriber

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