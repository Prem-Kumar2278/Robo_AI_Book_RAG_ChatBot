---
sidebar_position: 3
title: "Module 1 - Week 2: ROS 2 Nodes and Topics"
---

# Module 1 - Week 2: ROS 2 Nodes and Topics

## Learning Objectives

By the end of this week, you will be able to:
- Create and manage ROS 2 nodes using both Python and C++
- Implement publisher-subscriber communication patterns
- Understand Quality of Service (QoS) settings and their impact
- Debug and monitor node communication
- Design efficient topic-based architectures for robotics applications

## Deep Dive into ROS 2 Nodes

A node is the fundamental building block of a ROS 2 program. Nodes are designed to perform specific tasks and communicate with other nodes through topics, services, and actions.

### Node Lifecycle

The lifecycle of a ROS 2 node involves several states:
- **Unconfigured**: Node is created but not configured
- **Inactive**: Node is configured but not active
- **Active**: Node is running and processing callbacks
- **Finalized**: Node is shutting down

### Creating Nodes in Python

Let's explore more advanced node creation patterns in Python:

```python
# my_robot_nodes/my_robot_nodes/advanced_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Create a QoS profile for sensor data (best effort)
        sensor_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Create a QoS profile for control commands (reliable)
        cmd_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', cmd_qos)
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Subscribers
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, sensor_qos)
        self.cmd_subscriber = self.create_subscription(
            String, 'robot_command', self.command_callback, 10)

        # Timer for periodic tasks
        self.timer = self.create_timer(0.1, self.control_loop)

        # Node parameters
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 1.0)

        self.get_logger().info('Robot Controller Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        # Find minimum distance
        min_distance = min(msg.ranges)
        if min_distance < 1.0:  # If obstacle is closer than 1 meter
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def command_callback(self, msg):
        """Process incoming commands"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        if command == 'stop':
            self.stop_robot()
        elif command == 'forward':
            self.move_forward()

    def control_loop(self):
        """Main control loop running at 10Hz"""
        # This function runs every 0.1 seconds
        pass

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Robot stopped')

    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = self.get_parameter('linear_speed').get_parameter_value().double_value
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Moving forward')

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating Nodes in C++

For performance-critical applications, you might prefer C++:

```cpp
// my_robot_nodes_cpp/src/robot_controller.cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

class RobotController : public rclcpp::Node
{
public:
    RobotController() : Node("robot_controller_cpp")
    {
        // Create publishers
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "cmd_vel", 10);
        status_publisher_ = this->create_publisher<std_msgs::msg::String>(
            "robot_status", 10);

        // Create subscribers
        laser_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&RobotController::laser_callback, this, std::placeholders::_1));

        cmd_subscriber_ = this->create_subscription<std_msgs::msg::String>(
            "robot_command", 10,
            std::bind(&RobotController::command_callback, this, std::placeholders::_1));

        // Create timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&RobotController::control_loop, this));

        RCLCPP_INFO(this->get_logger(), "C++ Robot Controller Node initialized");
    }

private:
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Find minimum distance
        float min_distance = *std::min_element(msg->ranges.begin(), msg->ranges.end());
        if (min_distance < 1.0) {
            RCLCPP_WARN(this->get_logger(), "Obstacle detected at %.2f m", min_distance);
        }
    }

    void command_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        if (msg->data == "stop") {
            stop_robot();
        } else if (msg->data == "forward") {
            move_forward();
        }
    }

    void control_loop()
    {
        // Main control loop
    }

    void stop_robot()
    {
        auto cmd = geometry_msgs::msg::Twist();
        cmd.linear.x = 0.0;
        cmd.angular.z = 0.0;
        cmd_vel_publisher_->publish(cmd);
        RCLCPP_INFO(this->get_logger(), "Robot stopped");
    }

    void move_forward()
    {
        auto cmd = geometry_msgs::msg::Twist();
        cmd.linear.x = 0.5; // Default speed
        cmd.angular.z = 0.0;
        cmd_vel_publisher_->publish(cmd);
        RCLCPP_INFO(this->get_logger(), "Moving forward");
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_subscriber_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cmd_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclpy::init(argc, argv);
    rclpy::spin(std::make_shared<RobotController>());
    rclpy::shutdown();
    return 0;
}
```

## Topic Communication Patterns

### Publisher-Subscriber Pattern

The publisher-subscriber pattern is the most common communication pattern in ROS 2. Publishers send messages to topics, and subscribers receive messages from topics.

### Quality of Service (QoS) Settings

QoS settings allow you to configure how messages are delivered:

```python
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSLivelinessPolicy
)

# Example QoS configurations

# For real-time sensor data (best effort, keep last)
sensor_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE
)

# For critical commands (reliable, keep all)
command_qos = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_ALL,
    durability=QoSDurabilityPolicy.VOLATILE
)

# For configuration parameters (transient local)
config_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
)
```

### Message Types and Custom Messages

ROS 2 provides standard message types in packages like `std_msgs`, `sensor_msgs`, and `geometry_msgs`. You can also create custom message types.

Example of a custom message definition (`msg/RobotStatus.msg`):
```
string robot_name
float64 battery_level
bool is_moving
int32[] sensor_readings
geometry_msgs/Point position
```

## Advanced Topic Patterns

### Latched Topics

Latched topics keep the last published message and send it to new subscribers immediately:

```python
from rclpy.qos import QoSLivelinessPolicy

latched_qos = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE
)

# This publisher will send the last message to new subscribers
latched_publisher = self.create_publisher(String, 'latched_topic', latched_qos)
```

### Multiple Publishers/Single Subscriber

Multiple publishers can send to the same topic, and a single subscriber receives all messages:

```python
class MultiSourceSubscriber(Node):
    def __init__(self):
        super().__init__('multi_source_subscriber')

        # Multiple subscribers to the same message type but different topics
        self.source1_sub = self.create_subscription(
            String, 'source1', self.source1_callback, 10)
        self.source2_sub = self.create_subscription(
            String, 'source2', self.source2_callback, 10)

    def source1_callback(self, msg):
        self.get_logger().info(f'Source 1: {msg.data}')

    def source2_callback(self, msg):
        self.get_logger().info(f'Source 2: {msg.data}')
```

## Debugging and Monitoring

### Using ROS 2 Command Line Tools

```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo <topic_name> <message_type>

# Publish a message to a topic
ros2 topic pub <topic_name> <message_type> '<message_content>'

# Get information about a topic
ros2 topic info <topic_name>
```

### Using rqt Tools

```bash
# Install rqt tools
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins

# Launch rqt with various plugins
rqt
```

## Performance Considerations

### Topic Bandwidth

Consider the bandwidth implications of your topics:
- High-frequency sensor data (e.g., camera images) can consume significant bandwidth
- Use appropriate QoS settings to balance reliability and performance
- Consider data compression for large messages

### Message Size

Keep messages appropriately sized:
- Small messages: Good for high-frequency communication
- Large messages: May impact real-time performance
- Consider using message filters for large datasets

## Best Practices

### Node Design Principles

1. **Single Responsibility**: Each node should have a clear, focused purpose
2. **Loose Coupling**: Nodes should communicate through well-defined interfaces
3. **High Cohesion**: Related functionality should be grouped within a node
4. **Error Handling**: Implement proper error handling and recovery mechanisms

### Topic Naming Conventions

- Use descriptive, lowercase names with underscores
- Group related topics with common prefixes: `/robot1/sensors/laser_scan`
- Use consistent naming across your system
- Avoid generic names that could conflict with other packages

## Exercises

1. **Advanced Node Implementation**: Create a node that subscribes to multiple sensor topics and publishes fused data
2. **QoS Experimentation**: Implement the same communication with different QoS profiles and observe the differences
3. **Custom Message Creation**: Define and use a custom message type in your nodes
4. **Node Parameters**: Implement a node that uses parameters to configure its behavior
5. **Debugging Exercise**: Use ROS 2 command-line tools to debug a communication issue between nodes

## Code Example: Multi-Sensor Fusion Node

```python
# multi_sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np

class MultiSensorFusionNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Initialize sensor data storage
        self.laser_data = None
        self.imu_data = None

        # Create subscribers for different sensor types
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publisher for fused sensor data
        self.fused_pub = self.create_publisher(
            Float32MultiArray, 'fused_sensor_data', 10)

        # Timer for fusion processing
        self.timer = self.create_timer(0.05, self.fusion_callback)  # 20 Hz

        self.get_logger().info('Multi-Sensor Fusion Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = {
            'ranges': msg.ranges,
            'min_range': min(msg.ranges) if msg.ranges else float('inf'),
            'max_range': max(msg.ranges) if msg.ranges else 0.0
        }

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def fusion_callback(self):
        """Perform sensor fusion and publish results"""
        if self.laser_data and self.imu_data:
            # Simple fusion: combine sensor data into a single array
            fused_data = Float32MultiArray()

            # Add laser data (first 3 elements: min, max, count)
            laser_info = [
                self.laser_data['min_range'],
                self.laser_data['max_range'],
                len(self.laser_data['ranges'])
            ]

            # Add IMU orientation data (next 4 elements)
            imu_orientation = self.imu_data['orientation']

            # Combine all data
            fused_data.data = laser_info + imu_orientation

            # Publish the fused data
            self.fused_pub.publish(fused_data)

            self.get_logger().info(f'Published fused data: {fused_data.data}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiSensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Ethical Considerations

When designing node architectures and topic communications:

- **Data Privacy**: Consider what sensor data is being shared and with whom
- **System Security**: Implement proper access controls for critical topics
- **Fail-Safe Mechanisms**: Design systems that can safely degrade when nodes fail
- **Transparency**: Make communication patterns clear and understandable

## Summary

In this week, we've covered:
- Advanced node creation patterns in both Python and C++
- Quality of Service (QoS) settings and their applications
- Topic communication patterns and best practices
- Debugging and monitoring techniques
- Performance considerations for topic-based communication
- Multi-sensor fusion example

## References

1. ROS 2 Documentation - Nodes and Topics. (2023). Retrieved from https://docs.ros.org/en/humble/
2. DDS Specification. (2014). Object Management Group.
3. Quigley, M., et al. (2009). ROS: an open-source Robot Operating System.

---

**Next Week**: [ROS 2 Services and Actions](./week-3.md)