---
sidebar_position: 4
title: "Module 1 - Week 3: ROS 2 Services and Actions"
---

# Module 1 - Week 3: ROS 2 Services and Actions

## Learning Objectives

By the end of this week, you will be able to:
- Understand the differences between topics, services, and actions in ROS 2
- Implement request/response communication using services
- Create and use action servers and clients for long-running tasks
- Design appropriate communication patterns for different robotics scenarios
- Debug service and action communication issues
- Implement custom service and action message types

## Introduction to Services and Actions

In previous weeks, we explored the publisher-subscriber pattern using topics for asynchronous communication. This week, we'll examine two other important communication patterns in ROS 2: services for synchronous request/response communication and actions for long-running, cancellable tasks with feedback.

### Topics vs Services vs Actions

| Pattern | Communication Type | Use Case | Characteristics |
|---------|-------------------|----------|-----------------|
| Topics | Asynchronous | Continuous data streams | Fire-and-forget, many-to-many |
| Services | Synchronous | Request/response | Blocking call, one-to-one |
| Actions | Asynchronous with feedback | Long-running tasks | Cancelable, status updates |

## ROS 2 Services

Services provide a request/response communication pattern. A service client sends a request to a service server, which processes the request and returns a response. Unlike topics, services are synchronous - the client waits for the response before continuing.

### Service Architecture

A service consists of:
- **Service Definition**: Defines the request and response message types
- **Service Server**: Processes requests and sends responses
- **Service Client**: Sends requests and receives responses

### Creating Custom Service Types

Service definitions are stored in `.srv` files within the `srv/` directory of a package. Here's the structure:

```
# Example: AddTwoInts.srv
int64 a
int64 b
---
int64 sum
```

The `---` separates the request fields (above) from the response fields (below).

### Service Server Implementation

Let's create a service server that calculates the distance between two points:

```python
# my_robot_services/my_robot_services/distance_service.py
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
from geometry_msgs.msg import Point
from std_srvs.srv import SetBool

class DistanceServiceNode(Node):
    def __init__(self):
        super().__init__('distance_service_node')

        # Create a service that calculates distance between two points
        self.distance_service = self.create_service(
            SetBool,  # Using SetBool as a simple example service type
            'calculate_distance',
            self.calculate_distance_callback
        )

        # Example custom service (would need custom .srv file)
        # self.custom_distance_service = self.create_service(
        #     CalculateDistance,  # Custom service type
        #     'calculate_distance_custom',
        #     self.calculate_distance_custom_callback
        # )

        self.get_logger().info('Distance Service Node initialized')

    def calculate_distance_callback(self, request, response):
        """Calculate distance between two points"""
        # For this example, we'll use the SetBool service
        # In a real implementation, you'd create a custom service
        self.get_logger().info(f'Received request: {request.data}')

        # Simulate some processing
        import time
        time.sleep(0.1)  # Simulate processing time

        response.success = True
        response.message = f'Distance calculated successfully: {request.data}'

        return response

def main(args=None):
    rclpy.init(args=args)
    node = DistanceServiceNode()

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

### Service Client Implementation

Here's how to create a service client:

```python
# my_robot_services/my_robot_services/distance_client.py
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class DistanceClientNode(Node):
    def __init__(self):
        super().__init__('distance_client_node')

        # Create a client for the distance service
        self.client = self.create_client(SetBool, 'calculate_distance')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('Distance Client Node initialized')

    def send_request(self, data):
        """Send a request to the distance service"""
        request = SetBool.Request()
        request.data = data

        # Call the service asynchronously
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        """Handle the response from the service"""
        try:
            response = future.result()
            self.get_logger().info(f'Response: success={response.success}, message={response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DistanceClientNode()

    # Send a request
    node.send_request(True)

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

### C++ Service Implementation

For performance-critical applications, you might prefer C++:

```cpp
// my_robot_services_cpp/src/distance_service.cpp
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <memory>

class DistanceService : public rclcpp::Node
{
public:
    DistanceService() : Node("distance_service_cpp")
    {
        // Create service
        service_ = this->create_service<std_srvs::srv::SetBool>(
            "calculate_distance_cpp",
            [this](
                const std::shared_ptr<rmw_request_id_t> request_header,
                const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                const std::shared_ptr<std_srvs::srv::SetBool::Response> response)
            {
                RCLCPP_INFO(this->get_logger(), "Received request: %s",
                           request->data ? "true" : "false");

                // Simulate processing
                response->success = true;
                response->message = "Distance calculated successfully";
            });

        RCLCPP_INFO(this->get_logger(), "C++ Distance Service initialized");
    }

private:
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_;
};

int main(int argc, char * argv[])
{
    rclpy::init(argc, argv);
    rclpy::spin(std::make_shared<DistanceService>());
    rclpy::shutdown();
    return 0;
}
```

## ROS 2 Actions

Actions are designed for long-running tasks that require feedback, goal cancellation, and status reporting. Unlike services, actions are asynchronous and provide ongoing communication during execution.

### Action Architecture

An action consists of:
- **Action Definition**: Defines goal, feedback, and result message types
- **Action Server**: Executes the goal and provides feedback
- **Action Client**: Sends goals and receives feedback/results

### Creating Custom Action Types

Action definitions are stored in `.action` files:

```
# Example: Fibonacci.action
int32 order
---
int32[] sequence
---
int32[] feedback
```

The `---` separates goal (top), result (middle), and feedback (bottom).

### Action Server Implementation

Here's an implementation of an action server that simulates a robot navigation task:

```python
# my_robot_actions/my_robot_actions/navigation_action.py
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Point
from robot_interfaces.action import NavigateToPose  # Custom action type
import time

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')

        # Create action server with reentrant callback group for multiple goals
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publisher for robot position (for simulation)
        self.position_publisher = self.create_publisher(Point, 'robot_position', 10)

        self.get_logger().info('Navigation Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject a goal"""
        self.get_logger().info(f'Received goal: ({goal_request.pose.position.x}, {goal_request.pose.position.y})')

        # Accept all goals for this example
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the navigation goal"""
        self.get_logger().info('Executing navigation goal...')

        # Get the target pose from the goal
        target_pose = goal_handle.request.pose
        target_x = target_pose.position.x
        target_y = target_pose.position.y

        # Simulate robot movement with feedback
        current_x = 0.0  # Starting position
        current_y = 0.0
        step_size = 0.1  # Movement step size

        feedback_msg = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        while goal_handle.is_active:
            # Calculate distance to target
            distance = ((target_x - current_x)**2 + (target_y - current_y)**2)**0.5

            # Check if goal is reached
            if distance < 0.1:  # Within 0.1 units
                goal_handle.succeed()
                result.success = True
                result.message = f'Reached target ({target_x}, {target_y})'
                self.get_logger().info(f'Goal reached: {result.message}')
                break

            # Simulate movement toward target
            direction_x = (target_x - current_x) / distance
            direction_y = (target_y - current_y) / distance

            current_x += direction_x * step_size
            current_y += direction_y * step_size

            # Publish current position
            current_pos = Point()
            current_pos.x = current_x
            current_pos.y = current_y
            current_pos.z = 0.0
            self.position_publisher.publish(current_pos)

            # Publish feedback
            feedback_msg.current_pose.position.x = current_x
            feedback_msg.current_pose.position.y = current_y
            feedback_msg.distance_remaining = distance
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f'Moving to target: ({current_x:.2f}, {current_y:.2f}), distance: {distance:.2f}')

            # Sleep to simulate processing time
            time.sleep(0.5)

            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Goal canceled'
                self.get_logger().info('Goal canceled')
                break

        return result

def main(args=None):
    rclpy.init(args=args)

    node = NavigationActionServer()

    try:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

Here's how to create an action client:

```python
# my_robot_actions/my_robot_actions/navigation_client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose
from robot_interfaces.action import NavigateToPose  # Custom action type

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')

        # Create action client
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

    def send_goal(self, target_x, target_y):
        """Send a navigation goal"""
        # Wait for the action server to be available
        self._action_client.wait_for_server()

        # Create the goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = Pose()
        goal_msg.pose.position.x = target_x
        goal_msg.pose.position.y = target_y
        goal_msg.pose.position.z = 0.0
        # Set orientation to face forward (unit quaternion)
        goal_msg.pose.orientation.w = 1.0

        # Send the goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        self.get_logger().info(f'Sent navigation goal: ({target_x}, {target_y})')

    def goal_response_callback(self, future):
        """Handle the goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during execution"""
        feedback = feedback_msg.feedback
        current_pos = feedback.current_pose.position
        remaining = feedback.distance_remaining

        self.get_logger().info(
            f'Feedback: At ({current_pos.x:.2f}, {current_pos.y:.2f}), '
            f'distance remaining: {remaining:.2f}'
        )

    def get_result_callback(self, future):
        """Handle the final result"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, {result.message}')

def main(args=None):
    rclpy.init(args=args)

    action_client = NavigationActionClient()

    # Send a navigation goal
    action_client.send_goal(5.0, 3.0)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Node interrupted by user')
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Service and Action Patterns

### Service Composition

For complex operations, you might need to compose multiple services:

```python
class ServiceCompositionNode(Node):
    def __init__(self):
        super().__init__('service_composition_node')

        # Create clients for multiple services
        self.navigation_client = self.create_client(SetBool, 'navigate_to_location')
        self.manipulation_client = self.create_client(SetBool, 'manipulate_object')
        self.perception_client = self.create_client(SetBool, 'perceive_environment')

        # Timer to trigger composite tasks
        self.timer = self.create_timer(5.0, self.execute_composite_task)

    def execute_composite_task(self):
        """Execute a task that requires multiple services"""
        # First, perceive the environment
        self.perception_client.call_async(SetBool.Request(data=True))

        # Then navigate to location
        self.navigation_client.call_async(SetBool.Request(data=True))

        # Finally, manipulate object
        self.manipulation_client.call_async(SetBool.Request(data=True))
```

### Quality of Service for Services and Actions

While services and actions don't use the same QoS settings as topics, they do have their own configuration options:

```python
from rclpy.qos import QoSProfile

# Service QoS can be configured
def create_service_with_qos(self):
    qos_profile = QoSProfile(depth=10)

    service = self.create_service(
        SetBool,
        'my_service',
        self.service_callback,
        qos_profile=qos_profile
    )
```

## Debugging Services and Actions

### Using ROS 2 Command Line Tools

```bash
# List all services
ros2 service list

# Get information about a specific service
ros2 service info <service_name>

# Call a service from command line
ros2 service call /my_service std_srvs/srv/SetBool "{data: true}"

# Find service type
ros2 service type <service_name>

# Echo service requests/responses (if available)
ros2 service echo <service_name>
```

### Debugging Actions

```bash
# List all action servers
ros2 action list

# Get information about an action
ros2 action info <action_name>

# Send a goal from command line
ros2 action send_goal <action_name> <action_type> <goal_arguments>
```

## Performance Considerations

### Service Performance

- **Synchronous Nature**: Services block the client until response, consider using actions for long-running operations
- **Timeout Handling**: Always implement proper timeout handling in clients
- **Load Balancing**: For high-frequency requests, consider multiple service servers

### Action Performance

- **Feedback Frequency**: Balance feedback frequency with performance - too frequent feedback can overwhelm the system
- **Goal Preemption**: Implement proper goal preemption for responsive systems
- **Resource Management**: Actions can run for extended periods, ensure proper resource cleanup

## Best Practices

### Service Design Principles

1. **Use Services for Simple Operations**: Services are best for quick, synchronous operations
2. **Error Handling**: Always implement proper error handling and return appropriate status codes
3. **Validation**: Validate input parameters before processing
4. **Timeouts**: Implement reasonable timeouts on both client and server sides

### Action Design Principles

1. **Use Actions for Long-Running Tasks**: Actions are ideal for operations that take time and need feedback
2. **Provide Meaningful Feedback**: Include useful information in feedback messages
3. **Implement Cancellation**: Handle cancellation requests gracefully
4. **State Management**: Maintain clear state information during execution

### Communication Pattern Selection

Choose the right communication pattern based on your use case:

- **Topics**: Continuous data streams, sensor data, status updates
- **Services**: Simple request/response, configuration changes, quick computations
- **Actions**: Long-running tasks, navigation, manipulation, complex operations

## Exercises

1. **Service Implementation**: Create a service that calculates the factorial of a number
2. **Action Implementation**: Implement an action that simulates a robot arm movement with feedback
3. **Service Composition**: Create a node that coordinates multiple services to perform a complex task
4. **Error Handling**: Add comprehensive error handling to your service implementations
5. **Performance Testing**: Test the performance of your services and actions under various load conditions

## Code Example: Advanced Service with Error Handling

```python
# advanced_service_example.py
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class AdvancedServiceNode(Node):
    def __init__(self):
        super().__init__('advanced_service_node')

        # Create callback group for service
        service_callback_group = MutuallyExclusiveCallbackGroup()

        # Create service with error handling
        self.advanced_service = self.create_service(
            Trigger,
            'advanced_operation',
            self.advanced_operation_callback,
            callback_group=service_callback_group
        )

        # Track service statistics
        self.service_call_count = 0
        self.error_count = 0

        self.get_logger().info('Advanced Service Node initialized')

    def advanced_operation_callback(self, request, response):
        """Advanced service with comprehensive error handling"""
        self.service_call_count += 1

        try:
            # Simulate some processing that might fail
            import random
            if random.random() < 0.1:  # 10% chance of simulated error
                raise Exception("Simulated processing error")

            # Perform the actual operation
            self.get_logger().info(f'Processing advanced operation, call #{self.service_call_count}')

            # Simulate processing time
            import time
            time.sleep(0.1)

            response.success = True
            response.message = f'Advanced operation completed successfully (call #{self.service_call_count})'

        except Exception as e:
            self.error_count += 1
            self.get_logger().error(f'Service error: {str(e)}')
            response.success = False
            response.message = f'Service failed: {str(e)}'

        # Log statistics periodically
        if self.service_call_count % 10 == 0:
            self.get_logger().info(
                f'Service stats: {self.service_call_count} calls, {self.error_count} errors'
            )

        return response

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedServiceNode()

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

## Ethical Considerations

When implementing services and actions for robotics applications:

- **Safety**: Ensure that service and action implementations include safety checks and validation
- **Reliability**: Design services and actions to handle failures gracefully without compromising robot safety
- **Transparency**: Provide clear feedback and status information so users understand what the robot is doing
- **Accountability**: Log service and action calls for debugging and audit purposes

## Summary

In this week, we've covered:

- The differences between topics, services, and actions in ROS 2
- How to implement service servers and clients for request/response communication
- How to create action servers and clients for long-running tasks with feedback
- Advanced patterns for service composition and error handling
- Performance considerations and best practices
- Debugging techniques for services and actions
- When to use each communication pattern appropriately

## References

1. ROS 2 Documentation - Services and Actions. (2023). Retrieved from https://docs.ros.org/en/humble/
2. ROS 2 Actions Design. (2023). Open Robotics.
3. Faconti, P., et al. (2018). ROS 2 Design Overview. Open Robotics.

---

**Next Week**: [URDF Humanoid Setup](./week-4.md)