---
sidebar_position: 6
title: "Module 1 - Week 5: URDF Humanoid Setup Continued"
---

# Module 1 - Week 5: URDF Humanoid Setup Continued

## Learning Objectives

By the end of this week, you will be able to:
- Implement advanced humanoid kinematic constraints and safety limits
- Integrate sensors into humanoid robot models using URDF
- Configure control systems for humanoid robots with ROS 2
- Simulate humanoid locomotion and balance in Gazebo
- Debug and optimize humanoid robot models for performance
- Prepare humanoid models for real-world deployment considerations

## Advanced Humanoid Kinematic Constraints

### Joint Limit Optimization

Proper joint limits are crucial for humanoid safety and realistic movement. Let's explore how to optimize these constraints:

```xml
<!-- Example of optimized joint limits for a humanoid shoulder -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <origin xyz="0.0 0.15 0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <!-- Optimized limits based on human anatomy -->
  <limit lower="-2.0" upper="1.5" effort="50.0" velocity="2.0"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<joint name="left_shoulder_yaw" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_lower_arm"/>
  <origin xyz="0.0 0.0 -0.3" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <!-- Conservative limits to prevent self-collision -->
  <limit lower="-0.5" upper="1.0" effort="40.0" velocity="1.5"/>
  <dynamics damping="0.3" friction="0.05"/>
</joint>
```

### Kinematic Chain Validation

Implementing kinematic chain validation to ensure proper movement:

```python
# kinematic_validator.py
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from tf2_ros import Buffer, TransformListener
import math

class KinematicValidator(Node):
    def __init__(self):
        super().__init__('kinematic_validator')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Create TF buffer for kinematic calculations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for validation checks
        self.timer = self.create_timer(0.1, self.validate_kinematics)

        # Store joint limits
        self.joint_limits = {
            'torso_to_head': (-0.5, 0.5),
            'left_shoulder_pitch': (-2.0, 1.5),
            'left_shoulder_yaw': (-0.5, 1.0),
            'left_elbow_pitch': (-1.57, 0.0),
            # Add more joint limits as needed
        }

        self.current_joints = {}

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joints[name] = msg.position[i]

    def validate_kinematics(self):
        """Validate current joint positions against limits"""
        violations = []

        for joint_name, position in self.current_joints.items():
            if joint_name in self.joint_limits:
                lower, upper = self.joint_limits[joint_name]
                if position < lower or position > upper:
                    violations.append(f'{joint_name}: {position:.3f} (limit: {lower:.3f} to {upper:.3f})')

        if violations:
            for violation in violations:
                self.get_logger().warn(f'Kinematic violation: {violation}')
        else:
            self.get_logger().debug('All joints within limits')

    def calculate_workspace(self, joint_name):
        """Calculate reachable workspace for a specific joint chain"""
        # This would implement forward kinematics to determine workspace
        # For simplicity, we'll just return a placeholder
        return {
            'reachable_volume': 0.0,
            'min_position': Point(x=0.0, y=0.0, z=0.0),
            'max_position': Point(x=1.0, y=1.0, z=1.0)
        }

def main(args=None):
    rclpy.init(args=args)
    node = KinematicValidator()

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

## Sensor Integration in Humanoid Robots

### IMU Integration

Integrating IMU sensors for balance and orientation:

```xml
<!-- IMU sensor attached to torso -->
<joint name="torso_to_imu" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
</joint>

<link name="imu_link">
  <inertial>
    <mass value="0.1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<!-- Gazebo plugin for IMU -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Camera Integration

Adding cameras for perception:

```xml
<!-- Head-mounted camera -->
<joint name="head_to_camera" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0.0 0.05" rpy="0 0 0"/>
</joint>

<link name="camera_link">
  <inertial>
    <mass value="0.05"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
  </inertial>
</link>

<!-- Gazebo camera plugin -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### Force/Torque Sensors

Adding force/torque sensors for contact detection:

```xml
<!-- Force/Torque sensor in foot -->
<joint name="right_lower_leg_to_right_foot" type="fixed">
  <parent link="right_lower_leg"/>
  <child link="right_foot"/>
  <origin xyz="0.0 0.0 -0.25" rpy="0 0 0"/>
</joint>

<link name="right_foot">
  <visual>
    <geometry>
      <box size="0.2 0.1 0.05"/>
    </geometry>
    <material name="white"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.2 0.1 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.008"/>
  </inertial>
</link>

<!-- Gazebo F/T sensor -->
<gazebo reference="right_lower_leg_to_right_foot">
  <sensor name="right_foot_ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
  </sensor>
</gazebo>
```

## Control System Configuration

### Joint Controller Setup

Setting up controllers for humanoid joints:

```yaml
# config/humanoid_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_controller:
      type: joint_state_broadcaster/JointStateBroadcaster

    torso_to_head_position_controller:
      type: position_controllers/JointPositionController

    left_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    left_leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

left_arm_controller:
  ros__parameters:
    joints:
      - torso_to_left_shoulder
      - left_shoulder_to_elbow
      - left_elbow_to_wrist
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
    state_publish_rate: 50.0
    action_monitor_rate: 20.0
    allow_partial_joints_goal: false
    constraints:
      stopped_velocity_tolerance: 0.01
      goal_time: 0.0
```

### Balance Controller Implementation

Implementing a basic balance controller:

```python
# balance_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import WrenchStamped, Vector3
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy import signal
import math

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Subscribe to sensor data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu_sensor',
            self.imu_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.balance_control_loop)  # 100Hz

        # Balance control parameters
        self.kp_balance = 50.0  # Proportional gain for balance
        self.kd_balance = 10.0  # Derivative gain for balance
        self.target_roll = 0.0
        self.target_pitch = 0.0

        # State variables
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.roll_velocity = 0.0
        self.pitch_velocity = 0.0
        self.last_time = self.get_clock().now()

        # Joint position storage
        self.joint_positions = {}
        self.joint_velocities = {}

        self.get_logger().info('Balance Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Convert quaternion to roll/pitch
        quat = msg.orientation
        self.current_roll, self.current_pitch = self.quaternion_to_rpy(
            quat.w, quat.x, quat.y, quat.z
        )[:2]

        # Calculate angular velocities
        self.roll_velocity = msg.angular_velocity.x
        self.pitch_velocity = msg.angular_velocity.y

    def joint_callback(self, msg):
        """Update joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def quaternion_to_rpy(self, w, x, y, z):
        """Convert quaternion to roll, pitch, yaw"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def balance_control_loop(self):
        """Main balance control loop"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt == 0:
            return

        # Calculate balance errors
        roll_error = self.target_roll - self.current_roll
        pitch_error = self.target_pitch - self.current_pitch

        # Calculate control outputs using PD controller
        roll_control = self.kp_balance * roll_error - self.kd_balance * self.roll_velocity
        pitch_control = self.kp_balance * pitch_error - self.kd_balance * self.pitch_velocity

        # Generate joint commands to correct balance
        joint_commands = self.calculate_balance_joint_commands(roll_control, pitch_control)

        # Publish joint commands
        cmd_msg = JointState()
        cmd_msg.header.stamp = current_time.to_msg()
        cmd_msg.name = list(joint_commands.keys())
        cmd_msg.position = list(joint_commands.values())

        self.joint_cmd_pub.publish(cmd_msg)

    def calculate_balance_joint_commands(self, roll_control, pitch_control):
        """Calculate joint commands to maintain balance"""
        # This is a simplified model - in reality, you'd use inverse kinematics
        # or a more sophisticated balance control algorithm

        commands = {}

        # Adjust leg joints to maintain balance
        if 'left_hip_pitch' in self.joint_positions:
            commands['left_hip_pitch'] = self.joint_positions['left_hip_pitch'] + pitch_control * 0.01
        if 'right_hip_pitch' in self.joint_positions:
            commands['right_hip_pitch'] = self.joint_positions['right_hip_pitch'] + pitch_control * 0.01

        # Adjust ankle joints for fine balance
        if 'left_ankle_pitch' in self.joint_positions:
            commands['left_ankle_pitch'] = self.joint_positions['left_ankle_pitch'] - pitch_control * 0.02
        if 'right_ankle_pitch' in self.joint_positions:
            commands['right_ankle_pitch'] = self.joint_positions['right_ankle_pitch'] - pitch_control * 0.02

        # Adjust arm positions for balance
        if 'left_shoulder_pitch' in self.joint_positions:
            commands['left_shoulder_pitch'] = self.joint_positions['left_shoulder_pitch'] + roll_control * 0.05
        if 'right_shoulder_pitch' in self.joint_positions:
            commands['right_shoulder_pitch'] = self.joint_positions['right_shoulder_pitch'] - roll_control * 0.05

        return commands

def main(args=None):
    rclpy.init(args=args)
    node = BalanceController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Balance controller stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Considerations

### Gazebo Configuration for Humanoids

Optimizing Gazebo for humanoid simulation:

```xml
<!-- Gazebo world configuration -->
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Your humanoid robot -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Performance Optimization

For efficient humanoid simulation:

```python
# simulation_optimizer.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class SimulationOptimizer(Node):
    def __init__(self):
        super().__init__('simulation_optimizer')

        # Publisher for simulation parameters
        self.param_pub = self.create_publisher(Int32, '/simulation_params', 10)

        # Timer for optimization checks
        self.timer = self.create_timer(1.0, self.optimize_simulation)

        self.simulation_quality = 1  # 0=low, 1=medium, 2=high

    def optimize_simulation(self):
        """Adjust simulation parameters based on performance"""
        # This would monitor simulation performance and adjust parameters
        # For example, reducing physics update rate when CPU usage is high
        pass

def main(args=None):
    rclpy.init(args=args)
    node = SimulationOptimizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Locomotion and Gait Planning

### Basic Walking Pattern

Implementing a simple walking gait:

```python
# gait_planner.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool
import math
import numpy as np

class GaitPlanner(Node):
    def __init__(self):
        super().__init__('gait_planner')

        # Publisher for walking trajectories
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )

        # Subscriber for walk commands
        self.walk_sub = self.create_subscription(
            Bool,
            '/walk_command',
            self.walk_callback,
            10
        )

        # Timer for gait generation
        self.gait_timer = self.create_timer(0.1, self.generate_gait)

        self.is_walking = False
        self.gait_phase = 0.0
        self.step_frequency = 0.5  # steps per second

    def walk_callback(self, msg):
        """Start/stop walking"""
        self.is_walking = msg.data
        if not self.is_walking:
            self.gait_phase = 0.0

    def generate_gait(self):
        """Generate walking gait trajectory"""
        if not self.is_walking:
            return

        # Update gait phase
        self.gait_phase += 2 * math.pi * self.step_frequency * 0.1
        if self.gait_phase > 2 * math.pi:
            self.gait_phase -= 2 * math.pi

        # Generate gait pattern using sine functions
        left_hip = math.sin(self.gait_phase) * 0.3
        right_hip = math.sin(self.gait_phase + math.pi) * 0.3
        left_knee = math.sin(self.gait_phase + math.pi/2) * 0.2
        right_knee = math.sin(self.gait_phase + 3*math.pi/2) * 0.2

        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [
            'left_hip_pitch', 'right_hip_pitch',
            'left_knee_pitch', 'right_knee_pitch'
        ]

        point = JointTrajectoryPoint()
        point.positions = [left_hip, right_hip, left_knee, right_knee]
        point.velocities = [0.0] * 4  # Set appropriate velocities
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds

        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GaitPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Debugging Techniques

### URDF Debugging Tools

```python
# urdf_debugger.py
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import xml.etree.ElementTree as ET

class URDFDebugger(Node):
    def __init__(self):
        super().__init__('urdf_debugger')

        # Publisher for debug markers
        self.marker_pub = self.create_publisher(MarkerArray, '/debug_markers', 10)

        # Timer for publishing debug info
        self.debug_timer = self.create_timer(1.0, self.publish_debug_info)

    def publish_debug_info(self):
        """Publish debug markers for URDF visualization"""
        marker_array = MarkerArray()

        # Example: Publish center of mass markers
        com_marker = Marker()
        com_marker.header.frame_id = "base_link"
        com_marker.header.stamp = self.get_clock().now().to_msg()
        com_marker.ns = "urdf_debug"
        com_marker.id = 1
        com_marker.type = Marker.SPHERE
        com_marker.action = Marker.ADD
        com_marker.pose.position.x = 0.0
        com_marker.pose.position.y = 0.0
        com_marker.pose.position.z = 0.8  # Approximate CoM height
        com_marker.pose.orientation.w = 1.0
        com_marker.scale.x = 0.05
        com_marker.scale.y = 0.05
        com_marker.scale.z = 0.05
        com_marker.color.a = 1.0
        com_marker.color.r = 1.0
        com_marker.color.g = 0.0
        com_marker.color.b = 0.0

        marker_array.markers.append(com_marker)
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = URDFDebugger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deployment Considerations

### Real Robot Considerations

When transitioning from simulation to real hardware:

1. **Actuator Limits**: Ensure commands don't exceed physical actuator capabilities
2. **Safety Limits**: Implement hard limits to prevent damage
3. **Calibration**: Account for manufacturing tolerances and assembly variations
4. **Sensor Noise**: Handle sensor noise and drift appropriately

### Hardware-in-the-Loop Testing

```python
# hardware_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
import threading
import time

class HardwareInterface(Node):
    def __init__(self):
        super().__init__('hardware_interface')

        # Publishers for hardware commands
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/hardware_joint_commands',
            10
        )

        # Subscribers for hardware feedback
        self.hardware_state_sub = self.create_subscription(
            JointState,
            '/hardware_joint_states',
            self.hardware_state_callback,
            10
        )

        # Timer for hardware communication
        self.hw_timer = self.create_timer(0.01, self.hardware_communication)

        # Thread for hardware communication
        self.hw_thread = threading.Thread(target=self.hardware_thread)
        self.hw_thread.daemon = True
        self.hw_thread.start()

        self.hardware_joint_positions = {}
        self.hardware_joint_velocities = {}
        self.commanded_positions = {}

    def hardware_state_callback(self, msg):
        """Update hardware state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.hardware_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.hardware_joint_velocities[name] = msg.velocity[i]

    def hardware_communication(self):
        """Main hardware communication loop"""
        # This would interface with real hardware
        # For simulation, we'll just log the state
        if self.hardware_joint_positions:
            self.get_logger().debug(f'Hardware joints: {list(self.hardware_joint_positions.keys())}')

    def hardware_thread(self):
        """Dedicated thread for hardware communication"""
        while rclpy.ok():
            # Hardware communication would happen here
            time.sleep(0.001)  # 1kHz hardware loop

def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid URDF

### Model Optimization

1. **Collision Simplification**: Use simplified geometries for collision detection
2. **Mass Distribution**: Ensure realistic mass properties for stable simulation
3. **Joint Limits**: Set appropriate limits to prevent self-collision and damage
4. **Inertia Tensors**: Calculate accurate inertia values for realistic dynamics

### Performance Tips

- Use fixed joints where possible to reduce computational overhead
- Optimize mesh resolution for visual elements
- Implement level-of-detail (LOD) systems for complex models
- Use appropriate physics parameters for your use case

## Exercises

1. **Sensor Integration**: Add IMU, camera, and force/torque sensors to your humanoid model
2. **Balance Control**: Implement a basic balance controller using IMU feedback
3. **Walking Gait**: Create a simple walking pattern for your humanoid
4. **Simulation Optimization**: Optimize your model for better simulation performance
5. **Hardware Interface**: Design a basic interface for transitioning to real hardware

## Code Example: Complete Humanoid Control Node

```python
# complete_humanoid_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import math

class CompleteHumanoidController(Node):
    def __init__(self):
        super().__init__('complete_humanoid_controller')

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu_sensor', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10)

        # Timers
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz
        self.safety_timer = self.create_timer(0.1, self.safety_check)    # 10Hz

        # State variables
        self.current_joints = {}
        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.commanded_velocity = Twist()
        self.last_control_time = self.get_clock().now()

        # Control parameters
        self.balance_enabled = True
        self.walk_enabled = False

        self.get_logger().info('Complete Humanoid Controller initialized')

    def joint_callback(self, msg):
        """Update joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joints[name] = msg.position[i]

    def imu_callback(self, msg):
        """Update IMU data"""
        # Convert quaternion to RPY
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        self.imu_data['roll'] = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        self.imu_data['pitch'] = math.asin(2*(w*y - z*x))
        self.imu_data['yaw'] = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def cmd_vel_callback(self, msg):
        """Update commanded velocity"""
        self.commanded_velocity = msg

    def control_loop(self):
        """Main control loop"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time

        if dt == 0:
            return

        commands = JointState()
        commands.header.stamp = current_time.to_msg()

        if self.balance_enabled:
            self.apply_balance_control(commands)

        if self.walk_enabled:
            self.apply_walk_pattern(commands, dt)

        # Publish joint commands
        if commands.name:  # Only publish if we have commands
            self.joint_cmd_pub.publish(commands)

    def apply_balance_control(self, commands):
        """Apply balance control to maintain upright position"""
        # Simple balance control based on pitch angle
        pitch_error = -self.imu_data['pitch']  # Negative to correct forward lean
        balance_adjustment = pitch_error * 0.5  # Gain factor

        # Adjust hip joints to maintain balance
        if 'left_hip_pitch' in self.current_joints:
            commands.name.append('left_hip_pitch')
            commands.position.append(self.current_joints['left_hip_pitch'] + balance_adjustment)
        if 'right_hip_pitch' in self.current_joints:
            commands.name.append('right_hip_pitch')
            commands.position.append(self.current_joints['right_hip_pitch'] + balance_adjustment)

        # Adjust ankle joints for fine balance
        if 'left_ankle_pitch' in self.current_joints:
            commands.name.append('left_ankle_pitch')
            commands.position.append(self.current_joints['left_ankle_pitch'] - balance_adjustment)
        if 'right_ankle_pitch' in self.current_joints:
            commands.name.append('right_ankle_pitch')
            commands.position.append(self.current_joints['right_ankle_pitch'] - balance_adjustment)

    def apply_walk_pattern(self, commands, dt):
        """Apply walking pattern based on commanded velocity"""
        # This is a simplified walking pattern
        # In reality, you'd implement a more sophisticated gait planner
        pass

    def safety_check(self):
        """Perform safety checks"""
        # Check for dangerous joint positions
        for joint_name, position in self.current_joints.items():
            # Example safety check - adjust limits as needed
            if abs(position) > 3.0:  # 3 radians is quite extreme
                self.get_logger().warn(f'Dangerous joint position: {joint_name} = {position}')

        # Check IMU for dangerous tilts
        if abs(self.imu_data['pitch']) > 1.0:  # More than ~57 degrees
            self.get_logger().error(f'Dangerous pitch angle: {self.imu_data["pitch"]}')
            # Emergency stop logic would go here

def main(args=None):
    rclpy.init(args=args)
    node = CompleteHumanoidController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Humanoid controller stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Ethical Considerations

When developing humanoid robots:

- **Safety**: Prioritize safety in all design decisions, especially joint limits and control systems
- **Privacy**: Consider data collection and storage implications of sensors
- **Accessibility**: Design for diverse users and use cases
- **Transparency**: Make robot capabilities and limitations clear to users

## Summary

In this week, we've covered:

- Advanced kinematic constraints and validation for humanoid robots
- Sensor integration including IMU, cameras, and force/torque sensors
- Control system configuration for humanoid balance and movement
- Simulation optimization techniques for humanoid models
- Locomotion and gait planning basics
- Advanced debugging techniques for complex URDF models
- Deployment considerations for transitioning to real hardware
- Best practices for humanoid robot design

## References

1. Kajita, S. (2019). Humanoid Robot Control: A Reference for Readers and Beginners. Morgan & Claypool.
2. Sardain, P., & Bessonnet, G. (2004). Forces acting on a biped robot. Center of pressure-zero moment point. IEEE Transactions on Systems, Man, and Cybernetics.
3. ROS Control Documentation. (2023). Retrieved from https://control.ros.org/

---

**Next Week**: [Gazebo Simulation Environment](../module-2/week-6.md)