---
sidebar_position: 5
title: "Module 1 - Week 4: URDF Humanoid Setup"
---

# Module 1 - Week 4: URDF Humanoid Setup

## Learning Objectives

By the end of this week, you will be able to:
- Understand the Unified Robot Description Format (URDF) and its role in robotics
- Create complex robot models using URDF with multiple links and joints
- Design humanoid robot kinematic structures with proper joint configurations
- Implement visual and collision properties for robot models
- Simulate humanoid robots in ROS 2 with proper physical properties
- Debug and validate URDF models using ROS 2 tools

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines the robot's physical structure, including links (rigid bodies), joints (connections between links), and properties like mass, inertia, and visual/collision geometry.

### URDF Fundamentals

URDF describes a robot as a collection of rigid bodies (links) connected by joints in a tree structure. Each link can have:
- Visual properties (for display)
- Collision properties (for physics simulation)
- Inertial properties (for dynamics)
- Material properties (for rendering)

### Basic URDF Structure

A basic URDF file follows this structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Define materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <!-- Define links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Define joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.0 0.3 0.0" rpy="0.0 0.0 0.0"/>
  </joint>

  <link name="wheel_link">
    <!-- Link definition -->
  </link>
</robot>
```

## URDF Links and Geometry

### Link Definition

A link represents a rigid body with physical properties:

```xml
<link name="link_name">
  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Geometry type -->
    </geometry>
    <material name="material_name"/>
  </visual>

  <!-- Collision properties for physics -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Same or different geometry -->
    </geometry>
  </collision>

  <!-- Inertial properties for dynamics -->
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
</link>
```

### Geometry Types

URDF supports several geometry types:

```xml
<!-- Box geometry -->
<geometry>
  <box size="1.0 2.0 3.0"/>
</geometry>

<!-- Cylinder geometry -->
<geometry>
  <cylinder radius="0.5" length="1.0"/>
</geometry>

<!-- Sphere geometry -->
<geometry>
  <sphere radius="0.5"/>
</geometry>

<!-- Mesh geometry -->
<geometry>
  <mesh filename="package://my_robot/meshes/link.dae" scale="1.0 1.0 1.0"/>
</geometry>
```

## URDF Joints

### Joint Types

URDF supports several joint types:

- **fixed**: No movement, rigid connection
- **continuous**: Continuous rotation (like a wheel)
- **revolute**: Limited rotation (like an elbow)
- **prismatic**: Linear movement (like a piston)
- **floating**: 6-DOF movement (rarely used)
- **planar**: Planar movement (rarely used)

### Joint Definition

```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>

  <!-- For revolute and prismatic joints -->
  <axis xyz="1.0 0.0 0.0"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>

  <!-- For continuous joints -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Humanoid Robot Kinematics

### Humanoid Structure

A humanoid robot typically has the following structure:
- Torso/Body (base link)
- Head
- Two arms (left and right)
- Two legs (left and right)
- Various joints connecting these parts

### Example Humanoid Skeleton

```xml
<!-- Simplified humanoid structure -->
<robot name="simple_humanoid">
  <!-- Base body -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.3" rpy="0.0 0.0 0.0"/>
    <axis xyz="0.0 1.0 0.0"/>
    <limit lower="-0.5" upper="0.5" effort="1.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>
</robot>
```

## Advanced URDF Features

### Transmissions

Transmissions define how actuators connect to joints:

```xml
<transmission name="transmission_torso_to_head">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="torso_to_head">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="head_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo Plugins

Gazebo-specific properties can be added:

```xml
<gazebo reference="head">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <self_collide>false</self_collide>
</gazebo>
```

## Complete Humanoid URDF Example

Here's a more complete humanoid robot model:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.6"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.15" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="head">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0144" ixy="0.0" ixz="0.0" iyy="0.0144" iyz="0.0" izz="0.0144"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.0 0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="15.0" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_shoulder_to_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0.0 0.0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm (similar to left arm) -->
  <joint name="torso_to_right_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.0 -0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="15.0" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_shoulder_to_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0.0 0.0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="torso_to_left_hip" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.0 0.08 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_hip_to_knee" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0.0 0.0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.0" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Right Leg (similar to left leg) -->
  <joint name="torso_to_right_hip" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="0.0 -0.08 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_hip_to_knee" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0.0 0.0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.0" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Transmissions for control -->
  <transmission name="transmission_torso_to_head">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="torso_to_head">
      <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="head_servo">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Additional transmissions for arms and legs would go here -->
</robot>
```

## Xacro for Complex Models

Xacro (XML Macros) helps manage complex URDF models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="body_width" value="0.2" />
  <xacro:property name="body_depth" value="0.3" />
  <xacro:property name="body_height" value="0.6" />

  <!-- Macro for arm segments -->
  <xacro:macro name="arm_segment" params="name parent xyz length radius mass color">
    <joint name="${parent}_to_${name}" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
    </joint>

    <link name="${name}">
      <visual>
        <origin xyz="0 0 -${length/2}" rpy="0 0 0"/>
        <geometry>
          <capsule radius="${radius}" length="${length}"/>
        </geometry>
        <material name="${color}"/>
      </visual>
      <collision>
        <origin xyz="0 0 -${length/2}" rpy="0 0 0"/>
        <geometry>
          <capsule radius="${radius}" length="${length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 -${length/2}" rpy="0 0 0"/>
        <inertia ixx="${mass*radius*radius/2}" ixy="0" ixz="0"
                 iyy="${mass*(3*radius*radius + length*length)/12}" iyz="0"
                 izz="${mass*(3*radius*radius + length*length)/12}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro to create arms -->
  <xacro:arm_segment name="left_upper_arm" parent="torso"
                     xyz="0.0 0.15 0.1" length="0.3" radius="0.05" mass="2.0" color="blue"/>
  <xacro:arm_segment name="right_upper_arm" parent="torso"
                     xyz="0.0 -0.15 0.1" length="0.3" radius="0.05" mass="2.0" color="red"/>

</robot>
```

## URDF Validation and Debugging

### URDF Tools

ROS 2 provides several tools for validating and debugging URDF:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Display robot model information
urdf_to_graphiz /path/to/robot.urdf

# Parse and display the kinematic tree
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat robot.urdf)'
```

### Common URDF Issues

1. **Missing parent/child links**: Ensure all referenced links exist
2. **Inconsistent units**: Use consistent units throughout (typically meters, kilograms, radians)
3. **Invalid inertia matrices**: Ensure inertia values are physically realistic
4. **Self-collisions**: Check for unintended collisions between adjacent links

## Integration with ROS 2

### Robot State Publisher

The robot_state_publisher node publishes joint states as transforms:

```python
# robot_state_publisher_example.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer to publish state
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Initialize joint positions
        self.joint_positions = {
            'torso_to_head': 0.0,
            'torso_to_left_shoulder': 0.0,
            'left_shoulder_to_elbow': 0.0,
            # Add more joints as needed
        }

    def publish_joint_states(self):
        """Publish joint states message"""
        msg = JointState()
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()

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

## Simulation Considerations

### Physics Parameters

When creating humanoid robots for simulation, consider:

- **Mass distribution**: Realistic mass values for stable simulation
- **Inertia tensors**: Properly calculated for realistic dynamics
- **Joint limits**: Appropriate limits to prevent damage or unrealistic poses
- **Damping and friction**: Realistic values for stable simulation

### Collision Detection

For humanoid robots:
- Use simplified collision geometries for better performance
- Consider multiple collision elements per link for accuracy
- Balance detail with computational efficiency

## Best Practices

### URDF Design Principles

1. **Start Simple**: Begin with basic geometry and add complexity gradually
2. **Use Xacro**: For complex robots, use Xacro to manage repetitive elements
3. **Realistic Physics**: Use realistic mass and inertia values
4. **Modular Design**: Organize URDF in logical, reusable components
5. **Validation**: Regularly validate URDF with ROS 2 tools

### Humanoid-Specific Considerations

1. **Kinematic Chains**: Ensure proper parent-child relationships
2. **Balance**: Consider center of mass for stable locomotion
3. **Degrees of Freedom**: Provide sufficient DOF for intended tasks
4. **Safety**: Include joint limits to prevent damage

## Exercises

1. **Basic Humanoid**: Create a simple humanoid model with torso, head, and limbs
2. **Xacro Implementation**: Convert your basic model to use Xacro macros
3. **Kinematic Analysis**: Analyze the forward and inverse kinematics of your model
4. **Simulation Setup**: Load your URDF into Gazebo and test basic movement
5. **Validation**: Use ROS 2 tools to validate your URDF and fix any issues

## Code Example: URDF Validation Node

```python
# urdf_validator.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF
import os

class URDFValidator(Node):
    def __init__(self):
        super().__init__('urdf_validator')

        # Load URDF from parameter or file
        self.declare_parameter('urdf_file', '')
        urdf_file = self.get_parameter('urdf_file').get_parameter_value().string_value

        if urdf_file and os.path.exists(urdf_file):
            self.load_and_validate_urdf(urdf_file)
        else:
            self.get_logger().warn('No URDF file specified or file does not exist')

        # Publisher for validation results
        self.validation_pub = self.create_publisher(JointState, 'urdf_validation', 10)

    def load_and_validate_urdf(self, urdf_file):
        """Load and validate URDF file"""
        try:
            with open(urdf_file, 'r') as file:
                urdf_string = file.read()

            # Parse URDF
            robot = URDF.from_xml_string(urdf_string)

            self.get_logger().info(f'URDF loaded successfully: {robot.name}')
            self.get_logger().info(f'Links: {len(robot.links)}, Joints: {len(robot.joints)}')

            # Validate joint limits
            for joint in robot.joints:
                if joint.limit:
                    self.get_logger().info(
                        f'Joint {joint.name}: limits [{joint.limit.lower}, {joint.limit.upper}]'
                    )

            # Check for proper kinematic tree
            if robot.has_limits():
                self.get_logger().info('URDF has proper joint limits')
            else:
                self.get_logger().warn('URDF may have missing joint limits')

        except Exception as e:
            self.get_logger().error(f'URDF validation failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = URDFValidator()

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

When designing humanoid robots:

- **Safety**: Ensure models include appropriate safety constraints and limits
- **Accessibility**: Consider diverse body types and capabilities in design
- **Privacy**: Be mindful of sensor placement and data collection capabilities
- **Transparency**: Design robots with clear indicators of their capabilities and limitations

## Summary

In this week, we've covered:

- The fundamentals of URDF and its role in robot description
- How to create complex robot models with multiple links and joints
- Humanoid robot kinematic structures and joint configurations
- Visual and collision properties for robot models
- Advanced features like transmissions and Gazebo plugins
- Xacro for managing complex models
- Validation and debugging techniques
- Integration with ROS 2 systems

## References

1. ROS URDF Documentation. (2023). Retrieved from https://docs.ros.org/en/humble/
2. URDF/XML Format Specification. (2023). Open Robotics.
3. Corke, P. (2017). Robotics, Vision and Control: Fundamental Algorithms In MATLAB. Springer.

---

**Next Week**: [URDF Humanoid Setup Continued / Week 5](./week-5.md)