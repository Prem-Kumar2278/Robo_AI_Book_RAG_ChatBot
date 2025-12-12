---
sidebar_position: 7
title: "Module 2 - Week 6: Gazebo Simulation Environment"
---

# Module 2 - Week 6: Gazebo Simulation Environment

## Learning Objectives

By the end of this week, you will be able to:
- Understand the Gazebo simulation environment and its architecture
- Install and configure Gazebo for ROS 2 integration
- Create custom simulation worlds with objects and environments
- Integrate robots with sensors into Gazebo simulations
- Configure physics properties and parameters for realistic simulation
- Debug and optimize simulation performance
- Implement simulation-specific controllers and behaviors

## Introduction to Gazebo

Gazebo is a 3D simulation environment that enables accurate and efficient testing of robotics applications. It provides high-fidelity physics simulation, realistic rendering, and various sensors, making it an essential tool for robotics development and testing.

### Gazebo Architecture

Gazebo consists of several key components:
- **Server (gzserver)**: Runs the physics simulation and sensor updates
- **Client (gzclient)**: Provides the graphical user interface
- **Plugins**: Extend functionality for custom behaviors and ROS 2 integration
- **Models**: 3D objects and robots with physical properties
- **Worlds**: Environments containing models, lighting, and physics properties

### Gazebo vs. Other Simulation Environments

| Feature | Gazebo | Unity | Webots | PyBullet |
|---------|--------|-------|--------|----------|
| Physics Engine | ODE, Bullet, Simbody | PhysX | Custom | Bullet |
| ROS Integration | Excellent | Good (Unity Robotics) | Good | Basic |
| Visual Quality | Good | Excellent | Good | Basic |
| Real-time Simulation | Yes | Yes | Yes | Yes |
| Open Source | Yes | No* | Yes | Yes |

*Unity has free tier with limitations

## Gazebo Installation and Setup

### Installing Gazebo Garden

For ROS 2 Humble, we recommend Gazebo Garden:

```bash
# Add Gazebo packages repository
sudo curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gz-garden

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### ROS 2 Integration

Verify Gazebo integration with ROS 2:

```bash
# Check if Gazebo is properly installed
gz version

# Launch Gazebo with ROS 2 bridge
ros2 launch gazebo_ros gazebo.launch.py
```

## Creating Custom Worlds

### World File Structure

Gazebo worlds are defined using SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models -->
    <model name="my_robot">
      <pose>0 0 0.5 0 0 0</pose>
      <include>
        <uri>model://my_robot_model</uri>
      </include>
    </model>

    <!-- Static objects -->
    <model name="table">
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Advanced World Configuration

Creating a more complex world with custom lighting and terrain:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="complex_world">
    <!-- Physics configuration -->
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

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Custom terrain -->
    <model name="terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>model://my_terrain/heightmap.png</uri>
              <size>100 100 20</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <uri>model://my_terrain/heightmap.png</uri>
              <size>100 100 20</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/GroundSand</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Objects with textures -->
    <model name="textured_box">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Blue</name>
            </script>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0.0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Robot Integration in Gazebo

### Adding Gazebo Plugins to URDF

To integrate a robot with Gazebo, add the following plugins to your URDF:

```xml
<!-- Robot URDF with Gazebo plugins -->
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include ROS 2 control -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <!-- Joint definitions would go here -->
  </ros2_control>

  <!-- Gazebo-specific configurations -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

  <!-- Gazebo ROS 2 control plugin -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Example sensor integration -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
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

</robot>
```

### Controller Configuration

Create a controller configuration file for your robot:

```yaml
# config/my_robot_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    position_controller:
      type: position_controllers/JointGroupPositionController

velocity_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
    interface_name: velocity

position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
    interface_name: position
```

## Physics Configuration

### Understanding Physics Parameters

Physics parameters significantly affect simulation realism and performance:

```xml
<physics type="ode">
  <!-- Time step for physics simulation -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor (1.0 = real-time, >1 = faster than real-time) -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Update rate in Hz -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- ODE-specific parameters -->
  <ode>
    <!-- Solver parameters -->
    <solver>
      <type>quick</type>  <!-- quick or world -->
      <iters>10</iters>   <!-- Number of iterations -->
      <sor>1.3</sor>     <!-- Successive over-relaxation parameter -->
    </solver>

    <!-- Constraint parameters -->
    <constraints>
      <cfm>0.0</cfm>  <!-- Constraint Force Mixing -->
      <erp>0.2</erp>  <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Physics Optimization for Different Scenarios

Different simulation scenarios require different physics settings:

```xml
<!-- For stable, slow simulations -->
<physics type="ode">
  <max_step_size>0.0005</max_step_size>  <!-- Smaller steps for stability -->
  <real_time_factor>0.5</real_time_factor> <!-- Slower than real-time -->
  <real_time_update_rate>2000</real_time_update_rate>
  <ode>
    <solver>
      <iters>20</iters>  <!-- More iterations for accuracy -->
      <sor>1.0</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>  <!-- Lower CFM for tighter constraints -->
      <erp>0.1</erp>   <!-- Lower ERP for less error -->
    </constraints>
  </ode>
</physics>

<!-- For faster, less accurate simulations -->
<physics type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Larger steps for speed -->
  <real_time_factor>2.0</real_time_factor> <!-- Faster than real-time -->
  <real_time_update_rate>100</real_time_update_rate>
  <ode>
    <solver>
      <iters>5</iters>  <!-- Fewer iterations -->
      <sor>1.5</sor>
    </solver>
    <constraints>
      <cfm>1e-3</cfm>  <!-- Higher CFM for more compliance -->
      <erp>0.8</erp>   <!-- Higher ERP for more forgiveness -->
    </constraints>
  </ode>
</physics>
```

## Sensor Integration

### Camera Sensors

Integrating cameras for visual perception:

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="narrow_stereo_camera">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
      <update_rate>30.0</update_rate>
      <robot_namespace>/my_robot</robot_namespace>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Sensors

Adding LiDAR for 3D mapping and navigation:

```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
          <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensors

Integrating IMU for orientation and acceleration:

```xml
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
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.017</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.017</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.017</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

## Simulation Launch and Control

### Launch Files

Creating ROS 2 launch files for simulation:

```python
# launch/gazebo_simulation.launch.py
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_my_robot_description = get_package_share_directory('my_robot_description')

    # World file
    world = LaunchConfiguration('world', default=os.path.join(
        pkg_my_robot_description, 'worlds', 'my_world.sdf'
    ))

    # Launch Gazebo server and client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': open(
                os.path.join(pkg_my_robot_description, 'urdf', 'my_robot.urdf')
            ).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

### Simulation Control Node

Creating a node to control simulation behavior:

```python
# simulation_controller.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import ModelState
import math

class SimulationController(Node):
    def __init__(self):
        super().__init__('simulation_controller')

        # Subscriptions
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

        # Publishers
        self.velocity_pub = self.create_publisher(
            Twist, '/robot_velocity', 10)
        self.sim_control_pub = self.create_publisher(
            Bool, '/simulation_control', 10)

        # Services
        self.set_state_client = self.create_client(
            SetEntityState, '/set_entity_state')
        self.get_state_client = self.create_client(
            GetEntityState, '/get_entity_state')

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # State variables
        self.current_cmd_vel = Twist()
        self.laser_data = None
        self.imu_data = None
        self.robot_position = [0.0, 0.0, 0.0]
        self.robot_orientation = [0.0, 0.0, 0.0, 1.0]

        self.get_logger().info('Simulation Controller initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.current_cmd_vel = msg

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg

    def control_loop(self):
        """Main control loop"""
        # Process sensor data
        if self.laser_data:
            # Check for obstacles
            min_distance = min(self.laser_data.ranges)
            if min_distance < 0.5:  # Less than 0.5m to obstacle
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

        # Process IMU data
        if self.imu_data:
            # Extract orientation from quaternion
            q = self.imu_data.orientation
            self.robot_orientation = [q.x, q.y, q.z, q.w]

        # Apply velocity commands
        if self.current_cmd_vel.linear.x != 0 or self.current_cmd_vel.angular.z != 0:
            self.apply_velocity_command()

    def apply_velocity_command(self):
        """Apply velocity command to robot"""
        # In simulation, we might directly set the robot's velocity
        # or publish to a velocity controller
        cmd = Twist()
        cmd.linear.x = self.current_cmd_vel.linear.x
        cmd.angular.z = self.current_cmd_vel.angular.z
        self.velocity_pub.publish(cmd)

    def get_robot_state(self):
        """Get current robot state from Gazebo"""
        if not self.get_state_client.service_is_ready():
            return None

        request = GetEntityState.Request()
        request.name = 'my_robot'
        request.reference_frame = 'world'

        future = self.get_state_client.call_async(request)
        # In a real implementation, you'd handle the future response
        return future

def main(args=None):
    rclpy.init(args=args)
    node = SimulationController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Simulation controller stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Simulation Features

### Dynamic Obstacles

Creating dynamic obstacles in simulation:

```xml
<!-- Dynamic obstacle model -->
<sdf version="1.7">
  <model name="dynamic_obstacle">
    <pose>5 0 0.5 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.3</radius>
          </sphere>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Red</name>
          </script>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <sphere>
            <radius>0.3</radius>
          </sphere>
        </geometry>
      </collision>
    </link>

    <!-- Model plugin for movement -->
    <plugin name="model_move_plugin" filename="libgazebo_ros_p3d.so">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <body_name>link</body_name>
      <topic_name>obstacle_pose</topic_name>
      <gaussian_noise>0.0</gaussian_noise>
      <frame_name>world</frame_name>
    </plugin>
  </model>
</sdf>
```

### Wind and Environmental Effects

Adding environmental effects to simulation:

```xml
<world name="wind_world">
  <!-- Physics -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
  </physics>

  <!-- Wind effect -->
  <wind>
    <linear_velocity>0.5 0 0</linear_velocity>  <!-- 0.5 m/s in X direction -->
    <force>0.1 0 0</force>  <!-- Constant force applied -->
  </wind>

  <!-- Lighting -->
  <light name="sun" type="directional">
    <cast_shadows>true</cast_shadows>
    <pose>0 0 10 0 0 0</pose>
    <diffuse>0.8 0.8 0.8 1</diffuse>
    <specular>0.2 0.2 0.2 1</specular>
    <direction>-0.3 0.0 -1.0</direction>
  </light>

  <!-- Ground plane -->
  <include>
    <uri>model://ground_plane</uri>
  </include>

  <!-- Robot -->
  <model name="my_robot">
    <pose>0 0 0.5 0 0 0</pose>
    <include>
      <uri>model://my_robot_model</uri>
    </include>
  </model>
</world>
```

## Performance Optimization

### Simulation Performance Tips

1. **Reduce Physics Update Rate**: Lower `real_time_update_rate` for less computationally intensive simulations
2. **Simplify Collision Geometry**: Use simpler shapes for collision detection
3. **Limit Sensor Update Rates**: Reduce sensor update rates where possible
4. **Use Fixed Joints**: Replace complex joints with fixed joints when appropriate
5. **Optimize Meshes**: Use lower-poly meshes for visual elements

### Multi-threaded Simulation

For complex simulations, consider multi-threaded physics:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <thread_position_correction>true</thread_position_correction>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
      <use_dynamic_moi_rescaling>false</use_dynamic_moi_rescaling>
    </solver>
  </ode>
</physics>
```

## Debugging and Troubleshooting

### Common Simulation Issues

1. **Robot Falling Through Ground**:
   - Check collision geometries
   - Verify mass and inertia values
   - Adjust physics parameters (CFM, ERP)

2. **Jittery Movement**:
   - Increase solver iterations
   - Decrease time step
   - Adjust constraint parameters

3. **High CPU Usage**:
   - Reduce physics update rate
   - Simplify collision meshes
   - Limit sensor update rates

### Debugging Tools

```python
# simulation_debugger.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import psutil
import time

class SimulationDebugger(Node):
    def __init__(self):
        super().__init__('simulation_debugger')

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Publishers
        self.debug_info_pub = self.create_publisher(
            String, '/simulation_debug_info', 10)
        self.debug_markers_pub = self.create_publisher(
            MarkerArray, '/debug_markers', 10)

        # Timer for performance monitoring
        self.debug_timer = self.create_timer(1.0, self.performance_monitor)

        self.joint_states = None
        self.last_time = time.time()

    def joint_state_callback(self, msg):
        """Monitor joint states"""
        self.joint_states = msg

    def performance_monitor(self):
        """Monitor simulation performance"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        # Get system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Create debug message
        debug_msg = f'CPU: {cpu_percent}%, Memory: {memory_percent}%, '
        if self.joint_states:
            debug_msg += f'Joints: {len(self.joint_states.name)}'

        debug_str = String()
        debug_str.data = debug_msg
        self.debug_info_pub.publish(debug_str)

        # Publish performance markers
        self.publish_performance_markers(cpu_percent, memory_percent)

    def publish_performance_markers(self, cpu, memory):
        """Publish visualization markers for performance"""
        marker_array = MarkerArray()

        # CPU usage bar
        cpu_marker = Marker()
        cpu_marker.header.frame_id = "world"
        cpu_marker.header.stamp = self.get_clock().now().to_msg()
        cpu_marker.ns = "performance"
        cpu_marker.id = 1
        cpu_marker.type = Marker.CUBE
        cpu_marker.action = Marker.ADD
        cpu_marker.pose.position.x = -5.0
        cpu_marker.pose.position.y = 0.0
        cpu_marker.pose.position.z = 1.0
        cpu_marker.pose.orientation.w = 1.0
        cpu_marker.scale.x = cpu / 10.0  # Scale based on CPU usage
        cpu_marker.scale.y = 0.2
        cpu_marker.scale.z = 0.1
        cpu_marker.color.a = 1.0
        cpu_marker.color.r = cpu / 100.0  # Red for high CPU
        cpu_marker.color.g = (100 - cpu) / 100.0  # Green for low CPU
        cpu_marker.color.b = 0.0

        marker_array.markers.append(cpu_marker)
        self.debug_markers_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationDebugger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Simulation debugger stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### Simulation Design Principles

1. **Start Simple**: Begin with basic models and gradually add complexity
2. **Validate Against Reality**: Compare simulation results with real-world data when possible
3. **Performance vs. Accuracy**: Balance simulation fidelity with computational requirements
4. **Modular Design**: Organize models and worlds in reusable, modular components
5. **Documentation**: Maintain clear documentation of simulation parameters and assumptions

### Testing Strategies

1. **Unit Testing**: Test individual components in isolation
2. **Integration Testing**: Test component interactions
3. **Regression Testing**: Ensure changes don't break existing functionality
4. **Performance Testing**: Monitor simulation performance under various conditions

## Exercises

1. **World Creation**: Create a custom world with obstacles and environmental features
2. **Robot Integration**: Integrate your humanoid robot model into Gazebo with proper plugins
3. **Sensor Integration**: Add multiple sensor types to your robot and verify data publication
4. **Physics Tuning**: Experiment with different physics parameters and observe effects
5. **Performance Optimization**: Optimize your simulation for better performance while maintaining accuracy

## Code Example: Complete Simulation Setup

```python
# complete_simulation_setup.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, Imu
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster
from gazebo_msgs.srv import SetEntityState
import math
import numpy as np

class CompleteSimulationSetup(Node):
    def __init__(self):
        super().__init__('complete_simulation_setup')

        # Subscriptions
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Services
        self.set_state_client = self.create_client(
            SetEntityState, '/set_entity_state')

        # Timer
        self.sim_timer = self.create_timer(0.01, self.simulation_loop)  # 100Hz

        # State variables
        self.cmd_vel = Twist()
        self.joint_positions = {}
        self.scan_data = None
        self.imu_data = None

        # Robot state
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion
        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]

        self.get_logger().info('Complete Simulation Setup initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.cmd_vel = msg

    def joint_state_callback(self, msg):
        """Update joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg
        # Update orientation from IMU
        self.orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

    def simulation_loop(self):
        """Main simulation loop"""
        dt = 0.01  # 100Hz

        # Update robot pose based on velocity commands
        self.update_robot_pose(dt)

        # Publish odometry
        self.publish_odometry()

        # Broadcast transforms
        self.broadcast_transforms()

        # Process sensor data
        self.process_sensor_data()

    def update_robot_pose(self, dt):
        """Update robot position based on velocity"""
        # Simple differential drive kinematics
        linear_x = self.cmd_vel.linear.x
        angular_z = self.cmd_vel.angular.z

        # Update position (simple integration)
        self.position[0] += linear_x * math.cos(self.orientation[2]) * dt
        self.position[1] += linear_x * math.sin(self.orientation[2]) * dt
        # Note: This is simplified - in real simulation, physics engine handles this

    def publish_odometry(self):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.position[0]
        odom_msg.pose.pose.position.y = self.position[1]
        odom_msg.pose.pose.position.z = self.position[2]
        odom_msg.pose.pose.orientation.x = self.orientation[0]
        odom_msg.pose.pose.orientation.y = self.orientation[1]
        odom_msg.pose.pose.orientation.z = self.orientation[2]
        odom_msg.pose.pose.orientation.w = self.orientation[3]

        # Set velocity
        odom_msg.twist.twist.linear.x = self.cmd_vel.linear.x
        odom_msg.twist.twist.angular.z = self.cmd_vel.angular.z

        self.odom_pub.publish(odom_msg)

    def broadcast_transforms(self):
        """Broadcast TF transforms"""
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.position[0]
        t.transform.translation.y = self.position[1]
        t.transform.translation.z = self.position[2]

        t.transform.rotation.x = self.orientation[0]
        t.transform.rotation.y = self.orientation[1]
        t.transform.rotation.z = self.orientation[2]
        t.transform.rotation.w = self.orientation[3]

        self.tf_broadcaster.sendTransform(t)

    def process_sensor_data(self):
        """Process and validate sensor data"""
        if self.scan_data:
            # Check for valid range data
            valid_ranges = [r for r in self.scan_data.ranges if
                           not (math.isnan(r) or math.isinf(r))]
            if valid_ranges:
                min_range = min(valid_ranges)
                if min_range < 0.3:  # Obstacle within 30cm
                    self.get_logger().warn(f'Close obstacle detected: {min_range:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    node = CompleteSimulationSetup()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Simulation setup stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Ethical Considerations

When developing simulation environments:

- **Realism vs. Safety**: Balance simulation fidelity with safety considerations
- **Data Privacy**: Be mindful of data generated and collected during simulation
- **Environmental Impact**: Consider computational resource usage for large-scale simulations
- **Accessibility**: Ensure simulation tools are accessible to diverse users

## Summary

In this week, we've covered:

- Gazebo simulation environment architecture and setup
- Creating custom simulation worlds with objects and environments
- Integrating robots and sensors into Gazebo simulations
- Physics configuration and optimization for realistic simulation
- Advanced simulation features like dynamic obstacles and environmental effects
- Performance optimization techniques
- Debugging and troubleshooting strategies
- Best practices for simulation development

## References

1. Gazebo Documentation. (2023). Retrieved from https://gazebosim.org/
2. Koenig, N., & Howard, A. (2004). Design and use paradigms for Gazebo. IEEE/RSJ International Conference on Intelligent Robots and Systems.
3. ROS 2 with Gazebo Tutorial. (2023). Open Robotics.

---

**Next Week**: [Unity Integration](./week-7.md)