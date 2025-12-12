---
sidebar_position: 11
title: "Module 3 - Week 10: Isaac Deployment"
---

# Module 3 - Week 10: Isaac Deployment

## Learning Objectives

By the end of this week, you will be able to:
- Deploy Isaac applications to physical robots and edge devices
- Configure Isaac for real-time performance and optimization
- Implement hardware-specific optimizations for NVIDIA platforms
- Integrate Isaac with physical sensors and actuators
- Perform simulation-to-reality transfer and domain adaptation
- Monitor and debug deployed Isaac applications
- Optimize resource utilization on embedded systems
- Implement fail-safe mechanisms and error handling for deployment

## Introduction to Isaac Deployment

Isaac deployment involves transitioning from simulation to real-world applications, requiring careful consideration of hardware constraints, real-time performance requirements, and safety considerations. Unlike simulation, deployment must account for sensor noise, actuator delays, and environmental uncertainties.

### Deployment Architecture

The typical Isaac deployment architecture includes:

1. **Edge Computing Platform**: NVIDIA Jetson or similar platforms
2. **Sensor Integration**: Cameras, LiDAR, IMU, encoders
3. **Actuator Control**: Motors, servos, grippers
4. **Communication Layer**: ROS 2 nodes and Isaac ROS packages
5. **AI Inference Engine**: TensorRT for optimized neural networks
6. **Safety and Monitoring**: Error handling and system monitoring

### Hardware Platforms for Isaac Deployment

| Platform | Compute Capability | Power Consumption | Use Case |
|----------|-------------------|-------------------|----------|
| Jetson AGX Orin | 275 TOPS INT8 | 15-60W | High-performance robotics |
| Jetson Orin NX | 100 TOPS INT8 | 10-25W | Medium-performance robotics |
| Jetson AGX Xavier | 32 TOPS INT8 | 10-30W | Mobile robotics |
| Jetson Nano | 0.5 TOPS INT8 | 5-10W | Entry-level applications |

## Hardware Integration and Setup

### NVIDIA Jetson Platform Setup

Setting up NVIDIA Jetson for Isaac deployment:

```bash
# Flash Jetson with appropriate OS
# Use NVIDIA SDK Manager for initial setup

# Install Isaac ROS packages
sudo apt update
sudo apt install ros-humble-isaac-ros-common
sudo apt install ros-humble-isaac-ros-perception
sudo apt install ros-humble-isaac-ros-nav2

# Install TensorRT and CUDA
sudo apt install nvidia-jetpack

# Verify installation
nvidia-smi
nvcc --version
```

### Sensor Integration

Integrating various sensors with Isaac for deployment:

```python
# sensor_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan, Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

class IsaacSensorIntegration(Node):
    def __init__(self):
        super().__init__('isaac_sensor_integration')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Sensor data storage
        self.camera_data = {}
        self.lidar_data = None
        self.imu_data = None
        self.joint_data = None

        # Setup sensor subscriptions
        self.setup_sensor_subscriptions()

        # Setup actuator publishers
        self.setup_actuator_publishers()

        # Performance monitoring
        self.last_sensor_time = time.time()
        self.sensor_frequency = 0.0

        self.get_logger().info('Isaac sensor integration initialized')

    def setup_sensor_subscriptions(self):
        """Setup subscriptions for all sensors"""
        # RGB camera
        self.camera_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        # Depth camera
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # LiDAR
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        # IMU
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Joint states
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

    def setup_actuator_publishers(self):
        """Setup publishers for actuators"""
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

    def camera_callback(self, msg):
        """Process RGB camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process image with Isaac's GPU-accelerated pipelines
            processed_result = self.process_camera_image(cv_image)

            # Store processed result
            self.camera_data['processed'] = processed_result
            self.camera_data['timestamp'] = time.time()

            # Calculate sensor frequency
            current_time = time.time()
            self.sensor_frequency = 1.0 / (current_time - self.last_sensor_time)
            self.last_sensor_time = current_time

        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def depth_callback(self, msg):
        """Process depth camera data"""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.camera_data['depth'] = cv_depth
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.lidar_data = msg
        # Process LiDAR data for navigation and obstacle detection
        obstacles = self.detect_obstacles_from_lidar(msg)
        self.handle_obstacles(obstacles)

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        # Use IMU data for robot orientation and stabilization
        orientation = self.extract_orientation_from_imu(msg)
        self.update_robot_orientation(orientation)

    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_data = msg
        # Monitor joint positions and velocities
        self.monitor_joint_states(msg)

    def process_camera_image(self, image):
        """Process camera image using Isaac's optimized pipelines"""
        # This would use Isaac ROS packages for:
        # - Object detection
        # - Semantic segmentation
        # - Depth estimation
        # - Feature extraction

        # Example: Simple processing for demonstration
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Extract center region for focus
        center_region = image[center_y-50:center_y+50, center_x-50:center_x+50]

        # Return processed information
        return {
            'center_region': center_region,
            'image_shape': image.shape,
            'timestamp': time.time()
        }

    def detect_obstacles_from_lidar(self, scan_msg):
        """Detect obstacles from LiDAR data"""
        obstacles = []
        min_distance = float('inf')
        min_angle = 0

        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)):
                if range_val < min_distance and range_val > scan_msg.range_min:
                    min_distance = range_val
                    min_angle = scan_msg.angle_min + i * scan_msg.angle_increment

        if min_distance < 1.0:  # Obstacle within 1 meter
            obstacles.append({
                'distance': min_distance,
                'angle': min_angle,
                'timestamp': time.time()
            })

        return obstacles

    def handle_obstacles(self, obstacles):
        """Handle detected obstacles"""
        if obstacles:
            closest = min(obstacles, key=lambda x: x['distance'])
            if closest['distance'] < 0.5:  # Emergency stop
                self.emergency_stop()
            elif closest['distance'] < 1.0:  # Obstacle avoidance
                self.avoid_obstacle(closest['angle'])

    def extract_orientation_from_imu(self, imu_msg):
        """Extract orientation from IMU data"""
        # Convert quaternion to Euler angles
        w, x, y, z = (imu_msg.orientation.w, imu_msg.orientation.x,
                      imu_msg.orientation.y, imu_msg.orientation.z)

        # Convert to roll, pitch, yaw
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return {'roll': roll, 'pitch': pitch, 'yaw': yaw}

    def update_robot_orientation(self, orientation):
        """Update robot orientation based on IMU data"""
        # Use orientation for robot stabilization
        pitch_error = orientation['pitch']
        roll_error = orientation['roll']

        # Apply stabilization if needed
        if abs(pitch_error) > 0.1 or abs(roll_error) > 0.1:
            self.apply_stabilization(pitch_error, roll_error)

    def monitor_joint_states(self, joint_msg):
        """Monitor joint states for safety"""
        for i, name in enumerate(joint_msg.name):
            if i < len(joint_msg.position):
                pos = joint_msg.position[i]
                vel = joint_msg.velocity[i] if i < len(joint_msg.velocity) else 0

                # Check for dangerous conditions
                if abs(pos) > 3.0:  # Joint limit exceeded
                    self.get_logger().warn(f'Joint {name} position limit exceeded: {pos}')

                if abs(vel) > 5.0:  # Velocity limit exceeded
                    self.get_logger().warn(f'Joint {name} velocity limit exceeded: {vel}')

    def emergency_stop(self):
        """Emergency stop procedure"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().error('EMERGENCY STOP ACTIVATED')

    def avoid_obstacle(self, obstacle_angle):
        """Obstacle avoidance maneuver"""
        cmd = Twist()
        if obstacle_angle > 0:  # Obstacle on right, turn left
            cmd.angular.z = 0.5
        else:  # Obstacle on left, turn right
            cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(cmd)

    def apply_stabilization(self, pitch_error, roll_error):
        """Apply stabilization control"""
        cmd = Twist()
        cmd.angular.y = -pitch_error * 2.0  # Correct pitch
        cmd.angular.x = -roll_error * 2.0   # Correct roll
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacSensorIntegration()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Sensor integration stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Actuator Control Integration

Controlling physical actuators with Isaac:

```python
# actuator_control.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
import time

class IsaacActuatorControl(Node):
    def __init__(self):
        super().__init__('isaac_actuator_control')

        # Actuator configuration
        self.actuator_config = {
            'max_velocity': 2.0,      # rad/s
            'max_effort': 100.0,      # Nm
            'position_tolerance': 0.01,  # rad
            'velocity_tolerance': 0.1   # rad/s
        }

        # Joint control
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}

        # Setup publishers and subscribers
        self.setup_actuator_control()

        # Safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info('Isaac actuator control initialized')

    def setup_actuator_control(self):
        """Setup actuator control interfaces"""
        # Joint command publisher
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)

        # Velocity command publisher
        self.vel_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Command subscribers
        self.position_cmd_sub = self.create_subscription(
            JointState, '/position_commands', self.position_command_callback, 10)
        self.velocity_cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_command_callback, 10)

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def position_command_callback(self, msg):
        """Handle position commands"""
        # Validate commands before sending to actuators
        validated_positions = self.validate_position_commands(msg)

        if validated_positions:
            # Send validated commands to actuators
            self.send_position_commands(validated_positions)

    def velocity_command_callback(self, msg):
        """Handle velocity commands"""
        # Validate velocity commands
        validated_cmd = self.validate_velocity_command(msg)

        if validated_cmd:
            # Send validated commands to actuators
            self.send_velocity_command(validated_cmd)

    def validate_position_commands(self, joint_msg):
        """Validate position commands for safety"""
        validated_msg = JointState()
        validated_msg.header = joint_msg.header
        validated_msg.name = []
        validated_msg.position = []
        validated_msg.velocity = []
        validated_msg.effort = []

        for i, name in enumerate(joint_msg.name):
            if i < len(joint_msg.position):
                pos = joint_msg.position[i]

                # Check position limits (these would come from URDF)
                if self.check_position_limits(name, pos):
                    validated_msg.name.append(name)
                    validated_msg.position.append(pos)

                    # Add velocity and effort if provided
                    if i < len(joint_msg.velocity):
                        vel = min(joint_msg.velocity[i], self.actuator_config['max_velocity'])
                        validated_msg.velocity.append(vel)
                    else:
                        validated_msg.velocity.append(0.0)

                    if i < len(joint_msg.effort):
                        eff = min(joint_msg.effort[i], self.actuator_config['max_effort'])
                        validated_msg.effort.append(eff)
                    else:
                        validated_msg.effort.append(0.0)
                else:
                    self.get_logger().warn(f'Position command for {name} exceeds limits: {pos}')

        return validated_msg if validated_msg.name else None

    def validate_velocity_command(self, twist_msg):
        """Validate velocity commands for safety"""
        validated_cmd = Twist()

        # Limit linear and angular velocities
        validated_cmd.linear.x = max(
            -self.actuator_config['max_velocity'],
            min(self.actuator_config['max_velocity'], twist_msg.linear.x)
        )
        validated_cmd.angular.z = max(
            -self.actuator_config['max_velocity'],
            min(self.actuator_config['max_velocity'], twist_msg.angular.z)
        )

        # Add other components as needed
        validated_cmd.linear.y = max(
            -self.actuator_config['max_velocity'],
            min(self.actuator_config['max_velocity'], twist_msg.linear.y)
        )
        validated_cmd.angular.y = max(
            -self.actuator_config['max_velocity'],
            min(self.actuator_config['max_velocity'], twist_msg.angular.y)
        )
        validated_cmd.angular.x = max(
            -self.actuator_config['max_velocity'],
            min(self.actuator_config['max_velocity'], twist_msg.angular.x)
        )

        return validated_cmd

    def check_position_limits(self, joint_name, position):
        """Check if position is within joint limits"""
        # In a real implementation, this would check against URDF limits
        # For demonstration, we'll use generic limits
        joint_limits = {
            'joint1': (-3.14, 3.14),
            'joint2': (-2.0, 2.0),
            'joint3': (-3.14, 3.14),
            'joint4': (-2.0, 2.0),
            'joint5': (-3.14, 3.14),
            'joint6': (-2.0, 2.0),
            'joint7': (-3.14, 3.14)
        }

        if joint_name in joint_limits:
            min_limit, max_limit = joint_limits[joint_name]
            return min_limit <= position <= max_limit
        else:
            # Assume generic limits if not specified
            return -6.28 <= position <= 6.28

    def send_position_commands(self, joint_msg):
        """Send validated position commands to actuators"""
        # Apply smooth trajectory generation
        smooth_commands = self.generate_smooth_trajectory(joint_msg)

        # Publish commands
        self.joint_cmd_pub.publish(smooth_commands)

    def send_velocity_command(self, twist_msg):
        """Send validated velocity commands to actuators"""
        self.vel_cmd_pub.publish(twist_msg)

    def generate_smooth_trajectory(self, joint_msg):
        """Generate smooth trajectory for position commands"""
        # This would implement trajectory smoothing algorithms
        # like trapezoidal velocity profiles or cubic splines
        smoothed_msg = JointState()
        smoothed_msg.header = joint_msg.header
        smoothed_msg.name = joint_msg.name[:]
        smoothed_msg.position = joint_msg.position[:]
        smoothed_msg.velocity = joint_msg.velocity[:]
        smoothed_msg.effort = joint_msg.effort[:]

        # Add smooth velocity profiles (simplified)
        for i in range(len(smoothed_msg.velocity)):
            if smoothed_msg.velocity[i] == 0.0:
                # If no velocity specified, calculate based on position error
                joint_name = smoothed_msg.name[i]
                if joint_name in self.joint_positions:
                    pos_error = abs(smoothed_msg.position[i] - self.joint_positions[joint_name])
                    smoothed_msg.velocity[i] = min(pos_error * 2.0, self.actuator_config['max_velocity'])

        return smoothed_msg

    def safety_check(self):
        """Perform safety checks on actuator states"""
        # Check for dangerous joint states
        for name, pos in self.joint_positions.items():
            if abs(pos) > 6.0:  # Extreme position
                self.get_logger().error(f'Dangerous joint position for {name}: {pos}')
                self.emergency_stop()

        # Check for excessive velocities
        for name, vel in self.joint_velocities.items():
            if abs(vel) > self.actuator_config['max_velocity'] * 1.5:
                self.get_logger().error(f'Excessive joint velocity for {name}: {vel}')
                self.emergency_stop()

    def emergency_stop(self):
        """Emergency stop all actuators"""
        # Send zero commands to all joints
        zero_cmd = JointState()
        zero_cmd.name = list(self.joint_positions.keys())
        zero_cmd.position = [0.0] * len(zero_cmd.name)
        zero_cmd.velocity = [0.0] * len(zero_cmd.name)
        zero_cmd.effort = [0.0] * len(zero_cmd.name)

        self.joint_cmd_pub.publish(zero_cmd)

        # Stop base movement
        stop_cmd = Twist()
        self.vel_cmd_pub.publish(stop_cmd)

        self.get_logger().error('EMERGENCY STOP - All actuators stopped')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacActuatorControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Actuator control stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-Time Performance Optimization

### GPU Optimization for Edge Deployment

Optimizing Isaac applications for real-time performance on edge devices:

```python
# gpu_optimization.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import torch
import torch.nn as nn
from torch2trt import torch2trt
import numpy as np
import time
from threading import Lock

class GPUOptimizer(Node):
    def __init__(self):
        super().__init__('gpu_optimizer')

        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensorrt_enabled = False

        # Performance monitoring
        self.inference_times = []
        self.frame_processing_times = []
        self.gpu_utilization = []

        # Lock for thread safety
        self.processing_lock = Lock()

        # Setup optimization
        self.setup_gpu_optimization()

        # Performance monitoring
        self.performance_pub = self.create_publisher(Float32, '/performance_metrics', 10)
        self.performance_timer = self.create_timer(1.0, self.publish_performance_metrics)

        self.get_logger().info(f'GPU optimizer initialized on {self.device}')

    def setup_gpu_optimization(self):
        """Setup GPU optimization for Isaac deployment"""
        if self.device.type == 'cuda':
            # Set GPU memory growth (if supported)
            torch.cuda.empty_cache()

            # Optimize for inference
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Initialize optimized models
            self.initialize_optimized_models()

    def initialize_optimized_models(self):
        """Initialize optimized neural network models"""
        try:
            # Example: Initialize a detection model
            self.detection_model = self.create_detection_model()

            # Convert to TensorRT if available
            if self.is_tensorrt_available():
                self.optimize_model_with_tensorrt()
                self.tensorrt_enabled = True
                self.get_logger().info('TensorRT optimization enabled')
            else:
                self.get_logger().info('TensorRT not available, using PyTorch')

        except Exception as e:
            self.get_logger().error(f'Error initializing optimized models: {e}')

    def create_detection_model(self):
        """Create a detection model for optimization"""
        # This is a placeholder - in reality, you'd load a pre-trained model
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)  # 10 classes
        )

        model.to(self.device)
        model.eval()
        return model

    def is_tensorrt_available(self):
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False

    def optimize_model_with_tensorrt(self):
        """Optimize model with TensorRT"""
        try:
            # Create example input for TensorRT conversion
            x = torch.ones((1, 3, 224, 224)).cuda()

            # Convert model to TensorRT
            self.detection_model_trt = torch2trt(
                self.detection_model,
                [x],
                fp16_mode=True,  # Use FP16 for better performance
                max_workspace_size=1<<25  # 32MB workspace
            )

            self.get_logger().info('Model optimized with TensorRT')
        except Exception as e:
            self.get_logger().error(f'Error optimizing with TensorRT: {e}')

    def optimize_tensor_processing(self, tensor):
        """Optimize tensor processing for GPU"""
        with self.processing_lock:
            if tensor.device != self.device:
                tensor = tensor.to(self.device)

            # Ensure tensor is in the right format for GPU processing
            if tensor.dtype != torch.float16 and self.tensorrt_enabled:
                tensor = tensor.half()  # Use FP16 for TensorRT

            return tensor

    def process_frame_gpu_optimized(self, image_array):
        """Process frame with GPU optimizations"""
        start_time = time.time()

        try:
            # Convert numpy array to tensor
            tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(self.device)

            # Normalize for model input
            tensor = tensor / 255.0

            # Optimize tensor for processing
            tensor = self.optimize_tensor_processing(tensor)

            # Run inference
            with torch.no_grad():
                if self.tensorrt_enabled:
                    output = self.detection_model_trt(tensor)
                else:
                    output = self.detection_model(tensor)

            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Keep only recent measurements (last 100)
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]

            return output, inference_time

        except Exception as e:
            self.get_logger().error(f'GPU processing error: {e}')
            return None, 0.0

    def publish_performance_metrics(self):
        """Publish performance metrics"""
        if self.inference_times:
            avg_inference_time = np.mean(self.inference_times)
            fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

            # Create performance metric
            perf_msg = Float32()
            perf_msg.data = float(fps)
            self.performance_pub.publish(perf_msg)

            # Log performance
            self.get_logger().info(
                f'Performance: {fps:.2f} FPS, '
                f'Avg inference: {avg_inference_time*1000:.2f}ms'
            )

    def get_gpu_memory_info(self):
        """Get GPU memory information"""
        if self.device.type == 'cuda':
            import torch.cuda
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            memory_total = torch.cuda.get_device_properties(self.device).total_memory

            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'total': memory_total,
                'utilization': memory_allocated / memory_total if memory_total > 0 else 0
            }
        return None

class RealTimePerformanceMonitor(Node):
    def __init__(self):
        super().__init__('real_time_performance_monitor')

        # Performance metrics
        self.frame_times = []
        self.cpu_usage = []
        self.memory_usage = []

        # Setup monitoring
        self.monitoring_timer = self.create_timer(0.1, self.monitor_performance)

        # Publishers for performance data
        self.fps_pub = self.create_publisher(Float32, '/fps', 10)
        self.cpu_pub = self.create_publisher(Float32, '/cpu_usage', 10)

        # Initialize timing
        self.last_frame_time = time.time()

    def monitor_performance(self):
        """Monitor real-time performance"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Calculate FPS
        fps = 1.0 / frame_time if frame_time > 0 else 0.0

        # Store frame time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:  # Keep last 10 frames
            self.frame_times = self.frame_times[-10:]

        # Calculate average FPS
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

            # Publish FPS
            fps_msg = Float32()
            fps_msg.data = float(avg_fps)
            self.fps_pub.publish(fps_msg)

            # Monitor CPU usage (simplified)
            import psutil
            cpu_percent = psutil.cpu_percent()
            cpu_msg = Float32()
            cpu_msg.data = float(cpu_percent)
            self.cpu_pub.publish(cpu_msg)

        # Log performance if below threshold
        if fps < 10:  # Below 10 FPS
            self.get_logger().warn(f'Low FPS detected: {fps:.2f}')

def main(args=None):
    rclpy.init(args=args)

    # Create both nodes
    gpu_optimizer = GPUOptimizer()
    perf_monitor = RealTimePerformanceMonitor()

    # Use MultiThreadedExecutor for better performance
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()

    try:
        executor.add_node(gpu_optimizer)
        executor.add_node(perf_monitor)
        executor.spin()
    except KeyboardInterrupt:
        gpu_optimizer.get_logger().info('GPU optimizer stopped by user')
        perf_monitor.get_logger().info('Performance monitor stopped by user')
    finally:
        gpu_optimizer.destroy_node()
        perf_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Memory Management for Edge Devices

Managing memory efficiently on resource-constrained edge devices:

```python
# memory_management.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32
import numpy as np
import gc
import psutil
import torch
from collections import deque
import threading

class MemoryManager(Node):
    def __init__(self):
        super().__init__('memory_manager')

        # Memory configuration for edge device
        self.max_memory_usage = 0.8  # 80% of available memory
        self.memory_buffer_size = 10  # Number of frames to buffer
        self.low_memory_threshold = 0.9  # 90% memory usage triggers cleanup

        # Memory buffers
        self.image_buffer = deque(maxlen=self.memory_buffer_size)
        self.pointcloud_buffer = deque(maxlen=5)  # Point clouds are larger

        # Memory monitoring
        self.memory_usage_pub = self.create_publisher(Int32, '/memory_usage', 10)
        self.memory_monitor_timer = self.create_timer(1.0, self.monitor_memory)

        # Memory cleanup
        self.cleanup_timer = self.create_timer(5.0, self.cleanup_memory)

        # Setup memory-efficient processing
        self.setup_memory_efficient_processing()

        self.get_logger().info('Memory manager initialized')

    def setup_memory_efficient_processing(self):
        """Setup memory-efficient processing pipelines"""
        # Configure PyTorch for memory efficiency
        if torch.cuda.is_available():
            # Set memory fraction to limit GPU memory usage
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

        # Configure batch sizes for memory constraints
        self.processing_batch_size = self.calculate_optimal_batch_size()

    def calculate_optimal_batch_size(self):
        """Calculate optimal batch size based on available memory"""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = total_memory * self.max_memory_usage
                # Estimate memory needed per sample (simplified)
                memory_per_sample = 10 * 1024 * 1024  # 10MB per sample estimate
                optimal_batch_size = int(available_memory / memory_per_sample / 2)
                return max(1, min(optimal_batch_size, 8))  # Cap at reasonable size
            else:
                return 1  # Conservative for CPU processing
        except Exception:
            return 4  # Default batch size

    def monitor_memory(self):
        """Monitor system and GPU memory usage"""
        # System memory
        system_memory = psutil.virtual_memory()
        memory_percent = system_memory.percent

        # Publish memory usage
        mem_msg = Int32()
        mem_msg.data = int(memory_percent)
        self.memory_usage_pub.publish(mem_msg)

        # Check if memory usage is too high
        if memory_percent > self.low_memory_threshold * 100:
            self.get_logger().warn(f'High memory usage: {memory_percent}%')
            self.trigger_memory_cleanup()

        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory

            gpu_usage = (gpu_memory_allocated / gpu_memory_total) * 100
            if gpu_usage > self.low_memory_threshold * 100:
                self.get_logger().warn(f'High GPU memory usage: {gpu_usage:.1f}%')
                torch.cuda.empty_cache()  # Clear GPU cache

    def cleanup_memory(self):
        """Perform memory cleanup operations"""
        # Clear image buffer if too large
        if len(self.image_buffer) > self.memory_buffer_size * 0.8:
            # Keep only recent images
            recent_images = list(self.image_buffer)[-int(self.memory_buffer_size * 0.5):]
            self.image_buffer.clear()
            self.image_buffer.extend(recent_images)

        # Clear point cloud buffer
        if len(self.pointcloud_buffer) > 3:  # Keep only latest 3
            recent_clouds = list(self.pointcloud_buffer)[-3:]
            self.pointcloud_buffer.clear()
            self.pointcloud_buffer.extend(recent_clouds)

        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            self.get_logger().info(f'Garbage collected {collected} objects')

        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def trigger_memory_cleanup(self):
        """Trigger immediate memory cleanup"""
        # Clear all buffers
        self.image_buffer.clear()
        self.pointcloud_buffer.clear()

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.get_logger().info('Memory cleanup triggered')

    def process_image_memory_efficient(self, image_msg):
        """Process image with memory efficiency in mind"""
        try:
            # Convert image to numpy array (more memory efficient than ROS image)
            # This would use cv_bridge in practice
            image_array = np.frombuffer(image_msg.data, dtype=np.uint8)
            image_array = image_array.reshape((image_msg.height, image_msg.width, -1))

            # Resize if too large to save memory
            if image_array.size > 1000000:  # If image is larger than 1MB
                import cv2
                h, w = image_array.shape[:2]
                new_h, new_w = int(h * 0.5), int(w * 0.5)
                image_array = cv2.resize(image_array, (new_w, new_h))

            # Add to buffer
            self.image_buffer.append(image_array)

            # Process the image (placeholder)
            processed_result = self.process_with_memory_constraints(image_array)

            return processed_result

        except Exception as e:
            self.get_logger().error(f'Memory-efficient image processing error: {e}')
            return None

    def process_with_memory_constraints(self, data):
        """Process data while respecting memory constraints"""
        # Check current memory usage
        memory_percent = psutil.virtual_memory().percent

        if memory_percent > self.low_memory_threshold * 100:
            # Reduce processing complexity when memory is low
            return self.process_low_memory(data)
        else:
            # Normal processing
            return self.process_normal(data)

    def process_low_memory(self, data):
        """Process with reduced memory footprint"""
        # Use smaller models or reduced resolution
        # Skip expensive operations
        # Use quantized models
        return data  # Placeholder

    def process_normal(self, data):
        """Normal processing"""
        # Full processing pipeline
        return data  # Placeholder

class IsaacMemoryOptimizedNode(Node):
    def __init__(self):
        super().__init__('isaac_memory_optimized_node')

        # Initialize memory manager
        self.memory_manager = MemoryManager()

        # Setup topics with memory considerations
        self.setup_memory_efficient_topics()

        self.get_logger().info('Isaac memory-optimized node initialized')

    def setup_memory_efficient_topics(self):
        """Setup topics with memory efficiency in mind"""
        # Use lower resolution topics when possible
        # Use compressed image transport
        # Use appropriate QoS settings to reduce memory usage

        # Example: Setup with reduced frequency for memory conservation
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 5)  # Lower QoS

    def image_callback(self, msg):
        """Process image with memory management"""
        processed_data = self.memory_manager.process_image_memory_efficient(msg)
        # Continue with processing...

def main(args=None):
    rclpy.init(args=args)
    node = IsaacMemoryOptimizedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Memory-optimized node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation-to-Reality Transfer

### Domain Adaptation Techniques

Transferring models from simulation to reality requires addressing the reality gap:

```python
# domain_adaptation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time

class DomainAdaptationNode(Node):
    def __init__(self):
        super().__init__('domain_adaptation_node')

        # Initialize domain adaptation components
        self.sim_model = None
        self.adaptation_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Domain adaptation parameters
        self.adaptation_enabled = True
        self.adaptation_rate = 0.01
        self.similarity_threshold = 0.7

        # Setup for domain adaptation
        self.setup_domain_adaptation()

        # Publishers and subscribers
        self.similarity_pub = self.create_publisher(Float32, '/domain_similarity', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)

        self.get_logger().info('Domain adaptation node initialized')

    def setup_domain_adaptation(self):
        """Setup domain adaptation models"""
        # Load pre-trained simulation model
        self.load_simulation_model()

        # Initialize domain adaptation network
        self.initialize_adaptation_network()

    def load_simulation_model(self):
        """Load the pre-trained simulation model"""
        # This would load a model trained in Isaac Sim
        # For demonstration, we'll create a simple model
        self.sim_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)  # 10 output classes
        ).to(self.device)

        # Set to evaluation mode
        self.sim_model.eval()

    def initialize_adaptation_network(self):
        """Initialize domain adaptation network"""
        # Domain adaptation network to adjust for reality gap
        self.adaptation_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),  # Output same channels as input
            nn.Sigmoid()  # Normalize output
        ).to(self.device)

        # Initialize with identity mapping
        with torch.no_grad():
            self.adaptation_model[-2].weight.fill_(0.0)
            self.adaptation_model[-2].bias.fill_(0.0)

    def image_callback(self, msg):
        """Process image with domain adaptation"""
        try:
            # Convert ROS image to tensor
            image_tensor = self.ros_image_to_tensor(msg)

            # Apply domain adaptation
            adapted_image = self.apply_domain_adaptation(image_tensor)

            # Run through simulation model
            with torch.no_grad():
                output = self.sim_model(adapted_image)

            # Calculate domain similarity
            similarity = self.calculate_domain_similarity(image_tensor, adapted_image)
            self.publish_similarity(similarity)

        except Exception as e:
            self.get_logger().error(f'Domain adaptation error: {e}')

    def ros_image_to_tensor(self, msg):
        """Convert ROS image message to tensor"""
        # This would use cv_bridge in practice
        # For now, we'll simulate the conversion
        height, width = msg.height, msg.width
        image_array = np.frombuffer(msg.data, dtype=np.uint8)
        image_array = image_array.reshape((height, width, -1))

        # Convert to tensor and normalize
        tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0  # Normalize to [0, 1]
        tensor = tensor.to(self.device)

        return tensor

    def apply_domain_adaptation(self, image_tensor):
        """Apply domain adaptation to image"""
        if not self.adaptation_enabled:
            return image_tensor

        # Apply adaptation network
        adapted_tensor = self.adaptation_model(image_tensor)

        # Combine original and adapted images
        # This could be a learned combination or simple blending
        alpha = 0.5  # Adaptation strength
        result = alpha * adapted_tensor + (1 - alpha) * image_tensor

        return result

    def calculate_domain_similarity(self, original, adapted):
        """Calculate similarity between domains"""
        # Calculate similarity using various metrics
        # This could include texture, color distribution, edge features, etc.

        # Simple L2 distance between original and adapted
        diff = torch.mean((original - adapted) ** 2)
        similarity = torch.exp(-diff)  # Convert distance to similarity

        return similarity.item()

    def publish_similarity(self, similarity):
        """Publish domain similarity metric"""
        sim_msg = Float32()
        sim_msg.data = similarity
        self.similarity_pub.publish(sim_msg)

        if similarity < self.similarity_threshold:
            self.get_logger().warn(f'Low domain similarity: {similarity:.3f}')

class Sim2RealAdapter:
    def __init__(self):
        # Initialize ROS
        rclpy.init()

        # Create domain adaptation node
        self.da_node = DomainAdaptationNode()

        # Setup online adaptation
        self.setup_online_adaptation()

    def setup_online_adaptation(self):
        """Setup online domain adaptation"""
        # This would involve setting up:
        # - Online learning algorithms
        # - Continuous model updates
        # - Performance monitoring
        pass

    def run_adaptation(self):
        """Run the domain adaptation process"""
        try:
            rclpy.spin(self.da_node)
        except KeyboardInterrupt:
            self.da_node.get_logger().info('Domain adaptation stopped by user')
        finally:
            self.da_node.destroy_node()
            rclpy.shutdown()

def main():
    adapter = Sim2RealAdapter()
    adapter.run_adaptation()

if __name__ == '__main__':
    main()
```

### Reality Gap Compensation

Compensating for differences between simulation and reality:

```python
# reality_gap_compensation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import numpy as np
import time
from collections import deque

class RealityGapCompensator(Node):
    def __init__(self):
        super().__init__('reality_gap_compensator')

        # Compensation parameters
        self.scan_compensation = 1.1  # LiDAR range compensation
        self.imu_drift_compensation = 0.01  # IMU drift compensation
        self.joint_friction_compensation = 0.1  # Joint friction compensation

        # Compensation learning
        self.compensation_history = deque(maxlen=100)
        self.calibration_mode = True

        # Setup compensation system
        self.setup_compensation_system()

        # Publishers and subscribers
        self.setup_compensated_interfaces()

        self.get_logger().info('Reality gap compensator initialized')

    def setup_compensation_system(self):
        """Setup the compensation system"""
        # Initialize compensation models
        self.scan_compensator = self.initialize_scan_compensator()
        self.imu_compensator = self.initialize_imu_compensator()
        self.motion_compensator = self.initialize_motion_compensator()

    def setup_compensated_interfaces(self):
        """Setup compensated sensor and control interfaces"""
        # Subscriptions for raw sensor data
        self.raw_scan_sub = self.create_subscription(
            LaserScan, '/raw_scan', self.raw_scan_callback, 10)
        self.raw_imu_sub = self.create_subscription(
            Imu, '/raw_imu', self.raw_imu_callback, 10)
        self.raw_joint_sub = self.create_subscription(
            JointState, '/raw_joint_states', self.raw_joint_callback, 10)

        # Publishers for compensated data
        self.compensated_scan_pub = self.create_publisher(
            LaserScan, '/scan', 10)
        self.compensated_imu_pub = self.create_publisher(
            Imu, '/imu/data', 10)
        self.compensated_joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)

        # Control compensation
        self.raw_cmd_sub = self.create_subscription(
            Twist, '/raw_cmd_vel', self.raw_cmd_callback, 10)
        self.compensated_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

    def raw_scan_callback(self, msg):
        """Process raw LiDAR scan with compensation"""
        compensated_scan = self.compensate_scan_data(msg)
        self.compensated_scan_pub.publish(compensated_scan)

    def raw_imu_callback(self, msg):
        """Process raw IMU data with compensation"""
        compensated_imu = self.compensate_imu_data(msg)
        self.compensated_imu_pub.publish(compensated_imu)

    def raw_joint_callback(self, msg):
        """Process raw joint data with compensation"""
        compensated_joint = self.compensate_joint_data(msg)
        self.compensated_joint_pub.publish(compensated_joint)

    def raw_cmd_callback(self, msg):
        """Process raw commands with compensation"""
        compensated_cmd = self.compensate_command(msg)
        self.compensated_cmd_pub.publish(compensated_cmd)

    def compensate_scan_data(self, scan_msg):
        """Compensate LiDAR scan data for reality gap"""
        # Create new message with compensated data
        compensated_msg = LaserScan()
        compensated_msg.header = scan_msg.header
        compensated_msg.angle_min = scan_msg.angle_min
        compensated_msg.angle_max = scan_msg.angle_max
        compensated_msg.angle_increment = scan_msg.angle_increment
        compensated_msg.time_increment = scan_msg.time_increment
        compensated_msg.scan_time = scan_msg.scan_time
        compensated_msg.range_min = scan_msg.range_min
        compensated_msg.range_max = scan_msg.range_max

        # Apply range compensation
        compensated_ranges = []
        for range_val in scan_msg.ranges:
            if not (np.isnan(range_val) or np.isinf(range_val)):
                # Apply compensation factor
                compensated_range = range_val * self.scan_compensation
                # Add noise to simulate real sensor behavior
                noise = np.random.normal(0, 0.02)  # 2cm noise
                compensated_range += noise
                compensated_ranges.append(compensated_range)
            else:
                compensated_ranges.append(range_val)

        compensated_msg.ranges = compensated_ranges
        return compensated_msg

    def compensate_imu_data(self, imu_msg):
        """Compensate IMU data for drift and noise"""
        compensated_imu = Imu()
        compensated_imu.header = imu_msg.header

        # Apply drift compensation to orientation
        # This would involve more complex drift modeling in practice
        compensated_imu.orientation = imu_msg.orientation

        # Add realistic noise to angular velocity
        noise_std = 0.01  # 10 mrad/s noise
        compensated_imu.angular_velocity.x = (
            imu_msg.angular_velocity.x +
            np.random.normal(0, noise_std)
        )
        compensated_imu.angular_velocity.y = (
            imu_msg.angular_velocity.y +
            np.random.normal(0, noise_std)
        )
        compensated_imu.angular_velocity.z = (
            imu_msg.angular_velocity.z +
            np.random.normal(0, noise_std)
        )

        # Add realistic noise to linear acceleration
        acc_noise_std = 0.1  # 0.1 m/s² noise
        compensated_imu.linear_acceleration.x = (
            imu_msg.linear_acceleration.x +
            np.random.normal(0, acc_noise_std)
        )
        compensated_imu.linear_acceleration.y = (
            imu_msg.linear_acceleration.y +
            np.random.normal(0, acc_noise_std)
        )
        compensated_imu.linear_acceleration.z = (
            imu_msg.linear_acceleration.z +
            np.random.normal(0, acc_noise_std)
        )

        return compensated_imu

    def compensate_joint_data(self, joint_msg):
        """Compensate joint state data for friction and backlash"""
        compensated_joint = JointState()
        compensated_joint.header = joint_msg.header
        compensated_joint.name = joint_msg.name[:]
        compensated_joint.position = []
        compensated_joint.velocity = []
        compensated_joint.effort = []

        for i, name in enumerate(joint_msg.name):
            pos = joint_msg.position[i] if i < len(joint_msg.position) else 0.0
            vel = joint_msg.velocity[i] if i < len(joint_msg.velocity) else 0.0
            eff = joint_msg.effort[i] if i < len(joint_msg.effort) else 0.0

            # Apply friction compensation
            friction_compensation = self.joint_friction_compensation * np.sign(vel)
            compensated_pos = pos + friction_compensation

            # Apply velocity damping to simulate real joint behavior
            damping_factor = 0.95
            compensated_vel = vel * damping_factor

            compensated_joint.position.append(compensated_pos)
            compensated_joint.velocity.append(compensated_vel)
            compensated_joint.effort.append(eff)

        return compensated_joint

    def compensate_command(self, cmd_msg):
        """Compensate velocity commands for actuator dynamics"""
        compensated_cmd = Twist()

        # Apply actuator delay simulation
        delay = 0.05  # 50ms delay
        # In a real system, you'd implement actual delay

        # Apply actuator saturation
        max_linear_vel = 1.0  # m/s
        max_angular_vel = 1.0  # rad/s

        compensated_cmd.linear.x = np.clip(
            cmd_msg.linear.x, -max_linear_vel, max_linear_vel)
        compensated_cmd.angular.z = np.clip(
            cmd_msg.angular.z, -max_angular_vel, max_angular_vel)

        # Add actuator dynamics (first-order system)
        time_constant = 0.1  # 100ms time constant
        # This would implement actual actuator dynamics

        return compensated_cmd

    def update_compensation_parameters(self):
        """Update compensation parameters based on performance"""
        if not self.compensation_history:
            return

        # Calculate performance metrics from history
        # Update compensation parameters accordingly
        # This would involve machine learning algorithms

    def calibrate_compensation(self):
        """Calibrate compensation parameters"""
        # This would involve comparing simulation vs reality data
        # and adjusting compensation parameters
        pass

def main(args=None):
    rclpy.init(args=args)
    node = RealityGapCompensator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Reality gap compensator stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deployment Monitoring and Debugging

### System Health Monitoring

Monitoring deployed Isaac applications for performance and safety:

```python
# system_monitoring.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32
from sensor_msgs.msg import BatteryState
import psutil
import time
import subprocess
import threading
from collections import deque
import json

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        # System monitoring parameters
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        self.temperature_threshold = 70.0
        self.battery_threshold = 20.0

        # Performance history
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.temperature_history = deque(maxlen=100)

        # Setup monitoring
        self.setup_system_monitoring()

        # Publishers for system metrics
        self.cpu_pub = self.create_publisher(Float32, '/system/cpu_usage', 10)
        self.memory_pub = self.create_publisher(Float32, '/system/memory_usage', 10)
        self.temperature_pub = self.create_publisher(Float32, '/system/temperature', 10)
        self.status_pub = self.create_publisher(String, '/system/status', 10)
        self.battery_pub = self.create_publisher(BatteryState, '/system/battery', 10)

        # Timer for monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        self.get_logger().info('System monitor initialized')

    def setup_system_monitoring(self):
        """Setup system monitoring components"""
        # Initialize monitoring variables
        self.last_monitor_time = time.time()
        self.system_status = "NORMAL"

    def monitor_system(self):
        """Monitor system resources and performance"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        battery_percent = self.get_battery_level()
        temperature = self.get_system_temperature()

        # Store in history
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.temperature_history.append(temperature)

        # Publish metrics
        self.publish_metrics(cpu_percent, memory_percent, temperature, battery_percent)

        # Check for issues
        self.check_system_health(cpu_percent, memory_percent, temperature, battery_percent)

    def get_battery_level(self):
        """Get battery level (if available)"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
            else:
                return 100.0  # Assume full if no battery sensor
        except:
            return 100.0

    def get_system_temperature(self):
        """Get system temperature"""
        try:
            # Try to get temperature from system sensors
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                # Intel CPU temperature
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                # Raspberry Pi/ARM temperature
                return temps['cpu_thermal'][0].current
            else:
                # Fallback - try nvidia-smi for Jetson
                result = subprocess.run(['nvidia-smi', 'dmon', '-s', 't', '-d', '1', '-c', '1'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        temp_line = lines[1]
                        # Parse temperature from nvidia-smi output
                        parts = temp_line.split()
                        if len(parts) > 3:  # Temperature is usually in 4th column
                            return float(parts[3])
        except:
            pass
        return 30.0  # Default temperature if unable to read

    def publish_metrics(self, cpu, memory, temperature, battery):
        """Publish system metrics"""
        # CPU usage
        cpu_msg = Float32()
        cpu_msg.data = float(cpu)
        self.cpu_pub.publish(cpu_msg)

        # Memory usage
        memory_msg = Float32()
        memory_msg.data = float(memory)
        self.memory_pub.publish(memory_msg)

        # Temperature
        temp_msg = Float32()
        temp_msg.data = float(temperature)
        self.temperature_pub.publish(temp_msg)

        # Battery (if available)
        if battery >= 0:
            battery_msg = BatteryState()
            battery_msg.percentage = float(battery)
            battery_msg.voltage = 12.6  # Placeholder voltage
            battery_msg.current = 0.0   # Placeholder current
            self.battery_pub.publish(battery_msg)

    def check_system_health(self, cpu, memory, temperature, battery):
        """Check system health and update status"""
        issues = []

        if cpu > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu:.1f}%")

        if memory > self.memory_threshold:
            issues.append(f"High memory usage: {memory:.1f}%")

        if temperature > self.temperature_threshold:
            issues.append(f"High temperature: {temperature:.1f}°C")

        if battery < self.battery_threshold:
            issues.append(f"Low battery: {battery:.1f}%")

        # Update system status
        if issues:
            self.system_status = "WARNING"
            for issue in issues:
                self.get_logger().warn(f"System issue: {issue}")
        else:
            self.system_status = "NORMAL"

        # Publish status
        status_msg = String()
        status_msg.data = self.system_status
        self.status_pub.publish(status_msg)

class IsaacDeploymentMonitor:
    def __init__(self):
        # Initialize ROS
        rclpy.init()

        # Create system monitor
        self.system_monitor = SystemMonitor()

        # Create additional monitoring components
        self.setup_deployment_monitoring()

    def setup_deployment_monitoring(self):
        """Setup comprehensive deployment monitoring"""
        # This would include:
        # - Isaac-specific monitoring
        # - GPU monitoring
        # - Sensor health monitoring
        # - Actuator monitoring
        # - Network connectivity monitoring
        pass

    def run_monitoring(self):
        """Run the deployment monitoring system"""
        try:
            rclpy.spin(self.system_monitor)
        except KeyboardInterrupt:
            self.system_monitor.get_logger().info('System monitoring stopped by user')
        finally:
            self.system_monitor.destroy_node()
            rclpy.shutdown()

def main():
    monitor = IsaacDeploymentMonitor()
    monitor.run_monitoring()

if __name__ == '__main__':
    main()
```

### Error Handling and Recovery

Implementing robust error handling and recovery mechanisms:

```python
# error_handling.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import time
import traceback
from enum import Enum
from collections import deque
import threading

class SystemState(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"

class IsaacErrorHandler(Node):
    def __init__(self):
        super().__init__('isaac_error_handler')

        # System state management
        self.current_state = SystemState.NORMAL
        self.error_history = deque(maxlen=50)
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

        # Error detection and handling
        self.sensor_errors = set()
        self.actuator_errors = set()
        self.communication_errors = set()

        # Setup error handling
        self.setup_error_handling()

        # Publishers for error and diagnostic information
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.error_pub = self.create_publisher(String, '/system_errors', 10)
        self.state_pub = self.create_publisher(String, '/system_state', 10)

        # Timer for error monitoring
        self.error_monitor_timer = self.create_timer(0.5, self.monitor_errors)

        # Emergency stop publisher
        self.emergency_stop_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Isaac error handler initialized')

    def setup_error_handling(self):
        """Setup error handling components"""
        # Subscribe to various system topics to detect errors
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        """Monitor joint states for errors"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                pos = msg.position[i]
                if i < len(msg.velocity):
                    vel = msg.velocity[i]

                    # Check for dangerous joint positions
                    if abs(pos) > 5.0:  # Extreme position
                        self.log_error(f"Dangerous joint position for {name}: {pos}")
                        self.handle_joint_error(name, "EXTREME_POSITION", pos, vel)

                    # Check for excessive velocities
                    if abs(vel) > 10.0:  # Excessive velocity
                        self.log_error(f"Excessive velocity for {name}: {vel}")
                        self.handle_joint_error(name, "EXCESSIVE_VELOCITY", pos, vel)

    def handle_joint_error(self, joint_name, error_type, position, velocity):
        """Handle specific joint errors"""
        self.actuator_errors.add(joint_name)

        if error_type == "EXTREME_POSITION":
            # Stop the specific joint
            self.emergency_stop_joint(joint_name)
        elif error_type == "EXCESSIVE_VELOCITY":
            # Reduce velocity command
            self.reduce_joint_velocity(joint_name)

    def emergency_stop_joint(self, joint_name):
        """Emergency stop for a specific joint"""
        # This would send emergency stop command to specific joint
        self.get_logger().error(f"Emergency stop for joint: {joint_name}")

    def reduce_joint_velocity(self, joint_name):
        """Reduce velocity for a specific joint"""
        # This would send velocity reduction command
        self.get_logger().warn(f"Reducing velocity for joint: {joint_name}")

    def log_error(self, error_msg):
        """Log error with timestamp and context"""
        error_entry = {
            'timestamp': time.time(),
            'message': error_msg,
            'stack_trace': traceback.format_stack()
        }
        self.error_history.append(error_entry)
        self.get_logger().error(error_msg)

    def monitor_errors(self):
        """Monitor system for various types of errors"""
        # Check for different error conditions
        self.check_sensor_health()
        self.check_communication_health()
        self.check_system_resources()

        # Update system state based on error conditions
        self.update_system_state()

        # Publish diagnostic information
        self.publish_diagnostics()

    def check_sensor_health(self):
        """Check sensor health and functionality"""
        # This would check:
        # - Sensor data validity
        # - Sensor timeouts
        # - Sensor calibration status
        # - Data quality metrics

        # Example: Check if sensors are publishing data
        # This would require monitoring message rates
        pass

    def check_communication_health(self):
        """Check communication health"""
        # Check for network connectivity
        # Check for ROS communication issues
        # Check for sensor/actuator communication
        pass

    def check_system_resources(self):
        """Check system resource availability"""
        import psutil

        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        if cpu_percent > 95:
            self.log_error(f"Critical CPU usage: {cpu_percent}%")
            self.transition_to_emergency()
        elif memory_percent > 95:
            self.log_error(f"Critical memory usage: {memory_percent}%")
            self.trigger_memory_cleanup()

    def update_system_state(self):
        """Update system state based on current conditions"""
        if self.current_state == SystemState.EMERGENCY:
            # Stay in emergency until manually cleared
            return

        if self.actuator_errors or self.sensor_errors:
            if len(self.actuator_errors) + len(self.sensor_errors) > 5:
                # Too many errors, go to emergency
                self.transition_to_emergency()
            else:
                # Some errors, go to degraded
                self.transition_to_degraded()
        else:
            # No errors, maintain normal state
            if self.current_state != SystemState.NORMAL:
                self.transition_to_normal()

    def transition_to_emergency(self):
        """Transition to emergency state"""
        if self.current_state != SystemState.EMERGENCY:
            self.get_logger().error("TRANSITIONING TO EMERGENCY STATE")
            self.current_state = SystemState.EMERGENCY

            # Execute emergency procedures
            self.execute_emergency_stop()
            self.isolate_faulty_components()

            # Publish emergency state
            state_msg = String()
            state_msg.data = "EMERGENCY"
            self.state_pub.publish(state_msg)

    def transition_to_degraded(self):
        """Transition to degraded state"""
        if self.current_state != SystemState.DEGRADED:
            self.get_logger().warn("TRANSITIONING TO DEGRADED STATE")
            self.current_state = SystemState.DEGRADED

            # Publish degraded state
            state_msg = String()
            state_msg.data = "DEGRADED"
            self.state_pub.publish(state_msg)

    def transition_to_normal(self):
        """Transition to normal state"""
        if self.current_state != SystemState.NORMAL:
            self.get_logger().info("TRANSITIONING TO NORMAL STATE")
            self.current_state = SystemState.NORMAL

            # Publish normal state
            state_msg = String()
            state_msg.data = "NORMAL"
            self.state_pub.publish(state_msg)

    def execute_emergency_stop(self):
        """Execute emergency stop procedures"""
        # Stop all movement
        stop_cmd = Twist()
        self.emergency_stop_pub.publish(stop_cmd)

        # Stop all joint movements
        # This would send zero commands to all joints
        self.stop_all_joints()

    def stop_all_joints(self):
        """Stop all joint movements"""
        # Send zero position/velocity commands to all joints
        pass

    def isolate_faulty_components(self):
        """Isolate faulty system components"""
        # Disable faulty sensors
        for sensor in self.sensor_errors:
            self.disable_sensor(sensor)

        # Disable faulty actuators
        for actuator in self.actuator_errors:
            self.disable_actuator(actuator)

    def disable_sensor(self, sensor_name):
        """Disable a specific sensor"""
        self.get_logger().info(f"Disabling sensor: {sensor_name}")

    def disable_actuator(self, actuator_name):
        """Disable a specific actuator"""
        self.get_logger().info(f"Disabling actuator: {actuator_name}")

    def trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures"""
        import gc
        collected = gc.collect()
        self.get_logger().info(f"Memory cleanup collected {collected} objects")

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Create diagnostic status for overall system
        system_diag = DiagnosticStatus()
        system_diag.name = "Isaac System"
        system_diag.level = DiagnosticStatus.OK
        system_diag.message = f"System state: {self.current_state.value}"

        if self.current_state == SystemState.EMERGENCY:
            system_diag.level = DiagnosticStatus.ERROR
        elif self.current_state == SystemState.DEGRADED:
            system_diag.level = DiagnosticStatus.WARN

        # Add error count
        system_diag.values = [
            {'key': 'Error Count', 'value': str(len(self.error_history))},
            {'key': 'Actuator Errors', 'value': str(len(self.actuator_errors))},
            {'key': 'Sensor Errors', 'value': str(len(self.sensor_errors))}
        ]

        diag_array.status.append(system_diag)
        self.diagnostic_pub.publish(diag_array)

class IsaacRecoveryManager:
    def __init__(self):
        # Initialize ROS
        rclpy.init()

        # Create error handler
        self.error_handler = IsaacErrorHandler()

        # Setup recovery procedures
        self.setup_recovery_procedures()

    def setup_recovery_procedures(self):
        """Setup automated recovery procedures"""
        # This would include:
        # - Sensor recalibration procedures
        # - Actuator reinitialization
        # - System restart procedures
        # - Fallback behavior activation
        pass

    def run_error_handling(self):
        """Run the error handling system"""
        try:
            rclpy.spin(self.error_handler)
        except KeyboardInterrupt:
            self.error_handler.get_logger().info('Error handling stopped by user')
        finally:
            self.error_handler.destroy_node()
            rclpy.shutdown()

def main():
    recovery_manager = IsaacRecoveryManager()
    recovery_manager.run_error_handling()

if __name__ == '__main__':
    main()
```

## Deployment Best Practices

### Configuration Management

Managing configurations for different deployment environments:

```python
# configuration_manager.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import yaml
import os
from pathlib import Path
import hashlib

class IsaacConfigurationManager(Node):
    def __init__(self):
        super().__init__('isaac_config_manager')

        # Configuration paths
        self.config_dir = Path.home() / '.isaac_configs'
        self.current_config = {}
        self.config_history = []

        # Setup configuration management
        self.setup_configuration_system()

        # Publishers for configuration status
        self.config_status_pub = self.create_publisher(String, '/config_status', 10)

        self.get_logger().info('Isaac configuration manager initialized')

    def setup_configuration_system(self):
        """Setup the configuration management system"""
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)

        # Load default configuration
        self.load_default_configuration()

        # Load saved configuration
        self.load_saved_configuration()

    def load_default_configuration(self):
        """Load default Isaac configuration"""
        default_config = {
            'hardware': {
                'platform': 'jetson-agx-orin',
                'gpu_enabled': True,
                'memory_limit': '80%',
                'cpu_affinity': [0, 1, 2, 3]
            },
            'sensors': {
                'camera': {
                    'enabled': True,
                    'resolution': [640, 480],
                    'framerate': 30,
                    'format': 'bgr8'
                },
                'lidar': {
                    'enabled': True,
                    'range_min': 0.1,
                    'range_max': 25.0,
                    'resolution': 0.1
                },
                'imu': {
                    'enabled': True,
                    'rate': 100,
                    'calibration_required': True
                }
            },
            'performance': {
                'target_fps': 30,
                'max_latency': 0.1,
                'quality_settings': 'high'
            },
            'safety': {
                'max_velocity': 1.0,
                'max_acceleration': 2.0,
                'safety_margin': 0.5,
                'emergency_stop_enabled': True
            }
        }

        self.current_config = default_config

    def load_saved_configuration(self):
        """Load saved configuration from file"""
        config_file = self.config_dir / 'current_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.current_config.update(saved_config)
                    self.get_logger().info('Loaded saved configuration')
            except Exception as e:
                self.get_logger().error(f'Error loading saved config: {e}')

    def save_configuration(self, config_name=None):
        """Save current configuration"""
        if config_name is None:
            config_name = f"config_{int(time.time())}"

        # Create config history entry
        config_hash = hashlib.md5(json.dumps(self.current_config, sort_keys=True).encode()).hexdigest()
        config_entry = {
            'name': config_name,
            'timestamp': time.time(),
            'hash': config_hash,
            'config': self.current_config.copy()
        }

        self.config_history.append(config_entry)

        # Save to file
        config_file = self.config_dir / f'{config_name}.json'
        with open(config_file, 'w') as f:
            json.dump(self.current_config, f, indent=2)

        # Update current config file
        current_config_file = self.config_dir / 'current_config.json'
        with open(current_config_file, 'w') as f:
            json.dump(self.current_config, f, indent=2)

        self.get_logger().info(f'Configuration saved as {config_name}')

    def load_configuration(self, config_name):
        """Load a specific configuration"""
        config_file = self.config_dir / f'{config_name}.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    new_config = json.load(f)
                    self.current_config = new_config
                    self.save_configuration('current_config')
                    self.get_logger().info(f'Configuration {config_name} loaded')
                    return True
            except Exception as e:
                self.get_logger().error(f'Error loading config {config_name}: {e}')
        return False

    def validate_configuration(self, config=None):
        """Validate configuration parameters"""
        if config is None:
            config = self.current_config

        errors = []

        # Validate hardware settings
        if 'hardware' in config:
            hw = config['hardware']
            if hw.get('platform') not in ['jetson-agx-orin', 'jetson-orin-nx', 'jetson-agx-xavier', 'jetson-nano']:
                errors.append(f"Invalid platform: {hw.get('platform')}")

        # Validate sensor settings
        if 'sensors' in config:
            sensors = config['sensors']
            if 'camera' in sensors:
                cam = sensors['camera']
                if cam.get('framerate', 0) <= 0 or cam.get('framerate', 0) > 120:
                    errors.append("Invalid camera framerate")
                if cam.get('resolution'):
                    res = cam['resolution']
                    if len(res) != 2 or res[0] <= 0 or res[1] <= 0:
                        errors.append("Invalid camera resolution")

        # Validate performance settings
        if 'performance' in config:
            perf = config['performance']
            if perf.get('target_fps', 0) <= 0 or perf.get('target_fps', 0) > 120:
                errors.append("Invalid target FPS")

        # Validate safety settings
        if 'safety' in config:
            safety = config['safety']
            if safety.get('max_velocity', 0) <= 0:
                errors.append("Invalid max velocity")
            if safety.get('max_acceleration', 0) <= 0:
                errors.append("Invalid max acceleration")

        return len(errors) == 0, errors

    def apply_configuration(self):
        """Apply current configuration to system"""
        is_valid, errors = self.validate_configuration()
        if not is_valid:
            self.get_logger().error(f'Configuration validation failed: {errors}')
            return False

        # Apply configuration to system components
        self.apply_hardware_configuration()
        self.apply_sensor_configuration()
        self.apply_performance_configuration()
        self.apply_safety_configuration()

        # Save configuration
        self.save_configuration()

        self.get_logger().info('Configuration applied successfully')
        return True

    def apply_hardware_configuration(self):
        """Apply hardware-specific configuration"""
        hw_config = self.current_config.get('hardware', {})

        # Apply CPU affinity if specified
        cpu_affinity = hw_config.get('cpu_affinity', [])
        if cpu_affinity:
            # This would set CPU affinity for the process
            pass

        # Configure GPU settings
        if hw_config.get('gpu_enabled', True):
            # Initialize GPU components
            pass

    def apply_sensor_configuration(self):
        """Apply sensor configuration"""
        sensor_config = self.current_config.get('sensors', {})

        # Configure camera
        if 'camera' in sensor_config:
            cam_config = sensor_config['camera']
            if cam_config.get('enabled', True):
                # Apply camera settings
                pass

        # Configure LiDAR
        if 'lidar' in sensor_config:
            lidar_config = sensor_config['lidar']
            if lidar_config.get('enabled', True):
                # Apply LiDAR settings
                pass

        # Configure IMU
        if 'imu' in sensor_config:
            imu_config = sensor_config['imu']
            if imu_config.get('enabled', True):
                # Apply IMU settings
                pass

    def apply_performance_configuration(self):
        """Apply performance configuration"""
        perf_config = self.current_config.get('performance', {})

        # Apply FPS settings
        target_fps = perf_config.get('target_fps', 30)
        # This would configure processing rates accordingly

        # Apply latency settings
        max_latency = perf_config.get('max_latency', 0.1)
        # This would configure buffer sizes and processing pipelines

    def apply_safety_configuration(self):
        """Apply safety configuration"""
        safety_config = self.current_config.get('safety', {})

        # Apply velocity limits
        max_vel = safety_config.get('max_velocity', 1.0)
        # This would configure motion limits

        # Apply acceleration limits
        max_acc = safety_config.get('max_acceleration', 2.0)
        # This would configure motion profiles

        # Configure emergency stop
        if safety_config.get('emergency_stop_enabled', True):
            # Enable emergency stop functionality
            pass

    def get_configuration_summary(self):
        """Get summary of current configuration"""
        summary = {
            'hardware_platform': self.current_config.get('hardware', {}).get('platform', 'unknown'),
            'sensors_enabled': {
                'camera': self.current_config.get('sensors', {}).get('camera', {}).get('enabled', False),
                'lidar': self.current_config.get('sensors', {}).get('lidar', {}).get('enabled', False),
                'imu': self.current_config.get('sensors', {}).get('imu', {}).get('enabled', False)
            },
            'target_fps': self.current_config.get('performance', {}).get('target_fps', 30),
            'max_velocity': self.current_config.get('safety', {}).get('max_velocity', 1.0)
        }
        return summary

def main(args=None):
    rclpy.init(args=args)
    node = IsaacConfigurationManager()

    try:
        # Example: Apply current configuration
        node.apply_configuration()

        # Print configuration summary
        summary = node.get_configuration_summary()
        node.get_logger().info(f'Configuration summary: {summary}')

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Configuration manager stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. **Hardware Integration**: Integrate Isaac with physical sensors and actuators on an NVIDIA Jetson platform
2. **Performance Optimization**: Optimize an Isaac application for real-time performance on edge hardware
3. **Memory Management**: Implement memory-efficient processing for resource-constrained devices
4. **Domain Adaptation**: Apply domain adaptation techniques to transfer a model from simulation to reality
5. **Reality Gap Compensation**: Implement compensation algorithms for sensor and actuator differences
6. **System Monitoring**: Create a comprehensive monitoring system for deployed Isaac applications
7. **Error Handling**: Implement robust error detection and recovery mechanisms
8. **Configuration Management**: Create a configuration management system for different deployment scenarios

## Code Example: Complete Isaac Deployment System

```python
# complete_isaac_deployment.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from diagnostic_msgs.msg import DiagnosticArray
import numpy as np
import torch
import time
import psutil
from threading import Lock
import gc

class CompleteIsaacDeployment(Node):
    def __init__(self):
        super().__init__('complete_isaac_deployment')

        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_lock = Lock()

        # System state
        self.system_state = 'INITIALIZING'
        self.emergency_stop = False
        self.performance_metrics = {
            'fps': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0
        }

        # Initialize all systems
        self.initialize_sensors()
        self.initialize_actuators()
        self.initialize_ai_models()
        self.initialize_monitoring()

        # Setup ROS interfaces
        self.setup_ros_interfaces()

        # Timers
        self.performance_timer = self.create_timer(1.0, self.update_performance_metrics)
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info('Complete Isaac deployment system initialized')

    def initialize_sensors(self):
        """Initialize all sensor systems"""
        self.sensor_data = {
            'camera': None,
            'lidar': None,
            'imu': None,
            'joints': None
        }
        self.get_logger().info('Sensors initialized')

    def initialize_actuators(self):
        """Initialize all actuator systems"""
        self.actuator_commands = {
            'base_velocity': Twist(),
            'joint_positions': {}
        }
        self.get_logger().info('Actuators initialized')

    def initialize_ai_models(self):
        """Initialize AI models for deployment"""
        # This would load optimized models for edge deployment
        try:
            # Initialize perception model
            self.perception_model = self.load_optimized_perception_model()

            # Initialize control model
            self.control_model = self.load_optimized_control_model()

            self.get_logger().info('AI models initialized')
        except Exception as e:
            self.get_logger().error(f'Error initializing AI models: {e}')

    def load_optimized_perception_model(self):
        """Load optimized perception model"""
        # Placeholder for optimized model loading
        # This would load TensorRT optimized models
        return None

    def load_optimized_control_model(self):
        """Load optimized control model"""
        # Placeholder for optimized model loading
        return None

    def initialize_monitoring(self):
        """Initialize system monitoring"""
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.get_logger().info('Monitoring initialized')

    def setup_ros_interfaces(self):
        """Setup all ROS interfaces"""
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.performance_pub = self.create_publisher(Float32, '/performance_fps', 10)

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

    def image_callback(self, msg):
        """Process camera image"""
        with self.data_lock:
            self.sensor_data['camera'] = msg
            self.frame_count += 1

        # Process image with AI model
        if self.perception_model:
            self.process_camera_image(msg)

    def scan_callback(self, msg):
        """Process LiDAR scan"""
        with self.data_lock:
            self.sensor_data['lidar'] = msg

        # Process scan data for navigation
        obstacles = self.detect_obstacles_in_scan(msg)
        self.handle_obstacles(obstacles)

    def imu_callback(self, msg):
        """Process IMU data"""
        with self.data_lock:
            self.sensor_data['imu'] = msg

        # Use IMU for stabilization
        self.update_robot_stability(msg)

    def joint_callback(self, msg):
        """Process joint states"""
        with self.data_lock:
            self.sensor_data['joints'] = msg

        # Monitor joint health
        self.monitor_joint_health(msg)

    def process_camera_image(self, image_msg):
        """Process camera image with AI model"""
        try:
            # This would run the image through the perception model
            # For now, just increment frame counter
            pass
        except Exception as e:
            self.get_logger().error(f'Camera processing error: {e}')

    def detect_obstacles_in_scan(self, scan_msg):
        """Detect obstacles from LiDAR scan"""
        obstacles = []
        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)):
                if range_val < 1.0:  # Obstacle within 1 meter
                    angle = scan_msg.angle_min + i * scan_msg.angle_increment
                    obstacles.append({
                        'distance': range_val,
                        'angle': angle
                    })
        return obstacles

    def handle_obstacles(self, obstacles):
        """Handle detected obstacles"""
        if obstacles and not self.emergency_stop:
            closest = min(obstacles, key=lambda x: x['distance'])
            if closest['distance'] < 0.5:  # Emergency stop distance
                self.emergency_stop_robot()
            else:
                # Normal obstacle avoidance
                self.avoid_obstacle(closest)

    def update_robot_stability(self, imu_msg):
        """Update robot stability based on IMU"""
        # Extract orientation from IMU
        orientation = imu_msg.orientation
        # Apply stability control if needed
        pass

    def monitor_joint_health(self, joint_msg):
        """Monitor joint health and safety"""
        for i, name in enumerate(joint_msg.name):
            if i < len(joint_msg.position):
                pos = joint_msg.position[i]
                # Check for dangerous positions
                if abs(pos) > 3.0:  # Joint limit
                    self.get_logger().warn(f'Dangerous joint position: {name} = {pos}')
                    self.emergency_stop_robot()

    def update_performance_metrics(self):
        """Update system performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed > 0:
            self.performance_metrics['fps'] = 1.0 / elapsed
            self.last_frame_time = current_time

        # Update system metrics
        self.performance_metrics['cpu_usage'] = psutil.cpu_percent()
        self.performance_metrics['memory_usage'] = psutil.virtual_memory().percent

        # Publish performance
        fps_msg = Float32()
        fps_msg.data = float(self.performance_metrics['fps'])
        self.performance_pub.publish(fps_msg)

    def safety_check(self):
        """Perform safety checks"""
        # Check system resources
        if (self.performance_metrics['cpu_usage'] > 95 or
            self.performance_metrics['memory_usage'] > 95):
            self.get_logger().error('System resources critically high')
            self.emergency_stop_robot()

        # Check for emergency stop condition
        if self.emergency_stop:
            self.publish_system_status('EMERGENCY_STOP')
        else:
            self.publish_system_status('OPERATIONAL')

    def emergency_stop_robot(self):
        """Execute emergency stop"""
        if not self.emergency_stop:
            self.emergency_stop = True
            self.get_logger().error('EMERGENCY STOP ACTIVATED')

            # Stop all movement
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

    def avoid_obstacle(self, obstacle):
        """Execute obstacle avoidance"""
        cmd = Twist()
        if obstacle['angle'] > 0:  # Obstacle on right, turn left
            cmd.angular.z = 0.5
        else:  # Obstacle on left, turn right
            cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(cmd)

    def publish_system_status(self, status):
        """Publish system status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CompleteIsaacDeployment()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Complete Isaac deployment stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Ethical Considerations

When deploying Isaac applications:

- **Safety**: Ensure all safety mechanisms are properly implemented and tested
- **Privacy**: Be mindful of data collection and processing in deployed systems
- **Reliability**: Maintain high standards for system reliability and error handling
- **Transparency**: Document system capabilities, limitations, and failure modes
- **Accountability**: Establish clear responsibility for robot actions and decisions

## Summary

In this week, we've covered:

- Hardware integration and setup for Isaac deployment
- Sensor and actuator integration with Isaac
- Real-time performance optimization techniques
- Memory management for edge devices
- Simulation-to-reality transfer and domain adaptation
- Reality gap compensation methods
- System monitoring and debugging strategies
- Error handling and recovery mechanisms
- Configuration management for different deployment scenarios
- Best practices for Isaac deployment

## References

1. NVIDIA Isaac ROS Documentation. (2023). Retrieved from https://docs.nvidia.com/isaac-ros/
2. TensorRT Optimization Guide. (2023). NVIDIA Corporation.
3. ROS 2 Deployment Best Practices. (2023). Open Robotics.
4. Embedded AI Deployment Patterns. (2022). Research Papers.

---

**Next Week**: [Vision-Language-Action Models](../module-4/week-11.md)