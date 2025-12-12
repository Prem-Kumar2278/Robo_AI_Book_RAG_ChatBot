---
sidebar_position: 13
title: "Module 4 - Week 12: AI-Robot Integration"
---

# Module 4 - Week 12: AI-Robot Integration

## Learning Objectives

By the end of this week, you will be able to:
- Integrate AI models with robotic platforms for perception and control
- Implement real-time AI inference pipelines for robotics applications
- Design AI-robot communication architectures and protocols
- Optimize AI models for deployment on robotic hardware
- Implement safety mechanisms and validation for AI-driven robots
- Evaluate and benchmark AI-robot integration performance
- Design human-AI-robot interaction systems
- Troubleshoot common issues in AI-robot integration

## Introduction to AI-Robot Integration

AI-robot integration represents the convergence of artificial intelligence and robotics, enabling robots to perform complex tasks through intelligent decision-making. This integration transforms robots from simple programmable machines into adaptive, learning systems capable of operating in dynamic environments.

### Key Integration Components

AI-robot integration involves several key components working together:

1. **Perception Systems**: Computer vision, sensor processing, environment understanding
2. **Decision Making**: Planning, reasoning, task execution
3. **Control Systems**: Motor control, trajectory generation, actuation
4. **Learning Systems**: Adaptation, continuous improvement, skill acquisition
5. **Communication**: Human-robot interaction, multi-robot coordination

### Integration Architecture Patterns

The architecture of AI-robot integration systems typically follows these patterns:

#### 1. Hierarchical Integration
```
High-Level AI (Planning/Reasoning)
    ↓
Mid-Level AI (Task Planning/Execution)
    ↓
Low-Level AI (Control/Sensor Fusion)
    ↓
Hardware Interface (Actuators/Sensors)
```

#### 2. Parallel Integration
```
Perception AI ←→ Decision AI ←→ Control AI
    ↓              ↓              ↓
Sensors      Planning        Actuators
```

#### 3. Hybrid Integration
Combining hierarchical and parallel approaches for optimal performance.

## AI Model Integration Patterns

### Perception Integration

Integrating AI models for robotic perception involves processing sensor data to understand the environment:

```python
# perception_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from torchvision import transforms
import open3d as o3d

class AIPerceptionIntegration(Node):
    def __init__(self):
        super().__init__('ai_perception_integration')

        # Initialize components
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load AI models
        self.detection_model = self.load_detection_model()
        self.segmentation_model = self.load_segmentation_model()
        self.depth_model = self.load_depth_model()

        # Setup ROS interfaces
        self.setup_perception_interfaces()

        # Performance monitoring
        self.last_perception_time = self.get_clock().now()
        self.perception_frequency = 0.0

        self.get_logger().info('AI perception integration initialized')

    def setup_perception_interfaces(self):
        """Setup perception ROS interfaces"""
        # Image processing
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Point cloud processing
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/pointcloud', self.pointcloud_callback, 10)

        # Laser scan processing
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Detection results
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10)
        self.segmentation_pub = self.create_publisher(
            Image, '/segmentation', 10)

    def load_detection_model(self):
        """Load object detection model"""
        try:
            # Using a pre-trained model (e.g., YOLOv5, Detectron2, etc.)
            import torchvision.models.detection as detection_models
            model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading detection model: {e}')
            return None

    def load_segmentation_model(self):
        """Load semantic segmentation model"""
        try:
            import torchvision.models.segmentation as segmentation_models
            model = segmentation_models.deeplabv3_resnet50(pretrained=True)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading segmentation model: {e}')
            return None

    def load_depth_model(self):
        """Load depth estimation model"""
        try:
            # Using MiDaS for monocular depth estimation
            import torch.hub
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading depth model: {e}')
            return None

    def image_callback(self, msg):
        """Process RGB image with AI models"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run object detection
            detections = self.run_object_detection(cv_image)
            if detections:
                self.publish_detections(detections, msg.header)

            # Run semantic segmentation
            segmentation = self.run_segmentation(cv_image)
            if segmentation is not None:
                seg_msg = self.bridge.cv2_to_imgmsg(segmentation, encoding='mono8')
                seg_msg.header = msg.header
                self.segmentation_pub.publish(seg_msg)

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def run_object_detection(self, image):
        """Run object detection on image"""
        if self.detection_model is None:
            return None

        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.detection_model(input_tensor)

            # Process predictions
            detections = []
            for i, (box, score, label) in enumerate(zip(
                predictions[0]['boxes'],
                predictions[0]['scores'],
                predictions[0]['labels']
            )):
                if score > 0.5:  # Confidence threshold
                    detection = {
                        'bbox': box.cpu().numpy(),
                        'score': score.cpu().item(),
                        'label': label.cpu().item()
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            return None

    def run_segmentation(self, image):
        """Run semantic segmentation on image"""
        if self.segmentation_model is None:
            return None

        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out']
                predicted = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            return predicted.astype(np.uint8)

        except Exception as e:
            self.get_logger().error(f'Segmentation error: {e}')
            return None

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # Process depth information with AI models
            processed_depth = self.process_depth_with_ai(cv_depth)
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        try:
            # Convert PointCloud2 to Open3D format
            points = self.pointcloud2_to_array(msg)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Process with AI models (e.g., 3D object detection, segmentation)
            processed_pcd = self.process_pointcloud_with_ai(pcd)
        except Exception as e:
            self.get_logger().error(f'Point cloud processing error: {e}')

    def scan_callback(self, msg):
        """Process laser scan data"""
        try:
            # Process laser scan with AI models (e.g., for obstacle detection, mapping)
            processed_scan = self.process_scan_with_ai(msg)
        except Exception as e:
            self.get_logger().error(f'Scan processing error: {e}')

    def publish_detections(self, detections, header):
        """Publish object detections"""
        detection_msg = Detection2DArray()
        detection_msg.header = header

        for det in detections:
            detection_2d = Detection2D()
            detection_2d.bbox.center.x = float((det['bbox'][0] + det['bbox'][2]) / 2)
            detection_2d.bbox.center.y = float((det['bbox'][1] + det['bbox'][3]) / 2)
            detection_2d.bbox.size_x = float(det['bbox'][2] - det['bbox'][0])
            detection_2d.bbox.size_y = float(det['bbox'][3] - det['bbox'][1])

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['label'])
            hypothesis.hypothesis.score = det['score']
            detection_2d.results.append(hypothesis)

            detection_msg.detections.append(detection_2d)

        self.detection_pub.publish(detection_msg)

    def pointcloud2_to_array(self, msg):
        """Convert PointCloud2 message to numpy array"""
        import sensor_msgs.point_cloud2 as pc2
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def process_depth_with_ai(self, depth_image):
        """Process depth image with AI models"""
        # This would involve depth completion, obstacle detection, etc.
        return depth_image

    def process_pointcloud_with_ai(self, pointcloud):
        """Process point cloud with AI models"""
        # This would involve 3D object detection, segmentation, etc.
        return pointcloud

    def process_scan_with_ai(self, scan_msg):
        """Process laser scan with AI models"""
        # This would involve obstacle detection, classification, etc.
        return scan_msg

def main(args=None):
    rclpy.init(args=args)
    node = AIPerceptionIntegration()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('AI perception integration stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Decision Making Integration

Integrating AI models for decision making and planning:

```python
# decision_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point, Twist
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class AIDecisionIntegration(Node):
    def __init__(self):
        super().__init__('ai_decision_integration')

        # Initialize AI decision model
        self.decision_model = self.load_decision_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Robot state
        self.current_pose = None
        self.goal_pose = None
        self.map_data = None
        self.scan_data = None

        # Setup ROS interfaces
        self.setup_decision_interfaces()

        # Decision making timer
        self.decision_timer = self.create_timer(0.1, self.make_decisions)

        self.get_logger().info('AI decision integration initialized')

    def setup_decision_interfaces(self):
        """Setup decision-making ROS interfaces"""
        # State subscriptions
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.pose_callback, 10)
        self.goal_sub = self.create_subscription(
            Pose, '/goal_pose', self.goal_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Command publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)

        # Status publishers
        self.status_pub = self.create_publisher(String, '/decision_status', 10)

    def load_decision_model(self):
        """Load decision-making AI model"""
        # Simple neural network for decision making
        class DecisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 64)  # Input: robot state + goal + scan features
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 2)  # Output: linear velocity, angular velocity

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = torch.tanh(self.fc3(x))  # Output in [-1, 1] range
                return x

        model = DecisionModel()
        model.eval()
        return model

    def pose_callback(self, msg):
        """Update robot pose"""
        self.current_pose = msg

    def goal_callback(self, msg):
        """Update goal pose"""
        self.goal_pose = msg

    def map_callback(self, msg):
        """Update map data"""
        self.map_data = msg

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.scan_data = msg

    def make_decisions(self):
        """Make decisions using AI model"""
        if (self.current_pose is None or
            self.goal_pose is None or
            self.scan_data is None):
            return

        try:
            # Prepare input for decision model
            input_features = self.prepare_decision_input()
            if input_features is None:
                return

            # Run decision model
            with torch.no_grad():
                action = self.decision_model(input_features)

            # Convert action to robot command
            cmd = self.convert_action_to_command(action)
            self.cmd_pub.publish(cmd)

            # Publish status
            status_msg = String()
            status_msg.data = f'deciding, action: [{action[0]:.2f}, {action[1]:.2f}]'
            self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Decision making error: {e}')

    def prepare_decision_input(self):
        """Prepare input features for decision model"""
        if self.scan_data is None:
            return None

        # Extract features from laser scan
        scan_features = self.extract_scan_features(self.scan_data)

        # Extract goal features
        if self.current_pose and self.goal_pose:
            goal_features = self.calculate_goal_features(
                self.current_pose, self.goal_pose
            )
        else:
            goal_features = [0.0, 0.0, 0.0]  # [distance, angle, bearing]

        # Combine features
        features = scan_features + goal_features

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return features_tensor.to(self.device)

    def extract_scan_features(self, scan_msg):
        """Extract features from laser scan"""
        # Calculate various scan-based features
        ranges = [r for r in scan_msg.ranges if not (np.isnan(r) or np.isinf(r))]

        if not ranges:
            return [0.0] * 10  # Default features if no valid ranges

        # Statistical features
        features = [
            min(ranges) if ranges else float('inf'),  # Closest obstacle
            np.mean(ranges) if ranges else 0.0,      # Average distance
            np.std(ranges) if ranges else 0.0,       # Distance variance
            len([r for r in ranges if r < 1.0]),     # Obstacles within 1m
            len([r for r in ranges if r < 0.5]),     # Obstacles within 0.5m
            # Add more features as needed
        ]

        # Pad or truncate to fixed size (10 features)
        while len(features) < 10:
            features.append(0.0)
        return features[:10]

    def calculate_goal_features(self, current_pose, goal_pose):
        """Calculate features related to goal"""
        # Calculate distance to goal
        dx = goal_pose.position.x - current_pose.position.x
        dy = goal_pose.position.y - current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Calculate bearing to goal
        bearing = np.arctan2(dy, dx)

        # Calculate robot orientation
        # Simplified: assume orientation in z component of quaternion
        robot_yaw = 2 * np.arcsin(current_pose.orientation.z)

        # Calculate angle to goal relative to robot orientation
        angle_to_goal = bearing - robot_yaw
        # Normalize angle to [-π, π]
        while angle_to_goal > np.pi:
            angle_to_goal -= 2 * np.pi
        while angle_to_goal < -np.pi:
            angle_to_goal += 2 * np.pi

        return [distance, angle_to_goal, bearing]

    def convert_action_to_command(self, action):
        """Convert AI model action to robot command"""
        cmd = Twist()

        # Scale actions to appropriate ranges
        cmd.linear.x = float(action[0, 0].item()) * 0.5  # Scale to [-0.5, 0.5] m/s
        cmd.angular.z = float(action[0, 1].item()) * 1.0  # Scale to [-1.0, 1.0] rad/s

        return cmd

class AIPlanningIntegration(Node):
    def __init__(self):
        super().__init__('ai_planning_integration')

        # Planning AI model
        self.planning_model = self.load_planning_model()

        # Setup planning interfaces
        self.setup_planning_interfaces()

    def setup_planning_interfaces(self):
        """Setup planning ROS interfaces"""
        self.goal_sub = self.create_subscription(
            Pose, '/goal_pose', self.goal_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.path_pub = self.create_publisher(Path, '/plan', 10)

    def load_planning_model(self):
        """Load path planning AI model"""
        # This would be a more complex model for path planning
        # e.g., using reinforcement learning, graph neural networks, etc.
        class PlanningModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified model - in practice, this would be more complex
                self.encoder = nn.Sequential(
                    nn.Linear(100, 128),  # Map encoding
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(64 + 4, 32),  # +4 for start/end pose
                    nn.ReLU(),
                    nn.Linear(32, 2)  # Output: next position delta
                )

            def forward(self, map_features, start_pos, goal_pos):
                encoded_map = self.encoder(map_features)
                combined = torch.cat([encoded_map, start_pos, goal_pos], dim=1)
                return self.decoder(combined)

        model = PlanningModel()
        model.eval()
        return model

    def goal_callback(self, msg):
        """Handle goal pose for planning"""
        pass

    def map_callback(self, msg):
        """Handle map for planning"""
        pass

def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    decision_node = AIDecisionIntegration()
    planning_node = AIPlanningIntegration()

    # Use MultiThreadedExecutor
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(decision_node)
    executor.add_node(planning_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        decision_node.get_logger().info('AI decision integration stopped by user')
        planning_node.get_logger().info('AI planning integration stopped by user')
    finally:
        decision_node.destroy_node()
        planning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Control Integration

Integrating AI models for robot control and actuation:

```python
# control_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float32MultiArray
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import torch
import torch.nn as nn
import time

class AIControlIntegration(Node):
    def __init__(self):
        super().__init__('ai_control_integration')

        # Initialize control AI model
        self.control_model = self.load_control_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Robot state tracking
        self.current_joints = {}
        self.target_joints = {}
        self.current_pose = None
        self.desired_velocity = Twist()

        # Setup control interfaces
        self.setup_control_interfaces()

        # Control loop timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        self.get_logger().info('AI control integration initialized')

    def setup_control_interfaces(self):
        """Setup control ROS interfaces"""
        # State feedback
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.pose_callback, 10)
        self.velocity_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_callback, 10)

        # Command output
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10)
        self.velocity_cmd_pub = self.create_publisher(Twist, '/cmd_vel_out', 10)

    def load_control_model(self):
        """Load control AI model"""
        # Simple neural network for control
        class ControlModel(nn.Module):
            def __init__(self, input_size=14, output_size=7):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size)
                )

            def forward(self, x):
                return self.network(x)

        model = ControlModel()
        model.eval()
        return model

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joints[name] = msg.position[i]

    def pose_callback(self, msg):
        """Update current pose"""
        self.current_pose = msg

    def velocity_callback(self, msg):
        """Update desired velocity"""
        self.desired_velocity = msg

    def control_loop(self):
        """Main control loop with AI integration"""
        if not self.current_joints or not self.current_pose:
            return

        try:
            # Prepare input for control model
            input_features = self.prepare_control_input()
            if input_features is None:
                return

            # Run control model
            with torch.no_grad():
                control_output = self.control_model(input_features)

            # Convert AI output to robot commands
            commands = self.convert_control_output(control_output)

            # Publish commands
            self.publish_control_commands(commands)

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def prepare_control_input(self):
        """Prepare input features for control model"""
        # Get current joint positions (first 7 joints as example)
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        current_positions = []
        for name in joint_names:
            if name in self.current_joints:
                current_positions.append(self.current_joints[name])
            else:
                current_positions.append(0.0)

        # Get desired velocity
        desired_vel = [
            self.desired_velocity.linear.x,
            self.desired_velocity.linear.y,
            self.desired_velocity.linear.z,
            self.desired_velocity.angular.x,
            self.desired_velocity.angular.y,
            self.desired_velocity.angular.z
        ]

        # Combine features
        features = current_positions + desired_vel

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return features_tensor.to(self.device)

    def convert_control_output(self, control_output):
        """Convert AI control output to robot commands"""
        commands = control_output.squeeze().cpu().numpy()
        return commands

    def publish_control_commands(self, commands):
        """Publish control commands to robot"""
        # Create joint trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']

        point = JointTrajectoryPoint()
        point.positions = commands.tolist()
        point.velocities = [0.0] * len(commands)  # Zero velocities for simplicity
        point.accelerations = [0.0] * len(commands)  # Zero accelerations
        point.time_from_start = Duration(sec=0, nanosec=10000000)  # 10ms

        traj_msg.points = [point]
        self.joint_cmd_pub.publish(traj_msg)

class AIAdaptiveController(Node):
    def __init__(self):
        super().__init__('ai_adaptive_controller')

        # Adaptive control model
        self.adaptive_model = self.load_adaptive_model()

        # Learning parameters
        self.learning_rate = 0.001
        self.error_history = []

        # Setup adaptive control interfaces
        self.setup_adaptive_interfaces()

    def setup_adaptive_interfaces(self):
        """Setup adaptive control interfaces"""
        # Error feedback
        self.error_sub = self.create_subscription(
            Float32MultiArray, '/control_error', self.error_callback, 10)

        # Model update timer
        self.update_timer = self.create_timer(1.0, self.update_model)

    def load_adaptive_model(self):
        """Load adaptive control model"""
        # This would be a model that can be updated online
        class AdaptiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_network = nn.Sequential(
                    nn.Linear(14, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 7)
                )

            def forward(self, x):
                return self.base_network(x)

        model = AdaptiveModel()
        model.eval()
        return model

    def error_callback(self, msg):
        """Handle control error feedback"""
        error = np.array(msg.data)
        self.error_history.append(error)

        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]

    def update_model(self):
        """Update model based on error feedback"""
        if len(self.error_history) < 10:
            return

        # This would implement online learning/adaptive control
        # For now, we'll just log the average error
        avg_error = np.mean(np.array(self.error_history), axis=0)
        self.get_logger().info(f'Average control error: {avg_error}')

def main(args=None):
    rclpy.init(args=args)

    # Create control nodes
    control_node = AIControlIntegration()
    adaptive_node = AIAdaptiveController()

    # Use MultiThreadedExecutor
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(control_node)
    executor.add_node(adaptive_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        control_node.get_logger().info('AI control integration stopped by user')
        adaptive_node.get_logger().info('AI adaptive controller stopped by user')
    finally:
        control_node.destroy_node()
        adaptive_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Communication Architecture

### Real-Time Communication Patterns

Designing efficient communication between AI systems and robots:

```python
# communication_architecture.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Pose
from builtin_interfaces.msg import Time
import numpy as np
import time
import threading
from queue import Queue, PriorityQueue
import json

class AIBridgeNode(Node):
    def __init__(self):
        super().__init__('ai_bridge')

        # Communication queues for different priority levels
        self.high_priority_queue = PriorityQueue()
        self.normal_priority_queue = Queue()
        self.low_priority_queue = Queue()

        # Message buffers
        self.message_buffer = {}
        self.last_message_time = {}

        # Setup communication interfaces
        self.setup_communication_interfaces()

        # Communication timers
        self.high_priority_timer = self.create_timer(0.01, self.process_high_priority)  # 100Hz
        self.normal_priority_timer = self.create_timer(0.1, self.process_normal_priority)  # 10Hz
        self.low_priority_timer = self.create_timer(1.0, self.process_low_priority)  # 1Hz

        self.get_logger().info('AI bridge initialized')

    def setup_communication_interfaces(self):
        """Setup communication interfaces"""
        # High priority: sensor data, emergency commands
        self.sensor_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.high_priority_callback, 1)
        self.emergency_sub = self.create_subscription(
            String, '/emergency_stop', self.emergency_callback, 1)

        # Normal priority: control commands, state updates
        self.control_sub = self.create_subscription(
            Twist, '/cmd_vel', self.normal_priority_callback, 10)
        self.state_sub = self.create_subscription(
            Pose, '/robot_pose', self.state_callback, 10)

        # Low priority: logging, diagnostics
        self.log_sub = self.create_subscription(
            String, '/log_messages', self.low_priority_callback, 10)

        # Publishers
        self.ai_command_pub = self.create_publisher(String, '/ai_commands', 10)
        self.status_pub = self.create_publisher(String, '/communication_status', 10)

    def high_priority_callback(self, msg):
        """Handle high-priority messages"""
        # Add to high priority queue with timestamp
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.high_priority_queue.put((timestamp, 'sensor', msg))

    def normal_priority_callback(self, msg):
        """Handle normal-priority messages"""
        self.normal_priority_queue.put(('control', msg))

    def low_priority_callback(self, msg):
        """Handle low-priority messages"""
        self.low_priority_queue.put(('log', msg))

    def emergency_callback(self, msg):
        """Handle emergency messages"""
        # Emergency messages get highest priority
        self.high_priority_queue.put((time.time(), 'emergency', msg))

    def state_callback(self, msg):
        """Handle state messages"""
        self.message_buffer['state'] = msg
        self.last_message_time['state'] = time.time()

    def process_high_priority(self):
        """Process high-priority messages"""
        processed = 0
        while not self.high_priority_queue.empty() and processed < 10:  # Limit per cycle
            try:
                timestamp, msg_type, msg = self.high_priority_queue.get_nowait()
                self.handle_high_priority_message(msg_type, msg)
                processed += 1
            except:
                break

    def process_normal_priority(self):
        """Process normal-priority messages"""
        processed = 0
        while not self.normal_priority_queue.empty() and processed < 5:
            try:
                msg_type, msg = self.normal_priority_queue.get_nowait()
                self.handle_normal_priority_message(msg_type, msg)
                processed += 1
            except:
                break

    def process_low_priority(self):
        """Process low-priority messages"""
        processed = 0
        while not self.low_priority_queue.empty() and processed < 2:
            try:
                msg_type, msg = self.low_priority_queue.get_nowait()
                self.handle_low_priority_message(msg_type, msg)
                processed += 1
            except:
                break

    def handle_high_priority_message(self, msg_type, msg):
        """Handle high-priority message"""
        if msg_type == 'sensor':
            # Process sensor data immediately
            self.process_sensor_data(msg)
        elif msg_type == 'emergency':
            # Handle emergency immediately
            self.handle_emergency(msg)

    def handle_normal_priority_message(self, msg_type, msg):
        """Handle normal-priority message"""
        if msg_type == 'control':
            # Process control command
            self.process_control_command(msg)

    def handle_low_priority_message(self, msg_type, msg):
        """Handle low-priority message"""
        if msg_type == 'log':
            # Process logging
            self.process_log_message(msg)

    def process_sensor_data(self, sensor_msg):
        """Process sensor data for AI"""
        # This would forward sensor data to AI models
        pass

    def handle_emergency(self, emergency_msg):
        """Handle emergency situation"""
        self.get_logger().error(f'EMERGENCY: {emergency_msg.data}')
        # Publish emergency stop command
        stop_cmd = String()
        stop_cmd.data = 'EMERGENCY_STOP'
        self.ai_command_pub.publish(stop_cmd)

    def process_control_command(self, cmd_msg):
        """Process control command"""
        # This would validate and forward control commands
        pass

    def process_log_message(self, log_msg):
        """Process log message"""
        # This would handle logging
        pass

class CommunicationOptimizer:
    def __init__(self, node):
        self.node = node
        self.compression_enabled = True
        self.qos_profiles = self.setup_qos_profiles()

    def setup_qos_profiles(self):
        """Setup QoS profiles for different message types"""
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

        profiles = {
            'sensor': QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE
            ),
            'control': QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE
            ),
            'state': QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
            )
        }
        return profiles

    def compress_message(self, msg):
        """Compress message for efficient transmission"""
        if not self.compression_enabled:
            return msg

        # Simple compression example (in practice, use proper compression)
        if hasattr(msg, 'data'):
            if isinstance(msg.data, bytes):
                # For image data, use appropriate compression
                pass
        return msg

    def validate_message(self, msg_type, msg):
        """Validate message before processing"""
        # Check message validity
        if msg is None:
            return False

        # Validate based on type
        if msg_type == 'sensor':
            return self.validate_sensor_message(msg)
        elif msg_type == 'control':
            return self.validate_control_message(msg)

        return True

    def validate_sensor_message(self, msg):
        """Validate sensor message"""
        # Add sensor-specific validation
        return True

    def validate_control_message(self, msg):
        """Validate control message"""
        # Add control-specific validation
        return True

def main(args=None):
    rclpy.init(args=args)
    bridge_node = AIBridgeNode()
    optimizer = CommunicationOptimizer(bridge_node)

    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        bridge_node.get_logger().info('AI bridge stopped by user')
    finally:
        bridge_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Model Optimization for Robotics

Optimizing AI models for real-time robotics applications:

```python
# model_optimization.py
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np
import time

class ModelOptimizer:
    def __init__(self):
        self.optimization_techniques = [
            'quantization',
            'pruning',
            'knowledge_distillation',
            'model_compression'
        ]

    def quantize_model(self, model):
        """Quantize model for reduced memory and faster inference"""
        # Set model to evaluation mode
        model.eval()

        # Specify quantization configuration
        quantization_config = torch.quantization.get_default_qconfig('fbgemm')
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )

        return quantized_model

    def prune_model(self, model, sparsity=0.2):
        """Prune model to reduce size and improve speed"""
        import torch.nn.utils.prune as prune

        # Prune all linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                # Remove pruning reparameterization to make it permanent
                prune.remove(module, 'weight')

        return model

    def optimize_for_inference(self, model):
        """Optimize model for inference"""
        model.eval()
        # Use torch.jit for further optimization
        model = torch.jit.script(model)
        return model

    def benchmark_model(self, model, input_shapes, device='cpu'):
        """Benchmark model performance"""
        model.eval()
        model = model.to(device)

        # Create dummy inputs
        if isinstance(input_shapes, dict):
            dummy_inputs = {k: torch.randn(v).to(device) for k, v in input_shapes.items()}
        else:
            dummy_inputs = torch.randn(input_shapes).to(device)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                if isinstance(dummy_inputs, dict):
                    _ = model(**dummy_inputs)
                else:
                    _ = model(dummy_inputs)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                if isinstance(dummy_inputs, dict):
                    _ = model(**dummy_inputs)
                else:
                    _ = model(dummy_inputs)
                end = time.time()
                times.append(end - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': fps,
            'latency_ms': avg_time * 1000
        }

class EfficientAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use efficient layers
        self.backbone = self.create_efficient_backbone()
        self.head = self.create_efficient_head()

    def create_efficient_backbone(self):
        """Create efficient backbone using MobileNet or similar"""
        import torchvision.models as models
        # Use a lightweight model
        backbone = models.mobilenet_v2(pretrained=True)
        # Remove classifier
        backbone.classifier = nn.Identity()
        return backbone

    def create_efficient_head(self):
        """Create efficient task-specific head"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),  # MobileNetV2 feature size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 7-DOF output
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class HardwareAwareOptimizer:
    def __init__(self, target_hardware='cpu'):
        self.target_hardware = target_hardware
        self.hardware_specs = self.get_hardware_specs()

    def get_hardware_specs(self):
        """Get target hardware specifications"""
        if self.target_hardware == 'jetson_nano':
            return {
                'memory': 4 * 1024 * 1024 * 1024,  # 4GB
                'compute': '512 CUDA cores',
                'power': '10W'
            }
        elif self.target_hardware == 'jetson_agx_xavier':
            return {
                'memory': 32 * 1024 * 1024 * 1024,  # 32GB
                'compute': '512 CUDA cores',
                'power': '30W'
            }
        else:
            return {
                'memory': 8 * 1024 * 1024 * 1024,  # 8GB default
                'compute': 'CPU',
                'power': 'unlimited'
            }

    def optimize_for_hardware(self, model):
        """Optimize model based on hardware constraints"""
        if self.target_hardware.startswith('jetson'):
            # Apply GPU-specific optimizations
            return self.optimize_for_gpu(model)
        else:
            # Apply CPU-specific optimizations
            return self.optimize_for_cpu(model)

    def optimize_for_gpu(self, model):
        """Apply GPU-specific optimizations"""
        # Use TensorRT for NVIDIA GPUs
        try:
            from torch2trt import torch2trt
            # Create example input
            x = torch.ones((1, 3, 224, 224)).cuda()
            model_trt = torch2trt(
                model,
                [x],
                fp16_mode=True,  # Use FP16 for better performance
                max_workspace_size=1<<25  # 32MB workspace
            )
            return model_trt
        except ImportError:
            # Fallback to regular optimization
            return self.optimize_regular(model)

    def optimize_for_cpu(self, model):
        """Apply CPU-specific optimizations"""
        # Use CPU-specific optimizations
        return self.optimize_regular(model)

    def optimize_regular(self, model):
        """Apply regular optimizations"""
        optimizer = ModelOptimizer()
        # Apply quantization
        model = optimizer.quantize_model(model)
        # Apply pruning
        model = optimizer.prune_model(model, sparsity=0.1)
        return model

def optimize_robot_model(model, target_hardware='cpu'):
    """Complete model optimization pipeline for robotics"""
    # Create optimizer
    optimizer = ModelOptimizer()
    hw_optimizer = HardwareAwareOptimizer(target_hardware)

    # Apply hardware-specific optimizations
    optimized_model = hw_optimizer.optimize_for_hardware(model)

    # Benchmark performance
    benchmark_results = optimizer.benchmark_model(
        optimized_model,
        input_shapes=(1, 3, 224, 224),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Optimized model performance: {benchmark_results}")

    return optimized_model, benchmark_results

# Example usage
def main():
    # Create an example model
    model = EfficientAIModel()

    # Optimize for target hardware
    optimized_model, results = optimize_robot_model(model, 'jetson_nano')

    print(f"Model optimized for robotics: {results}")

if __name__ == '__main__':
    main()
```

## Safety and Validation

### Safety Mechanisms for AI-Robot Systems

Implementing safety mechanisms in AI-robot integration:

```python
# safety_validation.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, JointState
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import numpy as np
import time
from enum import Enum
from collections import deque

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"

class AISafetyValidator(Node):
    def __init__(self):
        super().__init__('ai_safety_validator')

        # Safety state
        self.safety_level = SafetyLevel.SAFE
        self.emergency_active = False
        self.safety_violations = deque(maxlen=50)

        # Robot state monitoring
        self.current_velocity = Twist()
        self.current_pose = Pose()
        self.joint_states = JointState()
        self.scan_data = None

        # Safety parameters
        self.velocity_limits = {
            'linear_x': 1.0,      # m/s
            'linear_y': 0.5,
            'angular_z': 1.0      # rad/s
        }
        self.joint_limits = {
            'position': 3.14,     # rad
            'velocity': 5.0       # rad/s
        }
        self.safety_margin = 0.5  # meters for obstacles

        # Setup safety interfaces
        self.setup_safety_interfaces()

        # Safety monitoring timers
        self.safety_monitor_timer = self.create_timer(0.1, self.safety_monitor)
        self.emergency_check_timer = self.create_timer(0.01, self.emergency_check)

        self.get_logger().info('AI safety validator initialized')

    def setup_safety_interfaces(self):
        """Setup safety monitoring interfaces"""
        # State monitoring
        self.velocity_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.pose_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Emergency control
        self.emergency_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, '/safety_diagnostics', 10)

    def velocity_callback(self, msg):
        """Monitor velocity commands"""
        self.current_velocity = msg
        self.check_velocity_safety()

    def pose_callback(self, msg):
        """Monitor robot pose"""
        self.current_pose = msg

    def joint_callback(self, msg):
        """Monitor joint states"""
        self.joint_states = msg
        self.check_joint_safety()

    def scan_callback(self, msg):
        """Monitor environment for obstacles"""
        self.scan_data = msg
        self.check_obstacle_safety()

    def check_velocity_safety(self):
        """Check if velocity commands are safe"""
        violations = []

        if abs(self.current_velocity.linear.x) > self.velocity_limits['linear_x']:
            violations.append(f"Linear X velocity limit exceeded: {self.current_velocity.linear.x}")
        if abs(self.current_velocity.linear.y) > self.velocity_limits['linear_y']:
            violations.append(f"Linear Y velocity limit exceeded: {self.current_velocity.linear.y}")
        if abs(self.current_velocity.angular.z) > self.velocity_limits['angular_z']:
            violations.append(f"Angular Z velocity limit exceeded: {self.current_velocity.angular.z}")

        if violations:
            self.log_safety_violation("VELOCITY", violations)
            if self.safety_level != SafetyLevel.EMERGENCY:
                self.safety_level = SafetyLevel.WARNING

    def check_joint_safety(self):
        """Check if joint states are safe"""
        if not self.joint_states.name:
            return

        violations = []

        for i, name in enumerate(self.joint_states.name):
            if i < len(self.joint_states.position):
                pos = self.joint_states.position[i]
                if abs(pos) > self.joint_limits['position']:
                    violations.append(f"Joint {name} position limit exceeded: {pos}")

            if i < len(self.joint_states.velocity):
                vel = self.joint_states.velocity[i]
                if abs(vel) > self.joint_limits['velocity']:
                    violations.append(f"Joint {name} velocity limit exceeded: {vel}")

        if violations:
            self.log_safety_violation("JOINT", violations)
            if self.safety_level != SafetyLevel.EMERGENCY:
                self.safety_level = SafetyLevel.WARNING

    def check_obstacle_safety(self):
        """Check for obstacle safety"""
        if self.scan_data is None:
            return

        violations = []
        min_distance = float('inf')

        for range_val in self.scan_data.ranges:
            if not (np.isnan(range_val) or np.isinf(range_val)):
                if range_val < min_distance:
                    min_distance = range_val

        if min_distance < self.safety_margin:
            violations.append(f"Obstacle too close: {min_distance:.2f}m < {self.safety_margin:.2f}m")

        if violations:
            self.log_safety_violation("OBSTACLE", violations)
            if min_distance < 0.1:  # Critical safety violation
                self.safety_level = SafetyLevel.EMERGENCY
            elif self.safety_level != SafetyLevel.EMERGENCY:
                self.safety_level = SafetyLevel.DANGER

    def log_safety_violation(self, violation_type, details):
        """Log safety violation"""
        violation = {
            'timestamp': time.time(),
            'type': violation_type,
            'details': details
        }
        self.safety_violations.append(violation)

        for detail in details:
            self.get_logger().warn(f"Safety violation - {violation_type}: {detail}")

    def safety_monitor(self):
        """Monitor overall safety status"""
        # Update safety status
        status_msg = String()
        status_msg.data = self.safety_level.value
        self.safety_status_pub.publish(status_msg)

        # Publish diagnostics
        self.publish_diagnostics()

    def emergency_check(self):
        """Check for emergency conditions"""
        if self.safety_level == SafetyLevel.EMERGENCY:
            if not self.emergency_active:
                self.activate_emergency_stop()
        else:
            if self.emergency_active:
                self.deactivate_emergency_stop()

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_active = True
        self.get_logger().error("EMERGENCY STOP ACTIVATED")

        # Publish emergency stop command
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_pub.publish(emergency_msg)

        # Stop all movement
        stop_cmd = Twist()
        # This would also publish to control topics to stop robot

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_active = False
        self.get_logger().info("Emergency stop deactivated")

        # Publish deactivate command
        emergency_msg = Bool()
        emergency_msg.data = False
        self.emergency_pub.publish(emergency_msg)

    def publish_diagnostics(self):
        """Publish safety diagnostics"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Overall safety status
        overall_diag = DiagnosticStatus()
        overall_diag.name = "AI-Robot Safety System"
        overall_diag.level = DiagnosticStatus.OK

        if self.safety_level == SafetyLevel.EMERGENCY:
            overall_diag.level = DiagnosticStatus.ERROR
            overall_diag.message = "EMERGENCY STOP ACTIVE"
        elif self.safety_level == SafetyLevel.DANGER:
            overall_diag.level = DiagnosticStatus.WARN
            overall_diag.message = "DANGER: Safety violations detected"
        elif self.safety_level == SafetyLevel.WARNING:
            overall_diag.level = DiagnosticStatus.WARN
            overall_diag.message = "WARNING: Potential safety issues"
        else:
            overall_diag.message = "All systems nominal"

        # Add safety metrics
        overall_diag.values = [
            {'key': 'Safety Level', 'value': self.safety_level.value},
            {'key': 'Emergency Active', 'value': str(self.emergency_active)},
            {'key': 'Violation Count', 'value': str(len(self.safety_violations))}
        ]

        diag_array.status.append(overall_diag)
        self.diagnostic_pub.publish(diag_array)

class AIValidationSystem:
    def __init__(self, node):
        self.node = node
        self.validation_results = {}
        self.performance_thresholds = {
            'accuracy': 0.8,
            'latency': 0.1,  # seconds
            'throughput': 10  # Hz
        }

    def validate_ai_model(self, model, test_data):
        """Validate AI model performance and safety"""
        results = {
            'accuracy': self.calculate_accuracy(model, test_data),
            'latency': self.measure_latency(model, test_data),
            'throughput': self.measure_throughput(model, test_data),
            'safety_compliance': self.check_safety_compliance(model)
        }

        self.validation_results = results
        return self.evaluate_validation_results(results)

    def calculate_accuracy(self, model, test_data):
        """Calculate model accuracy"""
        # This would implement accuracy calculation
        # based on the specific task
        return 0.95  # Placeholder

    def measure_latency(self, model, test_data):
        """Measure model inference latency"""
        import time
        times = []
        model.eval()

        with torch.no_grad():
            for sample in test_data[:100]:  # Test first 100 samples
                start = time.time()
                _ = model(sample['input'])
                end = time.time()
                times.append(end - start)

        return np.mean(times)

    def measure_throughput(self, model, test_data):
        """Measure model throughput"""
        import time
        start_time = time.time()
        count = 0

        model.eval()
        with torch.no_grad():
            for sample in test_data:
                _ = model(sample['input'])
                count += 1

                if count >= 100:  # Limit for reasonable test time
                    break

        elapsed = time.time() - start_time
        return count / elapsed if elapsed > 0 else 0

    def check_safety_compliance(self, model):
        """Check if model behavior is safe"""
        # This would implement safety checks
        # such as checking for adversarial examples,
        # out-of-distribution detection, etc.
        return True  # Placeholder

    def evaluate_validation_results(self, results):
        """Evaluate validation results against thresholds"""
        compliant = True

        for metric, threshold in self.performance_thresholds.items():
            if metric in results:
                if metric == 'latency':
                    if results[metric] > threshold:
                        self.node.get_logger().error(f"Latency violation: {results[metric]} > {threshold}")
                        compliant = False
                else:
                    if results[metric] < threshold:
                        self.node.get_logger().error(f"{metric} violation: {results[metric]} < {threshold}")
                        compliant = False

        return compliant

def main(args=None):
    rclpy.init(args=args)
    safety_node = AISafetyValidator()
    validation_system = AIValidationSystem(safety_node)

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        safety_node.get_logger().info('AI safety validator stopped by user')
    finally:
        safety_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Human-AI-Robot Interaction

### Designing Human-Centered AI-Robot Systems

Creating intuitive interfaces for human-AI-robot interaction:

```python
# human_ai_robot_interaction.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import time

class HumanAIInteractionNode(Node):
    def __init__(self):
        super().__init__('human_ai_interaction')

        # Initialize components
        self.bridge = CvBridge()
        self.speech_recognizer = sr.Recognizer()
        self.text_to_speech = pyttsx3.init()

        # Interaction state
        self.current_context = {}
        self.conversation_history = []
        self.user_attention = False
        self.interaction_mode = 'idle'  # idle, listening, processing, responding

        # Setup interaction interfaces
        self.setup_interaction_interfaces()

        # Interaction management
        self.interaction_timer = self.create_timer(0.1, self.interaction_management)

        self.get_logger().info('Human-AI interaction node initialized')

    def setup_interaction_interfaces(self):
        """Setup human interaction interfaces"""
        # Speech input/output
        self.speech_command_sub = self.create_subscription(
            String, '/speech_command', self.speech_command_callback, 10)
        self.speech_response_pub = self.create_publisher(
            String, '/speech_response', 10)

        # Visual attention
        self.attention_sub = self.create_subscription(
            Bool, '/user_attention', self.attention_callback, 10)

        # Gesture input
        self.gesture_sub = self.create_subscription(
            String, '/gesture_input', self.gesture_callback, 10)

        # AI command output
        self.ai_command_pub = self.create_publisher(
            String, '/ai_robot_command', 10)

        # Visualization for feedback
        self.visualization_pub = self.create_publisher(
            MarkerArray, '/interaction_markers', 10)

    def attention_callback(self, msg):
        """Handle user attention state"""
        self.user_attention = msg.data
        if msg.data and self.interaction_mode == 'idle':
            self.interaction_mode = 'listening'
            self.speak_response("Hello! How can I help you?")

    def speech_command_callback(self, msg):
        """Process speech commands"""
        command = msg.data.lower().strip()

        if self.interaction_mode in ['listening', 'processing']:
            self.interaction_mode = 'processing'
            self.process_speech_command(command)

    def gesture_callback(self, msg):
        """Process gesture commands"""
        gesture = msg.data.lower().strip()
        self.process_gesture_command(gesture)

    def process_speech_command(self, command):
        """Process natural language commands"""
        try:
            # Parse command
            parsed_command = self.parse_natural_command(command)

            if parsed_command:
                # Validate command safety
                if self.validate_command_safety(parsed_command):
                    # Execute command
                    self.execute_command(parsed_command)
                    self.interaction_mode = 'responding'
                    self.speak_response(f"Executing: {command}")
                else:
                    self.speak_response("Command not safe to execute")
            else:
                self.speak_response("I didn't understand that command")

        except Exception as e:
            self.get_logger().error(f'Command processing error: {e}')
            self.speak_response("Sorry, I encountered an error processing your command")

    def process_gesture_command(self, gesture):
        """Process gesture commands"""
        gesture_commands = {
            'wave': 'greet',
            'point': 'look_at',
            'thumbs_up': 'confirm',
            'thumbs_down': 'cancel'
        }

        if gesture in gesture_commands:
            command = gesture_commands[gesture]
            self.execute_command({'action': command, 'gesture': gesture})
            self.speak_response(f"Gesture recognized: {gesture}")

    def parse_natural_command(self, command):
        """Parse natural language command"""
        # Simple command parsing (in practice, use NLP models)
        command_lower = command.lower()

        # Navigation commands
        if any(word in command_lower for word in ['go to', 'move to', 'navigate to']):
            location = self.extract_location(command_lower)
            return {'action': 'navigate', 'target': location}

        # Manipulation commands
        elif any(word in command_lower for word in ['pick up', 'grasp', 'take']):
            object_name = self.extract_object(command_lower)
            return {'action': 'pick', 'object': object_name}

        # Place commands
        elif any(word in command_lower for word in ['place', 'put down', 'drop']):
            location = self.extract_location(command_lower)
            return {'action': 'place', 'location': location}

        # Follow commands
        elif 'follow' in command_lower:
            return {'action': 'follow', 'target': 'user'}

        # Stop commands
        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return {'action': 'stop'}

        return None

    def extract_location(self, command):
        """Extract location from command"""
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'shelf', 'counter']
        for loc in locations:
            if loc in command:
                return loc
        return 'unknown'

    def extract_object(self, command):
        """Extract object from command"""
        objects = ['cup', 'bottle', 'book', 'box', 'toy', 'ball', 'phone']
        for obj in objects:
            if obj in command:
                return obj
        return 'unknown'

    def validate_command_safety(self, command):
        """Validate command for safety"""
        unsafe_keywords = ['dangerous', 'unsafe', 'harm', 'break', 'damage']
        command_str = str(command).lower()

        for keyword in unsafe_keywords:
            if keyword in command_str:
                return False

        return True

    def execute_command(self, command):
        """Execute parsed command"""
        # Publish command to AI-robot system
        cmd_msg = String()
        cmd_msg.data = str(command)
        self.ai_command_pub.publish(cmd_msg)

        # Add to conversation history
        self.conversation_history.append({
            'timestamp': time.time(),
            'command': command,
            'status': 'executing'
        })

    def speak_response(self, message):
        """Speak response using text-to-speech"""
        def speak():
            self.text_to_speech.say(message)
            self.text_to_speech.runAndWait()

        # Run in separate thread to avoid blocking
        speak_thread = threading.Thread(target=speak)
        speak_thread.start()

        # Publish to ROS topic as well
        response_msg = String()
        response_msg.data = message
        self.speech_response_pub.publish(response_msg)

    def interaction_management(self):
        """Manage interaction state and flow"""
        # Reset interaction mode after timeout
        if (self.interaction_mode != 'idle' and
            len(self.conversation_history) > 0):
            last_interaction = self.conversation_history[-1]['timestamp']
            if time.time() - last_interaction > 30:  # 30 second timeout
                self.interaction_mode = 'idle'
                self.speak_response("Interaction timeout. Ready for new commands.")

        # Update visualization
        self.update_interaction_visualization()

    def update_interaction_visualization(self):
        """Update visualization markers for interaction"""
        marker_array = MarkerArray()

        # Add attention indicator
        attention_marker = Marker()
        attention_marker.header.frame_id = "base_link"
        attention_marker.header.stamp = self.get_clock().now().to_msg()
        attention_marker.ns = "interaction"
        attention_marker.id = 1
        attention_marker.type = Marker.SPHERE
        attention_marker.action = Marker.ADD
        attention_marker.pose.position.z = 1.0  # Above robot
        attention_marker.pose.orientation.w = 1.0
        attention_marker.scale.x = 0.2
        attention_marker.scale.y = 0.2
        attention_marker.scale.z = 0.2

        # Color based on attention state
        if self.user_attention:
            attention_marker.color.r = 0.0
            attention_marker.color.g = 1.0
            attention_marker.color.b = 0.0
        else:
            attention_marker.color.r = 1.0
            attention_marker.color.g = 0.0
            attention_marker.color.b = 0.0

        attention_marker.color.a = 1.0
        marker_array.markers.append(attention_marker)

        self.visualization_pub.publish(marker_array)

class AIIntentInterpreter:
    def __init__(self):
        self.intent_patterns = {
            'navigation': [
                'go to (.*)',
                'move to (.*)',
                'navigate to (.*)',
                'take me to (.*)'
            ],
            'manipulation': [
                'pick up (.*)',
                'grasp (.*)',
                'take (.*)',
                'get (.*)'
            ],
            'information': [
                'what is (.*)',
                'tell me about (.*)',
                'describe (.*)'
            ]
        }

    def interpret_intent(self, command):
        """Interpret user intent from command"""
        import re

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command.lower())
                if match:
                    return {
                        'intent': intent,
                        'parameters': match.groups(),
                        'confidence': 0.9
                    }

        return {
            'intent': 'unknown',
            'parameters': [],
            'confidence': 0.0
        }

def main(args=None):
    rclpy.init(args=args)
    interaction_node = HumanAIInteractionNode()
    intent_interpreter = AIIntentInterpreter()

    try:
        rclpy.spin(interaction_node)
    except KeyboardInterrupt:
        interaction_node.get_logger().info('Human-AI interaction stopped by user')
    finally:
        interaction_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting and Debugging

### Common Issues and Solutions

Addressing common challenges in AI-robot integration:

```python
# troubleshooting.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import numpy as np
import time
import traceback
from collections import defaultdict

class AIDebuggingNode(Node):
    def __init__(self):
        super().__init__('ai_debugging')

        # Debugging state
        self.debug_info = defaultdict(list)
        self.error_log = []
        self.performance_metrics = {}

        # Setup debugging interfaces
        self.setup_debugging_interfaces()

        # Debugging timer
        self.debug_timer = self.create_timer(1.0, self.debug_analysis)

        self.get_logger().info('AI debugging node initialized')

    def setup_debugging_interfaces(self):
        """Setup debugging interfaces"""
        # Debug publishers
        self.debug_pub = self.create_publisher(String, '/debug_info', 10)
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, '/ai_diagnostics', 10)

        # Error subscribers (from other nodes)
        self.error_sub = self.create_subscription(
            String, '/errors', self.error_callback, 10)

    def error_callback(self, msg):
        """Handle error messages from other nodes"""
        error_entry = {
            'timestamp': time.time(),
            'message': msg.data,
            'stack_trace': traceback.format_stack()
        }
        self.error_log.append(error_entry)

        # Log the error
        self.get_logger().error(f'Debug node received error: {msg.data}')

    def log_debug_info(self, category, info):
        """Log debug information by category"""
        self.debug_info[category].append({
            'timestamp': time.time(),
            'info': info
        })

        # Keep only recent entries
        if len(self.debug_info[category]) > 100:
            self.debug_info[category] = self.debug_info[category][-50:]

    def debug_analysis(self):
        """Analyze debug information and identify issues"""
        issues = []

        # Check for common problems
        issues.extend(self.check_performance_issues())
        issues.extend(self.check_data_issues())
        issues.extend(self.check_model_issues())

        # Publish diagnostics
        self.publish_diagnostics(issues)

        # Log issues
        for issue in issues:
            self.get_logger().warn(f'Debug issue: {issue}')

    def check_performance_issues(self):
        """Check for performance-related issues"""
        issues = []

        # Check if AI models are running slowly
        if 'inference_time' in self.debug_info:
            recent_times = [entry['info'] for entry in self.debug_info['inference_time'][-10:]]
            if recent_times:
                avg_time = np.mean(recent_times)
                if avg_time > 0.1:  # More than 100ms per inference
                    issues.append(f"Slow inference: {avg_time:.3f}s average")

        # Check for memory issues
        if 'memory_usage' in self.debug_info:
            recent_memory = [entry['info'] for entry in self.debug_info['memory_usage'][-5:]]
            if recent_memory:
                max_memory = max(recent_memory)
                if max_memory > 0.9:  # More than 90% memory usage
                    issues.append(f"High memory usage: {max_memory:.2f}")

        return issues

    def check_data_issues(self):
        """Check for data-related issues"""
        issues = []

        # Check for missing sensor data
        if 'sensor_data_count' in self.debug_info:
            recent_counts = [entry['info'] for entry in self.debug_info['sensor_data_count'][-10:]]
            if recent_counts and all(count == 0 for count in recent_counts[-5:]):
                issues.append("No sensor data received recently")

        # Check for data quality issues
        if 'data_quality' in self.debug_info:
            recent_quality = [entry['info'] for entry in self.debug_info['data_quality'][-10:]]
            if recent_quality:
                avg_quality = np.mean(recent_quality)
                if avg_quality < 0.5:  # Low data quality
                    issues.append(f"Low data quality: {avg_quality:.2f}")

        return issues

    def check_model_issues(self):
        """Check for model-related issues"""
        issues = []

        # Check for model output anomalies
        if 'model_output' in self.debug_info:
            recent_outputs = [entry['info'] for entry in self.debug_info['model_output'][-20:]]
            if recent_outputs:
                outputs_array = np.array(recent_outputs)
                if np.any(np.isnan(outputs_array)) or np.any(np.isinf(outputs_array)):
                    issues.append("Model producing NaN or Inf values")

                # Check for extreme outputs
                if np.any(np.abs(outputs_array) > 1000):
                    issues.append("Model producing extreme values")

        return issues

    def publish_diagnostics(self, issues):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Overall AI system status
        ai_diag = DiagnosticStatus()
        ai_diag.name = "AI-Robot Integration System"
        ai_diag.level = DiagnosticStatus.OK
        ai_diag.message = "All systems nominal"

        if issues:
            ai_diag.level = DiagnosticStatus.WARN
            ai_diag.message = f"{len(issues)} issues detected"
            ai_diag.values = [{'key': f'Issue {i+1}', 'value': issue} for i, issue in enumerate(issues)]

        diag_array.status.append(ai_diag)
        self.diagnostic_pub.publish(diag_array)

class CommonIssueResolver:
    def __init__(self, node):
        self.node = node

    def resolve_sensor_sync_issue(self):
        """Resolve sensor synchronization issues"""
        self.node.get_logger().info("Resolving sensor synchronization issue...")
        # This would involve:
        # - Checking timestamp alignment
        # - Adjusting QoS settings
        # - Implementing message filters
        pass

    def resolve_model_drift(self):
        """Resolve model drift issues"""
        self.node.get_logger().info("Resolving model drift issue...")
        # This would involve:
        # - Online model updates
        # - Domain adaptation
        # - Data distribution monitoring
        pass

    def resolve_performance_bottleneck(self):
        """Resolve performance bottleneck"""
        self.node.get_logger().info("Resolving performance bottleneck...")
        # This would involve:
        # - Model optimization
        # - Resource allocation
        # - Pipeline parallelization
        pass

    def resolve_communication_timeout(self):
        """Resolve communication timeout issues"""
        self.node.get_logger().info("Resolving communication timeout...")
        # This would involve:
        # - QoS adjustment
        # - Network optimization
        # - Retry mechanisms
        pass

def main(args=None):
    rclpy.init(args=args)
    debug_node = AIDebuggingNode()
    resolver = CommonIssueResolver(debug_node)

    try:
        rclpy.spin(debug_node)
    except KeyboardInterrupt:
        debug_node.get_logger().info('AI debugging node stopped by user')
    finally:
        debug_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. **Perception Integration**: Integrate an AI perception model with a robot's sensor system
2. **Decision Making**: Implement AI-based decision making for robot navigation
3. **Control Integration**: Create an AI controller for robot manipulation tasks
4. **Communication Optimization**: Design an efficient communication architecture for AI-robot systems
5. **Performance Optimization**: Optimize an AI model for real-time robotic applications
6. **Safety Implementation**: Implement safety mechanisms for AI-driven robot control
7. **Human Interaction**: Design a human-AI-robot interaction interface
8. **Troubleshooting**: Create debugging tools for AI-robot integration issues

## Code Example: Complete AI-Robot Integration System

```python
# complete_ai_robot_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, JointState
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import torch
import numpy as np
import time

class CompleteAIIntegrationSystem(Node):
    def __init__(self):
        super().__init__('complete_ai_integration')

        # Initialize components
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load AI models
        self.perception_model = self.load_perception_model()
        self.decision_model = self.load_decision_model()
        self.control_model = self.load_control_model()

        # System state
        self.current_sensor_data = {}
        self.ai_commands = {}
        self.system_status = 'initialized'

        # Setup ROS interfaces
        self.setup_integration_interfaces()

        # Performance monitoring
        self.inference_times = []
        self.last_inference_time = time.time()

        # Main processing timer
        self.processing_timer = self.create_timer(0.05, self.process_pipeline)  # 20Hz

        self.get_logger().info('Complete AI-robot integration system initialized')

    def setup_integration_interfaces(self):
        """Setup all integration interfaces"""
        # Sensor inputs
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Command outputs
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_status', 10)

    def load_perception_model(self):
        """Load perception AI model"""
        try:
            # This would load your specific perception model
            # For example: object detection, semantic segmentation, etc.
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 10)
            ).to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading perception model: {e}')
            return None

    def load_decision_model(self):
        """Load decision-making AI model"""
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2)  # linear, angular velocities
            ).to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading decision model: {e}')
            return None

    def load_control_model(self):
        """Load control AI model"""
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(14, 64),  # State + command
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 7)  # Joint commands
            ).to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading control model: {e}')
            return None

    def image_callback(self, msg):
        """Process image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            processed_image = self.preprocess_image(cv_image)
            self.current_sensor_data['image'] = processed_image
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.current_sensor_data['scan'] = msg

    def joint_callback(self, msg):
        """Process joint state data"""
        self.current_sensor_data['joints'] = msg

    def preprocess_image(self, image):
        """Preprocess image for AI models"""
        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)

    def process_pipeline(self):
        """Main AI-robot integration pipeline"""
        if not self.current_sensor_data:
            return

        try:
            # Run perception
            perception_output = self.run_perception()

            # Make decisions based on perception
            decisions = self.make_decisions(perception_output)

            # Generate control commands
            control_commands = self.generate_control_commands(decisions)

            # Execute commands
            self.execute_commands(control_commands)

            # Update status
            self.update_status()

        except Exception as e:
            self.get_logger().error(f'Processing pipeline error: {e}')

    def run_perception(self):
        """Run perception AI model"""
        if 'image' not in self.current_sensor_data or self.perception_model is None:
            return None

        with torch.no_grad():
            output = self.perception_model(self.current_sensor_data['image'])
            return output

    def make_decisions(self, perception_output):
        """Make decisions using decision model"""
        if perception_output is None or self.decision_model is None:
            return None

        # Prepare input for decision model
        # This would depend on your specific use case
        dummy_input = torch.randn(1, 10).to(self.device)  # Placeholder

        with torch.no_grad():
            decisions = self.decision_model(dummy_input)
            return decisions

    def generate_control_commands(self, decisions):
        """Generate control commands from decisions"""
        if decisions is None or self.control_model is None:
            return None

        # Prepare input for control model
        dummy_state = torch.randn(1, 14).to(self.device)  # Placeholder state

        with torch.no_grad():
            commands = self.control_model(dummy_state)
            return commands

    def execute_commands(self, commands):
        """Execute control commands"""
        if commands is None:
            return

        # Convert AI output to robot commands
        cmd = Twist()
        cmd.linear.x = float(commands[0, 0].item()) * 0.5  # Scale appropriately
        cmd.angular.z = float(commands[0, 1].item()) * 1.0

        self.cmd_pub.publish(cmd)

    def update_status(self):
        """Update system status"""
        status_msg = String()
        status_msg.data = f'running, models_loaded: {all(m is not None for m in [self.perception_model, self.decision_model, self.control_model])}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    ai_system = CompleteAIIntegrationSystem()

    try:
        rclpy.spin(ai_system)
    except KeyboardInterrupt:
        ai_system.get_logger().info('Complete AI integration system stopped by user')
    finally:
        ai_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Ethical Considerations

When integrating AI with robots:

- **Safety**: Ensure all AI-driven actions are safe for humans and environment
- **Privacy**: Protect user data and interactions
- **Transparency**: Make AI decision-making processes understandable
- **Accountability**: Establish clear responsibility for AI-robot actions
- **Bias**: Address potential biases in AI models
- **Human Oversight**: Maintain human control over critical decisions

## Summary

In this week, we've covered:

- AI-robot integration architecture and patterns
- Perception, decision-making, and control integration
- Communication architectures for AI-robot systems
- Performance optimization techniques for robotics
- Safety mechanisms and validation procedures
- Human-AI-robot interaction design
- Troubleshooting and debugging strategies
- Best practices for AI-robot integration

## References

1. Kober, J., et al. (2013). Reinforcement Learning in Robotics: A Survey. IJRR.
2. Argall, B.D., et al. (2009). A Survey of Imitation Learning. Robotics and Autonomous Systems.
3. OpenAI et al. (2022). Learning Dexterous In-Hand Manipulation. IJRR.
4. Levine, S., et al. (2016). Learning Deep Neural Network Policies with Continuous Actions. ICML.

---

**Next Week**: [Capstone Project](./week-13.md)