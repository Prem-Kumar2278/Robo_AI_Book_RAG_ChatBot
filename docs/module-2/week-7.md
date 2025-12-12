---
sidebar_position: 8
title: "Module 2 - Week 7: Unity Integration"
---

# Module 2 - Week 7: Unity Integration

## Learning Objectives

By the end of this week, you will be able to:
- Understand Unity's role in robotics simulation and visualization
- Install and configure Unity with ROS 2 integration tools
- Create 3D environments and robot models in Unity
- Implement ROS 2 communication within Unity using ROS TCP Connector
- Develop custom Unity scripts for robot control and sensor simulation
- Integrate Unity with existing ROS 2 workflows and tools
- Compare Unity with other simulation environments for robotics applications

## Introduction to Unity for Robotics

Unity is a powerful game engine that has gained significant traction in robotics development due to its advanced rendering capabilities, physics simulation, and cross-platform support. The Unity Robotics ecosystem provides tools specifically designed for robotics applications, making it an excellent choice for high-quality visualization and simulation.

### Unity vs. Traditional Robotics Simulators

| Feature | Unity | Gazebo | Webots | PyBullet |
|---------|-------|--------|--------|----------|
| Visual Quality | Excellent | Good | Good | Basic |
| Physics Accuracy | Good | Excellent | Good | Good |
| ROS Integration | Good (ROS TCP Connector) | Excellent | Good | Basic |
| Learning Curve | Moderate | Moderate | Moderate | Low |
| Performance | High (GPU-accelerated) | Moderate | Moderate | High |
| Commercial Use | Licensed | Open Source | Open Source | Open Source |

### Unity Robotics Hub

Unity provides the Robotics Hub, which includes:
- **ROS TCP Connector**: For communication between Unity and ROS 2
- **Unity Perception**: For generating synthetic training data
- **Unity ML-Agents**: For reinforcement learning applications
- **Synthetic Data Generation**: For creating labeled datasets

## Unity Installation and Setup

### Installing Unity

For robotics applications, we recommend Unity 2022.3 LTS or later:

1. Download Unity Hub from the Unity website
2. Install Unity Hub and create an account
3. Through Unity Hub, install Unity 2022.3 LTS or later
4. Install additional modules as needed (Linux Build Support, etc.)

### Unity ROS TCP Connector

The ROS TCP Connector enables communication between Unity and ROS 2:

```bash
# Clone the ROS TCP Connector repository
git clone https://github.com/Unity-Technologies/ROS-TCP-Connector.git

# Or install via Unity Package Manager (recommended)
# In Unity: Window > Package Manager > Add package from git URL
# Use: https://github.com/Unity-Technologies/ROS-TCP-Connector.git
```

### Setting Up Unity Project for Robotics

1. Create a new 3D project in Unity
2. Import the ROS TCP Connector package
3. Configure the ROS connector settings
4. Set up the scene with appropriate lighting and environment

## Creating Robot Models in Unity

### Importing Robot Models

Unity supports various 3D model formats, but for robotics integration, you'll typically work with:

1. **FBX**: Common format for 3D models
2. **URDF**: Can be converted using tools like `urdf2urdf`
3. **STL**: For basic geometric shapes

### Robot Model Structure in Unity

In Unity, a robot model typically consists of:

1. **Root GameObject**: The main robot object
2. **Link GameObjects**: Represent physical links from URDF
3. **Joint Components**: Represent joints between links
4. **Colliders**: For physics simulation
5. **Materials**: For visual appearance

```csharp
// RobotModel.cs - Basic robot model structure
using UnityEngine;
using System.Collections.Generic;

public class RobotModel : MonoBehaviour
{
    [System.Serializable]
    public class JointInfo
    {
        public string jointName;
        public GameObject linkObject;
        public JointType jointType;
        public float minAngle;
        public float maxAngle;
        public float currentAngle;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    public List<JointInfo> joints = new List<JointInfo>();
    public Transform rootLink;

    void Start()
    {
        InitializeRobot();
    }

    void InitializeRobot()
    {
        // Initialize joint information and setup constraints
        foreach (JointInfo joint in joints)
        {
            ConfigureJoint(joint);
        }
    }

    void ConfigureJoint(JointInfo joint)
    {
        // Configure joint constraints based on type
        ConfigurableJoint configJoint = joint.linkObject.GetComponent<ConfigurableJoint>();
        if (configJoint != null)
        {
            switch (joint.jointType)
            {
                case JointType.Revolute:
                    configJoint.xMotion = ConfigurableJointMotion.Locked;
                    configJoint.yMotion = ConfigurableJointMotion.Locked;
                    configJoint.zMotion = ConfigurableJointMotion.Locked;
                    configJoint.angularXMotion = ConfigurableJointMotion.Limited;
                    configJoint.angularYMotion = ConfigurableJointMotion.Locked;
                    configJoint.angularZMotion = ConfigurableJointMotion.Locked;

                    SoftJointLimit lowLimit = new SoftJointLimit();
                    lowLimit.limit = joint.minAngle;
                    configJoint.lowAngularXLimit = lowLimit;

                    SoftJointLimit highLimit = new SoftJointLimit();
                    highLimit.limit = joint.maxAngle;
                    configJoint.highAngularXLimit = highLimit;
                    break;

                case JointType.Prismatic:
                    configJoint.xMotion = ConfigurableJointMotion.Limited;
                    configJoint.yMotion = ConfigurableJointMotion.Locked;
                    configJoint.zMotion = ConfigurableJointMotion.Locked;
                    // Configure linear limits
                    break;

                case JointType.Fixed:
                    configJoint.xMotion = ConfigurableJointMotion.Locked;
                    configJoint.yMotion = ConfigurableJointMotion.Locked;
                    configJoint.zMotion = ConfigurableJointMotion.Locked;
                    configJoint.angularXMotion = ConfigurableJointMotion.Locked;
                    configJoint.angularYMotion = ConfigurableJointMotion.Locked;
                    configJoint.angularZMotion = ConfigurableJointMotion.Locked;
                    break;
            }
        }
    }

    public void SetJointPosition(string jointName, float angle)
    {
        JointInfo joint = joints.Find(j => j.jointName == jointName);
        if (joint != null)
        {
            joint.currentAngle = angle;
            // Update joint position based on new angle
            UpdateJointTransform(joint);
        }
    }

    void UpdateJointTransform(JointInfo joint)
    {
        // Apply the joint angle to the transform
        if (joint.jointType == JointType.Revolute)
        {
            joint.linkObject.transform.localRotation =
                Quaternion.Euler(joint.currentAngle, 0, 0);
        }
    }
}
```

### Materials and Textures for Robots

Creating realistic materials for robot models:

```csharp
// RobotMaterialManager.cs
using UnityEngine;

public class RobotMaterialManager : MonoBehaviour
{
    [Header("Material Presets")]
    public Material aluminumMaterial;
    public Material rubberMaterial;
    public Material plasticMaterial;
    public Material carbonFiberMaterial;

    [Header("Custom Properties")]
    public Color baseColor = Color.gray;
    public float metallic = 0.5f;
    public float smoothness = 0.5f;

    void Start()
    {
        ApplyMaterialsToRobot();
    }

    void ApplyMaterialsToRobot()
    {
        // Find all mesh renderers in child objects
        MeshRenderer[] renderers = GetComponentsInChildren<MeshRenderer>();

        foreach (MeshRenderer renderer in renderers)
        {
            // Apply appropriate material based on part type
            ApplyMaterialToPart(renderer);
        }
    }

    void ApplyMaterialToPart(MeshRenderer renderer)
    {
        // Determine part type based on name or tag
        string partName = renderer.gameObject.name.ToLower();

        if (partName.Contains("wheel") || partName.Contains("tire"))
        {
            renderer.material = CreateRubberMaterial();
        }
        else if (partName.Contains("arm") || partName.Contains("link"))
        {
            renderer.material = CreateAluminumMaterial();
        }
        else
        {
            renderer.material = CreateDefaultMaterial();
        }
    }

    Material CreateAluminumMaterial()
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = Color.gray;
        mat.SetFloat("_Metallic", 0.8f);
        mat.SetFloat("_Smoothness", 0.6f);
        return mat;
    }

    Material CreateRubberMaterial()
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = Color.black;
        mat.SetFloat("_Metallic", 0.1f);
        mat.SetFloat("_Smoothness", 0.2f);
        return mat;
    }

    Material CreateDefaultMaterial()
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = baseColor;
        mat.SetFloat("_Metallic", metallic);
        mat.SetFloat("_Smoothness", smoothness);
        return mat;
    }
}
```

## Unity Scene Setup for Robotics

### Creating Environments

Setting up a basic environment for robot simulation:

```csharp
// EnvironmentSetup.cs
using UnityEngine;
using System.Collections.Generic;

public class EnvironmentSetup : MonoBehaviour
{
    [Header("Environment Settings")]
    public float gravity = -9.81f;
    public Color skyColor = Color.blue;
    public Light sunLight;

    [Header("Obstacle Generation")]
    public GameObject obstaclePrefab;
    public int numberOfObstacles = 10;
    public Vector2 spawnArea = new Vector2(10, 10);

    void Start()
    {
        ConfigureEnvironment();
        GenerateObstacles();
    }

    void ConfigureEnvironment()
    {
        // Set gravity
        Physics.gravity = new Vector3(0, gravity, 0);

        // Configure lighting
        if (sunLight != null)
        {
            RenderSettings.ambientLight = Color.gray;
            sunLight.color = Color.white;
            sunLight.intensity = 1.0f;
        }

        // Set skybox color
        RenderSettings.fog = false;
        Camera.main.backgroundColor = skyColor;
    }

    void GenerateObstacles()
    {
        for (int i = 0; i < numberOfObstacles; i++)
        {
            Vector3 position = new Vector3(
                Random.Range(-spawnArea.x / 2, spawnArea.x / 2),
                0.5f, // Height above ground
                Random.Range(-spawnArea.y / 2, spawnArea.y / 2)
            );

            GameObject obstacle = Instantiate(obstaclePrefab, position, Quaternion.identity);
            obstacle.transform.localScale = new Vector3(
                Random.Range(0.5f, 2.0f),
                Random.Range(0.5f, 2.0f),
                Random.Range(0.5f, 2.0f)
            );
        }
    }

    public void AddObstacle(Vector3 position, Vector3 size)
    {
        GameObject obstacle = Instantiate(obstaclePrefab, position, Quaternion.identity);
        obstacle.transform.localScale = size;
    }
}
```

### Lighting and Visual Effects

Configuring lighting for realistic robot visualization:

```csharp
// LightingManager.cs
using UnityEngine;

public class LightingManager : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light mainLight;
    public Light[] additionalLights;
    public bool useRealisticLighting = true;

    [Header("Post-Processing")]
    public bool enablePostProcessing = false;
    public float exposure = 0.0f;
    public float bloomIntensity = 0.5f;

    void Start()
    {
        SetupLighting();
        ConfigurePostProcessing();
    }

    void SetupLighting()
    {
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.shadows = LightShadows.Soft;
            mainLight.shadowStrength = 0.8f;
            mainLight.intensity = 1.0f;
        }

        if (useRealisticLighting)
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.2f, 0.2f, 0.4f);
            RenderSettings.ambientEquatorColor = new Color(0.2f, 0.2f, 0.2f);
            RenderSettings.ambientGroundColor = new Color(0.2f, 0.2f, 0.2f);
        }
    }

    void ConfigurePostProcessing()
    {
        if (enablePostProcessing)
        {
            // This would integrate with Unity's Post-Processing Stack
            // Configure exposure, bloom, color grading, etc.
        }
    }
}
```

## ROS TCP Connector Integration

### Setting Up ROS Communication

The ROS TCP Connector allows Unity to communicate with ROS 2 nodes:

```csharp
// ROSConnector.cs
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class ROSConnector : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    public string laserScanTopic = "/scan";

    private ROSConnection ros;
    private float publishRate = 0.1f; // 10 Hz

    void Start()
    {
        ConnectToROS();
        SubscribeToTopics();
    }

    void ConnectToROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);
        Debug.Log($"Connected to ROS at {rosIPAddress}:{rosPort}");
    }

    void SubscribeToTopics()
    {
        // Subscribe to joint states
        ros.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.JointStateMsg>(
            jointStateTopic, OnJointStateReceived
        );

        // Subscribe to velocity commands
        ros.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TwistMsg>(
            cmdVelTopic, OnCmdVelReceived
        );
    }

    void OnJointStateReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.JointStateMsg jointState)
    {
        // Update robot model based on received joint states
        UpdateRobotJoints(jointState);
    }

    void OnCmdVelReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TwistMsg cmdVel)
    {
        // Process velocity commands
        ProcessVelocityCommand(cmdVel);
    }

    void UpdateRobotJoints(Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.JointStateMsg jointState)
    {
        RobotModel robot = GetComponent<RobotModel>();
        if (robot != null)
        {
            for (int i = 0; i < jointState.name.Count; i++)
            {
                string jointName = jointState.name[i];
                float jointPosition = jointState.position[i];
                robot.SetJointPosition(jointName, jointPosition);
            }
        }
    }

    void ProcessVelocityCommand(Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TwistMsg cmdVel)
    {
        // Process the velocity command
        Debug.Log($"Received velocity command: linear={cmdVel.linear.x}, angular={cmdVel.angular.z}");
    }

    public void PublishJointStates()
    {
        var jointState = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.JointStateMsg();
        jointState.header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.HeaderMsg();
        jointState.header.stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg();
        jointState.header.frame_id = "base_link";

        RobotModel robot = GetComponent<RobotModel>();
        if (robot != null)
        {
            foreach (var joint in robot.joints)
            {
                jointState.name.Add(joint.jointName);
                jointState.position.Add(joint.currentAngle);
                jointState.velocity.Add(0.0f); // Placeholder
                jointState.effort.Add(0.0f);   // Placeholder
            }
        }

        ros.Publish(jointStateTopic, jointState);
    }

    void Update()
    {
        // Publish joint states at specified rate
        if (Time.time % publishRate < Time.deltaTime)
        {
            PublishJointStates();
        }
    }
}
```

### Sensor Simulation in Unity

Implementing sensor simulation within Unity:

```csharp
// SensorSimulation.cs
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class SensorSimulation : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public Transform sensorMountPoint;
    public float sensorRange = 10.0f;
    public int laserRays = 360;
    public float fieldOfView = 60.0f;
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;

    private ROSConnection ros;
    private string laserScanTopic = "/scan";
    private string imageTopic = "/camera/image_raw";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }

    public void SimulateLaserScan()
    {
        var laserScan = new LaserScanMsg();
        laserScan.header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.HeaderMsg();
        laserScan.header.stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg();
        laserScan.header.frame_id = "laser_frame";

        laserScan.angle_min = -Mathf.PI;
        laserScan.angle_max = Mathf.PI;
        laserScan.angle_increment = (2 * Mathf.PI) / laserRays;
        laserScan.time_increment = 0.0f;
        laserScan.scan_time = 0.1f;
        laserScan.range_min = 0.1f;
        laserScan.range_max = sensorRange;

        // Simulate laser range measurements
        List<float> ranges = new List<float>();
        for (int i = 0; i < laserRays; i++)
        {
            float angle = laserScan.angle_min + i * laserScan.angle_increment;
            float distance = SimulateLaserRay(angle);
            ranges.Add(distance);
        }

        laserScan.ranges = ranges.ToArray();
        ros.Publish(laserScanTopic, laserScan);
    }

    float SimulateLaserRay(float angle)
    {
        // Perform raycast in the specified direction
        Vector3 direction = new Vector3(
            Mathf.Cos(angle),
            0,
            Mathf.Sin(angle)
        );

        RaycastHit hit;
        if (Physics.Raycast(sensorMountPoint.position,
                           sensorMountPoint.TransformDirection(direction),
                           out hit, sensorRange))
        {
            return hit.distance;
        }
        else
        {
            return float.PositiveInfinity; // No obstacle detected
        }
    }

    public void SimulateCameraImage()
    {
        // This would capture the camera view and convert to ROS image message
        // Implementation depends on Unity's camera capture capabilities
    }

    void Update()
    {
        // Simulate sensors at appropriate rates
        if (Time.frameCount % 10 == 0) // 10 Hz for laser scan
        {
            SimulateLaserScan();
        }
    }
}
```

## Advanced Unity Robotics Features

### Unity Perception Package

The Unity Perception package enables synthetic data generation:

```csharp
// PerceptionCamera.cs
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.Labeling;

public class PerceptionCamera : MonoBehaviour
{
    [Header("Perception Configuration")]
    public bool enableSemanticSegmentation = true;
    public bool enableInstanceSegmentation = true;
    public bool enableBoundingBoxes = true;

    [Header("Camera Settings")]
    public Camera perceptionCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float cameraFov = 60.0f;

    void Start()
    {
        SetupPerceptionCamera();
    }

    void SetupPerceptionCamera()
    {
        if (perceptionCamera == null)
        {
            perceptionCamera = GetComponent<Camera>();
        }

        if (perceptionCamera != null)
        {
            perceptionCamera.fieldOfView = cameraFov;
            perceptionCamera.targetTexture =
                new RenderTexture(imageWidth, imageHeight, 24);

            // Enable perception features
            if (enableSemanticSegmentation)
            {
                var semanticSegmentation = perceptionCamera.gameObject
                    .AddComponent<SemanticSegmentationLabeler>();
            }

            if (enableInstanceSegmentation)
            {
                var instanceSegmentation = perceptionCamera.gameObject
                    .AddComponent<InstanceSegmentationLabeler>();
            }

            if (enableBoundingBoxes)
            {
                var boundingBox = perceptionCamera.gameObject
                    .AddComponent<BoundingBoxLabeler>();
            }
        }
    }
}
```

### ML-Agents Integration

For reinforcement learning applications:

```csharp
// RobotAgent.cs
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class RobotAgent : Agent
{
    [Header("Robot Configuration")]
    public Transform target;
    public float moveSpeed = 5.0f;
    public float rotationSpeed = 100.0f;

    [Header("Reward Settings")]
    public float reachTargetReward = 10.0f;
    public float stepPenalty = -0.1f;
    public float distanceRewardMultiplier = 0.1f;

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot position
        transform.position = new Vector3(
            Random.Range(-5f, 5f),
            0.5f,
            Random.Range(-5f, 5f)
        );

        // Set new random target
        target.position = new Vector3(
            Random.Range(-8f, 8f),
            0.5f,
            Random.Range(-8f, 8f)
        );
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add robot position relative to target
        sensor.AddObservation((target.position - transform.position) / 20f);

        // Add robot rotation
        sensor.AddObservation(transform.rotation.eulerAngles.y / 180f);

        // Add distance to target
        sensor.AddObservation(Vector3.Distance(transform.position, target.position) / 20f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Process actions
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        float rotate = actions.ContinuousActions[2];

        // Apply movement
        Vector3 moveDirection = new Vector3(moveX, 0, moveZ).normalized;
        rb.velocity = new Vector3(moveDirection.x * moveSpeed, rb.velocity.y, moveDirection.z * moveSpeed);

        // Apply rotation
        transform.Rotate(Vector3.up, rotate * rotationSpeed * Time.deltaTime);

        // Calculate reward
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        SetReward(stepPenalty);

        // Bonus for getting closer to target
        AddReward(distanceRewardMultiplier / (distanceToTarget + 1f));

        // Check if target reached
        if (distanceToTarget < 1.0f)
        {
            SetReward(reachTargetReward);
            EndEpisode();
        }

        // End episode if too far from target
        if (distanceToTarget > 20f)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
        continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? -1f :
                                 Input.GetKey(KeyCode.E) ? 1f : 0f;
    }
}
```

## Integration with ROS 2 Workflows

### Launch Files for Unity Integration

Creating ROS 2 launch files that work with Unity:

```python
# launch/unity_robot_simulation.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    unity_ip = LaunchConfiguration('unity_ip', default='127.0.0.1')
    unity_port = LaunchConfiguration('unity_port', default='10000')

    # Package directories
    pkg_robot_description = get_package_share_directory('my_robot_description')
    pkg_unity_bridge = get_package_share_directory('unity_ros2_bridge')

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(
                os.path.join(pkg_robot_description, 'urdf', 'my_robot.urdf')
            ).read()
        }]
    )

    # Unity ROS bridge node
    unity_bridge = Node(
        package='unity_ros2_bridge',
        executable='unity_bridge_node',
        name='unity_bridge',
        parameters=[
            {'unity_ip': unity_ip},
            {'unity_port': unity_port},
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            os.path.join(pkg_robot_description, 'config', 'my_robot_controllers.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Spawn controllers
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['position_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        robot_state_publisher,
        unity_bridge,
        joint_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        robot_controller_spawner
    ])
```

### Custom ROS Messages for Unity

Creating custom messages for Unity-specific data:

```python
# msg/UnityTransform.msg
# Custom message for Unity-specific transform data
Header header
string object_name
geometry_msgs/Transform transform
float32 confidence
string status
```

### Unity-Specific ROS Node

Creating a ROS node that manages Unity communication:

```python
# unity_manager.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState, LaserScan
from tf2_ros import TransformBroadcaster
import socket
import json
import threading
import time

class UnityManager(Node):
    def __init__(self):
        super().__init__('unity_manager')

        # ROS publishers and subscribers
        self.unity_status_pub = self.create_publisher(String, '/unity_status', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/unity_joint_states', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/unity_scan', 10)

        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.joint_cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self.joint_cmd_callback, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Unity communication setup
        self.unity_ip = '127.0.0.1'
        self.unity_port = 10000
        self.unity_socket = None
        self.connect_to_unity()

        # Timers
        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.sensor_timer = self.create_timer(0.1, self.publish_sensor_data)

        # State variables
        self.cmd_vel = Twist()
        self.joint_commands = {}
        self.unity_connected = False

        self.get_logger().info('Unity Manager initialized')

    def connect_to_unity(self):
        """Connect to Unity via TCP"""
        try:
            self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.unity_socket.connect((self.unity_ip, self.unity_port))
            self.unity_connected = True
            self.get_logger().info(f'Connected to Unity at {self.unity_ip}:{self.unity_port}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Unity: {e}')
            self.unity_connected = False

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.cmd_vel = msg
        if self.unity_connected:
            self.send_command_to_unity('velocity', {
                'linear_x': msg.linear.x,
                'linear_y': msg.linear.y,
                'linear_z': msg.linear.z,
                'angular_x': msg.angular.x,
                'angular_y': msg.angular.y,
                'angular_z': msg.angular.z
            })

    def joint_cmd_callback(self, msg):
        """Handle joint commands"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_commands[name] = msg.position[i]

        if self.unity_connected:
            self.send_command_to_unity('joint_positions', dict(zip(msg.name, msg.position)))

    def send_command_to_unity(self, command_type, data):
        """Send command to Unity"""
        if not self.unity_connected:
            return

        message = {
            'type': command_type,
            'data': data,
            'timestamp': time.time()
        }

        try:
            json_message = json.dumps(message) + '\n'
            self.unity_socket.send(json_message.encode())
        except Exception as e:
            self.get_logger().error(f'Failed to send command to Unity: {e}')
            self.unity_connected = False

    def publish_status(self):
        """Publish Unity connection status"""
        status_msg = String()
        status_msg.data = 'connected' if self.unity_connected else 'disconnected'
        self.unity_status_pub.publish(status_msg)

    def publish_sensor_data(self):
        """Publish sensor data from Unity simulation"""
        if not self.unity_connected:
            return

        # Simulate receiving sensor data from Unity
        # In a real implementation, this would receive data from Unity
        self.simulate_sensor_data()

    def simulate_sensor_data(self):
        """Simulate sensor data for demonstration"""
        # Publish joint states
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ['joint1', 'joint2', 'joint3']
        joint_state.position = [0.0, 0.0, 0.0]  # Placeholder values
        joint_state.velocity = [0.0, 0.0, 0.0]
        joint_state.effort = [0.0, 0.0, 0.0]
        self.joint_state_pub.publish(joint_state)

        # Publish laser scan
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'laser_frame'
        scan.angle_min = -3.14
        scan.angle_max = 3.14
        scan.angle_increment = 0.0174533  # 1 degree
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [5.0] * 360  # Placeholder values
        self.scan_pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = UnityManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Unity manager stopped by user')
    finally:
        if node.unity_socket:
            node.unity_socket.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Unity Performance Tips

1. **Object Pooling**: Reuse objects instead of creating/destroying them frequently
2. **LOD Systems**: Use Level of Detail to reduce rendering complexity at distance
3. **Occlusion Culling**: Don't render objects that aren't visible
4. **Shader Optimization**: Use efficient shaders for real-time performance
5. **Physics Optimization**: Simplify collision meshes and adjust physics settings

### Network Optimization

For ROS-TCP communication:

1. **Message Batching**: Combine multiple small messages into larger ones
2. **Compression**: Use compression for large data like images
3. **Frequency Control**: Adjust publishing rates based on requirements
4. **Connection Management**: Handle connection failures gracefully

## Debugging and Troubleshooting

### Common Unity-ROS Integration Issues

1. **Connection Failures**:
   - Check IP addresses and ports
   - Verify firewall settings
   - Ensure both Unity and ROS nodes are running

2. **Synchronization Issues**:
   - Use appropriate time synchronization
   - Handle message ordering properly
   - Implement proper error handling

3. **Performance Problems**:
   - Monitor frame rates in Unity
   - Check network bandwidth usage
   - Optimize message publishing rates

### Debugging Tools

```csharp
// UnityDebugManager.cs
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;

public class UnityDebugManager : MonoBehaviour
{
    [Header("Debug Settings")]
    public bool showDebugInfo = true;
    public bool logROSConnection = true;
    public bool visualizeSensors = true;

    private GUIStyle debugStyle;
    private List<string> debugMessages = new List<string>();
    private float lastMessageTime;

    void Start()
    {
        debugStyle = new GUIStyle();
        debugStyle.normal.textColor = Color.white;
        debugStyle.fontSize = 14;
        debugStyle.alignment = TextAnchor.UpperLeft;
    }

    void OnGUI()
    {
        if (!showDebugInfo) return;

        string debugText = "Unity-ROS Debug Info:\n";
        debugText += $"ROS Connected: {IsROSConnected()}\n";
        debugText += $"Frame Rate: {GetFrameRate()}\n";
        debugText += $"Time Scale: {Time.timeScale}\n";

        // Add recent messages
        foreach (string msg in debugMessages)
        {
            debugText += msg + "\n";
        }

        GUI.Label(new Rect(10, 10, 400, 300), debugText, debugStyle);
    }

    bool IsROSConnected()
    {
        try
        {
            var ros = ROSConnection.GetOrCreateInstance();
            return ros != null;
        }
        catch
        {
            return false;
        }
    }

    float GetFrameRate()
    {
        return 1.0f / Time.deltaTime;
    }

    public void LogDebugMessage(string message)
    {
        string timestampedMessage = $"[{Time.time:0.00}] {message}";
        debugMessages.Add(timestampedMessage);

        // Keep only recent messages
        if (debugMessages.Count > 10)
        {
            debugMessages.RemoveAt(0);
        }

        lastMessageTime = Time.time;

        if (logROSConnection)
        {
            Debug.Log(timestampedMessage);
        }
    }

    void Update()
    {
        // Clean up old messages
        if (Time.time - lastMessageTime > 5.0f && debugMessages.Count > 0)
        {
            debugMessages.RemoveAt(0);
        }
    }
}
```

## Best Practices

### Unity for Robotics Best Practices

1. **Modular Design**: Create reusable components for different robot types
2. **Configuration-Driven**: Use scriptable objects for robot configurations
3. **Performance First**: Optimize for real-time performance from the start
4. **Testing Integration**: Implement automated testing for Unity-ROS integration
5. **Version Control**: Use appropriate version control for Unity assets

### Integration Best Practices

1. **Loose Coupling**: Minimize dependencies between Unity and ROS systems
2. **Error Handling**: Implement robust error handling for network communication
3. **State Management**: Properly manage state between Unity and ROS
4. **Data Validation**: Validate all data passed between systems
5. **Documentation**: Maintain clear documentation of integration points

## Exercises

1. **Unity Environment**: Create a Unity scene with a simple robot model and environment
2. **ROS Connection**: Implement ROS TCP Connector integration with basic communication
3. **Sensor Simulation**: Add laser scan simulation and publish to ROS
4. **Robot Control**: Implement basic robot movement controlled via ROS commands
5. **Performance Optimization**: Optimize your Unity scene for real-time performance

## Code Example: Complete Unity-ROS Integration

```csharp
// CompleteUnityROSIntegration.cs
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;

public class CompleteUnityROSIntegration : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public float publishRate = 0.1f; // 10 Hz

    [Header("Robot Configuration")]
    public Transform robotRoot;
    public List<JointInfo> joints = new List<JointInfo>();
    public Transform laserSensorMount;

    [Header("Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    public string laserScanTopic = "/scan";
    public string tfTopic = "/tf";

    private ROSConnection ros;
    private float lastPublishTime = 0f;
    private Vector3 robotVelocity = Vector3.zero;
    private float robotAngularVelocity = 0f;

    [System.Serializable]
    public class JointInfo
    {
        public string jointName;
        public Transform jointTransform;
        public float currentAngle;
        public float minAngle = -180f;
        public float maxAngle = 180f;
    }

    void Start()
    {
        InitializeROSConnection();
        SubscribeToROSTopics();
        InitializeRobot();
    }

    void InitializeROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);
        Debug.Log($"ROS Connection initialized: {rosIPAddress}:{rosPort}");
    }

    void SubscribeToROSTopics()
    {
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);
    }

    void InitializeRobot()
    {
        // Initialize joint angles
        foreach (var joint in joints)
        {
            joint.currentAngle = joint.jointTransform.localEulerAngles.y;
        }
    }

    void OnCmdVelReceived(TwistMsg cmdVel)
    {
        // Convert ROS velocity command to Unity movement
        robotVelocity = new Vector3(
            (float)cmdVel.linear.x,
            0f,
            (float)cmdVel.linear.y
        ) * Time.deltaTime;

        robotAngularVelocity = (float)cmdVel.angular.z * Time.deltaTime;

        // Apply movement to robot
        ApplyRobotMovement();
    }

    void ApplyRobotMovement()
    {
        // Apply linear velocity
        robotRoot.Translate(robotVelocity, Space.World);

        // Apply angular velocity
        robotRoot.Rotate(Vector3.up, robotAngularVelocity, Space.World);
    }

    void Update()
    {
        // Update joint angles based on transforms
        UpdateJointAngles();

        // Publish data at specified rate
        if (Time.time - lastPublishTime >= publishRate)
        {
            PublishJointStates();
            PublishLaserScan();
            lastPublishTime = Time.time;
        }
    }

    void UpdateJointAngles()
    {
        foreach (var joint in joints)
        {
            joint.currentAngle = joint.jointTransform.localEulerAngles.y;
        }
    }

    void PublishJointStates()
    {
        var jointState = new JointStateMsg();
        jointState.header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.HeaderMsg();
        jointState.header.stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg();
        jointState.header.frame_id = "base_link";

        foreach (var joint in joints)
        {
            jointState.name.Add(joint.jointName);
            jointState.position.Add(joint.currentAngle * Mathf.Deg2Rad);
            jointState.velocity.Add(0.0f); // Calculate from previous positions
            jointState.effort.Add(0.0f);
        }

        ros.Publish(jointStateTopic, jointState);
    }

    void PublishLaserScan()
    {
        var laserScan = new LaserScanMsg();
        laserScan.header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.HeaderMsg();
        laserScan.header.stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg();
        laserScan.header.frame_id = "laser_frame";

        laserScan.angle_min = -Mathf.PI;
        laserScan.angle_max = Mathf.PI;
        laserScan.angle_increment = (2 * Mathf.PI) / 360f; // 360 rays
        laserScan.time_increment = 0.0f;
        laserScan.scan_time = publishRate;
        laserScan.range_min = 0.1f;
        laserScan.range_max = 10.0f;

        // Simulate 360-degree laser scan
        List<float> ranges = new List<float>();
        for (int i = 0; i < 360; i++)
        {
            float angle = laserScan.angle_min + i * laserScan.angle_increment;
            float distance = SimulateLaserRay(angle);
            ranges.Add(distance);
        }

        laserScan.ranges = ranges.ToArray();
        ros.Publish(laserScanTopic, laserScan);
    }

    float SimulateLaserRay(float angle)
    {
        // Create direction vector in the laser's local space
        Vector3 direction = new Vector3(
            Mathf.Cos(angle),
            0f,
            Mathf.Sin(angle)
        );

        // Perform raycast in world space
        Vector3 worldDirection = laserSensorMount.TransformDirection(direction);
        RaycastHit hit;

        if (Physics.Raycast(laserSensorMount.position, worldDirection, out hit, 10f))
        {
            return hit.distance;
        }
        else
        {
            return laserScan.range_max; // No obstacle detected
        }
    }

    void OnValidate()
    {
        // Validate configuration in editor
        if (publishRate <= 0)
        {
            publishRate = 0.1f;
        }
    }
}
```

## Ethical Considerations

When developing Unity-ROS integration:

- **Data Privacy**: Be mindful of data generated and processed in simulation
- **Accessibility**: Ensure tools are accessible to users with different abilities
- **Environmental Impact**: Consider computational resource usage
- **Transparency**: Maintain clear documentation of system capabilities and limitations

## Summary

In this week, we've covered:

- Unity's role in robotics simulation and visualization
- Setting up Unity with ROS 2 integration tools
- Creating 3D environments and robot models in Unity
- Implementing ROS TCP Connector communication
- Developing custom Unity scripts for robot control and sensor simulation
- Advanced features like Unity Perception and ML-Agents
- Integration with existing ROS 2 workflows
- Performance optimization and debugging techniques
- Best practices for Unity-ROS integration

## References

1. Unity Robotics GitHub Repository. (2023). Retrieved from https://github.com/Unity-Technologies/Unity-Robotics-Hub
2. ROS TCP Connector Documentation. (2023). Unity Technologies.
3. Unity ML-Agents Toolkit. (2023). Unity Technologies.
4. Perception Package Documentation. (2023). Unity Technologies.

---

**Next Week**: [NVIDIA Isaac Overview](../module-3/week-8.md)