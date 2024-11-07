from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np

class DisturbanceInjector(Node):
    def __init__(self):
        super().__init__('disturbance_injector')
        self.get_logger().info('DisturbanceInjector node has started')
        self.publisher_ = self.create_publisher(Pose, '/disturbed_state', 10)
        self.subscriber_ = self.create_subscription(
            Pose,
            '/predicted_state',
            self.add_disturbance,
            10
        )

        # Initial position (x, y) and orientation (theta, which will be converted to quaternion)
        self.current_state = np.array([0.0, 0.0, 0.0])  # Position (x, y) and orientation (theta)

    def add_disturbance(self, msg):
        # Update the current state based on the predicted pose (position and orientation)
        self.current_state = np.array([msg.position.x, msg.position.y, self.quaternion_to_theta(msg.orientation)])

        # Apply random disturbance to position (x, y)
        disturbance = np.random.normal(0, 0.1, size=self.current_state.shape)
        disturbed_position = self.current_state + disturbance

        # Create a Pose message and set the disturbed position
        pose_msg = Pose()
        pose_msg.position.x = disturbed_position[0]
        pose_msg.position.y = disturbed_position[1]
        pose_msg.position.z = 0.0

        # Set orientation (theta as quaternion, for 2D just use cos/sin for the rotation around z-axis)
        theta = self.current_state[2]
        # Convert theta to quaternion (for simplicity, assume no roll/pitch in 2D)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = np.sin(theta / 2)  # 2D orientation as z-component
        pose_msg.orientation.w = np.cos(theta / 2)  # 2D orientation as w-component

        # Publish the disturbed pose
        self.publisher_.publish(pose_msg)
