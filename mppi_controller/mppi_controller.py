from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import numpy as np

class MPPIController(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self.get_logger().info('MPPIController node has started')
        self.twist_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_publisher_ = self.create_subscription(Pose, '/predicted_state', 10)
        self.state_subscriber_ = self.create_subscription(
            Pose,
            '/disturbed_state',
            self.update_state,
            10
        )

        self.current_state = np.array([0.0, 0.0, 0.0])                      # Starting position
        self.goal = np.array([5.0, 5.0, 0.0])                               # Goal position
        self.obstacles = [np.array([2.0, 2.0]), np.array([3.0, 4.0])]       # Dynamic obstacles
        self.num_samples = 100
        self.horizon = 10
        self.dt = 0.1

        # Kick-off the controller
        pose = Pose()
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        self.update_state(pose)

    def dynamics(self, state, control):
        x, y, theta = state
        v, w = control
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + w * self.dt
        return np.array([x_new, y_new, theta_new])

    def sample_trajectories(self, num_samples, horizon, state):
        trajectories = []
        controls = []
        for _ in range(num_samples):
            trajectory = [state]
            control_seq = []
            for _ in range(horizon):
                control = np.random.uniform(-1, 1, size=(2,))  # Random control for sampling
                new_state = self.dynamics(trajectory[-1], control)
                trajectory.append(new_state)
                control_seq.append(control)
            trajectories.append(np.array(trajectory))
            controls.append(control_seq)
        return trajectories, controls

    def cost_function(self, trajectory, goal, obstacles, control_effort_weight=0.1):
        goal_cost = np.linalg.norm(trajectory[-1][:2] - goal[:2])
        obstacle_cost = sum(np.exp(-np.linalg.norm(trajectory[:, :2] - obs[:2], axis=1)) for obs in obstacles)
        control_cost = control_effort_weight * np.sum(np.square(trajectory[1:] - trajectory[:-1]))
        return goal_cost + obstacle_cost + control_cost

    def select_best_trajectory(self, trajectories, controls, goal, obstacles):
        costs = [self.cost_function(traj, goal, obstacles) for traj in trajectories]
        best_index = np.argmin(costs)
        return trajectories[best_index], controls[best_index]

    def update_state(self, msg):
        # Update state based on the disturbed Pose (position and orientation)
        self.current_state = np.array([msg.position.x, msg.position.y, self.quaternion_to_theta(msg.orientation)])

        # Sample trajectories
        trajectories, controls = self.sample_trajectories(self.num_samples, self.horizon, self.current_state)
        best_trajectory, best_controls = self.select_best_trajectory(trajectories, controls, self.goal, self.obstacles)

        # Send control command
        cmd = Twist()
        cmd.linear.x = best_controls[0][0]
        cmd.angular.z = best_controls[0][1]
        self.twist_publisher_.publish(cmd)

        # Send updated state
        pose = Pose()
        predicted_pose = self.dynamics(self.current_state, (best_controls[0][0], best_controls[0][1]))
        pose.position.x = predicted_pose[0]
        pose.position.y = predicted_pose[1]
        pose.position.z = 0.0

        # Convert theta to quaternion (for simplicity, assume no roll/pitch in 2D)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = np.sin(self.current_state[2] / 2)  # 2D orientation as z-component
        pose.orientation.w = np.cos(self.current_state[2] / 2)  # 2D orientation as w-component

        self.state_publisher_.publish(pose)

    def quaternion_to_theta(self, quat):
        # Convert quaternion to 2D angle (theta), assuming the rotation is around the z-axis
        return 2 * np.arctan2(quat.z, quat.w)
