from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import numpy as np

class MPPIController(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self.get_logger().info('MPPIController node has started')
        self.twist_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.update_state)

        self.current_state = np.array([0.0, 0.0, 0.0])                      # Starting position
        self.goal = np.array([5.0, 5.0, 0.0])                               # Goal position
        self.obstacles = [np.array([2.0, 2.0]), np.array([3.0, 4.0])]       # Dynamic obstacles
        self.num_samples = 100
        self.horizon = 10
        self.dt = 0.1

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
        obstacle_cost = sum(np.sum(np.exp(-np.linalg.norm(trajectory[:, :2] - obs[:2], axis=1))) for obs in obstacles)
        control_cost = control_effort_weight * np.sum(np.square(trajectory[1:] - trajectory[:-1]))
        return goal_cost + obstacle_cost + control_cost

    def select_best_trajectory(self, trajectories, controls, goal, obstacles):
        costs = [self.cost_function(traj, goal, obstacles) for traj in trajectories]
        best_index = np.argmin(costs)
        return trajectories[best_index], controls[best_index]

    def update_state(self):
        # Sample trajectories
        trajectories, controls = self.sample_trajectories(self.num_samples, self.horizon, self.current_state)
        best_trajectory, best_controls = self.select_best_trajectory(trajectories, controls, self.goal, self.obstacles)

        # Send control command
        cmd = Twist()
        cmd.linear.x = best_controls[0][0]
        cmd.angular.z = best_controls[0][1]
        self.twist_publisher_.publish(cmd)

        # Compute the next state
        self.current_state = self.dynamics(self.current_state, best_controls[0])

        # Apply random disturbance to position (x, y)
        disturbance = np.random.normal(0, 0.05, size=self.current_state.shape)
        self.current_state = self.current_state + disturbance
