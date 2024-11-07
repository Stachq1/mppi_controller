from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np

class MPPIController(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_ = self.create_timer(0.1, self.control_loop)

        self.current_state = np.array([0.0, 0.0, 0.0])                      # Starting position
        self.goal = np.array([5.0, 5.0, 0.0])                               # Goal position
        self.obstacles = [np.array([2.0, 2.0]), np.array([3.0, 4.0])]       # Dynamic obstacles

    def cost_function(self, trajectory, goal, obstacles, control_effort_weight=0.1):
        goal_cost = np.linalg.norm(trajectory[-1][:2] - goal[:2])
        obstacle_cost = sum(np.exp(-np.linalg.norm(trajectory[:, :2] - obs[:2], axis=1)) for obs in obstacles)
        control_cost = control_effort_weight * np.sum(np.square(trajectory[1:] - trajectory[:-1]))
        return goal_cost + obstacle_cost + control_cost


    def select_best_trajectory(self, trajectories, controls, goal, obstacles):
        costs = [self.cost_function(traj, goal, obstacles) for traj in trajectories]
        best_index = np.argmin(costs)
        return trajectories[best_index], controls[best_index]


    def control_loop(self):
        num_samples = 100
        horizon = 10
        dt = 0.1

        # Sample trajectories
        trajectories, controls = self.sample_trajectories(num_samples, horizon, self.current_state, dt)
        best_trajectory, best_controls = self.select_best_trajectory(trajectories, controls, self.goal, self.obstacles)

        # Send first control command
        if best_controls:
            cmd = Twist()
            cmd.linear.x = best_controls[0][0]
            cmd.angular.z = best_controls[0][1]
            self.publisher_.publish(cmd)
