from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Pose, Point, Twist, Vector3
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

class MPPIController(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self.get_logger().info('MPPIController node has started')
        self.twist_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher_ = self.create_publisher(Marker, '/visualization_marker', 10)
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

    def cost_function(self, trajectory, goal, obstacles, control_effort_weight=0.05):
        goal_cost = np.linalg.norm(trajectory[-1][:2] - goal[:2])
        obstacle_cost = sum(np.sum(np.exp(-np.linalg.norm(trajectory[:, :2] - obs[:2], axis=1))) for obs in obstacles)
        control_cost = control_effort_weight * np.sum(np.square(trajectory[1:] - trajectory[:-1]))
        return goal_cost + obstacle_cost + control_cost

    def select_best_trajectory(self, trajectories, controls, goal, obstacles):
        costs = [self.cost_function(traj, goal, obstacles) for traj in trajectories]
        best_index = np.argmin(costs)
        return trajectories[best_index], controls[best_index]

    def visualize_robot_and_goal(self):
        # Create and initialize the Marker for the robot (a small sphere)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        # Visualize the robot as a red sphere
        robot_marker = Marker(
            header=header,
            ns="robot",
            id=0,
            type=Marker.SPHERE,
            action=Marker.ADD,
            pose=Pose(position=Point(x=self.current_state[0], y=self.current_state[1], z=0.0)),
            scale=Vector3(x=0.1, y=0.1, z=0.1),  # Size of the robot
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red color, full opacity
        )

        # Visualize the goal as a blue sphere (or any other color/shape)
        goal_marker = Marker(
            header=header,
            ns="goal",
            id=1,
            type=Marker.SPHERE,
            action=Marker.ADD,
            pose=Pose(position=Point(x=self.goal_state[0], y=self.goal_state[1], z=0.0)),  # Goal position
            scale=Vector3(x=0.1, y=0.1, z=0.1),  # Size of the goal
            color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue color, full opacity
        )

        # Publish both the robot and goal markers
        self.marker_publisher_.publish(robot_marker)
        self.marker_publisher_.publish(goal_marker)

    def visualize_trajectory(self, trajectory):
        # Create and initialize the Marker for the trajectory (line strip)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        marker = Marker(
            header=header,
            ns="trajectory",
            id=2,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            scale=Vector3(x=0.05, y=0.05, z=0.05),  # Thickness of the line
            color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue color for the trajectory
        )

        # Add points from the trajectory (z = 0 for 2D)
        marker.points = [Point(x=state[0], y=state[1], z=0.0) for state in trajectory]

        # Publish the trajectory marker
        self.marker_publisher_.publish(marker)

    def visualize_obstacles(self):
        # Create and publish a Marker for each obstacle
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        for i, obs in enumerate(self.obstacles):
            marker = Marker(
                header=header,
                ns="obstacle",
                id=i+3,  # Different ID for each obstacle
                type=Marker.SPHERE,
                action=Marker.ADD,
                pose=Pose(position=Point(x=obs[0], y=obs[1], z=0.0)),  # Assuming 2D plane
                scale=Vector3(x=0.2, y=0.2, z=0.2),  # Size of the sphere (obstacle size)
                color=ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # Green color, full opacity
            )

            # Publish the obstacle marker
            self.marker_publisher_.publish(marker)


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

        # Visualize the state
        self.visualize_robot_and_goal()
        self.visualize_trajectory(best_trajectory)
        self.visualize_obstacles()
