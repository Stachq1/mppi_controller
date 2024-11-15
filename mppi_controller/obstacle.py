import numpy as np

from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

class Obstacle:
    def __init__(self, start_pos, end_pos, vel, marker_publisher):
        # Initialize the points that the obstacle will move around
        self.start_pos = start_pos
        self.end_pos = end_pos

        # Initialize the current position of the obstacle and its velocity
        self.curr_pos = (start_pos + end_pos) / 2
        self.vel = vel

        # Generate on ID for each obstacle
        self.id = np.random.randint(10000)     # TODO: can get repeated!

        # Initialize the marker publisher
        self.marker_publisher_ = marker_publisher

    def move(self, dt):
        # Move the obstacle based on its velocity
        self.curr_pos += self.vel * dt

        # If the obstacle reaches the start/end position, change its direction
        if np.linalg.norm(self.curr_pos - self.end_pos) < 0.1:
            self.vel *= -1
        elif np.linalg.norm(self.curr_pos - self.start_pos) < 0.1:
            self.vel *= -1

    def visualize_obstacle(self, stamp):
        # Create and publish a Marker for each obstacle
        header = Header()
        header.stamp = stamp
        header.frame_id = "map"

        marker = Marker(
            header=header,
            ns="obstacle",
            id=self.id,
            type=Marker.SPHERE,
            action=Marker.ADD,
            pose=Pose(position=Point(x=self.curr_pos[0], y=self.curr_pos[1], z=0.0)),
            scale=Vector3(x=0.3, y=0.3, z=0.3),
            color=ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        )

        # Publish the obstacle marker
        self.marker_publisher_.publish(marker)