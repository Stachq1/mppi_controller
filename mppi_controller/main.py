import rclpy
from mppi_controller import MPPIController
from disturbance_injector import DisturbanceInjector

def main(args=None):
    rclpy.init(args=args)

    # Create both nodes
    mppi_node = MPPIController()
    disturbance_node = DisturbanceInjector()

    # Spin the nodes in parallel
    rclpy.spin(mppi_node)
    rclpy.spin(disturbance_node)

    # When finished, destroy the nodes
    mppi_node.destroy_node()
    disturbance_node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
