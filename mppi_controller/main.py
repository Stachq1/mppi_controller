import rclpy
from mppi_controller import MPPIController

def main(args=None):
    rclpy.init(args=args)
    node = MPPIController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
