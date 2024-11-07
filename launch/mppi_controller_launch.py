import launch
from launch import LaunchDescription
from launch.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node for the MPPIController
        Node(
            package='mppi_controller',
            executable='controller_and_disturbance_node',
            name='mppi_controller',
            output='screen',
            parameters=[]
        ),
        # Node for the DisturbanceInjector
        Node(
            package='mppi_controller',
            executable='controller_and_disturbance_node',
            name='disturbance_injector',
            output='screen',
            parameters=[]
        ),
    ])
