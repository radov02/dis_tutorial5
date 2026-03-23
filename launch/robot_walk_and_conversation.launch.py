from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Your existing robot simulation nodes
        # Node(package='gazebo_ros', ...),
        # Node(package='rviz2', ...),

        # 2. Your LLM bridge node
        Node(
            package='dis_tutorial5',
            executable='LLM.py',
            name='llm_node',
            output='screen'
        ),
        
        # 3. Your voice capture node
        Node(
            package='dis_tutorial5',
            executable='voice_capture',
            name='voice_node',
            output='screen'
        )
    ])