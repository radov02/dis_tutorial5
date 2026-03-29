# Copyright 2023 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='task1_blue_demo',
        choices=['task1_blue_demo', 'task1_green_demo', 'task1_yellow_demo'],
        description='Simulation World (from /home/erik/rins/worlds)'),
    DeclareLaunchArgument('model', default_value='standard', choices=['standard', 'lite'], description='Turtlebot4 Model'),
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false'], description='use_sim_time'),
    DeclareLaunchArgument('map_save_path', default_value='/home/erik/rins/maps/maps',
        description='Path (without extension) where the map will be saved on shutdown as .pgm + .yaml'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0', description=f'{pose_element} component of the robot pose.'))

def generate_launch_description():
    # Directories
    pkg_dis_tutorial5 = get_package_share_directory('dis_tutorial5')
    
    # Launch Files
    gazebo_launch = PathJoinSubstitution([pkg_dis_tutorial5, 'launch', 'sim.launch.py'])
    robot_spawn_launch = PathJoinSubstitution([pkg_dis_tutorial5, 'launch', 'turtlebot4_spawn.launch.py'])
    slam_launch = PathJoinSubstitution([pkg_dis_tutorial5, 'launch', 'slam.launch.py'])

    #Simulator and world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
            ('use_sim_time', LaunchConfiguration('use_sim_time'))
        ]
    )

    #Spawn turtlebot in the world
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_spawn_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('rviz', LaunchConfiguration('rviz')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('world', LaunchConfiguration('world')),
        ]
    )

    # SLAM
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time'))
        ]
    )

    # Save map to disk when Ctrl+C is pressed
    # map_saver_cli reads /map (transient_local QoS) so it gets the last published map
    # use_sim_time:=false because Gazebo is already shutting down
    save_map = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                LogInfo(msg=['Saving map to ', LaunchConfiguration('map_save_path'), ' ...']),
                ExecuteProcess(
                    cmd=[
                        'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                        '-f', LaunchConfiguration('map_save_path'),
                        '--ros-args', '-p', 'use_sim_time:=false'
                    ],
                    output='screen'
                )
            ]
        )
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gazebo)
    ld.add_action(robot_spawn)
    ld.add_action(slam)
    ld.add_action(save_map)
    return ld