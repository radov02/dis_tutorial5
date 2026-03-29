- sourcing:
    - shell script normally runs in a child process, when it exits, all the environment variables it set (like PATH, AMENT_PREFIX_PATH) are thrown away
    - "sourcing" (. script.sh or source script.sh) runs the script in your current shell process, so its variable changes stick
    - that's how `source install/setup.bash` permanently adds ROS 2 paths to your terminal session - without sourcing, ros2 launch wouldn't be able to find any packages

- `/urdf` folder:
    - holds robot's description files that define the robot's links, joints, visuals, collision geometry, inertial properties, sensors, and plugin hooks
    - used by tooling (e.g., robot_state_publisher, joint_state_publisher, Gazebo, RViz, controllers, spawn scripts), which read the URDF to publish transforms, visualize the robot, simulate physics, and configure controllers
    typical workflow is: generate URDF from XACRO: ros2 run xacro xacro path/to/robot.xacro > robot.urdf, then launch robot_state_publisher and rviz2 (or spawn in Gazebo) to view/simulate the robot

- scan the map using SLAM (e.g. `task1_blue_demo`):
    - t1: `ros2 run rmw_zenoh_cpp rmw_zenohd`
    - t2: `ros2 launch dis_tutorial5 sim_turtlebot_slam.launch.py world:=task1_blue_demo`
    - after map has been LiDAR scanned, in t3: `ros2 run nav2_map_server map_saver_cli -f /home/erik/rins/maps/task1_blue_demo`

- run navigation on saved scanned map (e.g. `task1_blue_demo`):
    - t1: `cd ~/rins && colcon build --packages-select dis_tutorial5`
    - t1: `source install/setup.bash`
    - t1: `ros2 run rmw_zenoh_cpp rmw_zenohd`
    - t2: `ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py map:=/home/erik/rins/maps/task1_blue_demo.yaml` (RViz (and map_server) gets the map from given .yaml (and its .pgm), the Gazebo Sim uses map that it gets from the .sdf file, linked inside the `sim.launch.py` (which forwards the file as `gz_args` into simulator launcher); if we want, we can specify the Gazebo Sim's world used as: `ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py world:=task1_blue_demo map:=/home/erik/rins/maps/task1_blue_demo.yaml`)
        - use 2D Pose Estimate in RViz (switch off the Controller, Amcl Particle Swarm, Bumper Hit and LaserScan to get rid of unnecessary processing)
    - run automatic floor sweep (looking for faces and rings):
        - t3: `ros2 run dis_tutorial5 detect_people2.py --ros-args -p map_yaml_path:=/home/erik/rins/maps/task1_blue_demo.yaml -p map_pgm_path:=/home/erik/rins/maps/task1_blue_demo.pgm` (make sure to delete the `people_detections.json` if debugging)
        - t4: `ros2 run dis_tutorial5 detect_rings.py`
        - t5: `ros2 run dis_tutorial5 autonomous_sweep.py`
        - in RViz enable /breadcrumbs Marker, /detected_rings MarkerArray and /people_marker_array MarkerArray

    - run robot commander (goes to people and rings): ...