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
    - scan map for faces and rings by:
        1) running automatic floor sweep
            - t3: `ros2 run dis_tutorial5 detect_people2.py --ros-args -p map_yaml_path:=/home/erik/rins/maps/task1_blue_demo.yaml -p map_pgm_path:=/home/erik/rins/maps/task1_blue_demo.pgm`
            - t4: `ros2 run dis_tutorial5 detect_rings.py`
            - t5: `ros2 run dis_tutorial5 autonomous_sweep.py`
            - in RViz enable /breadcrumbs Marker, /detected_rings MarkerArray and /people_marker_array MarkerArray
        2) or go manually through map and let robot detect
            - t3: `ros2 run dis_tutorial5 detect_people2.py --ros-args -p map_yaml_path:=/home/erik/rins/maps/task1_blue_demo.yaml -p map_pgm_path:=/home/erik/rins/maps/task1_blue_demo.pgm`
            - t4: `ros2 run dis_tutorial5 detect_rings.py  --ros-args -p world_name:=task1_blue_demo`
            - MOVE ONE OF THE .json FILES INTO FOLDER OF THE OTHER (because not ran at same time, they will go into different folders)
        

    - run robot commander (goes to people and rings - greedily on which is closest at given time) after map, faces and rings have been captured:
        - t1: `cd ~/rins && colcon build --packages-select dis_tutorial5`
        - t1: `source install/setup.bash`
        - t1: `ros2 run rmw_zenoh_cpp rmw_zenohd`
        - t2: `ollama serve` (or check if already running: `ollama list`)
        - t2: `ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py map:=/home/erik/rins/maps/task1_blue_demo.yaml` and use 2D Pose Estimate
        - t3: `cd /home/erik/rins && source install/setup.bash && ros2 run dis_tutorial5 LLM.py`
        - t4: `cd /home/erik/rins && source install/setup.bash && ros2 run dis_tutorial5 voice_capture --ros-args -p piper_model_path:=/home/erik/piper_models/en_US-lessac-medium/en_US-lessac-medium.onnx`
        - optionally in t5 to make one test request: `cd /home/erik/rins && source install/setup.bash && ros2 service call /human_detected robot_interfaces/srv/HumanDetected "{detect_signal: true}"`
        - t5: `ros2 run dis_tutorial5 robot_commander.py --detections-folder /home/erik/rins/maps/20260329_204956`


PROBLEMS:
    - does not go to all targets
    - does not go close enough to persons and rings