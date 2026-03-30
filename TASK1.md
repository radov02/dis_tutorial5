
# Use:

- scan the map using SLAM (e.g. `task1_blue_demo`):
    - you should have the `/worlds` folder in your workspace, where you have the `.sdf` files and mesh directories (because `sim_turtlebot_slam.launch.py` passes `world` to `sim.launch.py`, which sets the `GZ_SIM_RESOURCE_PATH` environment variable to include `/worlds` and passes `gz_args` = `task1_blue_demo.sdf -r -v 4 ...` to Gazebo)
    - t1: `ros2 run rmw_zenoh_cpp rmw_zenohd`
    - t2: `ros2 launch dis_tutorial5 sim_turtlebot_slam.launch.py world:=task1_blue_demo`
    - after map has been LiDAR scanned, in t3: `ros2 run nav2_map_server map_saver_cli -f /home/erik/rins/maps/task1_blue_demo`

- run navigation (for detection and autonomous walk) on saved scanned map (e.g. `task1_blue_demo`):
    - t1: `cd ~/rins && colcon build --packages-select dis_tutorial5`
    - t1: `source install/setup.bash`
    - t1: `ros2 run rmw_zenoh_cpp rmw_zenohd`
    - t2: `ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py map:=/home/erik/rins/maps/task1_blue_demo.yaml` (RViz (and map_server) gets the map from given .yaml (and its .pgm), the Gazebo Sim uses map that it gets from the .sdf file, linked inside the `sim.launch.py` (which forwards the file as `gz_args` into simulator launcher); if we want, we can specify the Gazebo Sim's world used as: `ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py world:=task1_blue_demo map:=/home/erik/rins/maps/task1_blue_demo.yaml`)
        - use 2D Pose Estimate in RViz (switch off the Controller, Amcl Particle Swarm, Bumper Hit and LaserScan to get rid of unnecessary processing)
        
    - scan map for faces and rings by:
        1) running automatic floor sweep
            - t3: `ros2 run dis_tutorial5 detect_people2.py --ros-args -p map_yaml_path:=/home/erik/rins/maps/task1_blue_demo.yaml -p map_pgm_path:=/home/erik/rins/maps/task1_blue_demo.pgm`
            - t4: `ros2 run dis_tutorial5 detect_rings.py --ros-args -p world_name:=task1_blue_demo`
            - t5: `ros2 run dis_tutorial5 autonomous_sweep.py`
            - in RViz enable /breadcrumbs Marker, /detected_rings MarkerArray and /people_marker_array MarkerArray
        2) run half automatic search (define positions on map which robot has to go to):
            - t3: `ros2 run dis_tutorial5 determine_search_points --ros-args -p output_file:=/home/erik/rins/maps/task1_blue_demo_search_positions.json` (note that the navigation stack should be running and publishing `/map` for this to work)
            - when done run:
                - t3: `ros2 run dis_tutorial5 detect_people2.py --ros-args -p map_yaml_path:=/home/erik/rins/maps/task1_blue_demo.yaml -p map_pgm_path:=/home/erik/rins/maps/task1_blue_demo.pgm`
                - t4: `ros2 run dis_tutorial5 detect_rings.py --ros-args -p world_name:=task1_blue_demo`
                - t5: `ros2 run dis_tutorial5 halfautonomous_search --ros-args -p search_positions_file:=/home/erik/rins/maps/task1_blue_demo_search_positions.json`
        3) or go manually through map and let robot detect
            - t3: `ros2 run dis_tutorial5 detect_people2.py --ros-args -p map_yaml_path:=/home/erik/rins/maps/task1_blue_demo.yaml -p map_pgm_path:=/home/erik/rins/maps/task1_blue_demo.pgm`
            - t4: `ros2 run dis_tutorial5 detect_rings.py --ros-args -p world_name:=task1_blue_demo`
        
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

- notes:
    - to remove or adjust spinning when robot reaches every goal, modify the `/home/erik/rins/src/dis_tutorial5/config/nav2.yaml`'s controller_server:ros__parameters:general_goal_checker:yaw_goal_tolerance: 6.28 (if you want zero yaw rotation when reached goal)
    - modify how robot handles goal visiting sequence in `halfautonomous_search.py` - modify how it stores the GoalEntry objects into priority queue
    - if robot is too sensitive to the obstacles (high costmap values), then adjust `COSTMAP_SAFETY_THRESHOLD` variable to e.g. 254


# About:

## World scanning into static map files for navigation
- PRECEPTION - use SLAM (mapping) procedure:
    - robot and world simulator is Gazebo Sim, which requires the `.sdf` file of the map and the world's images, textures
    - to scan the world into map using SLAM with LiDAR we use RViz, where it is simulated and shown
    - we use Gazebo Sim to control the robot in the world and the RViz is connected to Gazebo via bridge messages (`ros_gz_bridge`)
    - once we have walked the robot through our world so that we get good enough map, we save the map using `nav2_map_server` (part of Navigation2 - package of the standard navigation stack in ROS 2) into pair of .yaml (metadata) and .pgm (image) files that represent the static occupancy grid map produced by SLAM
- NAVIGATION - use NAV2 (navigation with a static map):
    - these map files are used by the robot [for localization (ACML), path planning and costmap building] when we are doing NAVIGATION WITH A STATIC MAP!
    - the `nav2_map_server` server publishes the topic `/map` to which the RViz subscribes to display what the robot system is already using
- NOTE: aside from perception and navigation the robot can also manipulate with environment, do decision making, have interaction with humans, communicates via network etc.
- files & folders: 
    - SLAM:
        - `slam.launch.py` runs the SLAM stack on robot for mapping
        - `sim_turtlebot_slam.launch.py` runs the simulation variant - spawns robot and runs SLAM in RViz simulator
    - NAVIGATION:
        - `nav2.launch.py` runs the Nav2 stack (planners, controllers, BT navigator etc.) for autonomous path planning and execution
        - `sim_turtlebot_nav.launch.py` runs the simulation variant - spawns robot and runs NAV2 in RViz simulator
        - other high-level launch configurations
            - TODO...
    - LOCALIZATION:
        - `localization.launch.py` starts the localization-related nodes (map server, AMCL node or NAV2 localization) so the robot can estimate its pose on a known map
    - SIMULATION HELPERS:
        - `turtlebot4_spawn.launch.py` handles spawning the turtlebot into the simulator (loads URDF/xacro, controllers, joint state publishers)
        - `sim.launch.py` generic simulation (world, simulator, robot model) launch description; serves as a base include for other sim-specific launches (nav or SLAM)
    - GEOFENCING:
        - if map has open areas that include drops or not allowed regions, we can fence the robot by using `map_geofence_tool.py`, which sets the ALLOWED_AREA_POLYGON geofence in robot_commander.py to the wanted bounds:
            - to open map plot where you can adjust the geofence: `python3 map_geofence_tool.py --map /home/erik/rins/maps/task1_yellow_demo.yaml --apply`
            - to disable geofencing: `python3 map_geofence_tool.py --turnoff` + rebuild with `colcon build --packages-select dis_tutorial5 --symlink-install` (or to reenable it `python3 map_geofence_tool.py --turnon`)

## LLM for people approach

- after installing the LLM locally, we run its service
- then the RobotCommander will use VoiceNode's service to request and receive the HumanDetected.srv message. VoiceNode will use the LLMNode's service to request and receive the LLMQuery.srv message. LLMNode uses HTTP requests to query the local LLM (on localhost) and HTTP responses are received
- files & folders: 
    - `LLM.py` (LLMNode class): spins LLMNode instance, which initializes LLM's parameters and URL and creates service that receives LLMQuery prompt and thus triggers the `_handle_query` method, which sends it to the LLM via HTTP request and when received returns it as response string in LLMQuery.srv message
    - `voice_capture.py` (VoiceNode class): spins the VoiceNode instance, which initializes default greeting and loads the Piper voice model; in `_speak` method it uses the voice file and plays the sound of provided text; the `_handle_human_detected` method is callback for when request comes (when human has just been detected), it then calls the helper `_get_llm_response`, which awaits for service of the LLM server and makes async request to it and waits for response with some timeout, it returns LLM's response and `_handle_human_detected` callback returns the response as string in HumanDetected.srv message
    - `robot_commander.py` (RobotCommander class): TODO...

## Face detection

- files & folders: 

## Ring detection

- files & folders: 

## Search for faces and rings

- files & folders: 

## Other

- sourcing:
    - shell script normally runs in a child process, when it exits, all the environment variables it set (like PATH, AMENT_PREFIX_PATH) are thrown away
    - "sourcing" (. script.sh or source script.sh) runs the script in your current shell process, so its variable changes stick
    - that's how `source install/setup.bash` permanently adds ROS 2 paths to your terminal session - without sourcing, ros2 launch wouldn't be able to find any packages

- `/urdf` folder:
    - holds robot's description files that define the robot's links, joints, visuals, collision geometry, inertial properties, sensors, and plugin hooks
    - used by tooling (e.g., robot_state_publisher, joint_state_publisher, Gazebo, RViz, controllers, spawn scripts), which read the URDF to publish transforms, visualize the robot, simulate physics, and configure controllers
    typical workflow is: generate URDF from XACRO: ros2 run xacro xacro path/to/robot.xacro > robot.urdf, then launch robot_state_publisher and rviz2 (or spawn in Gazebo) to view/simulate the robot