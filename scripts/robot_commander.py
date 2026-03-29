#! /usr/bin/env python3
# Mofidied from Samsung Research America
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


from enum import Enum
import json
import threading
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.duration import Duration as RclpyDuration
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from robot_interfaces.srv import HumanDetected

import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Range, Image, CameraInfo
from std_msgs.msg import ColorRGBA
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped, Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.qos import qos_profile_sensor_data


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

PERCEPTION_RADIUS_M = 0.3          # radius of robot's perception sweep (metres)
GLOBAL_COST_THRESHOLD = 20         # cells with cost strictly below this are candidates
LOCAL_OBSTACLE_THRESHOLD = 65      # local costmap cost above this = unexpected obstacle
VIEWPOINT_GRID_STEP_M = 0.4       # spacing between sampled viewpoints (metres)
RING_DEDUP_DISTANCE_M = 0.5       # merge ring detections closer than this (metres)

# Geofence: closed polygon defining the allowed operating area in map frame.
# The robot will refuse to navigate to any goal outside this polygon.
# Vertices are (x, y) in metres, listed in order (CW or CCW). Adjust to your map.
ALLOWED_AREA_POLYGON: list[tuple[float, float]] = [
    (-4.52, -8.10),
    (-4.48, 0.75),
    (3.07, 0.75),
    (3.02, -8.01),
]

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None

        # ROS2 subscribers
        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)

        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')
        self.human_interaction_client = self.create_client(HumanDetected, 'human_detected')

        timer_frequency = 1
        timer_period = 1/timer_frequency
        self.marker_id = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(Marker, "/breadcrumbs", QoSReliabilityPolicy.BEST_EFFORT)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # ---- Ring detection setup ----
        self._init_ring_detection()

        self.get_logger().info(f"Robot commander has been initialized!")

    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    def _wait_for_future(self, future, timeout=None):
        """Poll a future until done, relying on the background executor thread.

        Drop-in replacement for rclpy.spin_until_future_complete that is safe
        to call from any thread because it never touches the executor directly.
        """
        start = time.time()
        while not future.done():
            if timeout is not None and (time.time() - start) >= timeout:
                break
            time.sleep(0.02)

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        # ---- Geofence: reject goals outside the allowed polygon ----
        gx = pose.pose.position.x
        gy = pose.pose.position.y
        if not self._is_in_allowed_area(gx, gy):
            self.error(
                f'Goal ({gx:.2f}, {gy:.2f}) is OUTSIDE the allowed area '
                f'polygon – refusing to navigate.')
            return False

        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg, self._feedbackCallback)
        self._wait_for_future(send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' + str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = DurationMsg(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        self._wait_for_future(send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        self._wait_for_future(goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        self._wait_for_future(self.undock_result_future, timeout=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            self._wait_for_future(future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        self._wait_for_future(self.result_future, timeout=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            self._wait_for_future(future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return



    # -------------------- Human interaction helpers ------------------------------------

    def trigger_voice_interaction(self, prefetching, timeout_sec=60.0, max_attempts=3):

        for attempt in range(1, max_attempts + 1):

            self.get_logger().info(f'Triggering human interaction service...')
            if not self.human_interaction_client.wait_for_service(timeout_sec=5.0):
                self.warn('Human interaction service is unavailable.')
                return None

            # create the message and call the service:
            req = HumanDetected.Request()
            req.prefetching = prefetching
            future = self.human_interaction_client.call_async(req)
            self._wait_for_future(future, timeout=timeout_sec)

            if not future.done():
                future.cancel()
                self.warn(f'Human interaction service timed out after {timeout_sec:.1f}s.')
                return None

            try:
                response = future.result()
                if response and response.response_text:
                    if prefetching:
                        self.info(f'Received response from human interaction service for prefetched interaction text on attempt {attempt}/{max_attempts}: {response.response_text}')
                    else:
                        self.info(f'Received response from human interaction service for interaction playback confirmed on attempt {attempt}/{max_attempts}: {response.response_text}')
                    return response.response_text
                else:
                    if prefetching:
                        self.warn(f'Human interaction service returned an empty response for interaction prefetch failed on attempt {attempt}/{max_attempts}.')
                    else:
                        self.warn(f'Human interaction service returned an empty response for interaction playback not confirmed on attempt {attempt}/{max_attempts}.')
                    
                    if attempt < max_attempts:
                        time.sleep(0.5)
                        continue
                    else:
                        return None
            except Exception as e:
                self.error(f"Service call failed: {e}")
                return None

    def walk_to_persons_and_greet(self, detections_json_path):
        face_position_in_map_coordinates = []
        try:
            with open(detections_json_path, 'r') as f:
                data = json.load(f)
            face_position_in_map_coordinates = [tuple(d) for d in data]
            self.get_logger().info(f"Loaded {len(face_position_in_map_coordinates)} previous detections from {detections_json_path}")

            for face_tuple in face_position_in_map_coordinates:
                # skip invalid detections
                if any(math.isnan(v) for v in face_tuple[:2]):
                    self.get_logger().warn(f"Skipping detection with invalid coordinates: {face_tuple}")
                    continue

                # prefetch so LLM has the full travel time to respond:
                prefetch_result = [None]
                def run_prefetch():
                    prefetch_result[0] = self.trigger_voice_interaction(prefetching=True)
                prefetch_thread = threading.Thread(target=run_prefetch, daemon=True)
                prefetch_thread.start()

                # Compute yaw so the robot faces toward the person
                cur_x = self.current_pose.pose.position.x if hasattr(self, 'current_pose') else 0.0
                cur_y = self.current_pose.pose.position.y if hasattr(self, 'current_pose') else 0.0
                yaw = math.atan2(face_tuple[1] - cur_y, face_tuple[0] - cur_x)

                # set goal to reach and navigate there:
                face_pose = PoseStamped()
                face_pose.header.frame_id = 'map'
                face_pose.header.stamp = self.get_clock().now().to_msg()
                face_pose.pose.position.x = face_tuple[0]
                face_pose.pose.position.y = face_tuple[1]
                face_pose.pose.orientation = self.YawToQuaternion(yaw)
                self.goToPose(face_pose)

                # wait for this goal to complete before sending the next one
                while not self.isTaskComplete():
                    self.get_logger().info(f"Navigating to face detection [face: {face_tuple}]...")
                    time.sleep(1)

                # ensure prefetch has finished before requesting playback:
                prefetch_thread.join()

                if not prefetch_result[0]:
                    self.get_logger().warn(f"Skipping human [face: {face_tuple}] because prefetch returned no response text.")
                    continue
                
                # TODO: start listening for voice:
                # TODO: voice commands...
                # play voice sound:
                response_text = self.trigger_voice_interaction(prefetching=False)
                if response_text:
                    self.get_logger().info(f"Finished interaction with human [face: {face_tuple}] using response: {response_text}")
                else:
                    self.get_logger().warn(f"Skipping human [face: {face_tuple}] because playback could not be confirmed.")

        except FileNotFoundError:
            self.get_logger().info("No previous detections file found, starting fresh.")
        except Exception as e:
            self.get_logger().warn(f"Could not load detections: {e}")
    


    # -------------------- Autonomous search helpers ------------------------------------

    def _init_autonomous_search(self):
        """Set up state and subscribers needed for autonomous search."""
        self._global_costmap: OccupancyGrid | None = None
        self._local_costmap: OccupancyGrid | None = None
        self._cliff_detected: bool = False
        self._visited_viewpoints: set[tuple[int, int]] = set()

        # Costmap subscribers
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._globalCostmapCallback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._localCostmapCallback, qos_profile_sensor_data)
        # iRobot Create 3 cliff / IR drop sensor (adjust topic if needed)
        self.create_subscription(Range, 'ir_intensity_side_left', self._cliffCallback, qos_profile_sensor_data)

    def _globalCostmapCallback(self, msg: OccupancyGrid):
        self._global_costmap = msg

    def _localCostmapCallback(self, msg: OccupancyGrid):
        self._local_costmap = msg

    def _cliffCallback(self, msg: Range):
        # Range.range < threshold means the sensor is very close to a surface drop.
        # Tune this threshold to match your floor/table edge height.
        DROP_RANGE_THRESHOLD = 0.05   # metres
        self._cliff_detected = msg.range < DROP_RANGE_THRESHOLD
    
    def create_marker(self, point_stamped, marker_id, lifetime=180.0):
        """You can see the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.1
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        marker.lifetime = RclpyDuration(seconds=lifetime).to_msg()

        return marker
    
    def timer_callback(self):
        # Create a PointStamped in the /base_link frame of the robot
        # The point is located 0.1m in from of the robot
        # "Stamped" means that the message type contains a Header
        point_in_robot_frame = PointStamped()
        point_in_robot_frame.header.frame_id = "/base_link"
        point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        point_in_robot_frame.point.x = -0.1
        point_in_robot_frame.point.y = 0.
        point_in_robot_frame.point.z = 0.

        # Now we look up the transform between the base_link and the map frames
        # and then we apply it to our PointStamped
        time_now = rclpy.time.Time()
        timeout = RclpyDuration(seconds=0.1)
        try:
            # An example of how you can get a transform from /base_link frame to the /map frame
            # as it is at time_now, wait for timeout for it to become available
            trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
            self.get_logger().info(f"Looks like the transform is available.")

            # Now we apply the transform to transform the point_in_robot_frame to the map frame
            # The header in the result will be copied from the Header of the transform
            point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
            self.get_logger().info(f"We transformed a PointStamped!")

            # If the transformation exists, create a marker from the point, in order to visualize it in Rviz
            marker_in_map_frame = self.create_marker(point_in_map_frame, self.marker_id)

            # Publish the marker
            self.marker_pub.publish(marker_in_map_frame)
            self.get_logger().info(f"The marker has been published to /breadcrumbs (location: {point_in_map_frame.point}). You are able to visualize it in Rviz")

            # Increase the marker_id, so we dont overwrite the same marker.
            self.marker_id += 1

        except TransformException as te:
            self.get_logger().info(f"Cound not get the transform: {te}")

    def _costmap_to_numpy(self, costmap: OccupancyGrid) -> np.ndarray:
        """Return a 2D int8 array (row, col) from a flat OccupancyGrid data list."""
        w = costmap.info.width
        h = costmap.info.height
        return np.array(costmap.data, dtype=np.int8).reshape((h, w))

    def _world_to_cell(self, costmap: OccupancyGrid, wx: float, wy: float) -> tuple[int, int]:
        """Convert world (x, y) -> (row, col) in the costmap grid."""
        res = costmap.info.resolution
        ox  = costmap.info.origin.position.x
        oy  = costmap.info.origin.position.y
        col = int((wx - ox) / res)
        row = int((wy - oy) / res)
        return row, col

    def _cell_to_world(self, costmap: OccupancyGrid, row: int, col: int) -> tuple[float, float]:
        """Convert (row, col) -> world (x, y) at cell centre."""
        res = costmap.info.resolution
        ox  = costmap.info.origin.position.x
        oy  = costmap.info.origin.position.y
        wx  = ox + (col + 0.5) * res
        wy  = oy + (row + 0.5) * res
        return wx, wy

    def _is_in_allowed_area(self, wx: float, wy: float) -> bool:
        """Ray-casting point-in-polygon test against ALLOWED_AREA_POLYGON.

        Returns True if (wx, wy) is inside the geofence polygon, False otherwise.
        Works for any simple (non-self-intersecting) polygon, CW or CCW.
        """
        poly = ALLOWED_AREA_POLYGON
        n = len(poly)
        if n < 3:
            return True  # degenerate polygon -> no restriction
        inside = False
        px, py = wx, wy
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _is_cliff_safe(self) -> bool:
        if self._cliff_detected:
            self.warn("Cliff/drop sensor active - skipping this viewpoint.")
            return False
        return True

    def _is_local_costmap_clear(self, goal_wx: float, goal_wy: float) -> bool:
        """checks if the local costmap shows any unexpected obstacles near the goal"""
        if self._local_costmap is None:
            self.warn("Local costmap not received yet; assuming clear.")
            return True
        if self._global_costmap is None:
            return True

        local_grid  = self._costmap_to_numpy(self._local_costmap)
        local_h, local_w = local_grid.shape
        global_grid = self._costmap_to_numpy(self._global_costmap)

        # check a small window around the goal in the local costmap:
        radius_in_costmap_grid_cells = max(1, int(math.ceil(PERCEPTION_RADIUS_M / self._local_costmap.info.resolution)))
        goal_local_cell_coords_row, goal_local_cell_coords_col = self._world_to_cell(self._local_costmap, goal_wx, goal_wy)
        
        for drow in range(-radius_in_costmap_grid_cells, radius_in_costmap_grid_cells + 1):
            for dcol in range(-radius_in_costmap_grid_cells, radius_in_costmap_grid_cells + 1):

                if drow**2 + dcol**2 > radius_in_costmap_grid_cells**2:     # if outside the circular perception area, skip
                    continue

                lr, lc = goal_local_cell_coords_row + drow, goal_local_cell_coords_col + dcol
                if not (0 <= lr < local_h and 0 <= lc < local_w):
                    continue

                local_cost = int(local_grid[lr, lc])
                if local_cost < LOCAL_OBSTACLE_THRESHOLD:
                    continue  # this cell is fine

                # high cost in local costmap - is it also high globally?
                grow, gcol = self._world_to_cell(self._global_costmap, *self._cell_to_world(self._local_costmap, lr, lc))
                gh, gw = global_grid.shape
                if 0 <= grow < gh and 0 <= gcol < gw:
                    global_cost = int(global_grid[grow, gcol])
                    if global_cost < GLOBAL_COST_THRESHOLD:
                        # was free globally but costly locally -> new object
                        self.warn(f"Unexpected obstacle in local costmap at ({goal_wx:.2f}, {goal_wy:.2f}) - skipping viewpoint.")
                        return False
        return True

    def _order_viewpoints_by_proximity(self, viewpoints: list[tuple[float, float]], start_x: float, start_y: float) -> list[tuple[float, float]]:
        """greedy nearest-neighbour ordering starting from (start_x, start_y)"""
        remaining = list(viewpoints)
        ordered: list[tuple[float, float]] = []
        cx, cy = start_x, start_y

        while remaining:
            nearest = min(remaining, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            ordered.append(nearest)
            remaining.remove(nearest)
            cx, cy = nearest

        return ordered

    def _sample_candidate_viewpoints(self) -> list[tuple[float, float]]:
        """creates candidate goals for search trajectory"""
        if self._global_costmap is None:
            self.warn("Global costmap not yet received; cannot sample viewpoints.")
            return []
        
        candidates: list[tuple[float, float]] = []

        costmap_2D_grid = self._costmap_to_numpy(self._global_costmap)
        h, w = costmap_2D_grid.shape

        radius_in_costmap_grid_cells = int(math.ceil(PERCEPTION_RADIUS_M / self._global_costmap.info.resolution))   # perception radius in cells
        step_in_costmap_grid_cells = max(1, int(VIEWPOINT_GRID_STEP_M / self._global_costmap.info.resolution))

        # go through valid cells:
        for row in range(radius_in_costmap_grid_cells, h - radius_in_costmap_grid_cells, step_in_costmap_grid_cells):
            for col in range(radius_in_costmap_grid_cells, w - radius_in_costmap_grid_cells, step_in_costmap_grid_cells):

                cost_of_perception_circle = int(costmap_2D_grid[row, col])
                if cost_of_perception_circle < 0 or cost_of_perception_circle >= GLOBAL_COST_THRESHOLD:
                    # This cell is unknown (-1) or too costly - skip.
                    continue

                # quick circular neighbourhood check (vectorised slice for speed) to get valid patch:
                r0 = max(0, row - radius_in_costmap_grid_cells)
                r1 = min(h, row + radius_in_costmap_grid_cells + 1)
                c0 = max(0, col - radius_in_costmap_grid_cells)
                c1 = min(w, col + radius_in_costmap_grid_cells + 1)
                square_patch = costmap_2D_grid[r0:r1, c0:c1].astype(int)

                # build a boolean mask for cells inside the circle
                dr = np.arange(r0, r1) - row
                dc = np.arange(c0, c1) - col
                DC, DR = np.meshgrid(dc, dr)
                inside_circle_bool_mask = (DR**2 + DC**2) <= radius_in_costmap_grid_cells**2

                # if every cell inside the circle is either unknown or costly, skip
                inside_circle_costs = square_patch[inside_circle_bool_mask]
                if not np.any((inside_circle_costs >= 0) & (inside_circle_costs < GLOBAL_COST_THRESHOLD)):
                    continue

                # Reject viewpoints outside the allowed geofence polygon
                wx, wy = self._cell_to_world(self._global_costmap, row, col)
                if not self._is_in_allowed_area(wx, wy):
                    continue

                candidates.append((wx, wy))

        self.info(f"Sampled {len(candidates)} raw candidate viewpoints from global costmap.")
        return candidates


    # -------------------- Ring detection helpers -----------------------------------

    def _init_ring_detection(self):
        """Subscribe to /detected_rings published by the detect_rings node."""
        self.ring_detections = []       # list of (x, y, z, color_name) in map frame

        # Subscribe to ring detections from the detect_rings node
        self.ring_detection_sub = self.create_subscription(
            MarkerArray, "/detected_rings", self._detected_rings_callback,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1))

        self.info("Ring detection subscriber initialised (listening to /detected_rings).")

    def _detected_rings_callback(self, msg: MarkerArray):
        """Sync local ring list with the authoritative MarkerArray from detect_rings.

        detect_rings already handles deduplication and running-average position
        updates, so we simply mirror its current state rather than trying to
        re-add markers individually (which would accumulate duplicates as the
        running-average positions drift slightly across publishes).
        """
        fresh = []
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            color_name = marker.text if marker.text else "unknown"
            if math.isnan(x) or math.isnan(y):
                continue
            if not self._is_in_allowed_area(x, y):
                self.warn(f"Ring at ({x:.2f}, {y:.2f}) outside allowed area – ignoring.")
                continue
            fresh.append((x, y, z, color_name))

        if len(fresh) != len(self.ring_detections):
            self.ring_detections = fresh
            self.info(f"Ring list updated: {len(self.ring_detections)} ring(s) known.")

    # ---- accumulation & persistence ----

    def _add_ring_detection(self, x, y, z, color_name):
        """Store a ring detection, merging duplicates within RING_DEDUP_DISTANCE_M."""
        if math.isnan(x) or math.isnan(y):
            return
        if not self._is_in_allowed_area(x, y):
            self.warn(f"Ring at ({x:.2f}, {y:.2f}) is outside the allowed area polygon – ignoring.")
            return
        for ex in self.ring_detections:
            if math.sqrt((x - ex[0])**2 + (y - ex[1])**2) < RING_DEDUP_DISTANCE_M:
                return  # duplicate
        self.ring_detections.append((x, y, z, color_name))
        self.info(f"New ring detected! colour={color_name}  pos=({x:.2f}, {y:.2f}, {z:.2f})  total={len(self.ring_detections)}")

    def save_ring_detections(self, filepath):
        """Persist ring detections to a JSON file."""
        data = [{"x": x, "y": y, "z": z, "color": color}
                for x, y, z, color in self.ring_detections]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self.info(f"Saved {len(self.ring_detections)} ring detections to {filepath}")

    def walk_to_rings(self, detections_json_path=None):
        """Navigate to every detected (or previously saved) ring."""
        rings = list(self.ring_detections)

        if detections_json_path:
            try:
                with open(detections_json_path, 'r') as f:
                    data = json.load(f)
                rings = [(d["x"], d["y"], d["z"], d["color"]) for d in data]
                self.info(f"Loaded {len(rings)} ring detections from {detections_json_path}")
            except FileNotFoundError:
                self.info("No saved ring detections file found.")
            except Exception as e:
                self.warn(f"Could not load ring detections: {e}")

        if not rings:
            self.info("No ring detections to visit.")
            return

        RING_STANDOFF_M = 0.5  # stop this far from the ring and face it

        self.info(f"Navigating to {len(rings)} detected ring(s)...")
        for i, (x, y, z, color) in enumerate(rings):
            if math.isnan(x) or math.isnan(y):
                self.warn(f"Skipping ring {i+1} with invalid coordinates.")
                continue

            self.info(f"[{i+1}/{len(rings)}] Going to {color} ring at ({x:.2f}, {y:.2f})")

            # Compute standoff point: offset from ring toward current robot position
            cur_x = self.current_pose.pose.position.x if hasattr(self, 'current_pose') else 0.0
            cur_y = self.current_pose.pose.position.y if hasattr(self, 'current_pose') else 0.0
            dx = x - cur_x
            dy = y - cur_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > RING_STANDOFF_M:
                # Place goal RING_STANDOFF_M before the ring, facing it
                goal_x = x - (dx / dist) * RING_STANDOFF_M
                goal_y = y - (dy / dist) * RING_STANDOFF_M
            else:
                goal_x, goal_y = cur_x, cur_y  # already close enough

            # Geofence: skip rings whose approach point is outside the allowed area
            if not self._is_in_allowed_area(goal_x, goal_y):
                self.warn(f"Standoff goal ({goal_x:.2f}, {goal_y:.2f}) for {color} ring is outside allowed area – skipping.")
                continue

            # Yaw to face toward the ring from the standoff point
            yaw = math.atan2(y - goal_y, x - goal_x)

            ring_pose = PoseStamped()
            ring_pose.header.frame_id = 'map'
            ring_pose.header.stamp = self.get_clock().now().to_msg()
            ring_pose.pose.position.x = goal_x
            ring_pose.pose.position.y = goal_y
            ring_pose.pose.orientation = self.YawToQuaternion(yaw)

            self.goToPose(ring_pose)
            while not self.isTaskComplete():
                self.info(f"Navigating to {color} ring...")
                time.sleep(1)

            result = self.getResult()
            if result == TaskResult.SUCCEEDED:
                self.info(f"Reached {color} ring at ({x:.2f}, {y:.2f})")
            else:
                self.warn(f"Failed to reach {color} ring at ({x:.2f}, {y:.2f}): {result}")

        self.info("Finished visiting all rings.")

    def find_people_and_rings_autonomously(self):
        """sweep the robot's perception circle across the global costmap cells that have cost < `GLOBAL_COST_THRESHOLD` 
        by navigating to each candidate viewpoint (skipping those that have cliff or newly placed obstacle)"""
        self.get_logger().info("Starting autonomous search for people and rings...")

        self._init_autonomous_search()

        # wait a moment for costmap messages to arrive after subscribing
        self.info("Waiting for global costmap...")
        deadline = self.get_clock().now().nanoseconds + 10_000_000_000  # 10 s
        while self._global_costmap is None:
            time.sleep(0.5)   # callbacks are processed by the background executor
            if self.get_clock().now().nanoseconds > deadline:
                self.error("Global costmap never received. Aborting autonomous search.")
                return

        # 1. Sample all candidate viewpoints from the global costmap
        viewpoints = self._sample_candidate_viewpoints()
        if not viewpoints:
            self.error("No traversable viewpoints found in global costmap.")
            return

        # 2. Order them greedily from the robot's current position
        start_x = self.current_pose.pose.position.x if hasattr(self, 'current_pose') else 0.0
        start_y = self.current_pose.pose.position.y if hasattr(self, 'current_pose') else 0.0
        ordered = self._order_viewpoints_by_proximity(viewpoints, start_x, start_y)


        self.info(f"Visiting {len(ordered)} viewpoints.")
        found_people: list[tuple[float, float]] = []
        found_rings:  list[tuple[float, float]] = []

        for idx, (wx, wy) in enumerate(ordered):
            self.info(f"[{idx+1}/{len(ordered)}] Viewpoint ({wx:.2f}, {wy:.2f})")

            # 3a. Skip viewpoints too close to the robot's current position.
            # The robot's own body inflates cost in the local costmap without a
            # matching entry in the global costmap, which would incorrectly
            # trigger the "unexpected obstacle" check for nearby goals.
            dist_to_robot = math.sqrt((wx - start_x) ** 2 + (wy - start_y) ** 2)
            if dist_to_robot < 0.5:
                self.info(f"Viewpoint ({wx:.2f}, {wy:.2f}) is only {dist_to_robot:.2f} m away - too close; skipping.")
                continue

            # 3a2. Geofence check (should already be filtered by sampling, but
            #      double-check in case the costmap was updated between sampling
            #      and navigation)
            if not self._is_in_allowed_area(wx, wy):
                self.warn(f"Viewpoint ({wx:.2f}, {wy:.2f}) is outside the allowed area - skipping.")
                continue

            # 3b. Cliff / drop check (current sensor reading) -----------------
            time.sleep(0.05)   # let background executor process incoming messages
            if not self._is_cliff_safe():
                continue

            # 3c. Local costmap obstacle check --------------------------------
            if not self._is_local_costmap_clear(wx, wy):
                continue

            # 4. Navigate to viewpoint ----------------------------------------
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = wx
            goal.pose.position.y = wy
            # Face in the direction of travel (toward goal from current position)
            dx = wx - start_x
            dy = wy - start_y
            yaw = math.atan2(dy, dx)
            goal.pose.orientation = self.YawToQuaternion(yaw)

            if not self.goToPose(goal):
                self.warn(f"Goal ({wx:.2f}, {wy:.2f}) rejected by Nav2; skipping.")
                continue

            while not self.isTaskComplete():
                # Re-check safety while navigating
                time.sleep(0.1)   # callbacks processed by background executor
                if not self._is_cliff_safe() or not self._is_local_costmap_clear(wx, wy):
                    self.warn("Safety condition triggered mid-navigation; cancelling goal.")
                    self.cancelTask()
                    break
                time.sleep(0.1)

            if self.getResult() != TaskResult.SUCCEEDED:
                self.warn(f"Navigation to ({wx:.2f}, {wy:.2f}) did not succeed ({self.getResult()}); skipping.")
                continue

            # Update current position for next greedy step
            start_x, start_y = wx, wy

            # 5. Spin 360° so the full perception circle is scanned
            #self.spin(spin_dist=2 * math.pi, time_allowance=15)
            #while not self.isTaskComplete():
            #    rclpy.spin_once(self, timeout_sec=0.1)
            #    time.sleep(0.1)

            # 6. Ring detections are accumulated automatically via the
            #    detect_rings node publishing to /detected_rings.
            self.info(f"Covered viewpoint ({wx:.2f}, {wy:.2f}) - rings so far: {len(self.ring_detections)}")

        self.info(f"Autonomous search complete. Detected {len(self.ring_detections)} ring(s).")

        # Save ring detections to JSON (same approach as person detections)
        ring_detections_path = '/home/erik/rins/src/dis_tutorial5/ring_detections.json'
        self.save_ring_detections(ring_detections_path)


    #rc.walk_to_persons_and_greet('/home/erik/rins/src/dis_tutorial5/people_detections.json')    # go to all saved persons
    #rc.walk_to_rings('/home/erik/rins/src/dis_tutorial5/ring_detections.json')  # go to all detected rings


def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    # Spin the node continuously in a background thread so that futures can be
    # awaited from any thread (including the prefetch daemon) without competing
    # over the executor.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(rc)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        time.sleep(0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()


    rc.find_people_and_rings_autonomously()  # go through the map autonomously and search for people and rings to save them to JSON




    rc.spin(-0.57)

    rc.destroyNode()

if __name__=="__main__":
    main()