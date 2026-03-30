#!/usr/bin/env python3
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

PERCEPTION_RADIUS_M = 0.3          # robot perception radius (m)
GLOBAL_COST_THRESHOLD = 60         # max global costmap cost for traversable cells
LOCAL_OBSTACLE_THRESHOLD = 90      # local costmap cost indicating an obstacle (high to avoid robot footprint inflation FP)
VIEWPOINT_GRID_STEP_M = 0.4       # grid spacing between candidate viewpoints (m)
RING_DEDUP_DISTANCE_M = 0.5       # min distance to merge ring detections (m)
OBSTACLE_LOOKAHEAD_M = 0.6        # how far ahead (m) to scan for new obstacles

# Geofence polygon (map frame, x/y in metres, CW or CCW). Robot won't navigate outside.
GEOFENCE_ENABLED: bool = False
ALLOWED_AREA_POLYGON: list[tuple[float, float]] = [
    (-4.52, -8.10),
    (-4.48, 0.75),
    (3.07, 0.75),
    (3.02, -8.01),
]


class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace='', geofence_enabled=GEOFENCE_ENABLED):
        super().__init__(node_name=node_name, namespace=namespace)
        self._geofence_enabled = geofence_enabled
        self.pose_frame_id = 'map'
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None

        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')
        self.human_interaction_client = self.create_client(HumanDetected, 'human_detected')

        self.marker_id = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(Marker, "/breadcrumbs", QoSReliabilityPolicy.BEST_EFFORT)
        self.timer = self.create_timer(1.0, self.timer_callback)

        self._init_ring_detection()
        self._pending_face_approach: list[tuple[float, float]] = []
        self._pending_ring_inspect: list[tuple[float, float]] = []
        self._inspected_rings: set[tuple[float, float]] = set()
        self._approached_faces: set[tuple[float, float]] = set()
        self.get_logger().info("Robot commander initialized.")

    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    def _wait_for_future(self, future, timeout=None):
        """Block until future is done. Safe to call from any thread."""
        start = time.time()
        while not future.done():
            if timeout is not None and (time.time() - start) >= timeout:
                break
            time.sleep(0.02)

    def goToPose(self, pose, behavior_tree=''):
        """Send a NavigateToPose action goal."""
        gx = pose.pose.position.x
        gy = pose.pose.position.y
        if not self._is_in_allowed_area(gx, gy):
            self.error(f'Goal ({gx:.2f}, {gy:.2f}) is outside the geofence – refusing.')
            return False
        if not self._is_within_costmap_bounds(gx, gy):
            self.warn(f'Goal ({gx:.2f}, {gy:.2f}) is outside costmap bounds – skipping.')
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
        self._wait_for_future(send_goal_future, timeout=10.0)
        self.goal_handle = send_goal_future.result()

        if self.goal_handle is None or not self.goal_handle.accepted:
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
        """Return True if undock action has finished."""
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
        """Return True if the active action has finished (any outcome)."""
        if not self.result_future:
            return True
        self._wait_for_future(self.result_future, timeout=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task failed with status code: {self.status}')
                return True
        else:
            return False  # still in progress

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
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')
        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            future = state_client.call_async(req)
            self._wait_for_future(future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Node state: {state}')
            time.sleep(2)
    
    def YawToQuaternion(self, angle_z=0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)
        return Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])

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


    def _init_autonomous_search(self):
        """Set up state, subscribers, and timers needed for autonomous search."""
        self._global_costmap: OccupancyGrid | None = None
        self._local_costmap: OccupancyGrid | None = None
        self._cliff_detected: bool = False
        self._obstacle_blocked: bool = False
        self._visited_viewpoints: set[tuple[int, int]] = set()
        self._obstacle_debounce: int = 0  # consecutive obstacle detections

        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._globalCostmapCallback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._localCostmapCallback, qos_profile_sensor_data)
        self.create_subscription(Range, 'ir_intensity_side_left', self._cliffCallback, qos_profile_sensor_data)
        self._obstacle_ahead_timer = self.create_timer(0.5, self._check_obstacle_ahead_callback)

    def _globalCostmapCallback(self, msg: OccupancyGrid):
        self._global_costmap = msg

    def _localCostmapCallback(self, msg: OccupancyGrid):
        self._local_costmap = msg

    def _cliffCallback(self, msg: Range):
        DROP_RANGE_THRESHOLD = 0.05  # m – tune to floor/table edge height
        self._cliff_detected = msg.range < DROP_RANGE_THRESHOLD
    
    def create_marker(self, point_stamped, marker_id, lifetime=180.0):
        """Build a small sphere Marker at point_stamped's position."""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = marker_id

        scale = 0.1
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        marker.lifetime = RclpyDuration(seconds=lifetime).to_msg()

        return marker
    
    def timer_callback(self):
        """Publish a breadcrumb marker at the robot's current position (1 Hz)."""
        pt = PointStamped()
        pt.header.frame_id = "/base_link"
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = -0.1
        pt.point.y = 0.
        pt.point.z = 0.
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_link",
                                                    rclpy.time.Time(),
                                                    RclpyDuration(seconds=0.1))
            pt_map = tfg.do_transform_point(pt, trans)
            self.marker_pub.publish(self.create_marker(pt_map, self.marker_id))
            self.marker_id += 1
        except TransformException as te:
            self.get_logger().debug(f"TF not available: {te}")

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
        if not self._geofence_enabled:
            return True
        poly = ALLOWED_AREA_POLYGON
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > wy) != (yj > wy)) and (wx < (xj - xi) * (wy - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _is_within_costmap_bounds(self, wx: float, wy: float) -> bool:
        """Return True if (wx, wy) maps to a valid cell inside the global costmap.
        Prevents 'worldToMap failed' errors when Nav2 receives out-of-bounds goals."""
        if not hasattr(self, '_global_costmap') or self._global_costmap is None:
            return True
        row, col = self._world_to_cell(self._global_costmap, wx, wy)
        h = self._global_costmap.info.height
        w = self._global_costmap.info.width
        return 0 <= row < h and 0 <= col < w

    def _is_cliff_safe(self) -> bool:
        """Return False if the cliff/drop sensor is active."""
        if self._cliff_detected:
            self.warn("Cliff sensor active – skipping viewpoint.")
            return False
        return True

    def _is_local_costmap_clear(self, goal_wx: float, goal_wy: float) -> bool:
        """Return False if the local costmap shows a new obstacle near (goal_wx, goal_wy)
        that wasn't in the global costmap (i.e. dynamically appeared)."""
        if self._local_costmap is None or self._global_costmap is None:
            return True

        local_grid  = self._costmap_to_numpy(self._local_costmap)
        local_h, local_w = local_grid.shape
        global_grid = self._costmap_to_numpy(self._global_costmap)
        gh, gw = global_grid.shape

        r_cells = max(1, int(math.ceil(PERCEPTION_RADIUS_M / self._local_costmap.info.resolution)))
        cr, cc = self._world_to_cell(self._local_costmap, goal_wx, goal_wy)

        for drow in range(-r_cells, r_cells + 1):
            for dcol in range(-r_cells, r_cells + 1):
                if drow**2 + dcol**2 > r_cells**2:
                    continue
                lr, lc = cr + drow, cc + dcol
                if not (0 <= lr < local_h and 0 <= lc < local_w):
                    continue
                if int(local_grid[lr, lc]) < LOCAL_OBSTACLE_THRESHOLD:
                    continue
                # high local cost – check if it was free globally
                grow, gcol = self._world_to_cell(
                    self._global_costmap,
                    *self._cell_to_world(self._local_costmap, lr, lc))
                if 0 <= grow < gh and 0 <= gcol < gw:
                    if int(global_grid[grow, gcol]) < GLOBAL_COST_THRESHOLD:
                        self.warn(f"New obstacle near ({goal_wx:.2f}, {goal_wy:.2f}) – skipping.")
                        return False
        return True

    def _check_obstacle_ahead_callback(self):
        """Timer callback (2 Hz): cancel active navigation if a new obstacle appears
        in the path directly ahead of the robot.

        Looks OBSTACLE_LOOKAHEAD_M ahead in the robot's heading direction inside the
        local costmap. Any cell that is costly locally but was free globally triggers a
        goal cancellation so Nav2 can replan around the obstacle.
        """
        if self._local_costmap is None or self._global_costmap is None:
            return
        if not hasattr(self, 'current_pose'):
            return
        if self.result_future is None or self.isTaskComplete():
            return  # not currently navigating

        nav_start = getattr(self, '_nav_goal_start_time', None)
        if nav_start is None or (time.time() - nav_start) < 2.0:
            return

        q = self.current_pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y

        local_grid  = self._costmap_to_numpy(self._local_costmap)
        local_h, local_w = local_grid.shape
        global_grid = self._costmap_to_numpy(self._global_costmap)
        gh, gw = global_grid.shape
        res = self._local_costmap.info.resolution

        LOOKAHEAD_START_M = 0.35
        steps = max(1, int(OBSTACLE_LOOKAHEAD_M / res))
        start_step = max(1, int(LOOKAHEAD_START_M / res))
        width_cells = max(1, int(math.ceil((PERCEPTION_RADIUS_M / 2.0) / res)))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        for step in range(start_step, steps + 1):
            wx = rx + cos_yaw * step * res
            wy = ry + sin_yaw * step * res
            cr, cc = self._world_to_cell(self._local_costmap, wx, wy)

            for w in range(-width_cells, width_cells + 1):
                lr = cr + int(round(-sin_yaw * w))
                lc = cc + int(round( cos_yaw * w))
                if not (0 <= lr < local_h and 0 <= lc < local_w):
                    continue
                if int(local_grid[lr, lc]) < LOCAL_OBSTACLE_THRESHOLD:
                    continue
                grow, gcol = self._world_to_cell(
                    self._global_costmap,
                    *self._cell_to_world(self._local_costmap, lr, lc))
                if 0 <= grow < gh and 0 <= gcol < gw:
                    if int(global_grid[grow, gcol]) < GLOBAL_COST_THRESHOLD:
                        self._obstacle_debounce += 1
                        if self._obstacle_debounce < 3:
                            return
                        self.warn(
                            f"New obstacle ahead at ({wx:.2f}, {wy:.2f}) – "
                            "cancelling current goal to replan.")
                        self._obstacle_blocked = True
                        self._obstacle_debounce = 0
                        # Do NOT call cancelTask() here – it blocks the executor thread.
                        if self.goal_handle is not None:
                            self.goal_handle.cancel_goal_async()
                        return
        self._obstacle_debounce = 0

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
        """Return traversable viewpoints sampled on a regular grid from the global costmap."""
        if self._global_costmap is None:
            self.warn("Global costmap not yet received; cannot sample viewpoints.")
            return []

        candidates: list[tuple[float, float]] = []
        grid = self._costmap_to_numpy(self._global_costmap)
        h, w = grid.shape
        r_cells = int(math.ceil(PERCEPTION_RADIUS_M / self._global_costmap.info.resolution))
        step   = max(1, int(VIEWPOINT_GRID_STEP_M / self._global_costmap.info.resolution))

        for row in range(r_cells, h - r_cells, step):
            for col in range(r_cells, w - r_cells, step):
                cost = int(grid[row, col])
                if cost < 0 or cost >= GLOBAL_COST_THRESHOLD:
                    continue

                r0, r1 = max(0, row - r_cells), min(h, row + r_cells + 1)
                c0, c1 = max(0, col - r_cells), min(w, col + r_cells + 1)
                patch = grid[r0:r1, c0:c1].astype(int)
                dr = np.arange(r0, r1) - row
                dc = np.arange(c0, c1) - col
                DC, DR = np.meshgrid(dc, dr)
                mask = (DR**2 + DC**2) <= r_cells**2
                costs_in_circle = patch[mask]
                if not np.any((costs_in_circle >= 0) & (costs_in_circle < GLOBAL_COST_THRESHOLD)):
                    continue

                wx, wy = self._cell_to_world(self._global_costmap, row, col)
                if not self._is_in_allowed_area(wx, wy):
                    continue
                candidates.append((wx, wy))

        self.info(f"Sampled {len(candidates)} candidate viewpoints from global costmap.")
        return candidates


    def _init_ring_detection(self):
        """Subscribe to /detected_rings published by the detect_rings node."""
        self.ring_detections = []

        self.ring_detection_sub = self.create_subscription(
            MarkerArray, "/detected_rings", self._detected_rings_callback,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1))

        self.face_detection_sub = self.create_subscription(
            MarkerArray, "/detected_faces", self._detected_faces_callback,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1))

        self.info("Ring detection subscriber initialised (listening to /detected_rings and /detected_faces).")

    def _detected_rings_callback(self, msg: MarkerArray):
        """Mirror the ring list from detect_rings (dedup happens there)."""
        fresh = []
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            color_name = marker.text if marker.text else "unknown"
            if math.isnan(x) or math.isnan(y):
                continue
            fresh.append((x, y, z, color_name))

        prev_set = set((round(r[0], 2), round(r[1], 2), r[3]) for r in self.ring_detections)
        fresh_set = set((round(r[0], 2), round(r[1], 2), r[3]) for r in fresh)
        new_rounded = fresh_set - prev_set

        if len(fresh) != len(self.ring_detections):
            self.ring_detections = fresh
            self.info(f"Ring list updated: {len(self.ring_detections)} ring(s) known.")

        for nr in new_rounded:
            rx, ry, rcolor = nr
            match = next((r for r in fresh if round(r[0], 2) == rx and round(r[1], 2) == ry and r[3] == rcolor), None)
            if match is None:
                continue
            key = (round(match[0], 1), round(match[1], 1))
            if key in self._inspected_rings:
                continue
            if not any(math.hypot(match[0] - qx, match[1] - qy) < 0.5
                       for qx, qy in self._pending_ring_inspect):
                self._pending_ring_inspect.append((match[0], match[1]))
                self.info(f'Queued ring inspection at ({match[0]:.2f}, {match[1]:.2f})')

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

    def _detected_faces_callback(self, msg: MarkerArray):
        """Queue detected faces for approach by the main navigation loop.
        Expects markers with pose in the map frame (x,y).
        """
        if not msg.markers:
            return
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            if math.isnan(x) or math.isnan(y):
                continue
            key = (round(x, 1), round(y, 1))
            if key in self._approached_faces:
                continue
            # avoid duplicate queue entries
            if any(math.hypot(x - qx, y - qy) < 1.0
                   for qx, qy in self._pending_face_approach):
                continue
            self._pending_face_approach.append((x, y))
            self.info(f'Queued face approach at ({x:.2f}, {y:.2f})')

    def _approach_face(self, x: float, y: float, stop_distance: float = 0.8):
        """Navigate toward a face for better localization.  Called inline from the
        main navigation loop so there are no shared-state race conditions."""
        self._approached_faces.add((round(x, 1), round(y, 1)))
        if not hasattr(self, 'current_pose'):
            self.warn('No current_pose; cannot approach face.')
            return
        rx, ry = self.current_pose.pose.position.x, self.current_pose.pose.position.y
        dx, dy = x - rx, y - ry
        dist = math.hypot(dx, dy)
        if dist <= stop_distance:
            self.info('Already close to detected face.')
            return

        aim_x = x - (dx / dist) * stop_distance
        aim_y = y - (dy / dist) * stop_distance

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = aim_x
        goal.pose.position.y = aim_y
        goal.pose.orientation = self.YawToQuaternion(math.atan2(y - aim_y, x - aim_x))

        self.info(f'Approaching face at ({x:.2f},{y:.2f}) -> ({aim_x:.2f},{aim_y:.2f})')
        if not self.goToPose(goal):
            self.warn('Face-approach goal rejected.')
            return

        deadline = time.time() + 30.0
        while not self.isTaskComplete():
            time.sleep(0.1)
            if time.time() > deadline:
                self.warn('Face approach timed out; cancelling.')
                self.cancelTask()
                break
        time.sleep(2.0)

    def _inspect_ring(self, x: float, y: float, radius: float = 0.8, angle_offset: float = 0.6):
        """Visit two viewpoints (left/right) around a ring to improve its localization.
        Called inline from the main loop."""
        self._inspected_rings.add((round(x, 1), round(y, 1)))
        if not hasattr(self, 'current_pose'):
            self.warn('No current_pose; cannot inspect ring.')
            return
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        base_angle = math.atan2(ry - y, rx - x)

        for ang in [base_angle + angle_offset, base_angle - angle_offset]:
            px = x + math.cos(ang) * radius
            py = y + math.sin(ang) * radius

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = px
            goal.pose.position.y = py
            goal.pose.orientation = self.YawToQuaternion(math.atan2(y - py, x - px))

            self.info(f'Inspecting ring at ({x:.2f},{y:.2f}) from ({px:.2f},{py:.2f})')
            if not self.goToPose(goal):
                self.warn('Ring-inspect goal rejected; skipping viewpoint.')
                continue

            deadline = time.time() + 20.0
            while not self.isTaskComplete():
                time.sleep(0.1)
                if time.time() > deadline:
                    self.warn('Ring inspect timed out; cancelling.')
                    self.cancelTask()
                    break
            time.sleep(2.0)

    def _process_pending_detections(self):
        """Drain the face-approach and ring-inspect queues (called from main loop)."""
        while self._pending_face_approach:
            x, y = self._pending_face_approach.pop(0)
            self.info(f'>>> Processing face approach ({x:.2f}, {y:.2f})')
            self._approach_face(x, y)
        while self._pending_ring_inspect:
            x, y = self._pending_ring_inspect.pop(0)
            self.info(f'>>> Processing ring inspection ({x:.2f}, {y:.2f})')
            self._inspect_ring(x, y)

    def save_ring_detections(self, filepath):
        """Persist ring detections to a JSON file."""
        data = [{"x": x, "y": y, "z": z, "color": color}
                for x, y, z, color in self.ring_detections]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self.info(f"Saved {len(self.ring_detections)} ring detections to {filepath}")

    def find_people_and_rings_autonomously(self):
        """Navigate to all traversable viewpoints in the global costmap to search
        for people and rings.  Viewpoints are dynamically re-sorted by proximity
        to the robot's current position so the nearest unvisited one is always
        chosen next."""
        self.get_logger().info("Starting autonomous search for people and rings...")
        self._init_autonomous_search()

        # Wait for costmap (up to 10 s)
        self.info("Waiting for global costmap...")
        deadline = self.get_clock().now().nanoseconds + 10_000_000_000
        while self._global_costmap is None:
            time.sleep(0.5)
            if self.get_clock().now().nanoseconds > deadline:
                self.error("Global costmap never received. Aborting.")
                return

        remaining = self._sample_candidate_viewpoints()
        if not remaining:
            self.error("No traversable viewpoints found in global costmap.")
            return

        total = len(remaining)
        visited = 0
        MAX_RETRIES = 2
        retry_counts: dict[tuple[float, float], int] = {}

        while remaining:
            # --- always pick the closest viewpoint to the robot's current pose ---
            cx = self.current_pose.pose.position.x if hasattr(self, 'current_pose') else 0.0
            cy = self.current_pose.pose.position.y if hasattr(self, 'current_pose') else 0.0
            remaining.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            wx, wy = remaining.pop(0)
            visited += 1

            self.info(f"[{visited}/{total}] Viewpoint ({wx:.2f}, {wy:.2f})")

            # --- handle pending face / ring detections first ---
            self._process_pending_detections()

            dist_to_robot = math.hypot(wx - cx, wy - cy)
            if dist_to_robot < 0.5:
                self.info(f"  Too close ({dist_to_robot:.2f} m); skipping.")
                continue

            time.sleep(0.05)
            if not self._is_cliff_safe():
                continue
            if not self._is_local_costmap_clear(wx, wy):
                continue

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = wx
            goal.pose.position.y = wy
            goal.pose.orientation = self.YawToQuaternion(
                math.atan2(wy - cy, wx - cx))

            if not self.goToPose(goal):
                self.warn(f"  Goal rejected by Nav2; skipping.")
                continue

            self._obstacle_blocked = False
            self._nav_goal_start_time = time.time()
            nav_deadline = time.time() + 120.0
            detection_interrupted = False
            while not self.isTaskComplete():
                time.sleep(0.1)
                if not self._is_cliff_safe():
                    self.warn("Cliff detected mid-navigation; cancelling.")
                    self.cancelTask()
                    break
                if self._obstacle_blocked:
                    break
                if time.time() > nav_deadline:
                    self.warn(f"  Navigation to ({wx:.2f}, {wy:.2f}) timed out; cancelling.")
                    self.cancelTask()
                    break
                # If a face/ring was detected while navigating, cancel and attend to it
                if self._pending_face_approach or self._pending_ring_inspect:
                    self.info('Detection queued mid-navigation – cancelling to attend.')
                    self.cancelTask()
                    detection_interrupted = True
                    break

            # Process any pending detections that triggered the interruption
            if detection_interrupted:
                self._process_pending_detections()
                # Re-add the current viewpoint so it is reconsidered
                remaining.append((wx, wy))
                continue

            if self._obstacle_blocked:
                retries = retry_counts.get((wx, wy), 0) + 1
                retry_counts[(wx, wy)] = retries
                if retries < MAX_RETRIES:
                    self.info(f"  Obstacle blocked ({wx:.2f}, {wy:.2f}); requeueing ({retries}/{MAX_RETRIES}).")
                    remaining.append((wx, wy))
                else:
                    self.warn(f"  Obstacle blocked ({wx:.2f}, {wy:.2f}) {retries} times; giving up.")
                continue

            if self.getResult() != TaskResult.SUCCEEDED:
                self.warn(f"  Navigation did not succeed ({self.getResult()}); skipping.")
                continue

            self.info(f"  Covered. Rings so far: {len(self.ring_detections)}")

        self._process_pending_detections()
        self.info(f"Autonomous search complete. Detected {len(self.ring_detections)} ring(s).")
        self.save_ring_detections('/home/erik/rins/src/dis_tutorial5/ring_detections.json')


def main(args=None):
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--turnoff', action='store_true', help='Disable geofencing')
    parsed, remaining = parser.parse_known_args(sys.argv[1:] if args is None else args)

    rclpy.init(args=remaining)
    rc = RobotCommander(geofence_enabled=not parsed.turnoff)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(rc)
    threading.Thread(target=executor.spin, daemon=True).start()

    rc.waitUntilNav2Active()
    rc.info("Geofencing: " + ("DISABLED" if parsed.turnoff else "ENABLED"))

    while rc.is_docked is None:
        time.sleep(0.5)
    if rc.is_docked:
        rc.undock()

    rc.find_people_and_rings_autonomously()
    rc.spin(-0.57)
    rc.destroyNode()

if __name__=="__main__":
    main()