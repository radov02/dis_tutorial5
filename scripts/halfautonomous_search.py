#!/usr/bin/env python3
"""Half-autonomous search script.

Loads a list of pre-defined search positions from a JSON file and navigates
the robot to each one in order, performing face and ring detection at every
stop.  The JSON must contain a list of objects with at least "x" and "y"
fields (metres, map frame).  An optional "yaw" field (radians) sets the
robot's orientation when it arrives; if absent the robot faces the direction
of travel.

Run example
-----------
  ros2 run dis_tutorial5 halfautonomous_search \\
      --ros-args -p search_positions_file:=/home/erik/rins/maps/task1_blue_demo_search_positions.json
"""

from enum import Enum
import json
import math
import os
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
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy,
    QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data,
)

from robot_interfaces.srv import HumanDetected

import numpy as np
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Range
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPIN_AFTER_ARRIVAL   = True          # do a spin at each search position
SPIN_AMOUNT_RAD      = 3.14          # half circle – repeated twice for full coverage
SPIN_TIME_ALLOWANCE  = 10            # seconds per spin
ARRIVAL_PAUSE_S      = 1.0           # pause after arriving before spinning
POST_SPIN_PAUSE_S    = 1.0           # pause after spinning before moving on
RING_DEDUP_DISTANCE_M = 0.5          # minimum distance to merge ring detections
LOCAL_OBSTACLE_THRESHOLD = 90
GLOBAL_COST_THRESHOLD    = 20

amcl_pose_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

detection_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class TaskResult(Enum):
    UNKNOWN   = 0
    SUCCEEDED = 1
    CANCELED  = 2
    FAILED    = 3


# ---------------------------------------------------------------------------
# Robot commander
# ---------------------------------------------------------------------------

class RobotCommander(Node):

    def __init__(self, node_name: str = 'robot_commander', namespace: str = ''):
        super().__init__(node_name=node_name, namespace=namespace)

        # Declare parameter: path to the search-positions JSON
        self.declare_parameter('search_positions_file', '')

        self.pose_frame_id = 'map'
        self.goal_handle   = None
        self.result_future = None
        self.feedback      = None
        self.status        = None
        self.initial_pose_received = False
        self.is_docked     = None

        # Core subscriptions / publishers
        self.create_subscription(DockStatus, 'dock_status',
                                 self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 'amcl_pose',
            self._amclPoseCallback, amcl_pose_qos)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)

        # Action clients
        self.nav_to_pose_client   = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client           = ActionClient(self, Spin, 'spin')
        self.undock_action_client  = ActionClient(self, Undock, 'undock')
        self.dock_action_client    = ActionClient(self, Dock, 'dock')
        self.human_interaction_client = self.create_client(HumanDetected, 'human_detected')

        # TF + breadcrumbs
        self.marker_id   = 0
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub  = self.create_publisher(
            Marker, '/breadcrumbs', QoSReliabilityPolicy.BEST_EFFORT)
        self.timer = self.create_timer(1.0, self._timer_callback)

        # Search-point visualisation (RELIABLE + VOLATILE so RViz picks it up;
        # a timer re-publishes every 2 s for late subscribers)
        self._search_points_pub = self.create_publisher(
            MarkerArray, '/search_points', 10)
        self._search_point_markers: MarkerArray | None = None
        self._search_points_timer = self.create_timer(
            2.0, self._republish_search_points)

        # Detection reactive state
        self._pending_face_approach: list[tuple[float, float]] = []
        self._pending_ring_inspect:  list[tuple[float, float]] = []
        self._inspected_rings:       set[tuple[float, float]]  = set()
        self._approached_faces:      set[tuple[float, float]]  = set()

        # Ring / face detection subscribers
        self._init_detection_subscribers()

        # Obstacle / costmap helpers (initialised lazily when search starts)
        self._global_costmap: OccupancyGrid | None = None
        self._local_costmap:  OccupancyGrid | None = None
        self._cliff_detected  = False
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                 self._globalCostmapCallback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._localCostmapCallback, qos_profile_sensor_data)
        self.create_subscription(Range, 'ir_intensity_side_left',
                                 self._cliffCallback, qos_profile_sensor_data)

        self.ring_detections: list[tuple[float, float, float, str]] = []

        self.get_logger().info('HalfAutonomousSearch node initialised.')

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    # ------------------------------------------------------------------
    # Future helpers
    # ------------------------------------------------------------------

    def _wait_for_future(self, future, timeout=None):
        """Block (spin-waiting) until *future* completes.  Safe from any thread."""
        start = time.time()
        while not future.done():
            if timeout is not None and (time.time() - start) >= timeout:
                break
            time.sleep(0.02)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def goToPose(self, pose: PoseStamped, behavior_tree: str = '') -> bool:
        """Send a NavigateToPose goal.  Returns True if accepted."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' not available, waiting…")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info(f'Navigating to ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, self._feedbackCallback)
        self._wait_for_future(send_goal_future, timeout=10.0)
        self.goal_handle = send_goal_future.result()

        if self.goal_handle is None or not self.goal_handle.accepted:
            self.error(f'Goal to ({pose.pose.position.x:.2f}, '
                       f'{pose.pose.position.y:.2f}) was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist: float = 1.57, time_allowance: int = 10) -> bool:
        """Send a Spin action goal."""
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting…")

        goal_msg = Spin.Goal()
        goal_msg.target_yaw      = spin_dist
        goal_msg.time_allowance  = DurationMsg(sec=time_allowance)

        self.info(f'Spinning {spin_dist:.2f} rad…')
        send_goal_future = self.spin_client.send_goal_async(
            goal_msg, self._feedbackCallback)
        self._wait_for_future(send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def cancelTask(self):
        self.info('Cancelling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            self._wait_for_future(future)

    def isTaskComplete(self) -> bool:
        if not self.result_future:
            return True
        self._wait_for_future(self.result_future, timeout=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task ended – status {self.status}')
                return True
        else:
            return False
        self.debug('Task succeeded!')
        return True

    def getResult(self) -> TaskResult:
        if   self.status == GoalStatus.STATUS_SUCCEEDED: return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:   return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:  return TaskResult.CANCELED
        return TaskResult.UNKNOWN

    # ------------------------------------------------------------------
    # Undocking
    # ------------------------------------------------------------------

    def undock(self):
        self.info('Undocking…')
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)
        self._wait_for_future(goal_future)
        handle = goal_future.result()
        if not handle.accepted:
            self.error('Undock goal rejected')
            return
        result_future = handle.get_result_async()
        # wait until complete
        deadline = time.time() + 30.0
        while not result_future.done():
            time.sleep(0.1)
            if time.time() > deadline:
                self.warn('Undock timed out')
                break
        self.info('Undock complete')

    # ------------------------------------------------------------------
    # Nav2 lifecycle wait
    # ------------------------------------------------------------------

    def waitUntilNav2Active(self, navigator: str = 'bt_navigator',
                            localizer: str = 'amcl'):
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready!')

    def _waitForNodeToActivate(self, node_name: str):
        self.debug(f'Waiting for {node_name} to become active…')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} not available, waiting…')
        req   = GetState.Request()
        state = 'unknown'
        while state != 'active':
            future = state_client.call_async(req)
            self._wait_for_future(future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'  {node_name} state: {state}')
            time.sleep(2)

    # ------------------------------------------------------------------
    # Quaternion helper
    # ------------------------------------------------------------------

    def YawToQuaternion(self, angle_z: float = 0.) -> Quaternion:
        q = quaternion_from_euler(0, 0, angle_z)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _amclPoseCallback(self, msg: PoseWithCovarianceStamped):
        self.initial_pose_received = True
        self.current_pose = msg.pose

    def _feedbackCallback(self, msg):
        self.feedback = msg.feedback

    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def _globalCostmapCallback(self, msg: OccupancyGrid):
        self._global_costmap = msg

    def _localCostmapCallback(self, msg: OccupancyGrid):
        self._local_costmap = msg

    def _cliffCallback(self, msg: Range):
        self._cliff_detected = msg.range < 0.05

    # ------------------------------------------------------------------
    # Breadcrumb timer callback
    # ------------------------------------------------------------------

    def _timer_callback(self):
        pt = PointStamped()
        pt.header.frame_id = '/base_link'
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = -0.1
        try:
            trans  = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                RclpyDuration(seconds=0.1))
            pt_map = tfg.do_transform_point(pt, trans)
            marker = self._make_breadcrumb(pt_map)
            self.marker_pub.publish(marker)
            self.marker_id += 1
        except TransformException:
            pass

    def _make_breadcrumb(self, pt: PointStamped) -> Marker:
        m = Marker()
        m.header  = pt.header
        m.type    = Marker.SPHERE
        m.action  = Marker.ADD
        m.id      = self.marker_id
        m.scale.x = m.scale.y = m.scale.z = 0.08
        m.color.r = 0.2; m.color.g = 0.8; m.color.b = 0.2; m.color.a = 1.0
        m.pose.position.x = pt.point.x
        m.pose.position.y = pt.point.y
        m.pose.position.z = pt.point.z
        m.lifetime = RclpyDuration(seconds=600).to_msg()
        return m

    # ------------------------------------------------------------------
    # Detection subscribers
    # ------------------------------------------------------------------

    def _init_detection_subscribers(self):
        self.ring_detections: list[tuple[float, float, float, str]] = []

        self.create_subscription(
            MarkerArray, '/detected_rings',
            self._detected_rings_callback, detection_qos)
        self.create_subscription(
            MarkerArray, '/detected_faces',
            self._detected_faces_callback, detection_qos)
        self.info('Detection subscribers initialised'
                  ' (/detected_rings and /detected_faces).')

    def _detected_rings_callback(self, msg: MarkerArray):
        fresh = []
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            color = marker.text if marker.text else 'unknown'
            if math.isnan(x) or math.isnan(y):
                continue
            fresh.append((x, y, z, color))

        prev_set  = set((round(r[0], 2), round(r[1], 2), r[3]) for r in self.ring_detections)
        fresh_set = set((round(r[0], 2), round(r[1], 2), r[3]) for r in fresh)
        new_items = fresh_set - prev_set

        if len(fresh) != len(self.ring_detections):
            self.ring_detections = fresh
            self.info(f'Ring list updated: {len(self.ring_detections)} ring(s).')

        for nr in new_items:
            rx, ry, rcolor = nr
            match = next((r for r in fresh
                          if round(r[0], 2) == rx and round(r[1], 2) == ry
                          and r[3] == rcolor), None)
            if match is None:
                continue
            key = (round(match[0], 1), round(match[1], 1))
            if key in self._inspected_rings:
                continue
            if not any(math.hypot(match[0] - qx, match[1] - qy) < 0.5
                       for qx, qy in self._pending_ring_inspect):
                self._pending_ring_inspect.append((match[0], match[1]))
                self.info(f'Queued ring inspection at ({match[0]:.2f}, {match[1]:.2f})')

    def _detected_faces_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            if math.isnan(x) or math.isnan(y):
                continue
            key = (round(x, 1), round(y, 1))
            if key in self._approached_faces:
                continue
            if any(math.hypot(x - qx, y - qy) < 1.0
                   for qx, qy in self._pending_face_approach):
                continue
            self._pending_face_approach.append((x, y))
            self.info(f'Queued face approach at ({x:.2f}, {y:.2f})')

    # ------------------------------------------------------------------
    # Reactive behaviours (called from main loop only)
    # ------------------------------------------------------------------

    def _approach_face(self, x: float, y: float, stop_distance: float = 0.8):
        self._approached_faces.add((round(x, 1), round(y, 1)))
        if not hasattr(self, 'current_pose'):
            self.warn('No current_pose; cannot approach face.')
            return
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        dx, dy = x - rx, y - ry
        dist = math.hypot(dx, dy)
        if dist <= stop_distance:
            self.info('Already close to detected face.')
            return

        aim_x = x - (dx / dist) * stop_distance
        aim_y = y - (dy / dist) * stop_distance

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.position.x = aim_x
        goal.pose.position.y = aim_y
        goal.pose.orientation = self.YawToQuaternion(
            math.atan2(y - aim_y, x - aim_x))

        self.info(f'Approaching face at ({x:.2f},{y:.2f}) → ({aim_x:.2f},{aim_y:.2f})')
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

    def _inspect_ring(self, x: float, y: float,
                      radius: float = 0.8, angle_offset: float = 0.6):
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
            goal.header.stamp    = self.get_clock().now().to_msg()
            goal.pose.position.x = px
            goal.pose.position.y = py
            goal.pose.orientation = self.YawToQuaternion(
                math.atan2(y - py, x - px))

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
        """Drain face-approach and ring-inspect queues (call from main loop only)."""
        while self._pending_face_approach:
            x, y = self._pending_face_approach.pop(0)
            self.info(f'>>> Processing face approach ({x:.2f}, {y:.2f})')
            self._approach_face(x, y)
        while self._pending_ring_inspect:
            x, y = self._pending_ring_inspect.pop(0)
            self.info(f'>>> Processing ring inspection ({x:.2f}, {y:.2f})')
            self._inspect_ring(x, y)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_ring_detections(self, filepath: str):
        data = [{'x': x, 'y': y, 'z': z, 'color': c}
                for x, y, z, c in self.ring_detections]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self.info(f'Saved {len(self.ring_detections)} ring detections → {filepath}')

    # ------------------------------------------------------------------
    # Main search routine
    # ------------------------------------------------------------------

    def find_people_and_rings_at_search_points(self):
        """Navigate to each pre-defined search position from the JSON file."""

        json_path: str = self.get_parameter('search_positions_file').value
        if not json_path:
            self.error('Parameter search_positions_file is empty! '
                       'Pass it with --ros-args -p search_positions_file:=<path>')
            return

        if not os.path.isfile(json_path):
            self.error(f'Search positions file not found: {json_path}')
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        if not data:
            self.warn('Search positions file is empty – nothing to do.')
            return

        self.info(f'Loaded {len(data)} search positions from {json_path}')
        self._publish_search_point_markers(data)

        total   = len(data)
        visited = 0

        for idx, entry in enumerate(data):
            wx = float(entry['x'])
            wy = float(entry['y'])
            # Yaw: explicit value or face the next point, else use 0
            if 'yaw' in entry:
                wyaw = float(entry['yaw'])
            elif idx + 1 < len(data):
                nx = float(data[idx + 1]['x'])
                ny = float(data[idx + 1]['y'])
                wyaw = math.atan2(ny - wy, nx - wx)
            else:
                wyaw = 0.0

            visited += 1
            self.info(f'[{visited}/{total}]  →  ({wx:.2f}, {wy:.2f})  yaw={math.degrees(wyaw):.1f}°')

            # --- handle any accumulated detections before moving ---
            self._process_pending_detections()

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp    = self.get_clock().now().to_msg()
            goal.pose.position.x = wx
            goal.pose.position.y = wy
            goal.pose.orientation = self.YawToQuaternion(wyaw)

            if not self.goToPose(goal):
                self.warn(f'  Goal ({wx:.2f}, {wy:.2f}) rejected – skipping.')
                continue

            nav_deadline = time.time() + 180.0
            detection_interrupted = False

            while not self.isTaskComplete():
                time.sleep(0.1)
                if time.time() > nav_deadline:
                    self.warn(f'  Navigation to ({wx:.2f}, {wy:.2f}) timed out; cancelling.')
                    self.cancelTask()
                    break
                if self._pending_face_approach or self._pending_ring_inspect:
                    self.info('Detection queued mid-travel – pausing navigation to attend.')
                    self.cancelTask()
                    detection_interrupted = True
                    break

            if detection_interrupted:
                self._process_pending_detections()
                # Re-insert at front so we still visit this point
                data.insert(idx + 1, entry)
                total += 1
                continue

            result = self.getResult()
            if result not in (TaskResult.SUCCEEDED, TaskResult.UNKNOWN):
                self.warn(f'  Navigation result: {result}; continuing to next point.')

            # --- arrived: pause, then spin to observe (two half-turns = full circle) ---
            time.sleep(ARRIVAL_PAUSE_S)
            if SPIN_AFTER_ARRIVAL:
                for _ in range(2):
                    self.spin(SPIN_AMOUNT_RAD, SPIN_TIME_ALLOWANCE)
                    deadline = time.time() + SPIN_TIME_ALLOWANCE + 5.0
                    while not self.isTaskComplete():
                        time.sleep(0.1)
                        if time.time() > deadline:
                            self.cancelTask()
                            break
                    time.sleep(0.3)
                time.sleep(POST_SPIN_PAUSE_S)

            # --- process any detections gathered during / after the spin ---
            self._process_pending_detections()
            self.info(f'  Done.  Rings so far: {len(self.ring_detections)}')

        # Final pass: drain any remaining detections
        self._process_pending_detections()
        self.info(f'Half-autonomous search complete.  '
                  f'Found {len(self.ring_detections)} ring(s).')
        # Derive output path next to the JSON file
        base = os.path.splitext(json_path)[0]
        self.save_ring_detections(f'{base}_ring_detections.json')

    # ------------------------------------------------------------------
    # Search-point marker publisher
    # ------------------------------------------------------------------

    def _publish_search_point_markers(self, points: list[dict]):
        """Publish all search positions as a latched MarkerArray on /search_points."""
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, pt in enumerate(points):
            # Sphere at the position
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns     = 'search_points'
            m.id     = i * 2
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pt['x'])
            m.pose.position.y = float(pt['y'])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.18
            m.color.r = 1.0; m.color.g = 0.6; m.color.b = 0.0; m.color.a = 0.9
            m.lifetime.sec = 0  # persistent
            ma.markers.append(m)

            # Number label
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp    = now
            t.ns     = 'search_points_labels'
            t.id     = i * 2 + 1
            t.type   = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(pt['x'])
            t.pose.position.y = float(pt['y'])
            t.pose.position.z = 0.40
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
            label = pt.get('label', f'Point {i + 1}')
            t.text = f'{i + 1}: {label}' if label != f'Point {i + 1}' else str(i + 1)
            t.lifetime.sec = 0
            ma.markers.append(t)

        self._search_point_markers = ma       # store for periodic re-publish
        self._search_points_pub.publish(ma)
        self.info(f'Published {len(points)} search-point markers on /search_points')

    def _republish_search_points(self):
        """Re-publish search-point markers periodically so late RViz subscribers see them."""
        if self._search_point_markers is not None:
            self._search_points_pub.publish(self._search_point_markers)

    # ------------------------------------------------------------------
    # Logger shorthands
    # ------------------------------------------------------------------

    def info(self, msg: str):  self.get_logger().info(msg)
    def warn(self, msg: str):  self.get_logger().warn(msg)
    def error(self, msg: str): self.get_logger().error(msg)
    def debug(self, msg: str): self.get_logger().debug(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    rc = RobotCommander()

    # Spin ROS executor in a background thread so futures resolve without blocking
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(rc)
    threading.Thread(target=executor.spin, daemon=True).start()

    rc.waitUntilNav2Active()

    while rc.is_docked is None:
        time.sleep(0.5)
    if rc.is_docked:
        rc.undock()

    rc.find_people_and_rings_at_search_points()

    rc.destroyNode()


if __name__ == '__main__':
    main()
