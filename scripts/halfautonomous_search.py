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

SPIN_AT_GOAL_SECONDS     = 0             # spin for this many seconds at each search point (0 = no spin)
ARRIVAL_PAUSE_S          = 1.0           # pause after arriving before spinning
POST_SPIN_PAUSE_S    = 1.0           # pause after spinning before moving on
RING_DEDUP_DISTANCE_M = 0.5          # minimum distance to merge ring detections
LOCAL_OBSTACLE_THRESHOLD = 80
GLOBAL_COST_THRESHOLD    = 20
COSTMAP_SAFETY_THRESHOLD = 254  # local costmap cost triggering 180° turn (254 = lethal only)

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
# Priority queue for navigation goals
# ---------------------------------------------------------------------------

class GoalEntry:
    """A single navigation goal (search point, detected ring, or detected face)."""
    __slots__ = ('goal_type', 'x', 'y', 'yaw', 'label')

    def __init__(self, goal_type: str, x: float, y: float,
                 yaw: float | None = None, label: str = ''):
        self.goal_type = goal_type   # 'search' | 'ring' | 'face'
        self.x = x
        self.y = y
        self.yaw = yaw
        self.label = label

    def __repr__(self) -> str:
        return (f'GoalEntry({self.goal_type}, '
                f'x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw})')


class GoalPriorityQueue:
    """Priority queue that always yields the closest goal (priority = 1/dist)."""

    def __init__(self):
        self._goals: list[GoalEntry] = []

    def add(self, entry: GoalEntry):
        self._goals.append(entry)

    def pop_closest(self, robot_x: float, robot_y: float) -> GoalEntry | None:
        """Remove and return the goal nearest to (robot_x, robot_y)."""
        if not self._goals:
            return None
        best_idx = min(range(len(self._goals)),
                       key=lambda i: math.hypot(self._goals[i].x - robot_x,
                                                self._goals[i].y - robot_y))
        return self._goals.pop(best_idx)

    def has_closer_than(self, robot_x: float, robot_y: float,
                        threshold: float) -> bool:
        """Return True if any queued goal is closer than *threshold* metres."""
        return any(math.hypot(g.x - robot_x, g.y - robot_y) < threshold
                   for g in self._goals)

    def is_empty(self) -> bool:
        return len(self._goals) == 0

    def __len__(self) -> int:
        return len(self._goals)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for g in self._goals:
            counts[g.goal_type] = counts.get(g.goal_type, 0) + 1
        return ', '.join(f'{v} {k}' for k, v in counts.items()) or 'empty'


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

        # Unified goal priority queue (search points + detected rings/faces)
        self._goal_pq = GoalPriorityQueue()
        self._pq_lock = threading.Lock()
        self._pq_dirty = False          # set by detection callbacks
        self._visited_faces: set[tuple[float, float]] = set()
        self._visited_rings: set[tuple[float, float]] = set()

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
                self.debug(f'Task ended - status {self.status}')
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
    # Costmap safety helpers
    # ------------------------------------------------------------------

    def _get_robot_local_costmap_cost(self) -> int:
        """Return the max local costmap cost in a small radius around the robot.
        Returns 0 if costmap not available."""
        if self._local_costmap is None or not hasattr(self, 'current_pose'):
            return 0
        lc = self._local_costmap
        grid = np.array(lc.data, dtype=np.int8).reshape(
            (lc.info.height, lc.info.width))
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        res = lc.info.resolution
        ox = lc.info.origin.position.x
        oy = lc.info.origin.position.y
        col = int((rx - ox) / res)
        row = int((ry - oy) / res)
        h, w = grid.shape
        radius_cells = max(1, int(0.05 / res))  # ~0.05 m
        max_cost = 0
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < h and 0 <= c < w:
                    cost = int(grid[r, c])
                    if cost > max_cost:
                        max_cost = cost
        return max_cost

    def _get_robot_position(self) -> tuple[float | None, float | None]:
        """Return (x, y) of the robot in map frame, or (None, None)."""
        if hasattr(self, 'current_pose'):
            return (self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y)
        return (None, None)

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
            # Parse extended text format: "color|eccentricity|wall_normal_angle"
            raw_text = marker.text if marker.text else 'unknown'
            parts = raw_text.split('|')
            color = parts[0] if parts else 'unknown'
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
            # Skip if already visited (distance-based)
            if any(math.hypot(match[0] - vx, match[1] - vy) < RING_DEDUP_DISTANCE_M
                   for vx, vy in self._visited_rings):
                continue
            # Avoid duplicate PQ entries for nearby rings
            with self._pq_lock:
                already_queued = any(
                    g.goal_type == 'ring'
                    and math.hypot(g.x - match[0], g.y - match[1]) < RING_DEDUP_DISTANCE_M
                    for g in self._goal_pq._goals)
                if not already_queued:
                    self._goal_pq.add(GoalEntry('ring', match[0], match[1],
                                                label=f'ring ({rcolor})'))
                    self._pq_dirty = True
                    self.info(f'PQ: added ring at ({match[0]:.2f}, {match[1]:.2f}) [{rcolor}]')

    def _detected_faces_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            if math.isnan(x) or math.isnan(y):
                continue
            # Skip if already visited (distance-based)
            if any(math.hypot(x - vx, y - vy) < 1.0
                   for vx, vy in self._visited_faces):
                continue
            with self._pq_lock:
                already_queued = any(
                    g.goal_type == 'face'
                    and math.hypot(g.x - x, g.y - y) < 1.0
                    for g in self._goal_pq._goals)
                if not already_queued:
                    self._goal_pq.add(GoalEntry('face', x, y, label='face'))
                    self._pq_dirty = True
                    self.info(f'PQ: added face at ({x:.2f}, {y:.2f})')

    # ------------------------------------------------------------------
    # Reactive behaviours (called from main loop only)
    # ------------------------------------------------------------------

    def _approach_face(self, x: float, y: float, stop_distance: float = 0.0):
        self._visited_faces.add((x, y))
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

        self.info(f'\n\n\n>>> APPROACHING FACE at ({x:.2f},{y:.2f}) → ({aim_x:.2f},{aim_y:.2f}) <<<\n')
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
                      radius: float = 0.0, angle_offset: float = 0.6):
        self._visited_rings.add((x, y))
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

            self.info(f'\n\n\n>>> INSPECTING RING at ({x:.2f},{y:.2f}) from ({px:.2f},{py:.2f}) <<<\n')
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
        """Navigate using a unified priority queue of search points and detections.

        All search positions (from JSON), detected rings, and detected faces
        live in one priority queue keyed by 1/distance.  The robot always
        heads for the closest goal.  The PQ is re-evaluated whenever a new
        detection is added or the robot finishes a goal.
        """

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
            self.warn('Search positions file is empty - nothing to do.')
            return

        self.info(f'Loaded {len(data)} search positions from {json_path}')
        self._publish_search_point_markers(data)

        # Seed PQ with all search positions
        for idx, entry in enumerate(data):
            wx = float(entry['x'])
            wy = float(entry['y'])
            yaw = float(entry['yaw']) if 'yaw' in entry else None
            label = entry.get('label', f'Point {idx + 1}')
            self._goal_pq.add(GoalEntry('search', wx, wy, yaw=yaw, label=label))

        goals_completed = 0

        while True:
            # --- current robot position ---
            rx, ry = self._get_robot_position()
            if rx is None:
                self.warn('No robot pose available; waiting…')
                time.sleep(1.0)
                continue

            # --- pop the closest goal (priority = 1/dist) ---
            with self._pq_lock:
                current_goal = self._goal_pq.pop_closest(rx, ry)
            if current_goal is None:
                break  # PQ empty - done

            dist = math.hypot(current_goal.x - rx, current_goal.y - ry)
            goals_completed += 1
            self.info(
                f'\n\n--- [{goals_completed}] Next: {current_goal.goal_type} '
                f'"{current_goal.label}" at ({current_goal.x:.2f}, {current_goal.y:.2f}) '
                f'dist={dist:.2f}m  [PQ: {len(self._goal_pq)} remaining ({self._goal_pq.summary()})] ---\n')

            # ---- Face goal ----
            if current_goal.goal_type == 'face':
                # Skip if already visited since this goal was enqueued
                if any(math.hypot(current_goal.x - vx, current_goal.y - vy) < 1.0
                       for vx, vy in self._visited_faces):
                    self.info(f'  Face at ({current_goal.x:.2f}, {current_goal.y:.2f}) already visited – skipping.')
                else:
                    self._approach_face(current_goal.x, current_goal.y)
                    self.info(f'  Face done.  Rings so far: {len(self.ring_detections)}')
                continue

            # ---- Ring goal ----
            if current_goal.goal_type == 'ring':
                # Skip if already visited since this goal was enqueued
                if any(math.hypot(current_goal.x - vx, current_goal.y - vy) < RING_DEDUP_DISTANCE_M
                       for vx, vy in self._visited_rings):
                    self.info(f'  Ring at ({current_goal.x:.2f}, {current_goal.y:.2f}) already visited – skipping.')
                else:
                    self._inspect_ring(current_goal.x, current_goal.y)
                    self.info(f'  Ring done.  Rings so far: {len(self.ring_detections)}')
                continue

            # ---- Search-point goal ----
            # yaw_goal_tolerance is set to 6.28 in nav2.yaml so Nav2 will
            # NOT rotate in place at the goal.  We just need a valid quaternion.
            wyaw = current_goal.yaw
            if wyaw is None:
                wyaw = math.atan2(current_goal.y - ry, current_goal.x - rx)

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp    = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = current_goal.x
            goal_pose.pose.position.y = current_goal.y
            goal_pose.pose.orientation = self.YawToQuaternion(wyaw)

            if not self.goToPose(goal_pose):
                self.warn(f'  Goal ({current_goal.x:.2f}, {current_goal.y:.2f}) rejected - skipping.')
                continue

            nav_deadline = time.time() + 180.0
            interrupted = False
            self._pq_dirty = False  # reset before entering wait loop

            while not self.isTaskComplete():
                time.sleep(0.1)
                if time.time() > nav_deadline:
                    self.warn('  Navigation timed out; cancelling.')
                    self.cancelTask()
                    break

                # --- Costmap safety: 180° turn if local cost > threshold ---
                local_cost = self._get_robot_local_costmap_cost()
                if local_cost > COSTMAP_SAFETY_THRESHOLD:
                    self.info(
                        f'\n\n!!! HIGH COSTMAP COST ({local_cost}) - EXECUTING 180° TURN !!!\n')
                    self.cancelTask()
                    time.sleep(0.5)
                    self.spin(spin_dist=math.pi, time_allowance=10)
                    _spin_deadline = time.time() + 12.0
                    while not self.isTaskComplete():
                        time.sleep(0.1)
                        if time.time() > _spin_deadline:
                            self.cancelTask()
                            break
                    # Re-add the search point so it will be attempted again
                    with self._pq_lock:
                        self._goal_pq.add(current_goal)
                    interrupted = True
                    break

                # --- Check if a higher-priority (closer) goal appeared ---
                if self._pq_dirty:
                    self._pq_dirty = False
                    crx, cry = self._get_robot_position()
                    if crx is not None:
                        dist_to_goal = math.hypot(
                            current_goal.x - crx, current_goal.y - cry)
                        with self._pq_lock:
                            if self._goal_pq.has_closer_than(crx, cry, dist_to_goal):
                                self.info('PQ: closer goal detected - re-routing.')
                                self.cancelTask()
                                self._goal_pq.add(current_goal)  # put it back
                                interrupted = True
                                break

            if interrupted:
                continue

            result = self.getResult()
            if result not in (TaskResult.SUCCEEDED, TaskResult.UNKNOWN):
                self.warn(f'  Navigation result: {result}; continuing.')

            # --- arrived: pause ---
            time.sleep(ARRIVAL_PAUSE_S)

            # --- optional spin at search point ---
            if SPIN_AT_GOAL_SECONDS > 0:
                self.info(f'  Spinning for {SPIN_AT_GOAL_SECONDS}s at search point…')
                self.spin(spin_dist=2 * math.pi, time_allowance=SPIN_AT_GOAL_SECONDS)
                spin_deadline = time.time() + SPIN_AT_GOAL_SECONDS + 2.0
                while not self.isTaskComplete():
                    time.sleep(0.1)
                    if time.time() > spin_deadline:
                        self.warn('  Spin timed out; cancelling.')
                        self.cancelTask()
                        break
                time.sleep(POST_SPIN_PAUSE_S)

            self.info(f'  Done.  Rings so far: {len(self.ring_detections)}')

        self.info(
            f'\n\n=== HALF-AUTONOMOUS SEARCH COMPLETE ==='
            f'\nFound {len(self.ring_detections)} ring(s).\n')
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
