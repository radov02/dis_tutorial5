#!/usr/bin/env python3
"""Interactively pick and store search positions from the live /map topic.

Usage
-----
Start the navigation stack first (e.g. ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py
map:=/home/erik/rins/maps/task1_blue_demo.yaml), then run:

  ros2 run dis_tutorial5 determine_search_points \\
      --ros-args -p output_file:=/home/erik/rins/maps/task1_blue_demo_search_positions.json

Controls (OpenCV window)
------------------------
  Left-click    - add a search point at the clicked world position
  Right-click   - remove the nearest stored point (within 30 px)
  S             - save current points to JSON immediately
  C             - clear all points
  U             - undo last addition
  Q / ESC       - save to JSON and quit

Each saved entry has the fields: x, y, yaw (0.0), label ("Point N").
"""

import json
import math
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy,
    QoSProfile, QoSReliabilityPolicy,
)
from nav_msgs.msg import OccupancyGrid

qos_map = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# UI colours (BGR)
COLOUR_POINT   = (0,  200,  50)    # green dot
COLOUR_TEXT    = (255, 255, 255)
COLOUR_SHADOW  = (0,   0,   0)
BANNER_H       = 24                 # pixels reserved for status bar BELOW the map
REMOVE_SNAP_PX = 30                # click must be within this many pixels to remove


class SearchPointPicker(Node):

    def __init__(self):
        super().__init__('search_point_picker')

        self.declare_parameter('output_file', '')

        self._map_np      = None    # displayed (flipped) numpy image in grey
        self._map_colour  = None    # colour copy of the above (for redrawing)
        self._map_data    = {
            'resolution': None,
            'width': None,
            'height': None,
            'origin': None,           # [x, y, theta]
        }

        self._search_points: list[dict] = []   # list of {x, y, yaw, label}

        self.create_subscription(OccupancyGrid, '/map', self._map_callback, qos_map)

        self._window_name = 'Search Point Picker  |  L-click=add  R-click=remove  S=save  Q=quit'
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        self.get_logger().info(
            'Waiting for /map…  (Make sure the nav stack / map_server is running.)')
        self.get_logger().info(
            'Controls: [L-click] add point  [R-click] remove nearest point  '
            '[S] save  [C] clear  [U] undo  [Q/ESC] save & quit')

    # ------------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid):
        # Rebuild numpy array from occupancy grid data
        raw = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        raw = np.flipud(raw)          # origin at bottom-left → flip for display

        # Build a grayscale image matching the .pgm convention used in ROS
        display = np.zeros_like(raw, dtype=np.uint8)
        display[raw == 0]   = 200     # free space - light grey
        display[raw == 100] = 0       # occupied - black
        display[raw < 0]    = 127     # unknown   - mid grey

        self._map_np     = display
        self._map_colour = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        # Compute a dot radius that is ~1 % of the shorter map dimension, clamped [3, 12]
        self._point_radius = max(3, min(12, int(min(msg.info.width, msg.info.height) * 0.01)))

        # Store map meta-data
        self._map_data['resolution'] = msg.info.resolution
        self._map_data['width']      = msg.info.width
        self._map_data['height']     = msg.info.height
        ox_q = msg.info.origin.orientation
        yaw = math.atan2(
            2.0 * (ox_q.w * ox_q.z + ox_q.x * ox_q.y),
            1.0 - 2.0 * (ox_q.y * ox_q.y + ox_q.z * ox_q.z))
        self._map_data['origin'] = [
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            yaw,
        ]

        self.get_logger().info(
            f'Map received: {msg.info.width}×{msg.info.height} px, '
            f'res={msg.info.resolution:.3f} m/px, '
            f'origin=({self._map_data["origin"][0]:.2f}, '
            f'{self._map_data["origin"][1]:.2f})')

        self._redraw()

    # ------------------------------------------------------------------
    # Coordinate conversions (same convention as map_goals.py)
    # ------------------------------------------------------------------

    def _pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        """Convert OpenCV pixel (col, row) to ROS world coordinates (x, y)."""
        assert self._map_data['resolution'] is not None
        res    = self._map_data['resolution']
        ox     = self._map_data['origin'][0]
        oy     = self._map_data['origin'][1]
        height = self._map_data['height']
        wx = px * res + ox
        wy = (height - py) * res + oy
        return wx, wy

    def _world_to_pixel(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert ROS world coordinates (x, y) to OpenCV pixel (col, row)."""
        assert self._map_data['resolution'] is not None
        res    = self._map_data['resolution']
        ox     = self._map_data['origin'][0]
        oy     = self._map_data['origin'][1]
        height = self._map_data['height']
        px = int((wx - ox) / res)
        py = int(height - (wy - oy) / res)
        return px, py

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x: int, y: int, flags, params):
        if self._map_data['resolution'] is None:
            return  # map not yet received

        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = self._pixel_to_world(x, y)
            n      = len(self._search_points) + 1
            self._search_points.append({
                'x':    round(wx, 3),
                'y':    round(wy, 3),
                'yaw':  0.0,
                'label': f'Point {n}',
            })
            self.get_logger().info(
                f'Added Point {n}: world ({wx:.3f}, {wy:.3f})')
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self._search_points:
                return
            # Find nearest stored point (in pixel space)
            nearest_idx  = None
            nearest_dist = float('inf')
            for i, pt in enumerate(self._search_points):
                px, py = self._world_to_pixel(pt['x'], pt['y'])
                dist   = math.hypot(px - x, py - y)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx  = i
            if nearest_dist <= REMOVE_SNAP_PX and nearest_idx is not None:
                removed = self._search_points.pop(nearest_idx)
                self.get_logger().info(
                    f'Removed {removed["label"]}: ({removed["x"]:.3f}, {removed["y"]:.3f})')
                # Re-label remaining points
                for i, pt in enumerate(self._search_points):
                    pt['label'] = f'Point {i + 1}'
                self._redraw()
            else:
                self.get_logger().warn(
                    f'Right-click: no point within {REMOVE_SNAP_PX} px (nearest is {nearest_dist:.0f} px)')

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        if self._map_colour is None:
            return

        r = getattr(self, '_point_radius', 4)

        # Map image (not modified)
        map_canvas = self._map_colour.copy()

        for i, pt in enumerate(self._search_points):
            px, py = self._world_to_pixel(pt['x'], pt['y'])
            cv2.circle(map_canvas, (px, py), r, COLOUR_POINT, -1)
            cv2.circle(map_canvas, (px, py), r + 1, (255, 255, 255), 1)
            # Label next to dot
            label = str(i + 1)
            tx, ty = px + r + 2, py + 4
            cv2.putText(map_canvas, label, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_SHADOW, 2)
            cv2.putText(map_canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_TEXT, 1)

        # Status banner BELOW the map so it never covers it
        banner = np.full((BANNER_H, map_canvas.shape[1], 3), (40, 40, 40), dtype=np.uint8)
        status = (f'Points: {len(self._search_points)}   '
                  f'|   L-click=add   R-click=remove   '
                  f'S=save   C=clear   U=undo   Q/ESC=quit')
        cv2.putText(banner, status, (5, BANNER_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 200), 1)

        canvas = np.vstack([map_canvas, banner])
        cv2.imshow(self._window_name, canvas)

    # ------------------------------------------------------------------
    # Save / load helpers
    # ------------------------------------------------------------------

    def _output_path(self) -> str:
        path: str = self.get_parameter('output_file').value
        if not path:
            path = os.path.join(
                os.path.expanduser('~'), 'rins', 'maps',
                'search_positions.json')
        return path

    def save(self):
        if not self._search_points:
            self.get_logger().warn('No points to save.')
            return
        path = self._output_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._search_points, f, indent=2)
        self.get_logger().info(
            f'Saved {len(self._search_points)} search points → {path}')

    # ------------------------------------------------------------------
    # Main spin loop
    # ------------------------------------------------------------------

    def run(self):
        """Drive the OpenCV window event loop alongside ROS spinning."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self._map_np is not None:
                self._redraw()

            key = cv2.waitKey(50) & 0xFF

            # Check if the window was closed with the X button
            try:
                visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = -1
            if visible < 1:
                self.save()
                break

            if key in (ord('q'), ord('Q'), 27):        # Q or ESC
                self.save()
                break
            elif key in (ord('s'), ord('S')):
                self.save()
            elif key in (ord('c'), ord('C')):
                self._search_points.clear()
                self.get_logger().info('All points cleared.')
                self._redraw()
            elif key in (ord('u'), ord('U')):
                if self._search_points:
                    removed = self._search_points.pop()
                    self.get_logger().info(
                        f'Undo: removed {removed["label"]}')
                    self._redraw()

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = SearchPointPicker()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
