#!/usr/bin/python3

import os
import rclpy
from rclpy.node import Node
import cv2, math
import json
import numpy as np
from datetime import datetime
import tf2_geometry_msgs

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

qos_profile = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Physical size limits for valid rings (meters)
RING_MIN_DIAMETER_M = 0.001   # 0.1 cm
RING_MAX_DIAMETER_M = 0.40   # 40 cm

# Z-coordinate range in /map frame for valid rings (meters)
RING_MIN_Z_MAP_M = 0.15
RING_MAX_Z_MAP_M = 3.30

# Merge ring detections that are closer than this in the map frame (meters)
RING_DEDUP_DISTANCE_M = 0.8

# HoughCircles parameters
HOUGH_DP = 1.2        # inverse resolution ratio of the accumulator
HOUGH_MIN_DIST = 1        # minimum distance between detected circle centres (px)
                          # OpenCV clamps this to ≥1 internally anyway
HOUGH_PARAM1 = 87    # upper Canny threshold
HOUGH_PARAM2 = 27     # accumulator threshold (lower -> more circles detected)
HOUGH_MIN_RADIUS = 1
HOUGH_MAX_RADIUS = 50  # rings at ~0.5-3 m can span 5-60+ px in a 300 px image
# Two-pass Hough split radius.  Pass 1 covers [HOUGH_MIN_RADIUS, HOUGH_SPLIT_RADIUS],
# pass 2 covers [HOUGH_SPLIT_RADIUS, effective_max_radius].  Concentric ring edges
# that straddle the split land in different passes and are never mutually
# suppressed by minDist.  Must be strictly between HOUGH_MIN_RADIUS and
# HOUGH_MAX_RADIUS.  A value near the middle of the expected outer-radius range
# (rings typically 10-40 px outer radius) works well.
HOUGH_SPLIT_RADIUS = 15  # split point for two-pass Hough (px)

# Concentric-pair matching thresholds
# Adaptive: allow centre offset up to this fraction of the larger circle's
# radius.  0.25 means centres may differ by up to 25 % of outer_r.  Using a
# radius-relative threshold is far more robust than a fixed pixel count because
# rings of different sizes / distances produce different absolute offsets.
CONCENTRIC_CENTER_FRAC = 0.97  # max |centre offset| / outer_r
CONCENTRIC_RATIO_MIN = 0.01  # min inner_r / outer_r  (too small -> almost solid disc)
CONCENTRIC_RATIO_MAX = 0.98  # max inner_r / outer_r  (too large -> almost no ring)

# For thin rings, erode the depth visualisation before HoughCircles so that
# dark (close) ring-wall pixels are thickened.  In the depth viz, close objects
# are dark (low value) and background is white (255); cv2.erode is a min-filter
# that expands dark regions, making thin ring walls wide enough for HoughCircles
# to resolve two concentric edges.  Odd integer ≥ 3 (e.g. 3, 5, 7, 9).
DEPTH_EROSION_KERNEL = 3  # structuring-element side length for morphological erosion

# 3-D vs 2-D ring discrimination:
# A real hoop/ring has an open hole - the depth sensor sees through it to the
# background, so depth inside the inner circle > depth at ring band.  A flat
# drawing on a wall has uniform depth everywhere.
# At least one pixel inside the inner circle must be this much farther (m) than
# the median ring-band depth.  Scanning all pixels (not just the centre) is
# more robust when the hole is partially occluded or off-centre.
RING_3D_DEPTH_DIFF_MIN_M = 0.005   # minimum depth excess inside inner hole (m)

# When True, HoughCircles is run only on the upper half of the depth image.
# Rings are physical props elevated from the floor so they always appear in
# the upper portion of the camera view; restricting to the upper half reduces
# false positives from floor / table edges in the lower half.
HOUGH_UPPER_HALF_ONLY = True

# After the main two-pass Hough, any circle that remains unpaired triggers a
# targeted single-radius sweep to find a missed concentric partner.
# Steps: r-1, r-2, ..., 2  (smaller inner partner), then r+1, ..., r+MAX
# (larger outer partner, e.g. outer circle was missed by the main pass).
MAX_PX_FOR_OUTER_LOOKOUT = 10  # max upward radius offset for partner search


class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_yaml_path', '/home/erik/rins/maps/map.yaml'),
            ])

        map_yaml_path = self.get_parameter('map_yaml_path').get_parameter_value().string_value
        map_stem = os.path.splitext(os.path.basename(map_yaml_path))[0]
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.ring_detections_json_path = os.path.join(
            '/home/erik/rins/maps',
            f'ring_detections_{map_stem}._{_ts}.json'
        )

        self.bridge = CvBridge()

        # Latest depth frame (float32, NaN where invalid) + matching header
        self.latest_depth_image: np.ndarray | None = None
        self.latest_depth_header = None

        # Camera intrinsics - populated by camera_info_callback
        self.camera_info: CameraInfo | None = None

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/oakd/rgb/preview/camera_info", self.camera_info_callback, 1)

        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth eroded", cv2.WINDOW_NORMAL)

        # Accumulated confirmed ring detections: list of dicts
        # {"x", "y", "z", "color", "count"}
        self.ring_detections: list[dict] = []

        # Publish confirmed rings as a persistent MarkerArray so any node that
        # subscribes (even later) immediately gets the full current set.
        self.ring_marker_pub = self.create_publisher(
            MarkerArray, "/detected_rings", qos_profile)

        # Load previously saved detections if available
        self.load_detections()

        # Latest depth visualisation frame (uint8 grayscale) for HoughCircles
        self.latest_depth_viz: np.ndarray | None = None

        # State for change-based printing (avoids per-frame spam)
        self._prev_depth_circle_states: list | None = None

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def depth_callback(self, data: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        # Replace non-finite values with NaN so that all downstream code can
        # use np.isnan() to ignore invalid pixels.  The previous value of +inf
        # was used only for visualization (white = far/unknown); NaN serves the
        # same purpose after a dedicated viz conversion below.
        depth_image[~np.isfinite(depth_image)] = np.nan

        # Store for use in image_callback
        self.latest_depth_image = depth_image.copy()
        self.latest_depth_header = data.header

        # Visualisation: map finite values to [0, 255]; NaN -> 255 (white = unknown)
        valid_mask = np.isfinite(depth_image)
        image_viz = np.full(depth_image.shape, 255, dtype=np.uint8)
        if valid_mask.any():
            d_min = depth_image[valid_mask].min()
            d_max = depth_image[valid_mask].max()
            if d_max > d_min:
                image_viz[valid_mask] = (
                    (depth_image[valid_mask] - d_min) / (d_max - d_min) * 254
                ).astype(np.uint8)
            else:
                image_viz[valid_mask] = 0

        # Save for HoughCircles in image_callback; imshow is done there with
        # the circles overlaid so the depth window stays in sync with RGB.
        self.latest_depth_viz = image_viz

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------

    def _sample_depth_patch(self, px: int, py: int, patch_r: int,
                             rgb_h: int = 0, rgb_w: int = 0) -> float | None:
        """Return the median valid depth (metres) in a square patch of radius *patch_r*
        around pixel (*px*, *py*) in the depth image.

        Handles resolution mismatch: pixel coords can be given in RGB-image space
        and are scaled to depth-image space when rgb_h/rgb_w are provided.

        Returns None if no valid pixels are found or the result is out of range.
        """
        if self.latest_depth_image is None:
            return None
        dh, dw = self.latest_depth_image.shape[:2]

        if rgb_h > 0 and rgb_w > 0 and (rgb_h != dh or rgb_w != dw):
            dpx = int(px * dw / rgb_w)
            dpy = int(py * dh / rgb_h)
        else:
            dpx, dpy = px, py

        y0, y1 = max(0, dpy - patch_r), min(dh, dpy + patch_r + 1)
        x0, x1 = max(0, dpx - patch_r), min(dw, dpx + patch_r + 1)
        patch = self.latest_depth_image[y0:y1, x0:x1]

        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) == 0:
            return None

        depth = float(np.median(valid))
        # OAK-D reports depth in mm when > 100, otherwise already in metres
        depth_m = depth / 1000.0 if depth > 100 else depth
        if depth_m <= 0.05 or depth_m > 15.0:
            return None
        return depth_m

    def _get_ring_band_depth(self, px: int, py: int, outer_r: int, inner_r: int,
                              img_h: int, img_w: int) -> float | None:
        """Sample depth at four compass points on the ring band (midway between
        inner and outer radii) and return the median of valid readings."""
        mid_r = int((outer_r + inner_r) / 2)
        patch_r = max(3, mid_r // 4)   # small patch, stays within the band
        sample_pts = [
            (px + mid_r, py),
            (px - mid_r, py),
            (px, py + mid_r),
            (px, py - mid_r),
        ]
        depths = []
        for spx, spy in sample_pts:
            d = self._sample_depth_patch(spx, spy, patch_r, img_h, img_w)
            if d is not None:
                depths.append(d)
        return float(np.median(depths)) if depths else None

    def _get_center_depth(self, px: int, py: int, inner_r: int,
                           img_h: int, img_w: int) -> float | None:
        """Sample depth at the centre hole of a ring candidate.

        Uses a patch radius of at most inner_r // 3 to avoid bleeding into the
        ring band region.
        """
        patch_r = max(2, inner_r // 3)
        return self._sample_depth_patch(px, py, patch_r, img_h, img_w)

    def _is_3d_ring(self, px: int, py: int, outer_r: int, inner_r: int,
                    img_h: int, img_w: int) -> bool:
        """Return True if any pixel inside the outer circle is at least
        RING_3D_DEPTH_DIFF_MIN_M farther than the ring-band depth.

        Scanning all pixels within the outer circle (the full interior including
        the hole) is more robust than scanning only the inner circle, because it
        captures any background visible through the ring regardless of whether
        the inner edge was detected accurately.

        If the outer-circle check fails, falls back to checking only the inner
        circle (the true hole) to avoid false rejections when the outer circle
        includes ring-band pixels at the same depth as the ring.

        If depth data is unavailable the check is skipped (returns True) to
        avoid false rejections.
        """
        ring_depth = self._get_ring_band_depth(px, py, outer_r, inner_r, img_h, img_w)
        if ring_depth is None:
            print(f"    [3D] ({px},{py}): ring_depth=None => PASS (no band data)")
            return True  # no band data - pass through

        if self.latest_depth_image is None:
            print(f"    [3D] ({px},{py}): no depth image => PASS")
            return True
        dh, dw = self.latest_depth_image.shape[:2]

        # Map RGB-space centre and radius to depth-image space
        dpx = int(px * dw / img_w)
        dpy = int(py * dh / img_h)
        d_outer_r = max(2, int(outer_r * dw / img_w))
        d_inner_r = max(2, int(inner_r * dw / img_w))

        # First attempt: check within outer circle
        y0 = max(0, dpy - d_outer_r)
        y1 = min(dh, dpy + d_outer_r + 1)
        x0 = max(0, dpx - d_outer_r)
        x1 = min(dw, dpx + d_outer_r + 1)

        patch = self.latest_depth_image[y0:y1, x0:x1]

        # Circular mask within the bounding box
        ys, xs = np.ogrid[y0:y1, x0:x1]
        circle_mask = ((xs - dpx) ** 2 + (ys - dpy) ** 2) <= d_outer_r ** 2

        valid = patch[circle_mask & np.isfinite(patch) & (patch > 0)]
        if len(valid) > 0:
            # Convert to metres (OAK-D: > 100 means mm)
            depths_m = np.where(valid > 100, valid / 1000.0, valid)
            threshold = ring_depth + RING_3D_DEPTH_DIFF_MIN_M
            found = bool(np.any(depths_m > threshold))
            if found:
                return True
            print(f"    [3D-outer] ({px},{py}): ring_depth={ring_depth:.3f}m  max_inside={float(depths_m.max()):.3f}m  need>{threshold:.3f}m  => trying inner fallback")
        else:
            print(f"    [3D-outer] ({px},{py}): no valid depth data in outer circle => trying inner fallback")

        # Fallback: check within inner circle only (the true hole)
        y0_inner = max(0, dpy - d_inner_r)
        y1_inner = min(dh, dpy + d_inner_r + 1)
        x0_inner = max(0, dpx - d_inner_r)
        x1_inner = min(dw, dpx + d_inner_r + 1)

        patch_inner = self.latest_depth_image[y0_inner:y1_inner, x0_inner:x1_inner]
        ys_inner, xs_inner = np.ogrid[y0_inner:y1_inner, x0_inner:x1_inner]
        circle_mask_inner = ((xs_inner - dpx) ** 2 + (ys_inner - dpy) ** 2) <= d_inner_r ** 2

        valid_inner = patch_inner[circle_mask_inner & np.isfinite(patch_inner) & (patch_inner > 0)]
        if len(valid_inner) == 0:
            print(f"    [3D-inner] ({px},{py}): no valid depth data in inner circle => PASS")
            return True  # no depth data - pass through

        depths_m_inner = np.where(valid_inner > 100, valid_inner / 1000.0, valid_inner)
        threshold = ring_depth + RING_3D_DEPTH_DIFF_MIN_M
        found_inner = bool(np.any(depths_m_inner > threshold))
        
        if not found_inner:
            print(f"    [3D-inner] ({px},{py}): ring_depth={ring_depth:.3f}m  max_inside_inner={float(depths_m_inner.max()):.3f}m  need>{threshold:.3f}m  => FAILED")
        return found_inner

    # ------------------------------------------------------------------
    # 3-D position helpers
    # ------------------------------------------------------------------

    def _ring_physical_diameter_m(self, pixel_radius: float, depth_m: float) -> float | None:
        """Estimate the physical diameter (metres) of a circle from its pixel
        radius and the depth to the ring plane."""
        if self.camera_info is None:
            return None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        if fx == 0 or fy == 0:
            return None
        avg_f = (fx + fy) / 2.0
        return 2.0 * pixel_radius * depth_m / avg_f

    def _ring_map_position(self, pixel_x: int, pixel_y: int,
                            depth_m: float, image_header) -> tuple | None:
        """Back-project ring centre pixel to /map frame via TF2.

        Returns (x, y, z) in metres or None on failure.
        """
        if self.camera_info is None:
            return None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        if fx == 0 or fy == 0:
            return None

        # Back-project in optical-frame coordinates (Z = depth = forward).
        x_opt = (pixel_x - cx) * depth_m / fx
        y_opt = (pixel_y - cy) * depth_m / fy
        z_opt = depth_m

        # The TF static tree has identity rotation between oakd_rgb_camera_frame
        # and oakd_rgb_camera_optical_frame, meaning both frames share BODY
        # orientation (X-forward, Y-left, Z-up).  Feeding Z=depth into that frame
        # would be interpreted as "upward", rotating the map position ~90°.
        # Convert optical → body frame manually before giving the point to TF:
        #   body-X = opt-Z  (depth = forward)
        #   body-Y = -opt-X (camera-right = robot-left)
        #   body-Z = -opt-Y (camera-down  = robot-up)
        pt = PointStamped()
        pt.header.frame_id = "oakd_rgb_camera_frame"
        pt.header.stamp = image_header.stamp
        pt.point.x = float(z_opt)
        pt.point.y = float(-x_opt)
        pt.point.z = float(-y_opt)

        try:
            pt_map = self.tf_buffer.transform(
                pt, 'map', timeout=rclpy.duration.Duration(seconds=0.5))
            return (pt_map.point.x, pt_map.point.y, pt_map.point.z)
        except Exception as e:
            self.get_logger().debug(f"TF ring->map failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Main image processing
    # ------------------------------------------------------------------

    def image_callback(self, data: Image):
        """Detect ring candidates with HoughCircles on the depth image and classify
        colour from the RGB image annulus pixels.

        Pipeline:
          1. Detect circles with cv2.HoughCircles on the depth visualisation image.
          2. Find concentric pairs (same centre, different radii) in depth space.
          3. Scale concentric-pair coordinates to RGB image space.
          4. Discard 2-D paintings using a depth-hole check (centre deeper than band).
          5. Back-project valid candidates to /map frame via depth + TF2.
          6. Accept rings whose map-Z is in the expected elevation range.
          7. Sample ring colour from the RGB annulus pixels.
          8. Merge duplicates, publish MarkerArray.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        img_h, img_w = cv_image.shape[:2]

        if not getattr(self, '_diag_logged', False):
            dshape = self.latest_depth_image.shape \
                if self.latest_depth_image is not None else 'None'
            print(f"[DIAG] RGB: {img_w}x{img_h}  Depth: {dshape}")
            self._diag_logged = True

        if self.latest_depth_image is None:
            print("[WARN] No depth image received yet - skipping frame")
            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)
            return

        if self.camera_info is None:
            print("[WARN] No camera_info received yet - skipping frame")
            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)
            return

        # 1-2. HoughCircles on depth image + find concentric pairs ----------
        ring_candidates = []
        depth_circle_states = []  # for change-based logging

        if self.latest_depth_viz is not None:
            dh, dw = self.latest_depth_viz.shape[:2]

            # Erode depth viz so dark (close) ring-wall pixels are thickened.
            # Close objects = dark in the viz; erode is a min-filter that expands
            # dark regions, making thin walls wide enough for two Hough edges.
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (DEPTH_EROSION_KERNEL, DEPTH_EROSION_KERNEL))
            depth_erode = cv2.erode(self.latest_depth_viz, kernel)

            # Optionally restrict HoughCircles to the upper half of the image.
            # Rings (physical hoops on stands) always appear in the upper half
            # of the camera view; the lower half mostly contains floor / table
            # edges that generate spurious circles.
            if HOUGH_UPPER_HALF_ONLY:
                hough_input_eroded = cv2.medianBlur(depth_erode[:dh // 2, :], 5)
                hough_input_uneroded = cv2.medianBlur(self.latest_depth_viz[:dh // 2, :], 5)
            else:
                hough_input_eroded = cv2.medianBlur(depth_erode, 5)
                hough_input_uneroded = cv2.medianBlur(self.latest_depth_viz, 5)
            hough_input = hough_input_eroded  # for backward compat with sweep code

            # Clamp maxRadius so it doesn't exceed half the hough_input dimensions.
            # OpenCV requires the image to be at least 2*maxRadius+1 in each
            # dimension; violating this causes HoughCircles to return nothing.
            h_in, w_in = hough_input_eroded.shape[:2]
            effective_max_r = min(HOUGH_MAX_RADIUS, h_in // 2 - 1, w_in // 2 - 1)
            effective_max_r = max(effective_max_r, HOUGH_MIN_RADIUS + 2)
            split_r = min(HOUGH_SPLIT_RADIUS, effective_max_r - 1)
            split_r = max(split_r, HOUGH_MIN_RADIUS + 1)

            # Two-pass Hough: ring inner and outer edges straddle the split so
            # they land in different passes and are never suppressed by minDist.
            # Run on BOTH eroded and un-eroded depth viz to catch rings that are
            # too thin for erosion and rings that need erosion to be detected.
            _hough_kwargs = dict(
                method=cv2.HOUGH_GRADIENT,
                dp=HOUGH_DP,
                minDist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1,
                param2=HOUGH_PARAM2,
            )
            
            # Run on eroded image
            circles_inner_e = cv2.HoughCircles(
                hough_input_eroded,
                minRadius=HOUGH_MIN_RADIUS,
                maxRadius=split_r,
                **_hough_kwargs,
            )
            circles_outer_e = cv2.HoughCircles(
                hough_input_eroded,
                minRadius=split_r,
                maxRadius=effective_max_r,
                **_hough_kwargs,
            )
            
            # Run on un-eroded image
            circles_inner_u = cv2.HoughCircles(
                hough_input_uneroded,
                minRadius=HOUGH_MIN_RADIUS,
                maxRadius=split_r,
                **_hough_kwargs,
            )
            circles_outer_u = cv2.HoughCircles(
                hough_input_uneroded,
                minRadius=split_r,
                maxRadius=effective_max_r,
                **_hough_kwargs,
            )
            
            # Merge circles from both sources
            _parts = [c[0] for c in (circles_inner_e, circles_outer_e, 
                                      circles_inner_u, circles_outer_u) if c is not None]
            depth_circles_raw = (
                np.array([np.concatenate(_parts, axis=0)]) if _parts else None
            )
            depth_viz_bgr = cv2.cvtColor(self.latest_depth_viz, cv2.COLOR_GRAY2BGR)
            depth_erode_bgr = cv2.cvtColor(depth_erode, cv2.COLOR_GRAY2BGR)

            if depth_circles_raw is not None:
                # Circles are already in native depth-image space
                detected_d = np.around(depth_circles_raw[0]).astype(int)
                pair_indices: set[int] = set()
                # Best rejection reason per circle index
                # Priority: "no-close-circle" < "zero-radius" < "ratio OOR" < "border"
                _REASON_PRIORITY = {
                    "no-close-circle": 0,
                    "zero-radius":     1,
                    "ratio OOR":       2,
                    "border":          3,
                }
                reject_reason: dict[int, str] = {k: "no-close-circle" for k in range(len(detected_d))}

                def _update_reason(idx: int, reason: str):
                    current = reject_reason.get(idx, "no-close-circle")
                    if _REASON_PRIORITY.get(reason, 0) > _REASON_PRIORITY.get(current, 0):
                        reject_reason[idx] = reason

                # Nearest-neighbour concentric-pair matching.
                # For each circle i find the closest other circle j; accept as
                # a concentric pair when the centre offset is within
                # CONCENTRIC_CENTER_FRAC * outer_radius (adaptive, not fixed px).
                # Sort candidate partners by distance so we always try the
                # closest one first and record the best failure reason.
                n_circles = len(detected_d)
                for i in range(n_circles):
                    c1 = detected_d[i]
                    # Build list of (dist, j) for all j != i
                    partners = []
                    for j in range(n_circles):
                        if j == i:
                            continue
                        d = math.hypot(float(c1[0] - detected_d[j][0]),
                                       float(c1[1] - detected_d[j][1]))
                        partners.append((d, j))
                    partners.sort()

                    for dist, j in partners:
                        if i in pair_indices and j in pair_indices:
                            break  # already paired
                        c2 = detected_d[j]
                        outer, inner = (c1, c2) if c1[2] >= c2[2] else (c2, c1)
                        outer_r_d = int(outer[2])
                        if outer_r_d == 0:
                            _update_reason(i, "zero-radius")
                            continue

                        # Adaptive centre threshold
                        max_dist = CONCENTRIC_CENTER_FRAC * outer_r_d
                        if dist > max_dist:
                            # This is the closest candidate and it's still too far
                            _update_reason(i, f"no-close-circle (best dist={dist:.1f}, max={max_dist:.1f})")
                            break  # no closer partner exists

                        ratio = float(inner[2]) / float(outer_r_d)
                        if not (CONCENTRIC_RATIO_MIN <= ratio <= CONCENTRIC_RATIO_MAX):
                            _update_reason(i, f"ratio {ratio:.2f} OOR")
                            _update_reason(j, f"ratio {ratio:.2f} OOR")
                            continue

                        # Averaged centre in depth space, then scale to RGB
                        dpx = (int(outer[0]) + int(inner[0])) // 2
                        dpy = (int(outer[1]) + int(inner[1])) // 2
                        rgb_px  = int(dpx        * img_w / dw)
                        rgb_py  = int(dpy        * img_h / dh)
                        rgb_or  = int(outer_r_d  * img_w / dw)
                        rgb_ir  = int(inner[2]   * img_w / dw)

                        # Border check in RGB space
                        if not (img_h * 0.08 < rgb_py < img_h * 0.92):
                            _update_reason(i, "border")
                            _update_reason(j, "border")
                            continue

                        pair_indices.update([i, j])
                        ring_candidates.append({
                            "px": rgb_px, "py": rgb_py,
                            "outer_r": rgb_or, "inner_r": rgb_ir,
                        })
                        break  # i is paired; move to next i

                # ------------------------------------------------------------------
                # Targeted radius-sweep for circles still unpaired after the main pass.
                # For each unpaired circle at radius r, run HoughCircles at single-
                # radius bands: r-1, r-2, ..., down to 2 (looking for a smaller inner
                # partner), then r+1, ..., r+MAX_PX_FOR_OUTER_LOOKOUT (looking for a
                # larger outer partner that the main pass may have missed).
                # ------------------------------------------------------------------
                unpaired_now = [i for i in range(n_circles) if i not in pair_indices]
                for i in unpaired_now:
                    c1 = detected_d[i]
                    cx1, cy1, r1 = int(c1[0]), int(c1[1]), int(c1[2])

                    # Build the radius sequence: down first, then up
                    radii_to_try: list[int] = []
                    for delta in range(1, r1):  # r-1, r-2, ..., 2
                        t = r1 - delta
                        if t < 2:
                            break
                        radii_to_try.append(t)
                    for delta in range(1, MAX_PX_FOR_OUTER_LOOKOUT + 1):  # r+1 .. r+MAX
                        radii_to_try.append(r1 + delta)

                    found_pair = False
                    for target_r in radii_to_try:
                        # Skip if target radius would violate OpenCV size constraints
                        if target_r < 1 or target_r >= h_in // 2 or target_r >= w_in // 2:
                            continue
                        extra = cv2.HoughCircles(
                            hough_input,
                            minRadius=target_r,
                            maxRadius=target_r,
                            **_hough_kwargs,
                        )
                        if extra is None:
                            continue
                        for ec in extra[0]:
                            ecx = int(round(ec[0]))
                            ecy = int(round(ec[1]))
                            er  = int(round(ec[2]))
                            dist = math.hypot(cx1 - ecx, cy1 - ecy)
                            # Determine which is outer / inner
                            if r1 >= er:
                                outer_c, inner_c = c1, ec
                                outer_r_d = r1
                            else:
                                outer_c, inner_c = ec, c1
                                outer_r_d = er
                            if outer_r_d == 0:
                                continue
                            max_dist = CONCENTRIC_CENTER_FRAC * outer_r_d
                            if dist > max_dist:
                                continue
                            ratio = float(min(r1, er)) / float(outer_r_d)
                            if not (CONCENTRIC_RATIO_MIN <= ratio <= CONCENTRIC_RATIO_MAX):
                                continue
                            # Valid concentric pair - compute RGB-scaled coords
                            dpx_s = (cx1 + ecx) // 2
                            dpy_s = (cy1 + ecy) // 2
                            rgb_px  = int(dpx_s   * img_w / dw)
                            rgb_py  = int(dpy_s   * img_h / dh)
                            rgb_or  = int(outer_r_d          * img_w / dw)
                            rgb_ir  = int(min(r1, er)        * img_w / dw)
                            if not (img_h * 0.08 < rgb_py < img_h * 0.92):
                                continue
                            pair_indices.add(i)
                            ring_candidates.append({
                                "px": rgb_px, "py": rgb_py,
                                "outer_r": rgb_or, "inner_r": rgb_ir,
                            })
                            reject_reason[i] = f"paired-via-sweep (partner_r={target_r})"
                            found_pair = True
                            break
                        if found_pair:
                            break

                # Annotate depth window: green = part of concentric pair, orange = lone
                for idx, dc in enumerate(detected_d):
                    dpx, dpy, dr = int(dc[0]), int(dc[1]), int(dc[2])
                    is_pair = idx in pair_indices
                    reason_str = "" if is_pair else f" ({reject_reason.get(idx, '?')})"
                    label = "RING" if is_pair else f"no-pair{reason_str}"
                    colour = (0, 255, 0) if is_pair else (0, 165, 255)
                    # Native-resolution depth window
                    cv2.circle(depth_viz_bgr, (dpx, dpy), dr, colour, 1)
                    cv2.circle(depth_viz_bgr, (dpx, dpy), 2, colour, -1)
                    cv2.putText(depth_viz_bgr, label, (dpx + dr + 2, dpy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
                    # Eroded window (same native coords)
                    cv2.circle(depth_erode_bgr, (dpx, dpy), dr, colour, 2)
                    cv2.circle(depth_erode_bgr, (dpx, dpy), 3, colour, -1)
                    cv2.putText(depth_erode_bgr, label, (dpx + dr + 3, dpy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
                    depth_circle_states.append((dpx, dpy, dr, label))

            cv2.imshow("Depth eroded", depth_erode_bgr)

            cv2.imshow("Depth window", depth_viz_bgr)
            cv2.waitKey(1)

            # Log on change only
            if depth_circle_states != self._prev_depth_circle_states:
                if not depth_circle_states:
                    print("[INFO] Depth HoughCircles: no circles detected")
                else:
                    print(f"[INFO] Depth HoughCircles: {len(depth_circle_states)} circle(s), "
                          f"{len(ring_candidates)} concentric pair(s)")
                    if len(ring_candidates) > 0:
                        print('\n\n\n\n\n\n\n\n')
                    for dpx, dpy, dr, label in depth_circle_states:
                        print(f"  depth circle at ({dpx},{dpy}) r={dr}  [{label}]")
                self._prev_depth_circle_states = depth_circle_states

        # 3-7. Validate each candidate and build map position ---------------
        for c in ring_candidates:
            px, py  = c["px"], c["py"]
            outer_r = c["outer_r"]
            inner_r = c["inner_r"]

            # --- depth check: is the ring actually 3-D? ---
            # NOTE: _is_3d_ring reads self.latest_depth_image (raw float32 metres
            # from depth_callback), NOT the eroded uint8 visualisation used for
            # HoughCircles.  Erosion only affects circle pixel detection, not the
            # depth values sampled here.
            if not self._is_3d_ring(px, py, outer_r, inner_r, img_h, img_w):
                print(f"  Ring at ({px},{py}): depth hole not found - 2-D drawing, skipped")
                continue

            # --- depth at ring band ---
            depth_m = self._get_ring_band_depth(px, py, outer_r, inner_r, img_h, img_w)
            if depth_m is None:
                print(f"  Ring at ({px},{py}): no valid depth on ring band - skipped")
                continue

            # --- physical size filter ---
            diameter_m = self._ring_physical_diameter_m(float(outer_r), depth_m)
            if diameter_m is None or \
               not (RING_MIN_DIAMETER_M <= diameter_m <= RING_MAX_DIAMETER_M):
                print(f"  Ring at ({px},{py}): diameter {diameter_m} m out of range - skipped")
                continue

            # --- back-project to /map ---
            map_pos = self._ring_map_position(px, py, depth_m, data.header)
            if map_pos is None:
                print(f"  Ring at ({px},{py}): TF to map failed - skipped")
                continue
            wx, wy, wz = map_pos

            # --- elevation filter ---
            if not (RING_MIN_Z_MAP_M <= wz <= RING_MAX_Z_MAP_M):
                print(f"  Ring at ({px},{py}): map-Z {wz:.2f} m out of "
                      f"[{RING_MIN_Z_MAP_M}, {RING_MAX_Z_MAP_M}] - skipped")
                continue

            # --- colour from RGB annulus pixels (depth-derived coords scaled to RGB) ---
            color_name = self._get_ring_color(cv_image, px, py, outer_r, inner_r)

            print(f"  Valid ring at ({px},{py}): depth={depth_m:.2f} m  "
                  f"diam={diameter_m*100:.1f} cm  "
                  f"map=({wx:.2f}, {wy:.2f}, {wz:.2f})  colour={color_name}")

            self._process_ring_detection(wx, wy, wz, color_name)

            # Draw accepted ring on RGB image
            cv2.circle(cv_image, (px, py), outer_r, (0, 255, 0), 2)
            cv2.circle(cv_image, (px, py), inner_r, (0, 200, 0), 2)
            cv2.circle(cv_image, (px, py), 2, (0, 0, 255), 3)

        cv2.imshow("Detected rings", cv_image)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Colour classification
    # ------------------------------------------------------------------

    def _get_ring_color(self, cv_image: np.ndarray, cx: int, cy: int,
                         outer_r: int, inner_r: int) -> str:
        """Classify the dominant colour of the ring annulus in HSV space."""
        h, w = cv_image.shape[:2]
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer_mask, (cx, cy), outer_r, 255, -1)
        cv2.circle(inner_mask, (cx, cy), inner_r, 255, -1)
        ring_mask = cv2.subtract(outer_mask, inner_mask)

        if cv2.countNonZero(ring_mask) == 0:
            return "unknown"

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h_val, s_val, v_val = cv2.mean(hsv, mask=ring_mask)[:3]

        if s_val < 40:
            return "black" if v_val < 80 else "white"
        if h_val < 10 or h_val > 170:  return "red"
        if 10  <= h_val < 25:           return "orange"
        if 25  <= h_val < 35:           return "yellow"
        if 35  <= h_val < 85:           return "green"
        if 85  <= h_val < 130:          return "blue"
        if 130 <= h_val < 170:          return "purple"
        return "unknown"

    # ------------------------------------------------------------------
    # Detection accumulation & publishing
    # ------------------------------------------------------------------

    def _process_ring_detection(self, x, y, z, color_name):
        """Merge this /map-frame sighting into confirmed rings via running average.

        Every valid sighting is immediately confirmed (RING_MIN_CONFIRMATIONS=1).
        Subsequent sightings within RING_DEDUP_DISTANCE_M update the running
        average and republish, so the marker always reflects the latest position.
        """
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            return
        for ring in self.ring_detections:
            if math.sqrt((x - ring["x"])**2 + (y - ring["y"])**2) < RING_DEDUP_DISTANCE_M:
                # Update running average
                ring["count"] += 1
                n = ring["count"]
                ring["x"] += (x - ring["x"]) / n
                ring["y"] += (y - ring["y"]) / n
                ring["z"] += (z - ring["z"]) / n
                self.get_logger().debug(
                    f"UPDATED ring {ring['color']} pos=({ring['x']:.2f}, {ring['y']:.2f}, {ring['z']:.2f}) n={n}")
                self._publish_ring_markers()
                return
        # Brand-new ring
        self.ring_detections.append({"x": x, "y": y, "z": z, "color": color_name, "count": 1})
        self.get_logger().info(
            f"NEW ring: colour={color_name} pos=({x:.2f}, {y:.2f}, {z:.2f}) total={len(self.ring_detections)}")
        self._publish_ring_markers()

    def save_detections_to_json(self):
        """Persist ring detections to a JSON file."""
        try:
            with open(self.ring_detections_json_path, 'w') as f:
                json.dump(self.ring_detections, f, indent=2)
            self.get_logger().info(f"Saved {len(self.ring_detections)} ring detections to {self.ring_detections_json_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save ring detections: {e}")

    def load_detections(self):
        """Load previously saved ring detections from JSON file."""
        try:
            with open(self.ring_detections_json_path, 'r') as f:
                data = json.load(f)
            self.ring_detections = data
            self.get_logger().info(f"Loaded {len(self.ring_detections)} previous ring detections from {self.ring_detections_json_path}")
            self._publish_ring_markers()
        except FileNotFoundError:
            self.get_logger().info("No previous ring detections file found, starting fresh.")
        except Exception as e:
            self.get_logger().warn(f"Could not load ring detections: {e}")

    def _publish_ring_markers(self):
        """Publish all confirmed rings as a MarkerArray on /detected_rings."""
        ma = MarkerArray()
        for i, ring in enumerate(self.ring_detections):
            x, y, z, cname = ring["x"], ring["y"], ring["z"], ring["color"]
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "detected_rings"
            m.id = i
            m.type = 2

            scale = 0.15
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale

            if cname == "red":
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
            elif cname == "green":
                m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
            elif cname == "blue":
                m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0
            elif cname == "yellow":
                m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0
            elif cname == "orange":
                m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0
            elif cname == "purple":
                m.color.r = 0.5; m.color.g = 0.0; m.color.b = 1.0
            elif cname == "black":
                m.color.r = 0.1; m.color.g = 0.1; m.color.b = 0.1
            elif cname == "white":
                m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0
            else:
                m.color.r = 0.5; m.color.g = 0.5; m.color.b = 0.5
            m.color.a = 1.0

            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z

            ma.markers.append(m)
        self.ring_marker_pub.publish(ma)


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()
    try:
        rclpy.spin(rd_node)
    except KeyboardInterrupt:
        pass
    finally:
        rd_node.save_detections_to_json()
        rd_node.destroy_node()
        rclpy.shutdown()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()