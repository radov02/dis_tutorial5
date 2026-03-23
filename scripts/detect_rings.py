#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2, math
import numpy as np
import tf2_ros
import tf2_geometry_msgs

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3, Pose
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

# Physical size limits for valid rings (metres)
RING_MIN_DIAMETER_M = 0.02   # 2 cm
RING_MAX_DIAMETER_M = 0.25   # 25 cm

# Z-coordinate range in /map frame for valid rings (metres)
RING_MIN_Z_MAP_M = 1.20
RING_MAX_Z_MAP_M = 3.00

RING_DEDUP_DISTANCE_M = 0.5       # merge ring detections closer than this (metres)

# Geofence: same polygon as robot_commander.py – rings outside are discarded.
ALLOWED_AREA_POLYGON: list[tuple[float, float]] = [
    (-4.52, -8.10),
    (-4.48,  0.75),
    ( 3.07,  0.75),
    ( 3.02, -8.01),
]


def _point_in_polygon(wx: float, wy: float) -> bool:
    """Ray-casting point-in-polygon test for ALLOWED_AREA_POLYGON."""
    poly = ALLOWED_AREA_POLYGON
    n = len(poly)
    if n < 3:
        return True
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


class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 5
        timer_period = 1/timer_frequency

        # ellipse thresholds
        self.ecc_thr = 100
        self.ratio_thr = 1.5
        self.center_thr = 10

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Latest depth image – used for 3-D ring position from camera intrinsics
        self.latest_depth_image = None
        self.latest_depth_header = None

        # Camera intrinsics (populated by camera_info_callback, used for diameter estimate)
        self.camera_info = None

        # TF2 for transforming ring positions to /map frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, "/oakd/rgb/preview/camera_info", self.camera_info_callback, 1)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

        # Ring detections: list of dicts {"x", "y", "z", "color", "count"}
        # Every valid ring sighting immediately merges into this list.
        self.ring_detections = []

        # Publisher for confirmed ring detections
        self.ring_marker_pub = self.create_publisher(
            MarkerArray, "/detected_rings", qos_profile)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def _get_depth_at_pixel(self, px: int, py: int, rgb_h: int = 0, rgb_w: int = 0) -> float | None:
        """Return the median depth (metres) in a small patch around (px, py).

        Handles resolution mismatch between the RGB and depth images by scaling
        the pixel coordinates accordingly.

        Returns None when depth is unavailable or out of the valid range.
        """
        if self.latest_depth_image is None:
            return None
        dh, dw = self.latest_depth_image.shape[:2]

        # Scale pixel coordinates from RGB resolution to depth resolution
        if rgb_h > 0 and rgb_w > 0 and (rgb_h != dh or rgb_w != dw):
            scale_x = dw / rgb_w
            scale_y = dh / rgb_h
            dpx = int(px * scale_x)
            dpy = int(py * scale_y)
        else:
            dpx, dpy = px, py

        r = 7  # larger patch (15x15) to cope with sparse depth regions
        y0, y1 = max(0, dpy - r), min(dh, dpy + r + 1)
        x0, x1 = max(0, dpx - r), min(dw, dpx + r + 1)
        patch = self.latest_depth_image[y0:y1, x0:x1]
        valid = patch[(patch > 0) & (~np.isinf(patch)) & (~np.isnan(patch))]
        if len(valid) == 0:
            return None
        depth = float(np.median(valid))
        # OAK-D reports depth in mm when > 100, otherwise already in metres
        depth_m = depth / 1000.0 if depth > 100 else depth
        if depth_m <= 0.05 or depth_m > 15.0:
            return None
        return depth_m

    def _ring_physical_diameter_m(self, outer_ellipse, depth_m: float) -> float | None:
        """Estimate the physical diameter of the ring (outer ellipse) in metres."""
        if self.camera_info is None:
            return None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        if fx == 0 or fy == 0:
            return None
        # Use the average of both semi-axes in pixels as a representative radius
        pixel_radius = (outer_ellipse[1][0] + outer_ellipse[1][1]) / 4.0  # axes are diameters
        avg_f = (fx + fy) / 2.0
        physical_diameter = 2.0 * pixel_radius * depth_m / avg_f
        return physical_diameter

    def _ring_map_position(self, pixel_x: int, pixel_y: int, depth_m: float, image_header) -> tuple | None:
        """Return (x, y, z) of the ring centre in the /map frame, or None."""
        if self.camera_info is None:
            return None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        if fx == 0 or fy == 0:
            return None

        pt = PointStamped()
        # Use the frame_id from camera_info – it always matches the intrinsics
        # (K matrix).  Using the RGB image header.frame_id can differ and
        # causes the depth axis to fold onto the wrong map axis.
        pt.header.frame_id = self.camera_info.header.frame_id
        pt.header.stamp = image_header.stamp
        pt.point.x = float((pixel_x - cx) * depth_m / fx)
        pt.point.y = float((pixel_y - cy) * depth_m / fy)
        pt.point.z = float(depth_m)

        try:
            pt_map = self.tf_buffer.transform(pt, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
            return (pt_map.point.x, pt_map.point.y, pt_map.point.z)
        except Exception as e:
            self.get_logger().debug(f"TF ring->map failed: {e}")
            return None

    def image_callback(self, data):
        """Detect ring candidates in the RGB image and resolve their 3-D position.

        Uses the depth image + camera intrinsics to back-project the ring centre
        pixel into camera space, then transforms to /map frame via TF2.
        No point cloud is required.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        img_h, img_w = cv_image.shape[:2]

        # One-time diagnostic: log resolution of both images
        if not getattr(self, '_diag_logged', False):
            dshape = self.latest_depth_image.shape if self.latest_depth_image is not None else 'None'
            print(f"[DIAG] RGB image: {img_w}x{img_h}, Depth image: {dshape}")
            self._diag_logged = True

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Binarize the image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
        cv2.imshow("Binary Image", thresh)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 1)
        cv2.imshow("Detected contours", gray)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)

                # filter ellipses that are too eccentric
                e = ellipse[1]
                ecc1 = e[0]
                ecc2 = e[1]

                ratio = ecc1/ecc2 if ecc1>ecc2 else ecc2/ecc1
                if ratio<=self.ratio_thr and ecc1<self.ecc_thr and ecc2<self.ecc_thr:

                    elps.append(ellipse)

        # Find two elipses with same centers
        new_candidates = []
        for n in range(len(elps)):
            e1 = elps[n]

            # display candidates
            cv2.ellipse(cv_image, e1, (255, 255, 0), 1)
            cv2.circle(cv_image, (int(e1[0][0]), int(e1[0][1])), 1, (255, 255, 0), -1)

            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees

                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))

                # The centers of the two elipses should be within 10 pixels of each other
                if dist >= self.center_thr:
                    continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring

                px = int(le[0][0])
                py = int(le[0][1])

                # ---- Pixel-space filters (no depth needed) ----
                outer_area = le[1][0] * le[1][1]
                inner_area = se[1][0] * se[1][1]
                if outer_area == 0:
                    continue
                area_ratio = inner_area / outer_area
                if area_ratio < 0.20 or area_ratio > 0.85:
                    continue
                if le[1][0] < 15 or le[1][1] < 15:
                    continue
                if py < img_h * 0.08 or py > img_h * 0.92:
                    continue

                new_candidates.append({"px": px, "py": py, "le": le, "se": se,
                                       "cv_image": cv_image.copy()})

        # Process each candidate immediately using the depth image + camera intrinsics
        for c in new_candidates:
            px, py   = c["px"], c["py"]
            le, se   = c["le"], c["se"]
            cimg     = c["cv_image"]

            depth_m = self._get_depth_at_pixel(px, py, img_h, img_w)
            if depth_m is None:
                print(f"  Ring at ({px},{py}): no valid depth - skipped")
                continue

            diameter_m = self._ring_physical_diameter_m(le, depth_m)
            if diameter_m is None or not (RING_MIN_DIAMETER_M <= diameter_m <= RING_MAX_DIAMETER_M):
                print(f"  Ring at ({px},{py}): physical diameter {diameter_m} m out of range - skipped")
                continue

            map_pos = self._ring_map_position(px, py, depth_m, data.header)
            if map_pos is None:
                print(f"  Ring at ({px},{py}): TF to map failed - skipped")
                continue
            wx, wy, wz = map_pos

            if not (RING_MIN_Z_MAP_M <= wz <= RING_MAX_Z_MAP_M):
                print(f"  Ring at ({px},{py}): map-Z {wz:.2f} m out of range [{RING_MIN_Z_MAP_M}, {RING_MAX_Z_MAP_M}] - skipped")
                continue

            if not _point_in_polygon(wx, wy):
                print(f"  Ring at ({px},{py}): map pos ({wx:.2f}, {wy:.2f}) outside allowed area - discarded")
                continue

            print(f"  Valid ring at ({px},{py}): depth={depth_m:.2f} m, diameter={diameter_m*100:.1f} cm, "
                  f"map_pos=({wx:.2f}, {wy:.2f}, {wz:.2f})")

            color_name = self._get_ring_color(cimg, le, se)
            self._process_ring_detection(wx, wy, wz, color_name)

        cv2.imshow("Detected rings", cv_image)
        cv2.waitKey(1)

    def depth_callback(self, data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        depth_image[depth_image==np.inf] = 0

        # Store the latest depth image so image_callback can use it for 3-D filtering
        self.latest_depth_image = depth_image.copy()
        self.latest_depth_header = data.header
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_max = np.max(image_1)
        if image_max > 0:
            image_1 = image_1 / image_max * 255

        image_viz = np.array(image_1, dtype= np.uint8)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)

    # ---- colour helpers ----

    def _get_ring_color(self, cv_image, outer_ellipse, inner_ellipse):
        """Extract the dominant colour of the ring region between the two ellipses."""
        h, w = cv_image.shape[:2]
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(outer_mask, outer_ellipse, 255, -1)
        cv2.ellipse(inner_mask, inner_ellipse, 255, -1)
        ring_mask = cv2.subtract(outer_mask, inner_mask)

        if cv2.countNonZero(ring_mask) == 0:
            return "unknown"

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_image, mask=ring_mask)[:3]
        h_val, s_val, v_val = mean_hsv

        if s_val < 40:
            return "black" if v_val < 80 else "white"
        if h_val < 10 or h_val > 170:
            return "red"
        if 10 <= h_val < 25:
            return "orange"
        if 25 <= h_val < 35:
            return "yellow"
        if 35 <= h_val < 85:
            return "green"
        if 85 <= h_val < 130:
            return "blue"
        if 130 <= h_val < 170:
            return "purple"
        return "unknown"

    # ---- detection accumulation (single-pass, always in /map frame) ----

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
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.text = cname  # color name for subscribers to read
            c = ColorRGBA(a=1.0)
            if cname == "red":      c.r = 1.0
            elif cname == "green":  c.g = 1.0
            elif cname == "blue":   c.b = 1.0
            elif cname == "yellow": c.r = 1.0; c.g = 1.0
            elif cname == "orange": c.r = 1.0; c.g = 0.5
            elif cname == "purple": c.r = 0.5; c.b = 1.0
            elif cname == "black":  c.r = c.g = c.b = 0.1
            elif cname == "white":  c.r = c.g = c.b = 1.0
            else:                   c.r = c.g = c.b = 0.5
            m.color = c
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            ma.markers.append(m)
        self.ring_marker_pub.publish(ma)


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()