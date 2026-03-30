#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo

from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import yaml
import json
from datetime import datetime

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

from ultralytics import YOLO

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
				('map_yaml_path', '/home/erik/rins/maps/map.yaml'),
				('map_pgm_path', '/home/erik/rins/maps/map.pgm'),
		])

		marker_topic = "/people_marker"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, qos_profile_sensor_data)
		self.camera_info_sub = self.create_subscription(CameraInfo, "/oakd/rgb/preview/camera_info", self.camera_info_callback, qos_profile_sensor_data)

		self.marker_array_pub = self.create_publisher(MarkerArray, "/people_marker_array", QoSReliabilityPolicy.BEST_EFFORT)

		# Publish face detections on /detected_faces with transient-local QoS
		# so late-subscribing nodes (halfautonomous_search, RViz) get the full set.
		from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy as Rel
		_faces_qos = QoSProfile(
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			reliability=Rel.RELIABLE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1)
		self.detected_faces_pub = self.create_publisher(MarkerArray, '/detected_faces', _faces_qos)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		# Depth & camera intrinsics state (same approach as detect_rings)
		self.latest_depth_image = None
		self.latest_depth_header = None
		self.camera_info = None

		# we will save face positions in the map coordinates:
		self.face_position_in_map_coordinates = []
		self.face_best_alignment_score = []  # tracks best combined score per face
		# minimum distance (meters) between two detections to consider them different people
		self.dedup_distance = 1.5
		# max_detection_z: height in map frame (Z=up). A person's face is ~1.0-1.8 m.
		self.max_detection_z = 2.0

		# map info for saving detections to PGM
		self.map_yaml_path = self.get_parameter('map_yaml_path').get_parameter_value().string_value
		self.map_pgm_path = self.get_parameter('map_pgm_path').get_parameter_value().string_value

		map_stem = os.path.splitext(os.path.basename(self.map_yaml_path))[0]
		_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
		_subfolder = os.path.join('/home/erik/rins/maps', _ts)
		os.makedirs(_subfolder, exist_ok=True)
		self.detections_json_path = os.path.join(
			_subfolder,
			f'people_detections_{map_stem}.json'
		)

		# load previously saved detections if available
		self.load_detections()

		# we need TF2 Listener, which will store the /tf and /tf_static topic messages into buffer which we will use to make transformation 
		# between camera and map frames:
		self.tf_buf = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
		self.create_timer(2.0, self.log_face_positions)
		self.create_timer(2.0, self._republish_face_markers)

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				bboxes = x.boxes.xyxy
				if bboxes.nelement() == 0: # skip if empty
					continue

				for bbox in bboxes:
					self.get_logger().info(f"Person has been detected!")

					x1 = int(bbox[0])
					y1 = int(bbox[1])
					x2 = int(bbox[2])
					y2 = int(bbox[3])

					# draw rectangle
					cv_image = cv2.rectangle(cv_image, (x1, y1), (x2, y2), self.detection_color, 3)

					cx = int((x1 + x2) / 2)
					cy = int((y1 + y2) / 2)

					# draw the center of bounding box
					cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

					self.faces.append({'cx': cx, 'cy': cy, 'bbox': (x1, y1, x2, y2)})

			# Process detected faces using depth image + intrinsics
			# (same approach as detect_rings - proven to produce correct map coords)
			if self.faces and self.latest_depth_image is not None and self.camera_info is not None:
				self.process_detections(data, cv_image.shape[:2])

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def depth_callback(self, data):
		"""Store latest depth image for use in face processing."""
		try:
			depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
		except CvBridgeError as e:
			print(e)
			return
		depth_image[~np.isfinite(depth_image)] = np.nan
		self.latest_depth_image = depth_image
		self.latest_depth_header = data.header

	def camera_info_callback(self, msg):
		self.camera_info = msg

	def _sample_depth_at_pixel(self, px, py, img_h, img_w, patch_r=3):
		"""Sample median depth at pixel (px, py) in RGB-image space.
		Handles resolution mismatch between RGB and depth images."""
		if self.latest_depth_image is None:
			return None
		dh, dw = self.latest_depth_image.shape[:2]
		dpx = int(px * dw / img_w) if img_w != dw else int(px)
		dpy = int(py * dh / img_h) if img_h != dh else int(py)
		dpx = int(np.clip(dpx, 0, dw - 1))
		dpy = int(np.clip(dpy, 0, dh - 1))

		y0, y1 = max(0, dpy - patch_r), min(dh, dpy + patch_r + 1)
		x0, x1 = max(0, dpx - patch_r), min(dw, dpx + patch_r + 1)
		patch = self.latest_depth_image[y0:y1, x0:x1]

		valid = patch[np.isfinite(patch) & (patch > 0)]
		if len(valid) == 0:
			return None

		depth = float(np.median(valid))
		# OAK-D may report in mm (value > 100) or metres
		depth_m = depth / 1000.0 if depth > 100 else depth
		if depth_m <= 0.05 or depth_m > 15.0:
			return None
		return depth_m

	def _backproject_to_body(self, px, py, depth_m):
		"""Back-project RGB pixel to camera body frame using intrinsics.
		Uses the same optical-to-body conversion as detect_rings:
		  body_X =  opt_Z  (depth  -> forward)
		  body_Y = -opt_X  (right  -> left, negated)
		  body_Z = -opt_Y  (down   -> up, negated)
		"""
		fx = self.camera_info.k[0]
		fy = self.camera_info.k[4]
		cx = self.camera_info.k[2]
		cy = self.camera_info.k[5]

		x_opt = (px - cx) * depth_m / fx
		y_opt = (py - cy) * depth_m / fy
		z_opt = depth_m

		return (z_opt, -x_opt, -y_opt)

	def process_detections(self, rgb_msg, img_shape):
		"""Process detected faces using depth image + camera intrinsics.
		Uses the same back-projection approach as detect_rings to ensure
		correct map positioning."""
		img_h, img_w = img_shape
		source_frame = "oakd_rgb_camera_frame"
		stamp = rgb_msg.header.stamp

		# Log once for debugging
		if not hasattr(self, '_logged_source_frame'):
			self.get_logger().info(
				f"Using depth+intrinsics back-projection (same as detect_rings), "
				f"source_frame='{source_frame}'"
			)
			self._logged_source_frame = True
			self._diag_count = 0

		# get robot position for offset orientation and fallback
		robot_x = None
		robot_y = None
		try:
			trans = self.tf_buf.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=Duration(seconds=0.5))
			robot_x = trans.transform.translation.x
			robot_y = trans.transform.translation.y
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			self.get_logger().warn(f"Could not look up base_link -> map transform: {e}")

		for detection in self.faces:
			px = detection['cx']
			py = detection['cy']
			x1, y1, x2, y2 = detection['bbox']

			# Sample depth at bbox center
			depth_m = self._sample_depth_at_pixel(px, py, img_h, img_w)
			if depth_m is None:
				self.get_logger().warn("Skipping detection: no valid depth")
				continue

			# Back-project to body frame (same as detect_rings)
			body_xyz = self._backproject_to_body(px, py, depth_m)

			try:
				point_in_map = self.transform_point_to_map(body_xyz, source_frame, stamp)
				map_x = point_in_map.point.x
				map_y = point_in_map.point.y
				map_z = point_in_map.point.z

				# Diagnostic logging for first few detections
				diag_count = getattr(self, '_diag_count', 0)
				if diag_count < 5:
					self.get_logger().info(
						f"[DIAG {diag_count}] px=({px},{py}) depth={depth_m:.2f}m "
						f"body=({body_xyz[0]:.3f},{body_xyz[1]:.3f},{body_xyz[2]:.3f}) "
						f"-> map=({map_x:.2f},{map_y:.2f},{map_z:.2f}) "
						+ (f"robot=({robot_x:.2f},{robot_y:.2f})" if robot_x is not None else "robot=N/A")
					)
					self._diag_count = diag_count + 1

				if map_z > self.max_detection_z:
					self.get_logger().warn(f"Skipping: z={map_z:.2f}m > max {self.max_detection_z:.2f}m")
					continue

				# Compute wall tangent from left/right bbox corners in map space,
				# then place marker 0.5 m along the perpendicular toward the robot.
				offset_distance = 0.5
				offset_applied = False
				angle_of_view = None
				normal = None

				right_depth = self._sample_depth_at_pixel(x2, py, img_h, img_w)
				left_depth  = self._sample_depth_at_pixel(x1, py, img_h, img_w)
				point_in_map_right_bbox_corner = None
				point_in_map_left_bbox_corner  = None
				if right_depth is not None:
					right_body = self._backproject_to_body(x2, py, right_depth)
					point_in_map_right_bbox_corner = self.transform_point_to_map(right_body, source_frame, stamp)
				if left_depth is not None:
					left_body = self._backproject_to_body(x1, py, left_depth)
					point_in_map_left_bbox_corner = self.transform_point_to_map(left_body, source_frame, stamp)

				if (robot_x is not None and robot_y is not None
						and point_in_map_right_bbox_corner is not None
						and point_in_map_left_bbox_corner  is not None):
					tangent_dx = point_in_map_right_bbox_corner.point.x - point_in_map_left_bbox_corner.point.x
					tangent_dy = point_in_map_right_bbox_corner.point.y - point_in_map_left_bbox_corner.point.y
					tangent_norm = np.sqrt(tangent_dx**2 + tangent_dy**2)
					if tangent_norm > 0.01:
						to_robot_x = robot_x - map_x
						to_robot_y = robot_y - map_y
						cross_prod = tangent_dy * to_robot_x - tangent_dx * to_robot_y
						dot_prod   = tangent_dx * to_robot_x + tangent_dy * to_robot_y
						angle_of_view = np.arctan2(cross_prod, dot_prod)
						nx = -tangent_dy / tangent_norm
						ny =  tangent_dx / tangent_norm
						if nx * to_robot_x + ny * to_robot_y < 0:
							nx, ny = -nx, -ny
						normal = (nx, ny)

						# Apply 0.5 m offset along wall perpendicular
						map_x += nx * offset_distance
						map_y += ny * offset_distance
						offset_applied = True

				# Fallback: offset toward robot if wall tangent couldn't be computed
				if not offset_applied and robot_x is not None and robot_y is not None:
					to_robot_dx = robot_x - map_x
					to_robot_dy = robot_y - map_y
					to_robot_dist = np.sqrt(to_robot_dx**2 + to_robot_dy**2)
					if to_robot_dist > 0.01:
						map_x += (to_robot_dx / to_robot_dist) * offset_distance
						map_y += (to_robot_dy / to_robot_dist) * offset_distance
				dist_to_face = 100.0
				# compute combined alignment score for this observation
				combined_score = 0.0
				if angle_of_view is not None and normal is not None:
					# --- distance to face in map frame ---
					if robot_x is not None and robot_y is not None:
						dist_to_face = np.sqrt((robot_x - map_x)**2 + (robot_y - map_y)**2)
					else:
						dist_to_face = 2.0  # neutral fallback

					# Factor 1: bbox perpendicularity, sharpness scaled by distance
					min_k = 1.0
					max_k = 8.0
					ref_dist = 3.0
					dist_norm = np.clip(dist_to_face / ref_dist, 0.0, 1.0)
					perp_k = min_k + (max_k - min_k) * dist_norm

					deviation_norm = 1.0 - np.clip(abs(angle_of_view), 0.0, np.pi / 2) / (np.pi / 2)
					perp_factor = np.exp(-perp_k * deviation_norm)

					# Factor 2: camera view vector alignment with face normal
					try:
						cam_trans = self.tf_buf.lookup_transform(
							"map", "base_link", rclpy.time.Time(), timeout=Duration(seconds=0.5)
						)
						q = cam_trans.transform.rotation
						siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
						cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
						yaw = np.arctan2(siny_cosp, cosy_cosp)
						cam_forward_x = np.cos(yaw)
						cam_forward_y = np.sin(yaw)

						dot = cam_forward_x * normal[0] + cam_forward_y * normal[1]
						alignment = np.clip(abs(dot), 0.0, 1.0)

						view_min_k = 1.0
						view_max_k = 8.0
						view_k = view_min_k + (view_max_k - view_min_k) * dist_norm
						view_factor = np.exp(view_k * (alignment - 1.0))
					except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
						view_factor = 0.0

					combined_score = perp_factor * view_factor

				# deduplicate: only add if no existing detection is within self.dedup_distance
				is_new = True
				matched_idx = -1
				for i, (ex, ey, ez) in enumerate(self.face_position_in_map_coordinates):
					dist = np.sqrt((map_x - ex)**2 + (map_y - ey)**2)
					if dist < self.dedup_distance:
						is_new = False
						matched_idx = i
						break

				if is_new and dist_to_face < 3.0:
					self.face_position_in_map_coordinates.append((map_x, map_y, map_z))
					self.face_best_alignment_score.append(combined_score)
					self.get_logger().info(f"NEW face at ({map_x:.2f}, {map_y:.2f}, {map_z:.2f}), score: {combined_score:.3f}")
				elif not is_new and matched_idx >= 0:
					while len(self.face_best_alignment_score) <= matched_idx:
						self.face_best_alignment_score.append(0.0)
					prev_best = self.face_best_alignment_score[matched_idx]
					if combined_score > prev_best:
						self.face_position_in_map_coordinates[matched_idx] = (map_x, map_y, map_z)
						self.face_best_alignment_score[matched_idx] = combined_score
						self.get_logger().info(f"UPDATED face {matched_idx}: score {combined_score:.3f} > {prev_best:.3f}")
					else:
						self.get_logger().debug(f"Kept face {matched_idx}, score {combined_score:.3f} <= {prev_best:.3f}")

			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
				self.get_logger().warn(f"Could not transform to map frame: {e}")

		self.publish_all_markers()

	def transform_point_to_map(self, xyz_point, source_frame, stamp=None):
		point_s = PointStamped()
		point_s.header.frame_id = source_frame
		point_s.header.stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()
		point_s.point.x = float(xyz_point[0])
		point_s.point.y = float(xyz_point[1])
		point_s.point.z = float(xyz_point[2])
		return self.tf_buf.transform(point_s, "map", timeout=Duration(seconds=0.5))

	def publish_all_markers(self):
		"""Publish a MarkerArray with all stored face positions in the map frame."""
		marker_array = MarkerArray()
		for i, (mx, my, mz) in enumerate(self.face_position_in_map_coordinates):
			marker = Marker()
			marker.header.frame_id = "map"
			marker.header.stamp = self.get_clock().now().to_msg()
			marker.ns = "people_detections"
			marker.id = i
			marker.type = 2

			scale = 0.1
			marker.scale.x = scale
			marker.scale.y = scale
			marker.scale.z = scale

			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 1.0
			marker.color.a = 1.0

			marker.pose.position.x = mx
			marker.pose.position.y = my
			marker.pose.position.z = mz

			marker_array.markers.append(marker)

		self.marker_array_pub.publish(marker_array)
		self.detected_faces_pub.publish(marker_array)

	def _republish_face_markers(self):
		"""Re-publish face markers periodically for late RViz / halfautonomous_search subscribers."""
		if self.face_position_in_map_coordinates:
			self.publish_all_markers()

	def load_detections(self):
		"""Load previously saved detections from JSON file."""
		try:
			with open(self.detections_json_path, 'r') as f:
				data = json.load(f)
			self.face_position_in_map_coordinates = [tuple(d) for d in data]
			self.face_best_alignment_score = [0.0] * len(self.face_position_in_map_coordinates)
			self.get_logger().info(f"Loaded {len(self.face_position_in_map_coordinates)} previous detections from {self.detections_json_path}")
		except FileNotFoundError:
			self.get_logger().info("No previous detections file found, starting fresh.")
		except Exception as e:
			self.get_logger().warn(f"Could not load detections: {e}")

	def save_detections_to_json(self):
		"""Save current detections to a JSON file for persistence."""
		try:
			with open(self.detections_json_path, 'w') as f:
				json.dump(self.face_position_in_map_coordinates, f, indent=2)
			self.get_logger().info(f"Saved {len(self.face_position_in_map_coordinates)} detections to {self.detections_json_path}")
		except Exception as e:
			self.get_logger().error(f"Failed to save detections to JSON: {e}")

	def log_face_positions(self):
		if self.face_position_in_map_coordinates:
			self.get_logger().info(f"Detected {len(self.face_position_in_map_coordinates)} unique people in map coordinates:")
			for i, (x, y, z) in enumerate(self.face_position_in_map_coordinates):
				self.get_logger().info(f"  Person {i+1}: ({x:.2f}, {y:.2f}, {z:.2f})")
		else:
			self.get_logger().info("No faces detected yet")

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.save_detections_to_json()
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()
