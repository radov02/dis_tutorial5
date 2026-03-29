#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

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
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.marker_array_pub = self.create_publisher(MarkerArray, "/people_marker_array", QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		# we will save face positions in the map coordinates:
		self.face_position_in_map_coordinates = []
		self.face_best_alignment_score = []  # tracks best combined score per face
		# minimum distance (meters) between two detections to consider them different people
		self.dedup_distance = 1.5
		# max_detection_z: height in map frame (Z=up). A person's face is ~1.0–1.8 m.
		self.max_detection_z = 2.0

		# map info for saving detections to PGM
		self.map_yaml_path = self.get_parameter('map_yaml_path').get_parameter_value().string_value
		self.map_pgm_path = self.get_parameter('map_pgm_path').get_parameter_value().string_value

		map_stem = os.path.splitext(os.path.basename(self.map_yaml_path))[0]
		_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
		self.detections_json_path = os.path.join(
			'/home/erik/rins/maps',
			f'people_detections_{map_stem}._{_ts}.json'
		)

		# load previously saved detections if available
		self.load_detections()

		# we need TF2 Listener, which will store the /tf and /tf_static topic messages into buffer which we will use to make transformation 
		# between camera and map frames:
		self.tf_buf = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
		self.create_timer(2.0, self.log_face_positions)

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

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width

		# decode point cloud once per callback
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))

		# The PointCloud2 data comes in the optical frame (X=right, Y=down, Z=forward)
		# with frame_id = "oakd_rgb_camera_optical_frame".
		# The system URDF (turtlebot4_description) has the CORRECT rotation
		# rpy=(-pi/2, 0, -pi/2) for the optical→body joint, so TF2 handles the
		# axis conversion automatically.  Pass raw XYZ directly — do NOT apply
		# a manual opt_to_body conversion (that would double-rotate!).
		# (The local dis_tutorial5/urdf copy has an identity rotation, but it is
		# NOT used at runtime — the launch loads from turtlebot4_description.)
		#
		# detect_rings.py uses opt_to_body because it constructs optical XYZ from
		# pixel+depth+intrinsics, so it must convert manually before handing to
		# the body frame.  Here we read XYZ from the PointCloud2 which already
		# has the correct frame_id for TF2 to handle.
		source_frame = "oakd_rgb_camera_optical_frame"

		# Log frame_id once for debugging
		if not hasattr(self, '_logged_source_frame'):
			actual_frame = data.header.frame_id if data.header.frame_id else "(empty)"
			self.get_logger().info(
				f"PointCloud2 frame_id='{actual_frame}', "
				f"pointcloud size={data.height}x{data.width}, "
				f"using source_frame='{source_frame}' for TF"
			)
			self._logged_source_frame = True
			self._diag_count = 0  # count diagnostic detections

		# get robot position once for normal orientation and fallback
		robot_x = None
		robot_y = None
		try:
			trans = self.tf_buf.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=Duration(seconds=0.5))
			robot_x = trans.transform.translation.x
			robot_y = trans.transform.translation.y
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			self.get_logger().warn(f"Could not look up base_link -> map transform for offset orientation: {e}")

		# iterate over face coordinates
		for detection in self.faces:
			x = detection['cx']
			y = detection['cy']
			x1, y1, x2, y2 = detection['bbox']

			# read center coordinates
			d = self.get_valid_point_around(a, x, y)

			# skip invalid (NaN) depth readings
			if d is None:
				self.get_logger().warn("Skipping detection with NaN depth")
				continue

			# Transform the pointcloud XYZ directly from the optical frame to map.
			# TF2 applies the optical→body rotation from the system URDF.
			try:
				point_in_map = self.transform_point_to_map(d, source_frame)
				map_x = point_in_map.point.x
				map_y = point_in_map.point.y
				map_z = point_in_map.point.z

				# Diagnostic logging for first few detections
				diag_count = getattr(self, '_diag_count', 0)
				if diag_count < 5:
					self.get_logger().info(
						f"[DIAG {diag_count}] optical=({d[0]:.3f},{d[1]:.3f},{d[2]:.3f}) "
						f"→ map=({map_x:.2f},{map_y:.2f},{map_z:.2f}) "
						f"robot=({robot_x:.2f},{robot_y:.2f})" if robot_x is not None else
						f"[DIAG {diag_count}] optical=({d[0]:.3f},{d[1]:.3f},{d[2]:.3f}) "
						f"→ map=({map_x:.2f},{map_y:.2f},{map_z:.2f}) robot=N/A"
					)
					self._diag_count = diag_count + 1

				if map_z > self.max_detection_z:
					self.get_logger().warn(f"Skipping detection: center z={map_z:.2f}m exceeds max_detection_z={self.max_detection_z:.2f}m")
					continue

				# Compute wall tangent from left/right bbox corners in map space,
				# then place marker 0.5 m along the perpendicular toward the robot.
				offset_distance = 0.5
				offset_applied = False
				angle_of_view = None
				normal = None

				right_corner_xyz = self.get_valid_point_around(a, x2, y)
				left_corner_xyz  = self.get_valid_point_around(a, x1, y)
				point_in_map_right_bbox_corner = None
				point_in_map_left_bbox_corner  = None
				if right_corner_xyz is not None:
					point_in_map_right_bbox_corner = self.transform_point_to_map(right_corner_xyz, source_frame)
				if left_corner_xyz is not None:
					point_in_map_left_bbox_corner  = self.transform_point_to_map(left_corner_xyz, source_frame)
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
					# close = forgiving (low k), far = strict (high k)
					min_k = 1.0   # sharpness at ~0m (very forgiving)
					max_k = 8.0   # sharpness at ref_dist and beyond (very strict)
					ref_dist = 3.0  # distance at which max sharpness kicks in
					dist_norm = np.clip(dist_to_face / ref_dist, 0.0, 1.0)
					perp_k = min_k + (max_k - min_k) * dist_norm

					deviation_norm = 1.0 - np.clip(abs(angle_of_view), 0.0, np.pi / 2) / (np.pi / 2)
					perp_factor = np.exp(-perp_k * deviation_norm)

					# Factor 2: camera view vector alignment with face normal
					# also scaled by distance: far away you need tighter alignment
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

				if is_new and dist_to_face < 3.0:  # also ignore very far detections as they are likely false positives
					self.face_position_in_map_coordinates.append((map_x, map_y, map_z))
					self.face_best_alignment_score.append(combined_score)
					self.get_logger().info(f"NEW face in map coordinates: ({map_x:.2f}, {map_y:.2f}, {map_z:.2f}), score: {combined_score:.3f}")
				elif not is_new and matched_idx >= 0:
					while len(self.face_best_alignment_score) <= matched_idx:
						self.face_best_alignment_score.append(0.0)
					prev_best = self.face_best_alignment_score[matched_idx]
					if combined_score > prev_best:
						# new observation is better aligned — replace position entirely
						self.face_position_in_map_coordinates[matched_idx] = (map_x, map_y, map_z)
						self.face_best_alignment_score[matched_idx] = combined_score
						self.get_logger().info(f"UPDATED face {matched_idx} with better alignment score: {combined_score:.3f} > {prev_best:.3f}")
					else:
						self.get_logger().debug(f"Kept face {matched_idx}, current score {combined_score:.3f} <= best {prev_best:.3f}")

			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
				self.get_logger().warn(f"Could not transform to map frame: {e}")

		# re-publish all known detections as a MarkerArray so they all stay visible
		self.publish_all_markers()

	def get_valid_point_around(self, cloud_xyz, x, y, radius=3):
		"""Return a valid xyz point near (x, y), or None if none found."""
		# clip to camera frame's width and height
		height, width, _ = cloud_xyz.shape
		x = int(np.clip(x, 0, width - 1))
		y = int(np.clip(y, 0, height - 1))

		for r in range(radius + 1):
			x_min = max(0, x - r)
			x_max = min(width - 1, x + r)
			y_min = max(0, y - r)
			y_max = min(height - 1, y + r)

			for yy in range(y_min, y_max + 1):
				for xx in range(x_min, x_max + 1):
					p = cloud_xyz[yy, xx, :]
					if not (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2])):
						return p

		return None

	def transform_point_to_map(self, xyz_point, source_frame):
		point_in_source_frame = PointStamped()
		point_in_source_frame.header.frame_id = source_frame
		point_in_source_frame.header.stamp = rclpy.time.Time().to_msg()  # latest available
		point_in_source_frame.point.x = float(xyz_point[0])
		point_in_source_frame.point.y = float(xyz_point[1])
		point_in_source_frame.point.z = float(xyz_point[2])

		return self.tf_buf.transform(point_in_source_frame, "map", timeout=Duration(seconds=0.5))

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