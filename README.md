

# Tutorial 5: Point Cloud Library and ring detection

#### Development of Intelligent Systems, 2026

This exercise will show a few examples of how to use the [Point Cloud Library (PCL)](https://pointclouds.org/) and OpenCV to extract information from the RGBD camera. The PCL project contains a large number of [tutorials](https://pcl.readthedocs.io/projects/tutorials/en/master/) demonstrating how to use the library. From the code in this tutorial, you can extrapolate how to use the PCL library in ROS2. For our purposes, the tutorials on PointCloud [segmentation](https://pcl.readthedocs.io/projects/tutorials/en/master/#segmentation) are the most relevant. The given examples use the RANSAC algorithm to find planes and cylinders, and extract the inliers.

## Ring detection
In the script *detect_rings.py* you can find a rough demonstration for finding the rings in the image. This is one of the simplest possible approaches, for demonstration purposes. You are highly encouraged to develop your own approach. The given code is explained below, which you should at least customize to improve its performance:

First, convert the image to numpy form:
```
cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
```
Convert it to a grayscale image:
```
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
```

Apply thresholding to get a binary image. There are different possible approaches, here we use the adaptive algotihm:
```
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
```

Extract contours from the edges in the binary image:
```
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
```

Then, we fit ellipses to all contours that are longer than some threshold (20) and filter them based on eccentricity (so we ignore ellipses that are too flat):
```
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
```

We then evaluate all pairs of ellipses and try to eliminate all that do not form a ring. OpenCV returns ellipses as oriented bounding boxes. Each ellipse is represented as the coordinates of its the center point `e[0]`, the length of the minor and major axis `e[1]` and its rotation `e[2]`. The ellipses that represent the inner and outer circles of a ring have some similar properties. First, their centers should be roughly the same: 
```
e1 = elps[n]
e2 = elps[m]
dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))

# The centers of the two elipses should be within 5 pixels of each other
if dist >= self.center_thr:
    continue
```

You can do many other filters, including color, point cloud geometry, etc. You will likely need to include different assumptions and filters to correctly detect actual rings consistently and  remove false positives.

## Cylinder segmentation
***NOTE: This is the content from last year's task. It is included here as a point of interest only. You are not required to use it.*** 
You can localize cylinders in the scene by fitting a cylinder model to the geometry of the point cloud using RANSAC. The file `cylinder_segmentation.cpp` implements this using the PCL library. It republishes the filtered point cloud on the `/cylinder` topic, and the markers representing the cylinder locations to `/detected_cylinder` topic

Please note that the given node fits a cylinder to every point cloud it receives. It should be used to find the accurate position of a cylinder, but it is not reliable as a cylinder detector on its own. It can be used, but you need to filter out the false detections.

First we transform the ROS2 message to a PCL point cloud, and then to a type appropriate for processing:
```
// convert ROS msg to PointCloud2
pcl_conversions::toPCL(*msg, *pcl_pc);

// convert PointCloud2 to templated PointCloud
pcl::fromPCLPointCloud2(*pcl_pc, *cloud);
```

Keep only the points that have the x-dimension (view in front of the camera) between 0 and 10.
```
// Build a passthrough filter to remove spurious NaNs
pass.setInputCloud(cloud);
pass.setFilterFieldName("x");
pass.setFilterLimits(0, 10);
pass.filter(*cloud_filtered);
```

Then, we calculate normals to the points. For each point, we take a look at its neighbors and estimate the normal to the surface. This is one of many possible approaches to do this. In this way, we can also create a 3d mesh from 3d points:
```
// Estimate point normals
ne.setSearchMethod(tree);
ne.setInputCloud(cloud_filtered);
ne.setKSearch(50);
ne.compute(*cloud_normals);
```

We find the largest plane in the point cloud. This will usually be the ground plane. It will be simpler for RANSAC to find a good fit for the cylinder if we filter out all the points we are certain do not belong to the cylinder:
```
// Create the segmentation object for the planar model and set all the
// parameters
seg.setOptimizeCoefficients(true);
seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
seg.setNormalDistanceWeight(0.1);
seg.setMethodType(pcl::SAC_RANSAC);
seg.setMaxIterations(100);
seg.setDistanceThreshold(0.03);
seg.setInputCloud(cloud_filtered);
seg.setInputNormals(cloud_normals);

seg.segment(*inliers_plane, *coefficients_plane);
```

Remove all the points that belong to the plane from the point cloud:
```
// Extract the planar inliers from the input cloud
extract.setInputCloud(cloud_filtered);
extract.setIndices(inliers_plane);
extract.setNegative(false);
extract.filter(*cloud_plane);
```

Finally, run RANSAC and fit a cylinder model to the rest of the points:
```
// Create the segmentation object for cylinder segmentation and set all the
// parameters
seg.setOptimizeCoefficients(true);
seg.setModelType(pcl::SACMODEL_CYLINDER);
seg.setMethodType(pcl::SAC_RANSAC);
seg.setNormalDistanceWeight(0.1);
seg.setMaxIterations(100);
seg.setDistanceThreshold(0.05);
seg.setRadiusLimits(0.06, 0.2);
seg.setInputCloud(cloud_filtered2);
seg.setInputNormals(cloud_normals2);

// Obtain the cylinder inliers and coefficients
seg.segment(*inliers_cylinder, *coefficients_cylinder);
```

In the end, extract the points that belong to the cylinder and computer their centroid:
```
// extract cylinder
extract.setInputCloud(cloud_filtered2);
extract.setIndices(inliers_cylinder);
extract.setNegative(false);
pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
extract.filter(*cloud_cylinder);

// calculate marker
pcl::compute3DCentroid(*cloud_cylinder, centroid);
```

## TODO for students:
As part of Task 1, you need to find the faces and the 3D rings in the course. The code in this exercise will NOT perform this task reliably out of the box. You should either develop a completely new approach, or use this code as a starting point.
**General note:** Use markers in RVIZ to help with interpreting your robot's perception and knowledge. It is much easier than looking at terminal output. You can program markers to have different colors, sizes and shapes. Display the hardcoded points for movement, face locations, ring candidates, colors, etc.

### For face detection
Your person detector will process every image and detect the face multiple times when the robot moving around the course. You need to group the detections in the map coordinate space to establish a good estimation of the face's position. Additionally, for programming the robot approach, you might want to also calculate the direction of the wall on which the face is (i.e. the normal of the wall).

### For ring detection
There are two types of rings in the course, flat rings painted on boxes, and actual rings hanging in the air. You should only detect the free-standing rings and ignore the flat rings. You can use either or both 2D and 3D data to solve the task. You can also exploit the color information in the image (using color segmentation). For the 3D rings, there should be a hole in the inside ellipse, which can be verified from the point cloud or the depth image.




## Try out the walk & greet
- commands:
  - `cd /home/erik/rins && colcon build`
  - `source /home/erik/rins/install/setup.bash`
  - T1: `ros2 run rmw_zenoh_cpp rmw_zenohd`
  - T2: `ollama serve` (or check if already running: `ollama list`)
  - T2: `ros2 launch dis_tutorial5 sim_turtlebot_nav.launch.py map:=/home/erik/rins/maps/maps.yaml` and use 2D Pose Estimate
  - T3: `cd /home/erik/rins && source install/setup.bash && ros2 run dis_tutorial5 LLM.py`
  - T4: `cd /home/erik/rins && source install/setup.bash && ros2 run dis_tutorial5 voice_capture --ros-args -p piper_model_path:=/home/erik/piper_models/en_US-lessac-medium/en_US-lessac-medium.onnx`
  - optionally: T5 to make one test request: `cd /home/erik/rins && source install/setup.bash && ros2 service call /human_detected robot_interfaces/srv/HumanDetected "{detect_signal: true}"`
  - T5: `ros2 run dis_tutorial5 detect_people2.py`
  - T6: `ros2 run dis_tutorial5 detect_rings.py`
  - T7: `ros2 run dis_tutorial5 robot_commander.py`


## Geofencing
- once map had been scanned we want to adjust the allowed area to prevent robot from going on illegal places (island)
- firstly we run `python3 src/dis_tutorial5/scripts/map_geofence_tool.py --preview --apply` and adjust the area
- then when done press ENTER to get the fenced allowed area (copy the array into the `ALLOWED_AREA_POLYGON` used in the `robot_commander.py` if it did not change)