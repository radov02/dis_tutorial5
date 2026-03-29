cmake_minimum_required(VERSION 3.28)
project(dis_tutorial5)

# Set policies BEFORE any find_package calls that might trigger them
cmake_policy(SET CMP0144 NEW)
cmake_policy(SET CMP0074 NEW)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# These must come AFTER the policy is set
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})

set(dependencies rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs tf2 tf2_geometry_msgs pcl_conversions)

add_executable(cylinder_segmentation src/cylinder_segmentation.cpp)
ament_target_dependencies(cylinder_segmentation ${dependencies})

target_link_libraries(cylinder_segmentation
  ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
)

install(TARGETS
  cylinder_segmentation
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/autonomous_sweep.py
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/detect_rings.py
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/detect_people2.py
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/LLM.py
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/robot_commander.py
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS scripts/voice_capture.py
  RENAME voice_capture
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS scripts/map_geofence_tool.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

ament_package()