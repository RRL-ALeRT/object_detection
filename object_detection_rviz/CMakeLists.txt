cmake_minimum_required(VERSION 3.5)
project(object_detection_rviz)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_AUTOMOC ON)

add_library(object_detection_rviz SHARED
  src/capture_panel.cpp
  include/object_detection_rviz/capture_panel.hpp
)

target_include_directories(object_detection_rviz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(object_detection_rviz
  rclcpp
  rviz_common
  rviz_default_plugins
  sensor_msgs
  cv_bridge
  image_transport
)

target_link_libraries(object_detection_rviz
  Qt5::Widgets
  ${OpenCV_LIBRARIES}
)

ament_export_libraries(object_detection_rviz)
ament_export_dependencies(rclcpp rviz_common sensor_msgs cv_bridge)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(
  TARGETS object_detection_rviz
  EXPORT object_detection_rviz
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()