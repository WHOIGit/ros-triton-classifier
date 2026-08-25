cmake_minimum_required(VERSION 2.8.3)
project(triton_classifier)

# This package no longer defines its own messages; it publishes the standard
# vision_msgs types, so there is nothing to generate.
find_package(
  catkin REQUIRED COMPONENTS
    std_msgs
    vision_msgs
)

catkin_package(
  CATKIN_DEPENDS
    sensor_msgs
    std_msgs
    vision_msgs
)

catkin_install_python(
  PROGRAMS
    src/classifier_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
