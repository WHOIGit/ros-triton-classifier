cmake_minimum_required(VERSION 2.8.3)
project(triton_classifier)

# This package no longer defines its own messages; it publishes the standard
# vision_msgs types, so there is nothing to generate.
find_package(
  catkin REQUIRED COMPONENTS
    rospy
    sensor_msgs
    vision_msgs
)

# Invokes setup.py, which installs src/triton_api.py as a Python module; the
# installed classifier_node.py imports it, so without this the install space
# would be missing half the package.
catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    vision_msgs
)

catkin_install_python(
  PROGRAMS
    src/classifier_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  # The suite is plain pytest with no ROS runtime dependency; register it so
  # `catkin_make run_tests` actually executes it instead of silently passing.
  catkin_run_tests_target("pytest" tests "pytest-tests.xml"
    COMMAND "python3 -m pytest ${PROJECT_SOURCE_DIR}/tests --junit-xml=${CATKIN_TEST_RESULTS_DIR}/${PROJECT_NAME}/pytest-tests.xml"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
