cmake_minimum_required(VERSION 2.8.3)
project(triton_classifier)

find_package(
  catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
)

add_message_files(
  FILES
  Classification.msg
  ObjectHypothesisWithClassName.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

if(CATKIN_ENABLE_TESTING)
  # The suite is plain pytest with no ROS runtime dependency; register it so
  # `catkin_make run_tests` actually executes it instead of silently passing.
  catkin_run_tests_target("pytest" tests "pytest-tests.xml"
    COMMAND "python3 -m pytest ${PROJECT_SOURCE_DIR}/tests --junit-xml=${CATKIN_TEST_RESULTS_DIR}/${PROJECT_NAME}/pytest-tests.xml"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
