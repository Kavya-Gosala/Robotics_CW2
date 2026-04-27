cmake_minimum_required(VERSION 3.0.2)
project(com760cw2_com760group30)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf
  message_generation
)

add_service_files(
  FILES
  MineRescueSetBugStatus.srv
  MineRescueSetQLStatus.srv
)

add_message_files(
  FILES
  RLAction.msg
  RLReward.msg
  SurvivorDetected.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf
  message_runtime
)

catkin_install_python(PROGRAMS
  scripts/GoToPoint.py
  scripts/FollowWall.py
  scripts/Bug2.py
  scripts/QLearning.py
  scripts/SurvivorDetector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)