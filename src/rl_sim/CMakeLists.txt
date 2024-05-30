cmake_minimum_required(VERSION 3.0.2)
project(rl_sim)

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")
find_package(gazebo REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    controller_manager
    genmsg
    joint_state_controller
    robot_state_publisher
    roscpp
    gazebo_ros
    std_msgs
    tf
    geometry_msgs
    robot_msgs
    robot_joint_controller
)

# find_package(yaml-cpp REQUIRED)
link_directories(/usr/local/lib)
include_directories(${YAML_CPP_INCLUDE_DIR})

catkin_package(
    CATKIN_DEPENDS 
    robot_joint_controller
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    library/rl_sdk
    library/loop
)

add_library(rl_sdk library/rl_sdk/rl_sdk.cpp)
target_link_libraries(rl_sdk "${TORCH_LIBRARIES}")
set_property(TARGET rl_sdk PROPERTY CXX_STANDARD 14)

add_executable(rl_sim src/rl_sim.cpp )
target_link_libraries(rl_sim
  ${catkin_LIBRARIES} ${EXTRA_LIBS}
  rl_sdk yaml-cpp
)
