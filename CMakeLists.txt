cmake_minimum_required(VERSION 3.24)
project(network_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(network_project src/main.cpp src/Threshold_Func.cpp src/Layer.cpp src/Score_Func.cpp src/Network.cpp)
target_include_directories(network_project PRIVATE ${CMAKE_SOURCE_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (network_project Eigen3::Eigen)
