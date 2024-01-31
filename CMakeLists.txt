cmake_minimum_required(VERSION 3.24)
project(network_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(network_project src/main.cpp src/ThresholdFunc.cpp src/Layer.cpp src/ScoreFunc.cpp src/Network.cpp src/tests.cpp src/except.cpp)
target_include_directories(network_project PRIVATE eigen EigenRand)
