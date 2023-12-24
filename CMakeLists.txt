cmake_minimum_required(VERSION 3.24)
project(network_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(network_project network/main.cpp)
target_include_directories(network_project PRIVATE ${CMAKE_SOURCE_DIR})

