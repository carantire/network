cmake_minimum_required(VERSION 3.24)
project(network_project)

set(CMAKE_CXX_STANDARD 17)

find_package(MNIST PATHS mnist/)


add_executable(network_project src/main.cpp src/ThresholdFunc.cpp src/Layer.cpp src/ScoreFunc.cpp src/Network.cpp src/tests.cpp src/except.cpp src/Mnist_test.cpp src/sin_test.cpp src/Xor_test.cpp)
target_include_directories(network_project PRIVATE eigen EigenRand ${MNIST_INCLUDE_DIR})
target_compile_definitions(network_project PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
