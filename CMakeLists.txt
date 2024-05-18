cmake_minimum_required(VERSION 3.24)
project(network_project)

set(CMAKE_CXX_STANDARD 17)

find_package(MNIST PATHS mnist/)

add_library(network_lib
        src/ThresholdFunc.cpp
        src/Layer.cpp
        src/ScoreFunc.cpp
        src/Network.cpp
        src/LearningRate.cpp)
target_include_directories(network_lib PUBLIC eigen EigenRand ${MNIST_INCLUDE_DIR})

add_executable(network_test test/main.cpp test/tests.cpp test/except.cpp)
target_include_directories(
        network_test PUBLIC "${PROJECT_SOURCE_DIR}/src")
add_executable(sin_train test/sin_test.cpp)
target_include_directories(
        sin_train PUBLIC "${PROJECT_SOURCE_DIR}/src")
add_executable(mnist_train test/Mnist_test.cpp)
target_include_directories(
        mnist_train PUBLIC "${PROJECT_SOURCE_DIR}/src")
#add_executable(tests test/tests.cpp)
#target_include_directories(
#        tests PUBLIC "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(network_test PUBLIC network_lib)
target_link_libraries(sin_train PUBLIC network_lib)
target_link_libraries(mnist_train PUBLIC network_lib)

target_compile_definitions(mnist_train PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
