cmake_minimum_required(VERSION 3.20)
project(Halogen)

set(CMAKE_CXX_STANDARD 23)

add_executable(Halogen main.cpp)
add_executable(TensorTests tests/TensorTests.cpp)

target_include_directories(Halogen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(TensorTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
