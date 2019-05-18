cmake_minimum_required(VERSION 3.14)
project(Convolution)

set(CMAKE_CXX_STANDARD 14)

find_package (OpenCL REQUIRED)

add_executable(convolution main.cpp)
target_link_libraries (convolution ${OpenCL_LIBRARY})