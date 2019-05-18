cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 14)

find_package (OpenCL REQUIRED)

add_executable(prefix_sum main.cpp)
target_link_libraries(prefix_sum ${OpenCL_LIBRARY})