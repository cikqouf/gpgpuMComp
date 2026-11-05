cmake_minimum_required(VERSION 3.30)
project(VulkanComputeProj CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)

add_executable(app src/main.cpp)

target_include_directories(app PRIVATE include/)

target_link_libraries(app -lvulkan -fopenmp)

target_compile_options(app PRIVATE -O0 -g -mavx2 -mfma)
