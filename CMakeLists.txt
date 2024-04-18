cmake_minimum_required(VERSION 3.22)
project(project)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_subdirectory(external/pybind11)

# Get the path to the project directory
get_filename_component(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Set the path to the libtorch directory relative to the project directory
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_DIR}/external/libtorch")

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(src)
