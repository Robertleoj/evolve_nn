cmake_minimum_required(VERSION 3.22)
project(project)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external/fmt)
add_subdirectory(external/spdlog)
add_subdirectory(external/pybind11)

# Set to ON for profiling.
# Off by default because this consumes a lot of memory.
option(TRACY_ENABLE "" OFF)
add_subdirectory(external/tracy)

add_subdirectory(src)
add_subdirectory(cpp_scripts)
