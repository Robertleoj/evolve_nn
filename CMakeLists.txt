cmake_minimum_required(VERSION 3.22)
project(project)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external/pybind11)

# Get the path to the project directory
get_filename_component(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Get the PyTorch CMake path from the Poetry Python installation
set(POETRY_PYTHON_PATH $ENV{POETRY_PYTHON_PATH})
if(NOT POETRY_PYTHON_PATH)
  message(FATAL_ERROR "Poetry Python path not set. Please set POETRY_PYTHON_PATH environment variable.")
endif()

execute_process(
  COMMAND ${POETRY_PYTHON_PATH} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the PyTorch CMake path from Python installation
# find_package(Python3 COMPONENTS Interpreter REQUIRED)
# execute_process(
#   COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
#   OUTPUT_VARIABLE TORCH_CMAKE_PATH
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )
message(STATUS "Torch CMake path: ${TORCH_CMAKE_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

find_package(Torch REQUIRED)

message(STATUS "Torch flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)
add_subdirectory(src)