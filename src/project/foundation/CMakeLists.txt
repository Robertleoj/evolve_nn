find_package(pylon 7.4.0 REQUIRED)

set(FILES
    example.cpp
)

pybind11_add_module(foundation python.cpp ${FILES})
target_link_libraries(foundation PUBLIC spdlog::spdlog Tracy::TracyClient)

target_include_directories(foundation PUBLIC ../../include)


add_library(common_sources OBJECT ${FILES})

target_link_libraries(common_sources PUBLIC spdlog::spdlog Tracy::TracyClient)

target_include_directories(common_sources PUBLIC ../../include)