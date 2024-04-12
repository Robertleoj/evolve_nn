set(FILES
    example.cpp
)

pybind11_add_module(foundation python.cpp ${FILES})

target_link_libraries(foundation PRIVATE ${TORCH_LIBRARIES})

target_include_directories(foundation PUBLIC ../../include)

