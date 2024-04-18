set(FILES
    example.cpp
)

# pybind11_add_module(foundation python.cpp ${FILES})

add_library(foundation SHARED python.cpp)

target_link_libraries(foundation PRIVATE ${TORCH_LIBRARIES} Python::Python)

target_include_directories(foundation PUBLIC ../../include)
