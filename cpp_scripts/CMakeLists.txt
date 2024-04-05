# cpp_scripts/CMakeLists.txt
file(GLOB_RECURSE CPP_SCRIPTS_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

foreach(SOURCE_FILE ${CPP_SCRIPTS_SRC})
    get_filename_component(SOURCE_PATH ${SOURCE_FILE} PATH)
    get_filename_component(SOURCE_FILE_NAME ${SOURCE_FILE} NAME_WE)

    # Set the target name as the path and file name
    set(TARGET_NAME ${SOURCE_FILE_NAME})

    message(STATUS "Adding executable: ${TARGET_NAME}")

    # Create a directory structure in the build directory
    set(TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_PATH})
    file(MAKE_DIRECTORY ${TARGET_DIR})

    # Add an executable for this source file
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_DIR})

    # Link the executable with the necessary libraries
    target_link_libraries(${TARGET_NAME} spdlog::spdlog Tracy::TracyClient common_sources)

    target_include_directories(${TARGET_NAME} PUBLIC ../../include)
endforeach()
