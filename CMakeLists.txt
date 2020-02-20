cmake_minimum_required(VERSION 3.11.3)

project(VICTOR_MACHINE_LEARNING)

set(CMAKE_CXX_STANDARD 17)
set(the_description "Machine Learning Project from Scratch for Udacity C++ Capstone")

add_executable(capstone ${PROJECT_SOURCE_DIR}/src/main.cpp)
set_target_properties(capstone PROPERTIES
    LINKER_LANGUAGE CXX
)

add_subdirectory(src/network)

target_link_libraries(capstone neural_network)
target_link_libraries(capstone iris)
