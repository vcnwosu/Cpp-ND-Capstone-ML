cmake_minimum_required(VERSION 3.11.3)

project(VICTOR_NEURAL_NETWORK)

set(CMAKE_CXX_STANDARD 17)
set(the_description "The Neural Network Implementation")

file(GLOB libneural_network_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB libneural_network_hdrs ${PROJECT_SOURCE_DIR}/src/*.h)

add_library(neural_network SHARED ${libneural_network_srcs})
set_target_properties(neural_network PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
)
target_include_directories(neural_network PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(neural_network_test ${PROJECT_SOURCE_DIR}/test/main.cpp)
set_target_properties(neural_network_test PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test
)
target_include_directories(neural_network_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(neural_network_test neural_network)

add_subdirectory(${PROJECT_SOURCE_DIR}/../iris ${PROJECT_SOURCE_DIR}/build)

target_link_libraries(neural_network iris)
target_link_libraries(neural_network_test iris)
