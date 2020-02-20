cmake_minimum_required(VERSION 3.11.3)

project(VICTOR_DATALOADER)

set(CMAKE_CXX_STANDARD 17)
set(the_description "A dataloader to load the Iris iris")

file(GLOB libiris_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(iris SHARED ${libiris_srcs})
set_target_properties(iris PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
)
target_include_directories(iris PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(iris_test ${PROJECT_SOURCE_DIR}/test/main.cpp)
set_target_properties(iris_test PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test
)
target_include_directories(iris_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(iris_test iris)
