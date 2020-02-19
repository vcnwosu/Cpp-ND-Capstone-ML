cmake_minimum_required(VERSION 3.11.3)

project(VICTOR_DATALOADER)

set(CMAKE_CXX_STANDARD 17)
set(the_description "A dataloader to load the Iris dataset")

file(GLOB libdataset_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(dataset SHARED ${libdataset_srcs})
set_target_properties(dataset PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
)
target_include_directories(dataset PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(dataset_test ${PROJECT_SOURCE_DIR}/test/main.cpp)
set_target_properties(dataset_test PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test
)
target_include_directories(dataset_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dataset_test dataset)
