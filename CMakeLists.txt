cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(silk)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(silk src/main.cpp)
target_link_libraries(silk "${TORCH_LIBRARIES}")
set_property(TARGET silk PROPERTY CXX_STANDARD 14)

add_subdirectory("libs/polyscope")
target_link_libraries(silk polyscope)

add_subdirectory("libs/geometry-central")
target_link_libraries(silk geometry-central)
