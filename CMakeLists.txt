cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(silk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_executable(silk src/main.cpp)

# Torch
list(APPEND CMAKE_PREFIX_PATH "libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(silk "${TORCH_LIBRARIES}")
set_property(TARGET silk PROPERTY CXX_STANDARD 17)

add_subdirectory("libs/TinyAD")

# PolyScope and Geometry-central
add_subdirectory("libs/polyscope")
add_subdirectory("libs/geometry-central")
target_link_libraries(silk polyscope)
target_link_libraries(silk geometry-central)

# Executables for the examples
add_executable(two_triangles_pytorch examples/two_triangles/two_triangles_pytorch.cpp)
target_link_libraries(two_triangles_pytorch polyscope)
target_link_libraries(two_triangles_pytorch geometry-central)
target_link_libraries(two_triangles_pytorch "${TORCH_LIBRARIES}")

# Executables for the examples
add_executable(two_triangles_tinyad examples/two_triangles/two_triangles_tinyad.cpp)
target_link_libraries(two_triangles_tinyad polyscope)
target_link_libraries(two_triangles_tinyad geometry-central)
target_link_libraries(two_triangles_tinyad TinyAD)

