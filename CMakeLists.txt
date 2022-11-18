cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(silk)
add_executable(silk src/main.cpp)

# Torch
list(APPEND CMAKE_PREFIX_PATH "libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(silk "${TORCH_LIBRARIES}")
set_property(TARGET silk PROPERTY CXX_STANDARD 14)

# PolyScope and Geometry-central
add_subdirectory("libs/polyscope")
add_subdirectory("libs/geometry-central")
target_link_libraries(silk polyscope)
target_link_libraries(silk geometry-central)

# Executables for the examples
add_executable(two_triangles examples/two_triangles/two_triangles.cpp)
target_link_libraries(two_triangles "${TORCH_LIBRARIES}")
target_link_libraries(two_triangles polyscope)
target_link_libraries(two_triangles geometry-central)
