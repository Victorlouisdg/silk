cmake_minimum_required(VERSION 3.16)

project(silk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(eigen)
include(ipc_toolkit)
include(polyscope)
include(geometry_central)
include(tinyad)
include(googletest)
include(fabsim_recipe)

# add_library(fabsim)
# include(tinyad_examples)

set(LIBIGL_RESTRICTED_TRIANGLE ON)
include(libigl)
igl_include_optional(predicates)
igl_include_optional(restricted triangle)


# Find packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

message("Building silk in " ${CMAKE_BUILD_TYPE} " mode")
# add_subdirectory("libs/TinyAD-Examples")

file(GLOB_RECURSE SILK_HEADERS include/*.hh)
file(GLOB_RECURSE SILK_SOURCES src/*.cpp)
add_library(silk ${SILK_SOURCES} ${SILK_HEADERS})
message(${SILK_HEADERS})
message(${SILK_SOURCES})

target_compile_features(silk PUBLIC cxx_std_17)

target_include_directories(silk PUBLIC
    include
)

target_link_libraries(silk PUBLIC
    Eigen3::Eigen
    igl::core
    igl::predicates
    igl_restricted::triangle
    ipc::toolkit
    polyscope
    geometry-central
    TinyAD
    fabsim
    # TinyAD-Examples
)

function(add_example example_name main)
    message(STATUS "Adding example ${example_name}")
    add_executable(${example_name} ${main})
    add_dependencies(${example_name} silk)
    target_link_libraries(${example_name} PUBLIC silk)
endfunction()

file(GLOB_RECURSE SILK_EXAMPLES examples/*.cpp)
foreach(main ${SILK_EXAMPLES})
    get_filename_component(example_name ${main} NAME_WE)
    get_filename_component(dir ${main} DIRECTORY)
    get_filename_component(dir_name ${dir} NAME)
    add_example(${example_name} ${main})
endforeach()


include(GoogleTest)
function(add_test test_name main)
    message(STATUS "Adding test ${test_name}")
    add_executable(${test_name} ${main})
    add_dependencies(${test_name} silk)
    target_link_libraries(${test_name} PUBLIC silk GTest::gtest_main)
    gtest_discover_tests(${test_name})

endfunction()


file(GLOB_RECURSE SILK_TESTS test/*.cpp)
foreach(main ${SILK_TESTS})
    get_filename_component(test_name ${main} NAME_WE)
    get_filename_component(dir ${main} DIRECTORY)
    get_filename_component(dir_name ${dir} NAME)
    add_test(${test_name} ${main})
endforeach()

# enable_testing()

# add_executable(
#   deformation_gradient_test
#   test/deformation_gradient_test.cpp
# )
# target_link_libraries(
#   deformation_gradient_test
#   GTest::gtest_main
#   silk
# )

# # TODO: remove this duplication, e.g. with a function similar to above.
# add_executable(
#   energies_test
#   test/energies_test.cpp
# )
# target_link_libraries(
#   energies_test
#   GTest::gtest_main
#   silk
# )

# include(GoogleTest)
# gtest_discover_tests(deformation_gradient_test)
# gtest_discover_tests(energies_test)