set(GEOMETRY_CENTRAL_GIT_TAG master)
include(FetchContent)
FetchContent_Declare(
    geometry-central
    GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
    GIT_TAG ${GEOMETRY_CENTRAL_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(geometry-central)