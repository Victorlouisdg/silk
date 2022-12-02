set(POLYSCOPE_GIT_TAG master)
include(FetchContent)
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG ${POLYSCOPE_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(polyscope)