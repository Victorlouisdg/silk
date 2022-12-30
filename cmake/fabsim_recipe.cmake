set(FABSIM_GIT_TAG master)
include(FetchContent)
FetchContent_Declare(
    fabsim
    GIT_REPOSITORY https://github.com/DavidJourdan/fabsim.git
    GIT_TAG ${FABSIM_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fabsim)