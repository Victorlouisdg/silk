set(TINYAD_GIT_TAG main)
include(FetchContent)
FetchContent_Declare(
    TinyAD
    GIT_REPOSITORY https://github.com/patr-schm/TinyAD.git
    GIT_TAG ${TINYAD_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(TinyAD)