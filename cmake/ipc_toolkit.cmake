set(IPC_TOOLKIT_GIT_TAG v1.0.0)
include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG ${IPC_TOOLKIT_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ipc_toolkit)