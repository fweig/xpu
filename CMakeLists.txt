cmake_minimum_required(VERSION 3.18)

project(xpu)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
enable_language(CUDA)

add_library(xpu
    src/xpu/gpu.cpp
    src/xpu/dl_utils.cpp
    src/xpu/backend/cpu/CPUBackend.cpp
)

add_library(XPUBackendCUDA MODULE
    src/xpu/backend/cuda/CUDABackend.cu
)

target_link_libraries(xpu dl)

include_directories(src)

add_executable(vector_add
    examples/vector_add/test.cpp
    examples/vector_add/TestKernels.cpp
    examples/vector_add/vec_add.cpp
)

target_link_libraries(vector_add xpu)

add_library(TestKernelsCUDA MODULE
    examples/vector_add/TestKernelsCUDA.cu
)