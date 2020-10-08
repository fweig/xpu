cmake_minimum_required(VERSION 3.18)

project(cmake-project-template)

set(CMAKE_CXX_STANDARD 11)

add_library(xpu
    src/xpu/gpu.cpp
    src/xpu/backend/cpu/CPUBackend.cpp
)

include_directories(src)

add_executable(vector_add
    examples/vector_add/test.cpp
    examples/vector_add/TestKernels.cpp
    examples/vector_add/vec_add.cpp
)

target_link_libraries(vector_add xpu)