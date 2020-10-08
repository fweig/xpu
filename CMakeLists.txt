cmake_minimum_required(VERSION 3.18)

project(cmake-project-template)

set(CMAKE_CXX_STANDARD 11)

add_executable(test
    test.cpp
    gpu.cpp
    TestKernels.cpp
    backend/cpu/CPUBackend.cpp
    vec_add.cpp
)