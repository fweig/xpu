add_executable(vector_add vector_add.cpp)
xpu_add_device_library(VectorOps kernels.h VectorOps.cpp)
target_link_libraries(vector_add xpu VectorOps)