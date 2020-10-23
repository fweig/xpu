add_executable(vector_add vector_add.cpp)
xpu_add_device_library(VectorOps KERNEL_DEF kernels.h INTERFACE_H interface.h SRCS VectorOps.cpp)
target_link_libraries(vector_add xpu VectorOps)