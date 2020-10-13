add_executable(vector_add test.cpp)
xpu_add_device_library(VectorOps kernels.h vec_add.cpp)
target_link_libraries(vector_add xpu VectorOps)