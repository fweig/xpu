find_package(GTest REQUIRED)
xpu_add_device_library(test_kernels KERNEL_DEF test_kernels.def INTERFACE_H test_kernels_interface.h SRCS test_kernels.cpp)
add_executable(xpu_test xpu_test.cpp)
target_link_libraries(xpu_test xpu test_kernels GTest::GTest)