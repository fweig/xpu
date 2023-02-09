add_executable(xpuinfo xpuinfo.cpp)
target_compile_options(xpuinfo PUBLIC -fPIE -fPIC)
target_link_libraries(xpuinfo xpu)
