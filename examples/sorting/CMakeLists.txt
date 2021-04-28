include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(SortKernel SortKernel.cpp)
xpu_attach_device_library(SortKernel SortKernel SortKernel.def SortKernel.cpp)

add_executable(sort_struct sort_struct.cpp)
target_link_libraries(sort_struct xpu SortKernel)
