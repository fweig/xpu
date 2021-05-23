include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(SortKernel SHARED SortKernel.cpp)
xpu_attach(SortKernel SortKernel.cpp)

add_executable(sort_struct sort_struct.cpp)
target_link_libraries(sort_struct xpu SortKernel)
