include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(MergeKernel SHARED MergeKernel.cpp)
xpu_attach(MergeKernel MergeKernel.cpp)

add_executable(merge merge.cpp)
target_link_libraries(merge xpu MergeKernel)
