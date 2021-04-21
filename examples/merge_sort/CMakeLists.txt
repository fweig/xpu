include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(mergeSortOps mergeSortOps.cpp)
xpu_attach_device_library(mergeSortOps mergeSortOps mergeSortOps.def mergeSortOps.cpp)

add_executable(merge_sort merge_sort.cpp)
target_link_libraries(merge_sort xpu mergeSortOps)