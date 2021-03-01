include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(VectorOps VectorOps.cpp)
xpu_attach_device_library(VectorOps VectorOps VectorOps.def VectorOps.cpp)

add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add xpu VectorOps)
