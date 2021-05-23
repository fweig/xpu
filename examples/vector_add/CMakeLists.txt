include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(VectorOps SHARED VectorOps.cpp)
xpu_attach(VectorOps VectorOps.cpp)

add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add xpu VectorOps)
