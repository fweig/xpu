include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_compile_options("-O3")
add_definitions(-DNDEBUG)

set(deviceSrcs bench_device.cpp)
add_library(BenchDevice SHARED ${deviceSrcs})
xpu_attach(BenchDevice ${deviceSrcs})
add_executable(xpu_bench xpu_bench.cpp)
target_link_libraries(xpu_bench xpu BenchDevice)
