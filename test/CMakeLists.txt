include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  set(BUILD_GMOCK OFF)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(deviceSrcs TestKernels.cpp)
add_library(TestKernels SHARED ${deviceSrcs})
xpu_attach(TestKernels ${deviceSrcs})
add_executable(xpu_test xpu_test.cpp)
target_link_libraries(xpu_test xpu TestKernels gtest)

add_test(NAME xpu_test_cpu COMMAND xpu_test)
set_tests_properties(xpu_test_cpu PROPERTIES ENVIRONMENT "XPU_DEVICE=cpu")

if (XPU_ENABLE_CUDA)
  add_test(NAME xpu_test_cuda COMMAND xpu_test)
  set_tests_properties(xpu_test_cuda PROPERTIES ENVIRONMENT "XPU_DEVICE=cuda0")
endif()

if (XPU_ENABLE_HIP)
  add_test(NAME xpu_test_hip COMMAND xpu_test)
  set_tests_properties(xpu_test_hip PROPERTIES ENVIRONMENT "XPU_DEVICE=hip0")
endif()

if (XPU_ENABLE_SYCL)
  add_test(NAME xpu_test_sycl COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_sycl_test.sh ${CMAKE_CURRENT_BINARY_DIR}/../bin/xpu_test)
  set_tests_properties(xpu_test_sycl PROPERTIES ENVIRONMENT "XPU_DEVICE=sycl1")
endif()
