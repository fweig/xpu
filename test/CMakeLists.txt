include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        4ec4cd23f486bf70efcc5d2caa40f24368f752e3
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
