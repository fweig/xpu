include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  set(WARN_DEPRECATED_BACK ${CMAKE_WARN_DEPRECATED})
  set(CMAKE_WARN_DEPRECATED FALSE)
  set(BUILD_GMOCK OFF)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  set(CMAKE_WARN_DEPRECATED ${WARN_DEPRECATED_BACK})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(deviceSrcs TestKernels.cpp)
add_library(TestKernels SHARED ${deviceSrcs})
xpu_attach(TestKernels ${deviceSrcs})
add_executable(xpu_test xpu_test.cpp)
target_link_libraries(xpu_test xpu TestKernels gtest)
