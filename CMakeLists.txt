cmake_minimum_required(VERSION 3.18)

project(xpu)

if(NOT DEFINED XPU_BUILD_TESTS)
    set(XPU_BUILD_TESTS OFF)
endif()

if (NOT DEFINED XPU_BUILD_EXAMPLES)
    set(XPU_BUILD_EXAMPLES OFF)
endif()

if (NOT DEFINED XPU_ENABLE_CUDA)
check_language(CUDA)
if (${CMAKE_CUDA_COMPILER})
    set(XPU_ENABLE_CUDA ON)
else()
    set(XPU_ENABLE_CUDA OFF)
endif()
endif()

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(XPU_STANDALONE ON)
else()
    set(XPU_STANDALONE OFF)
endif()

if (XPU_STANDALONE)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wextra -Werror -Wfatal-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wextra -Werror")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set_property(GLOBAL PROPERTY XpuRootDir "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(GLOBAL PROPERTY XpuEnableCuda "${XPU_ENABLE_CUDA}")


function(xpu_create_subproject Library DeviceLibrary DriverType UnitySrcAbsolute)
    include(ExternalProject)
    set(DriverType "Cuda")
    set(SrcDir "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}_${DriverType}")
    get_property(RootDir GLOBAL PROPERTY XpuRootDir)
    get_target_property(IncludeDirectories ${Library} INCLUDE_DIRECTORIES)
    file(MAKE_DIRECTORY ${SrcDir})
    configure_file(
        "${RootDir}/templates/CMakeLists.txt.in" 
        "${SrcDir}/CMakeLists.txt"
    )
    ExternalProject_Add(${DeviceLibrary}_${DriverType}
        SOURCE_DIR "${SrcDir}"
        CMAKE_ARGS  
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
        BUILD_ALWAYS TRUE
        STEP_TARGETS build
        EXCLUDE_FROM_ALL TRUE
    )
    add_dependencies(${Library} ${DeviceLibrary}_${DriverType}-build)
endfunction()

function(xpu_attach_device_library Library DeviceLibrary KernelDecl)
    include(ExternalProject)

    set(oneValueArgs KERNEL_DECL)
    set(multiValueArgs SRCS)

    cmake_parse_arguments(DEV_LIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(DeviceSrcs "${ARGN}")
    set(UnitySrc "${DeviceLibrary}_Unity.cpp")
    set(UnitySrcAbsolute "${CMAKE_CURRENT_BINARY_DIR}/${UnitySrc}")

    get_filename_component(KernelDeclFullPath "${KernelDecl}" REALPATH)

    get_property(RootDir GLOBAL PROPERTY XpuRootDir)
    get_property(EnableCuda GLOBAL PROPERTY XpuEnableCuda)

    configure_file("${RootDir}/templates/device_library.h.in" "${DeviceLibrary}.h")
    configure_file("${RootDir}/templates/device_library.cpp.in" "${DeviceLibrary}.cpp")
    set(DeviceSrcs "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}.cpp" ${DeviceSrcs})

    target_include_directories(${Library} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
    target_sources(${Library} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}.cpp")

    file(REMOVE ${UnitySrcAbsolute})
    foreach(File ${DeviceSrcs})
        get_filename_component(FileAbsolute "${File}" REALPATH)
        file(APPEND ${UnitySrcAbsolute}
            "#include <${FileAbsolute}>\n"
        )
    endforeach()

    if (${EnableCuda})
        xpu_create_subproject(${Library} ${DeviceLibrary} Cuda ${UnitySrcAbsolute})
    endif()
endfunction()

include_directories(src)

add_library(xpu SHARED
    src/xpu/host.cpp
    src/xpu/dl_utils.cpp
    src/xpu/driver/cpu/cpu_driver.cpp
)
target_link_libraries(xpu dl)


if (${XPU_ENABLE_CUDA})
    enable_language(CUDA)
    xpu_create_subproject(xpu xpu_driver Cuda "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/driver/cuda/cuda_driver.cu")
endif()



if (XPU_BUILD_TESTS)
    add_subdirectory(test)
endif()

if (XPU_BUILD_EXAMPLES)
    add_subdirectory(examples/vector_add)
endif()
# add_subdirectory(examples/stsreco)