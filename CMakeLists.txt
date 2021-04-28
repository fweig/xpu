cmake_minimum_required(VERSION 3.11)

project(xpu)

if(NOT DEFINED XPU_BUILD_TESTS)
    set(XPU_BUILD_TESTS OFF)
endif()

if (NOT DEFINED XPU_BUILD_EXAMPLES)
    set(XPU_BUILD_EXAMPLES OFF)
endif()

if (NOT DEFINED XPU_ENABLE_CUDA)
    set(XPU_ENABLE_CUDA OFF)
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
if (DEFINED XPU_HIP_CXX)
    set_property(GLOBAL PROPERTY XpuHipCxx "${XPU_HIP_CXX}")
endif()

function(xpu_create_subproject Library DeviceLibrary DriverType UnitySrcAbsolute)
    include(ExternalProject)
    set(SrcDir "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}_${DriverType}")
    get_property(RootDir GLOBAL PROPERTY XpuRootDir)
    get_target_property(IncludeDirectories ${Library} INCLUDE_DIRECTORIES)
    get_property(HipCxx GLOBAL PROPERTY XpuHipCxx)

    if (DriverType STREQUAL "Hip")
        set(SubprojectCxx ${HipCxx})
    else()
        set(SubprojectCxx ${CMAKE_CXX_COMPILER})
    endif()

    file(MAKE_DIRECTORY ${SrcDir})
    configure_file(
        "${RootDir}/templates/CMakeLists.txt.in"
        "${SrcDir}/CMakeLists.txt"
    )

    set(OutputDir ${CMAKE_CURRENT_BINARY_DIR})

    # if (DEFINED CMAKE_INSTALL_LIBDIR)
    #     set(OutputDir ${CMAKE_INSTALL_LIBDIR})
    # else()
    # endif()

    ExternalProject_Add(${DeviceLibrary}_${DriverType}
        SOURCE_DIR "${SrcDir}"
        CMAKE_ARGS
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${OutputDir}
            -DCMAKE_CXX_COMPILER=${SubprojectCxx}
        BUILD_ALWAYS TRUE
        STEP_TARGETS build
        EXCLUDE_FROM_ALL TRUE
    )
    add_dependencies(${Library} ${DeviceLibrary}_${DriverType}-build)
endfunction()

function(xpu_attach_device_library Library DeviceLibrary KernelDecl)
    include(ExternalProject)

    # set(oneValueArgs KERNEL_DECL)
    # set(multiValueArgs SRCS)

    # cmake_parse_arguments(DEV_LIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(DeviceSrcs "${ARGN}")
    set(UnitySrc "${DeviceLibrary}_Unity.cpp")
    set(UnitySrcAbsolute "${CMAKE_CURRENT_BINARY_DIR}/${UnitySrc}")

    get_filename_component(KernelDeclFullPath "${KernelDecl}" REALPATH)

    get_property(RootDir GLOBAL PROPERTY XpuRootDir)
    get_property(EnableCuda GLOBAL PROPERTY XpuEnableCuda)
    get_property(EnableHip GLOBAL PROPERTY XpuHipCxx SET)

    configure_file("${RootDir}/templates/device_library.h.in" "${DeviceLibrary}.h")
    configure_file("${RootDir}/templates/device_library.cpp.in" "${DeviceLibrary}_Xpu_runtime.cpp")
    set(DeviceSrcs "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}_Xpu_runtime.cpp" ${DeviceSrcs})

    target_include_directories(${Library} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
    message(STATUS "Adding source ${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}.cpp to target ${Library}")
    target_sources(${Library} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}_Xpu_runtime.cpp")

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

    if (${EnableHip})
        xpu_create_subproject(${Library} ${DeviceLibrary} Hip ${UnitySrcAbsolute})
    endif()
endfunction()

include_directories(src)

add_library(xpu SHARED
    src/xpu/host.cpp
    src/xpu/detail/dl_utils.cpp
    src/xpu/driver/cpu/cpu_driver.cpp
)
target_link_libraries(xpu dl)

if (XPU_ENABLE_CUDA)
    enable_language(CUDA)
    xpu_create_subproject(
        xpu xpu_driver Cuda
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/driver/hip_cuda/cuda_driver.cu"
    )
endif()

if (DEFINED XPU_HIP_CXX)
    xpu_create_subproject(
        xpu xpu_driver Hip
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/driver/hip_cuda/hip_driver.cpp"
    )
endif()

if (XPU_BUILD_TESTS)
    add_subdirectory(test)
endif()

if (XPU_BUILD_EXAMPLES)
    add_subdirectory(examples/vector_add)
    add_subdirectory(examples/sorting)
endif()
