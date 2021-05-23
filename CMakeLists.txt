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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -g3 -Wall -Wextra -Werror -fmax-errors=5")
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

function(xpu_create_subproject Library DriverType UnitySrcAbsolute)
    include(ExternalProject)
    set(SrcDir "${CMAKE_CURRENT_BINARY_DIR}/${Library}_${DriverType}")
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

    ExternalProject_Add(${Library}_${DriverType}
        SOURCE_DIR "${SrcDir}"
        CMAKE_ARGS
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${OutputDir}
            -DCMAKE_CXX_COMPILER=${SubprojectCxx}
        BUILD_ALWAYS TRUE
        STEP_TARGETS build
        EXCLUDE_FROM_ALL TRUE
    )
    add_dependencies(${Library} ${Library}_${DriverType}-build)
endfunction()

function(xpu_attach Library)
    include(ExternalProject)

    set(DeviceSrcs "${ARGN}")
    set(UnitySrc "${Library}_Unity.cpp")
    set(UnitySrcAbsolute "${CMAKE_CURRENT_BINARY_DIR}/${UnitySrc}")

    get_filename_component(KernelDeclFullPath "${KernelDecl}" REALPATH)

    get_property(RootDir GLOBAL PROPERTY XpuRootDir)
    get_property(EnableCuda GLOBAL PROPERTY XpuEnableCuda)
    get_property(EnableHip GLOBAL PROPERTY XpuHipCxx SET)

    target_compile_definitions(${Library} PRIVATE XPU_IMAGE_FILE="${Library}")
    target_compile_options(${Library} PUBLIC -fPIC)

    file(REMOVE ${UnitySrcAbsolute})
    foreach(File ${DeviceSrcs})
        get_filename_component(FileAbsolute "${File}" REALPATH)
        file(APPEND ${UnitySrcAbsolute}
            "#include <${FileAbsolute}>\n"
        )
    endforeach()

    if (${EnableCuda})
        xpu_create_subproject(${Library} Cuda ${UnitySrcAbsolute})
    endif()

    if (${EnableHip})
        xpu_create_subproject(${Library} Hip ${UnitySrcAbsolute})
    endif()
endfunction()

include_directories(src)

add_library(xpu SHARED
    src/xpu/detail/dl_utils.cpp
    src/xpu/detail/runtime.cpp
    src/xpu/driver/cpu/cpu_driver.cpp
    src/xpu/driver/cpu/this_thread.cpp
)
target_link_libraries(xpu dl)

if (XPU_ENABLE_CUDA)
    enable_language(CUDA)
    xpu_create_subproject(
        xpu Cuda
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/driver/hip_cuda/cuda_driver.cu"
    )
endif()

if (DEFINED XPU_HIP_CXX)
    xpu_create_subproject(
        xpu Hip
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
