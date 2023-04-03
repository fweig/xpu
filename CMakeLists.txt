cmake_minimum_required(VERSION 3.11)

project(xpu)

set(XPU_BUILD_TESTS OFF CACHE BOOL "Build xpu unit tests.")
set(XPU_BUILD_EXAMPLES OFF CACHE BOOL "Build xpu examples.")
if (APPLE)
    set(XPU_ENABLE_OPENMP OFF CACHE BOOL "Use OpenMP for kernel parallelization.")
else()
    set(XPU_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP for kernel parallelization.")
endif()
set(XPU_ENABLE_CUDA OFF CACHE BOOL "Enable xpu cuda backend.")
set(XPU_CUDA_ARCH 75 CACHE STRING "Target cuda architectures.")
set(XPU_ENABLE_HIP OFF CACHE STRING "Enable xpu hip backend.")
set(XPU_HIP_ARCH "gfx906;gfx908" CACHE STRING "Target hip architectures.")
set(XPU_ROCM_ROOT /opt/rocm CACHE PATH "Path to rocm installation.")
set(XPU_ENABLE_SYCL OFF CACHE BOOL "Enable xpu sycl backend.")
set(XPU_SYCL_CXX "icpx" CACHE STRING "Path to sycl compiler.")
set(XPU_SYCL_TARGETS "spir64" CACHE STRING "Target sycl architectures. (Use 'x86_64' for better debugging.)")
set(XPU_DEBUG OFF CACHE BOOL "Enable debug options for GPU code.")
set(XPU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(XPU_STANDALONE ON)
else()
    set(XPU_STANDALONE OFF)
endif()

if (XPU_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
endif()

if (XPU_ENABLE_HIP)
    set(HipClang "${XPU_ROCM_ROOT}/llvm/bin/clang")
    if (NOT EXISTS "${HipClang}")
        message(FATAL_ERROR
            "Could not find hip-clang (looked in ${HipClang}). "
            "Please set XPU_ROCM_ROOT to your ROCM installation."
        )
    endif()
endif()

message(STATUS "xpu configuration:")
message(STATUS "  XPU_BUILD_TESTS:    ${XPU_BUILD_TESTS}")
message(STATUS "  XPU_BUILD_EXAMPLES: ${XPU_BUILD_EXAMPLES}")
message(STATUS "  XPU_ENABLE_OPENMP:  ${XPU_ENABLE_OPENMP}")
if (XPU_ENABLE_CUDA)
    message(STATUS "  XPU_ENABLE_CUDA:    ON (cc: ${CMAKE_CUDA_COMPILER}, arch: ${XPU_CUDA_ARCH})")
else()
    message(STATUS "  XPU_ENABLE_CUDA:    OFF")
endif()
if (XPU_ENABLE_HIP)
    message(STATUS "  XPU_ENABLE_HIP:     ON (cc: ${HipClang}, arch: ${XPU_HIP_ARCH})")
else()
    message(STATUS "  XPU_ENABLE_HIP:     OFF")
endif()
if (XPU_ENABLE_SYCL)
    message(STATUS "  XPU_ENABLE_SYCL:    ON (cc: ${XPU_SYCL_CXX}, targets: ${XPU_SYCL_TARGETS})")
else()
    message(STATUS "  XPU_ENABLE_SYCL:    OFF")
endif()
message(STATUS "  XPU_DEBUG:          ${XPU_DEBUG}")

if (XPU_STANDALONE)
    include(CheckCXXCompilerFlag)
    if (NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 17)
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wextra -Werror -Wfatal-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

    check_cxx_compiler_flag("-fmax-errors=5" CXX_HAS_MAX_ERRORS_FLAG)
    if (CXX_HAS_MAX_ERRORS_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=5")
    endif()
    if (XPU_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Og")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O2")
    endif()

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

function(xpu_create_subproject Library DriverType UnitySrcAbsolute)
    include(ExternalProject)

    get_target_property(OutputDir ${Library} LIBRARY_OUTPUT_DIRECTORY)
    set(SubLibraryName ${Library}_${DriverType})
    set(SubLibraryFullPath ${OutputDir}/${CMAKE_SHARED_LIBRARY_PREFIX}${SubLibraryName}${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(SrcDir "${CMAKE_CURRENT_BINARY_DIR}/${SubLibraryName}")
    set(HostCxx ${CMAKE_CXX_COMPILER})

    if (DriverType STREQUAL "Hip")
        set(HipClang "${XPU_ROCM_ROOT}/llvm/bin/clang")
        set(SubprojectCxx ${HipClang}++)
        set(SubprojectC ${HipClang})
    elseif(DriverType STREQUAL "Sycl")
        set(SubprojectCxx ${XPU_SYCL_CXX})
        set(SubprojectC ${CMAKE_C_COMPILER})
    elseif (DriverType STREQUAL "Cuda")
        set(SubprojectCxx ${CMAKE_CXX_COMPILER})
        set(SubprojectC ${CMAKE_C_COMPILER})
    else()
        message(FATAL_ERROR "Unknown driver type ${DriverType}.")
    endif()

    file(MAKE_DIRECTORY ${SrcDir})
    configure_file(
        "${XPU_ROOT_DIR}/templates/CMakeLists.txt.in"
        "${SrcDir}/CMakeLists.txt"
        @ONLY
    )

    message(STATUS "Generating library ${SubLibraryName}.")
    ExternalProject_Add(${SubLibraryName}
        SOURCE_DIR "${SrcDir}"
        CMAKE_ARGS
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${OutputDir}
            -DCMAKE_CXX_COMPILER=${SubprojectCxx}
            -DCMAKE_C_COMPILER=${SubprojectC}
            -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        PREFIX "${SrcDir}"
        BUILD_ALWAYS TRUE
        STEP_TARGETS build
        EXCLUDE_FROM_ALL TRUE
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_MERGED_STDOUTERR TRUE
        LOG_OUTPUT_ON_FAILURE TRUE
    )

    ExternalProject_Add_Step(${SubLibraryName} export-include-dirs
        COMMAND ${CMAKE_COMMAND} -E echo "include_directories($<TARGET_PROPERTY:${Library},INCLUDE_DIRECTORIES>)" > "${SubLibraryName}-include-dirs.cmake"
        WORKING_DIRECTORY ${SrcDir}
        BYPRODUCTS "${SrcDir}/${SubLibraryName}-include-dirs.cmake"
        # COMMENT "${SubLibraryName}: Export include directories"
        DEPENDERS configure
    )

    add_dependencies(${Library} ${SubLibraryName}-build)
endfunction()

function(xpu_attach Library)
    include(ExternalProject)

    set(DeviceSrcs "${ARGN}")
    set(UnitySrc "${Library}_Unity.cpp")
    set(UnitySrcAbsolute "${CMAKE_CURRENT_BINARY_DIR}/${UnitySrc}")

    get_filename_component(KernelDeclFullPath "${KernelDecl}" REALPATH)

    set_property(TARGET ${Library} APPEND PROPERTY COMPILE_DEFINITIONS XPU_IMAGE_FILE="${Library}")
    set_property(TARGET ${Library} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
    set_property(TARGET ${Library} APPEND PROPERTY LINK_LIBRARIES xpu)

    get_target_property(DeviceLibDir ${Library} LIBRARY_OUTPUT_DIRECTORY)
    set_property(TARGET ${Library} APPEND PROPERTY BUILD_RPATH ${DeviceLibDir})

    if (XPU_ENABLE_OPENMP)
        find_package(OpenMP)
        if (NOT OpenMP_CXX_FOUND)
            message(FATAL_ERROR "OpenMP not found. Try to disable OpenMP in xpu via '-DXPU_ENABLE_OPENMP=OFF'.")
        endif()
        set_property(TARGET ${Library} APPEND PROPERTY LINK_LIBRARIES OpenMP::OpenMP_CXX)
    endif()

    file(REMOVE ${UnitySrcAbsolute})
    foreach(File ${DeviceSrcs})
        get_filename_component(FileAbsolute "${File}" REALPATH)
        file(APPEND ${UnitySrcAbsolute}
            "#include <${FileAbsolute}>\n"
        )
    endforeach()

    if (XPU_ENABLE_CUDA)
        xpu_create_subproject(${Library} Cuda ${UnitySrcAbsolute})
    endif()

    if (XPU_ENABLE_HIP)
        xpu_create_subproject(${Library} Hip ${UnitySrcAbsolute})
    endif()

    if (XPU_ENABLE_SYCL)
        xpu_create_subproject(${Library} Sycl ${UnitySrcAbsolute})
    endif()
endfunction()

add_library(xpu SHARED
    src/xpu/detail/backend.cpp
    src/xpu/detail/buffer_registry.cpp
    src/xpu/detail/common.cpp
    src/xpu/detail/dl_utils.cpp
    src/xpu/detail/exceptions.cpp
    src/xpu/detail/log.cpp
    src/xpu/detail/queue_handle.cpp
    src/xpu/detail/runtime.cpp
    src/xpu/detail/platform/cpu/cpu_driver.cpp
    src/xpu/detail/platform/cpu/this_thread.cpp
)
target_link_libraries(xpu dl)
target_include_directories(xpu PUBLIC src)

install(TARGETS xpu
    LIBRARY
)

install(DIRECTORY src/ TYPE INCLUDE
    FILES_MATCHING PATTERN "*.h"
)

if (XPU_ENABLE_CUDA)
    xpu_create_subproject(
        xpu Cuda
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/detail/platform/hip_cuda/cuhip_driver.cpp"
    )
endif()

if (XPU_ENABLE_HIP)
    xpu_create_subproject(
        xpu Hip
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/detail/platform/hip_cuda/cuhip_driver.cpp"
    )
endif()

if (XPU_ENABLE_SYCL)
    xpu_create_subproject(
        xpu Sycl
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu/detail/platform/sycl/sycl_driver.cpp"
    )
endif()

get_target_property(DeviceLibDir xpu LIBRARY_OUTPUT_DIRECTORY)
set_property(TARGET xpu APPEND PROPERTY BUILD_RPATH ${DeviceLibDir})

if (XPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()

if (XPU_BUILD_EXAMPLES)
    add_subdirectory(examples/merging)
    add_subdirectory(examples/vector_add)
    add_subdirectory(examples/sorting)
    add_subdirectory(examples/xpuinfo)

    add_custom_target(examples
        DEPENDS
            merge
            vector_add
            sort_struct
            xpuinfo
    )
endif()
