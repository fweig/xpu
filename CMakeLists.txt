cmake_minimum_required(VERSION 3.18)

project(xpu)

enable_language(CUDA)

if(NOT DEFINED XPU_BUILD_TESTS)
    set(XPU_BUILD_TESTS OFF)
endif()

if (NOT DEFINED XPU_BUILD_EXAMPLES)
    set(XPU_BUILD_EXAMPLES OFF)
endif()

if (NOT DEFINED XPU_ENABLE_CUDA)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
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

add_library(xpu SHARED
    src/xpu/host.cpp
    src/xpu/dl_utils.cpp
    src/xpu/driver/cpu/cpu_driver.cpp
)

if (${XPU_ENABLE_CUDA})
    add_library(XPUBackendCUDA MODULE
        src/xpu/driver/cuda/cuda_driver.cu
    )
endif()


target_link_libraries(xpu dl)

include_directories(src)

set_property(GLOBAL PROPERTY xpu_generate_script "${CMAKE_CURRENT_SOURCE_DIR}/generate_device_library_src.sh")
set_property(GLOBAL PROPERTY xpu_root_dir "${CMAKE_CURRENT_SOURCE_DIR}")

function(xpu_attach_device_library Library DeviceLibrary KernelDecl)
    set(oneValueArgs KERNEL_DECL)
    set(multiValueArgs SRCS)

    cmake_parse_arguments(DEV_LIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(Srcs "${ARGN}")
    set(UnitySrc "${DeviceLibrary}_Unity.cpp")

    get_filename_component(KernelDeclFullPath "${KernelDecl}" REALPATH)

    get_property(RootDir GLOBAL PROPERTY xpu_root_dir)

    configure_file("${RootDir}/templates/device_library.h.in" "${DeviceLibrary}.h")
    configure_file("${RootDir}/templates/device_library.cpp.in" "${DeviceLibrary}.cpp")

    target_include_directories(${Library} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
    target_sources(${Library} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/${DeviceLibrary}.cpp")

    foreach(File ${Srcs})
        get_filename_component(FileAbsolute "${File}" REALPATH)
        file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/${UnitySrc}
            "#include <${FileAbsolute}>\n"
        )
    endforeach()

endfunction()

function(xpu_add_device_library libName)
    set(oneValueArgs KERNEL_DEF INTERFACE_H)
    set(multiValueArgs SRCS)

    cmake_parse_arguments(DEV_LIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(kernelDeclFile "${DEV_LIB_KERNEL_DEF}")
    set(sourceFiles "${DEV_LIB_SRCS}")
    set(interfaceFile "${DEV_LIB_INTERFACE_H}")
    set(frontendHeader "${libName}.h")
    set(frontendSrc "${libName}CPU.cpp")
    set(backendCudaName "${libName}CUDA")
    set(backendCudaSrc "${libName}CUDA.cu")

    get_property(generateScript GLOBAL PROPERTY xpu_generate_script)
    get_property(rootDir GLOBAL PROPERTY xpu_root_dir)


    # add_custom_command(
    #     OUTPUT ${frontendHeader}
    #     COMMAND bash ${generateScript} -t header -o ${CMAKE_CURRENT_BINARY_DIR}/${frontendHeader}
    #     DEPENDS ${generateScript}
    # )

    # add_custom_command(
    #     OUTPUT ${frontendSrc}
    #     COMMAND bash ${generateScript} -t frontend -s ${sourceFiles} -o ${CMAKE_CURRENT_BINARY_DIR}/${frontendSrc}
    #     DEPENDS ${generateScript} ${kernelDeclFile} ${sourceFiles} ${frontendHeader}
    # )
    set(backendName "${libName}CPU")
    configure_file(${rootDir}/src/xpu/device_library/template/frontend.h.in ${libName}.h)
    configure_file(${rootDir}/src/xpu/device_library/template/frontend.cpp.in ${libName}Frontend.cpp)
    configure_file(${rootDir}/src/xpu/device_library/template/backend.h.in ${libName}CPU.h)
    file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/${libName}CPUUnity.cpp
        "#include <${libName}Frontend.cpp>\n"
    )
    foreach(file ${sourceFiles})
        file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/${libName}CPUUnity.cpp
            "#include <${file}>\n"
        )
    endforeach()
    add_library(${libName}
        ${CMAKE_CURRENT_BINARY_DIR}/${libName}CPUUnity.cpp
    )
    target_include_directories(${libName}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}    
    )
    target_compile_definitions(${libName} 
        PRIVATE XPU_DEVICE_LIBRARY_BACKEND_NAME=${libName}CPU
    )

    if (${XPU_CUDA_ENABLED})
        set(backendName "${libName}CUDA")
        configure_file(${rootDir}/src/xpu/device_library/template/backend.h.in ${libName}CUDA.h)
        configure_file(${rootDir}/src/xpu/device_library/template/backend.cpp.in ${libName}CUDABackend.cpp)
        file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/${libName}CUDAUnity.cu
            "#include <${libName}CUDABackend.cpp>\n"
        )
        foreach(file ${sourceFiles})
            file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/${libName}CUDAUnity.cu
                "#include <${file}>\n"
            )
        endforeach()
        add_library(${backendCudaName} MODULE
            ${CMAKE_CURRENT_BINARY_DIR}/${libName}CUDAUnity.cu
        )
        target_include_directories(${backendCudaName}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        )
        target_compile_definitions(${backendCudaName}
            PRIVATE XPU_DEVICE_LIBRARY_BACKEND_NAME=${libName}CUDA
        )
    endif()
endfunction()

if (XPU_BUILD_TESTS)
    add_subdirectory(test)
endif()

if (XPU_BUILD_EXAMPLES)
    add_subdirectory(examples/vector_add)
endif()
# add_subdirectory(examples/stsreco)