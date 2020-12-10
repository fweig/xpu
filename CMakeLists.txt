cmake_minimum_required(VERSION 3.18)

project(xpu)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
enable_language(CUDA)

add_library(xpu
    src/xpu/host.cpp
    src/xpu/dl_utils.cpp
    src/xpu/driver/cpu/cpu_driver.cpp
)

add_library(XPUBackendCUDA MODULE
    src/xpu/driver/cuda/cuda_driver.cu
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wextra -Werror")

target_link_libraries(xpu dl)

include_directories(src)

set_property(GLOBAL PROPERTY xpu_generate_script "${CMAKE_CURRENT_SOURCE_DIR}/generate_device_library_src.sh")
set_property(GLOBAL PROPERTY xpu_root_dir "${CMAKE_CURRENT_SOURCE_DIR}")

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
endfunction()

add_subdirectory(examples/vector_add)
# add_subdirectory(examples/stsreco)