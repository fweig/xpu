find_package(Doxygen REQUIRED)

set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

doxygen_add_docs(docs
    common.h
    defines.h
    device.h
    host.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src/xpu
    COMMENT "Generate documentation"
)
