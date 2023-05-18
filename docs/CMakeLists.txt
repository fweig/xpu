find_package(Doxygen REQUIRED)

set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_SHOW_INCLUDE_FILES NO)
set(DOXYGEN_SHOW_NAMESPACES YES)
set(DOXYGEN_HAVE_DOT NO)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ../../README.md)
set(DOXYGEN_HTML_EXTRA_STYLESHEET
    ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-awesome.css
    ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-awesome-sidebar-only.css
)
set(DOXYGEN_HTML_COLORSTYLE LIGHT)

doxygen_add_docs(docs
    ../../README.md
    common.h
    defines.h
    device.h
    host.h
    ../../docs/namespace.dox
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src/xpu
    COMMENT "Generate documentation"
)
