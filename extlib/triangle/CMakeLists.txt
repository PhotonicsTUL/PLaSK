project(triangle C)
cmake_minimum_required(VERSION 2.8)

option(PLASK_TRIANGLE_SHARED "compile shared (static is default) version of Triangle library" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # uses by MSVC
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
endforeach()

if (PLASK_TRIANGLE_SHARED)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)  # all functions will be exported from windows DLL
    add_library(triangle SHARED triangle.c triangle.h)
else(PLASK_TRIANGLE_SHARED)
    add_library(triangle STATIC triangle.c triangle.h)
endif (PLASK_TRIANGLE_SHARED)

target_compile_options(triangle PRIVATE -DTRILIBRARY -DNO_TIMER -fPIC)
target_include_directories(triangle INTERFACE .)

if (PLASK_TRIANGLE_SHARED)
    if(WIN32)
        install(TARGETS triangle RUNTIME DESTINATION bin COMPONENT core
                              ARCHIVE DESTINATION lib COMPONENT dev)
    else(WIN32)
        install(TARGETS triangle LIBRARY DESTINATION lib COMPONENT core)
    endif(WIN32)
else(PLASK_TRIANGLE_SHARED)
    install(TARGETS triangle ARCHIVE DESTINATION lib COMPONENT dev)
endif (PLASK_TRIANGLE_SHARED)

install(FILES triangle.h DESTINATION include COMPONENT dev)
