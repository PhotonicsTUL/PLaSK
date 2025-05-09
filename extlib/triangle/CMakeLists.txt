project(triangle C)
cmake_minimum_required(VERSION 3.14)

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
    add_library(triangle SHARED triangle.cc triangle.h)
else()
    add_library(triangle STATIC triangle.cc triangle.h)
endif ()

target_compile_options(triangle PRIVATE -DTRILIBRARY -DNO_TIMER)
if (NOT WIN32)
    target_compile_options(triangle PRIVATE -fPIC)
elseif (MSVC)
    target_compile_options(triangle PRIVATE /wd4244 /wd4267 /wd4305 /wd4996)
endif()

target_include_directories(triangle INTERFACE .)

if (USE_SYSTEM_FMT)
    target_link_libraries(triangle PRIVATE fmt::fmt)
endif()

if(PLASK_TRIANGLE_SHARED)
    if(WIN32)
        if(INSTALL_EXTERNAL_DEV)
            install(TARGETS triangle RUNTIME DESTINATION bin COMPONENT core
                                     ARCHIVE DESTINATION lib COMPONENT dev)
        else()
            install(TARGETS triangle RUNTIME DESTINATION bin COMPONENT core)
        endif()
    else()
        install(TARGETS triangle LIBRARY DESTINATION lib COMPONENT core)
    endif()
elseif(INSTALL_EXTERNAL_DEV)
    install(TARGETS triangle ARCHIVE DESTINATION lib COMPONENT dev)
endif()

if(INSTALL_EXTERNAL_DEV)
    install(FILES triangle.h DESTINATION include COMPONENT dev)
endif()
