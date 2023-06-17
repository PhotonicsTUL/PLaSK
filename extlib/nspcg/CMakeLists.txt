project(nspcg Fortran)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # uses by MSVC
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
endforeach()
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)  # all functions will be exported from windows DLL

add_library(nspcg SHARED nspcg.f)
target_compile_options(nspcg PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-std=legacy>)

if(WIN32)
    if(INSTALL_DEV)
        install(TARGETS nspcg RUNTIME DESTINATION bin COMPONENT solvers
                              ARCHIVE DESTINATION lib COMPONENT dev)
        else()
            install(TARGETS nspcg RUNTIME DESTINATION bin COMPONENT solvers)
    endif()
else()
    install(TARGETS nspcg LIBRARY DESTINATION lib COMPONENT solvers)
endif()

install(FILES nspcg.hpp DESTINATION include COMPONENT dev)