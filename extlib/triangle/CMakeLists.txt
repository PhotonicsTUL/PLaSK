project(triangle C)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # uses by MSVC
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
endforeach()
#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)  #all functions will be exported from windows DLL

add_library(triangle STATIC triangle.c triangle.h)
target_compile_options(triangle PRIVATE -DTRILIBRARY -fPIC)
#target_include_directories(triangle INTERFACE .)

install(TARGETS triangle ARCHIVE DESTINATION lib COMPONENT dev)
install(FILES triangle.h DESTINATION include COMPONENT dev)
