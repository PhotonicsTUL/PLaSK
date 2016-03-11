if(WIN32)   # this PE parser is for windows only
# it comes from edd dbg project (boost license)
# see: https://bitbucket.org/edd/dbg/src/17b3135bcf73e4187826eb8c70f44665fd6cb816/src/windows/mingw/?at=default

project(PEparser)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # uses by MSVC
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
endforeach()

add_library(PEparser STATIC
    file.cpp file.hpp
    pe.cpp pe.hpp
    pe_symtab.cpp pe_symtab.hpp
    symbols.cpp symbols.hpp
    )

endif(WIN32)
