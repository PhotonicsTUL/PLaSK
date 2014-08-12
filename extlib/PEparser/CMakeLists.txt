if(WIN32)   # this PE parser is for windows only

project(PEparser)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(PEparser STATIC
    file.cpp file.hpp
    pe.cpp pe.hpp
    pe_symtab.cpp pe_symtap.hpp
    symbols.cpp)

endif(WIN32)
