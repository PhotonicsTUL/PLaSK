project(camos C)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(camos SHARED mach.c camos.c camos.h)

if(WIN32)
    install(TARGETS camos RUNTIME DESTINATION bin COMPONENT solvers
                          ARCHIVE DESTINATION lib COMPONENT dev)
else(WIN32)
    install(TARGETS camos LIBRARY DESTINATION lib COMPONENT solvers)
endif(WIN32)

install(FILES cmaos.h DESTINATION include/camos COMPONENT dev)