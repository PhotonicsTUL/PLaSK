# After calling 'cmake' use:
#  'make' to build all,
#  'make test' to run tests.

cmake_minimum_required (VERSION 2.8)
project (plask)

add_definitions(-Wall)        # all warnings
add_definitions(-std=c++0x)    # C++0x (C++11) standard

include_directories(.)        # external modules can include plask using <plask/....h>
include_directories(extlibs)    # place for local external libraries

# compress and strip executable (target)
MACRO(compress targetname)
    if (PACK_EXE)
        get_property(apppath TARGET ${targetname} PROPERTY LOCATION)
        if (CMAKE_STRIP)
            add_custom_command(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_STRIP}
                                ARGS ${apppath})
        endif(CMAKE_STRIP)
        if (SELF_PACKER_FOR_EXECUTABLE)
            add_custom_command(TARGET ${targetname} POST_BUILD COMMAND ${SELF_PACKER_FOR_EXECUTABLE}
                                ARGS ${SELF_PACKER_FOR_EXECUTABLE_FLAGS} "-q" ${apppath})    #TODO remove "-q" when SelfPackers will be fixed
        endif(SELF_PACKER_FOR_EXECUTABLE)
    endif(PACK_EXE)
ENDMACRO(compress targetname)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")

option(PACK_EXE "self-pack and strip executables" OFF)  #compress exe
if (PACK_EXE)
    find_package(SelfPackers)
    find_program(CMAKE_STRIP NAMES strip)
endif(PACK_EXE)

# ----------===== plask library =====--------------------------------------------
# creating configuration file for plask library
#CONFIGURE_FILE(plask/config.h.cmake plask/config.h)

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h)
file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp plask/*.cxx plask/*.h)
add_library(plask STATIC ${plask_src})

option (BUILD_SHARED_LIBS "build also shared library" ON)
if (BUILD_SHARED_LIBS)
    add_library(plask-shared SHARED ${plask_src})
    set_target_properties(plask-shared PROPERTIES OUTPUT_NAME plask)
    set_target_properties(plask-shared PROPERTIES COMPILE_FLAGS -fpic)
endif(BUILD_SHARED_LIBS)


# ----------===== Python =====--------------------------------------------

option (BUILD_PYTHON_MODULES "build and install Python bindings" ON)
if (BUILD_PYTHON_MODULES)
    # Find Python
    include(FindPythonInterp)
    include(FindPythonLibs)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/determineNumpyPath.py "try: import numpy; print numpy.get_include()\nexcept: pass\n")
    exec_program("${PYTHON_EXECUTABLE}" ARGS "\"${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/determineNumpyPath.py\"" OUTPUT_VARIABLE NUMPY_PATH})
#     if (NUMPY_PATH)
#     endif (NUMPY_PATH)
#   TODO: break if numpy not found

endif(BUILD_PYTHON_MODULES)

# ----------===== unit tests =====--------------------------------------------
option (MAKE_TESTS "build and run unit tests" ON)
if (MAKE_TESTS)
    enable_testing()
    find_package(Boost COMPONENTS unit_test_framework)
    if(Boost_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        file(GLOB_RECURSE test_plask_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.cxx tests/plask/*.h)
        add_executable(test_plask ${test_plask_src})
        target_link_libraries(test_plask ${Boost_LIBRARIES})

        add_test(plask ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask)

        if(BUILD_PYTHON_MODULES)
            file(GLOB plask_python_tests FOLLOW_SYMLINKS tests/python/test*.py)
            make_directory(${CMAKE_CURRENT_BINARY_DIR}/Testing/Python)
            foreach(plask_python_test ${plask_python_tests})
                configure_file(${plask_python_test} ${CMAKE_CURRENT_BINARY_DIR}/Testing/Python)
            endforeach(plask_python_test)
            add_test(plask-python "${PYTHON_EXECUTABLE}" -m unittest discover -t ${CMAKE_CURRENT_BINARY_DIR}/Testing/Python -s ${CMAKE_CURRENT_BINARY_DIR}/Testing/Python)
        endif(BUILD_PYTHON_MODULES)

        add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask)
    else(Boost_FOUND)
        message("Boost Test Library not found. Building without unit tests.")
    endif(Boost_FOUND)
endif(MAKE_TESTS)

# ----------===== documentation (api) =====--------------------------------------------
option(BUILD_DOCUMENTATION "build doxygen documentation" OFF)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_custom_target(doc ALL ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    else(DOXYGEN_FOUND)
        message("Doxygen not found. Build without documentation.")
    endif(DOXYGEN_FOUND)
endif(BUILD_DOCUMENTATION)

# ----------===== install =====--------------------------------------------
if (BUILD_SHARED_LIBS)
    install(TARGETS plask-shared LIBRARY DESTINATION lib COMPONENT libplask)
endif(BUILD_SHARED_LIBS)

install(TARGETS plask ARCHIVE DESTINATION lib COMPONENT libplask-dev)
install(FILES ${plask_headers} DESTINATION include/plask COMPONENT libplask-dev)
