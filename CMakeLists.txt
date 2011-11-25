# After calling 'cmake' use:
#  'make' to build all,
#  'make test' to run tests.

cmake_minimum_required (VERSION 2.8)
project (plask)

add_definitions(-Wall)          # all warnings
add_definitions(-std=c++0x)     # C++0x (C++11) standard

include_directories(.)          # external modules can include plask using <plask/....h>
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

# ---------- user options --------------------------------------------
option (BUILD_SHARED_LIBS "build also shared library" ON)
option (BUILD_PYTHON_MODULE "build and install Python module plask" ON)
option (MAKE_TESTS "build and run unit tests" ON)


# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
find_package(Boost COMPONENTS unit_test_framework python)

# ----------===== plask library =====--------------------------------------------
# creating configuration file for plask library
#CONFIGURE_FILE(plask/config.h.cmake plask/config.h)

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h)
file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp plask/*.cxx plask/*.h)

add_library(plask-static STATIC ${plask_src})
set_target_properties(plask-static PROPERTIES OUTPUT_NAME plask)

if (BUILD_SHARED_LIBS)
    add_library(plask-shared SHARED ${plask_src})
    set_target_properties(plask-shared PROPERTIES OUTPUT_NAME plask)
    set_target_properties(plask-shared PROPERTIES COMPILE_FLAGS -fPIC)
endif()

# ----------===== Python =====--------------------------------------------

if (BUILD_PYTHON_MODULE)
    # Find Python
    find_package(PythonInterp)
    if (NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "Python not found! Try builing without BUILD_PYTHON_MODULE.")
    endif ()
    find_package(PythonLibs)
    if (NOT PYTHON_INCLUDE_DIRS)
        message(FATAL_ERROR "Python includes not found! Try builing without BUILD_PYTHON_MODULE.")
    endif ()

    # Determine directory to install *.py files to (TODO: decide if they shouldn't be installed as local modules)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonInstallPath.py "from distutils import sysconfig\nprint sysconfig.get_python_lib(False)")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonInstallPath.py" OUTPUT_VARIABLE PYTHON_INSTALL_DIR)
    # Determine directory to install binary Python modules to (TODO: decide if they shouldn't be installed as local modules)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonModuleInstallPath.py "from distutils import sysconfig\nprint sysconfig.get_python_lib(True)")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonModuleInstallPath.py" OUTPUT_VARIABLE PYTHON_MODULE_INSTALL_DIR)

    # Find numpy
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DetermineNumpyPath.py "try: import numpy; print numpy.get_include()\nexcept: pass\n")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DetermineNumpyPath.py" OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS)
    if (NOT NUMPY_INCLUDE_DIRS)
        message(FATAL_ERROR "Python package 'numpy' not found! Try builing without BUILD_PYTHON_MODULE.")
    endif ()

    # --- Here build the modules ---


endif(BUILD_PYTHON_MODULE)

# ----------===== unit tests =====--------------------------------------------
if (MAKE_TESTS)

    enable_testing()
    if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        file(GLOB_RECURSE test_plask_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.cxx tests/plask/*.h)
        add_executable(test_plask ${test_plask_src})
        target_link_libraries(test_plask ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} plask-shared)
        add_test(plask ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask)
        add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask)
    else()
        message("Boost Test Library not found. Building without C++ unit tests.")
    endif()

    if(BUILD_PYTHON_MODULE)
        add_test(plask-python "${PYTHON_EXECUTABLE}" -B -m unittest discover -v -t ${CMAKE_CURRENT_SOURCE_DIR}/tests/python -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
        if(NOT Boost_UNIT_TEST_FRAMEWORK_FOUND) # Add test target only for Python tests
            add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure)
        endif()
    endif(BUILD_PYTHON_MODULE)

endif(MAKE_TESTS)

# ----------===== documentation (api) =====--------------------------------------------
option(BUILD_DOCUMENTATION "build doxygen documentation" OFF)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_custom_target(doc ALL ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    else()
        message("Doxygen not found. Build without documentation.")
    endif()
endif()

# ----------===== install =====--------------------------------------------
if (BUILD_SHARED_LIBS)
    install(TARGETS plask-shared LIBRARY DESTINATION lib COMPONENT libplask)
endif()

install(TARGETS plask-static ARCHIVE DESTINATION lib COMPONENT libplask-dev)
install(FILES ${plask_headers} DESTINATION include/plask COMPONENT libplask-dev)
