# CMake file for project PLaSK

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0017 OLD)	# ensure our CMake modules are loaded first

project(plask)

set(VERSION_MAJOR 0)            # MAJOR version (change to 1 on publication in Journal of Computer Physics)
set(VERSION_MINOR 1)            # MINOR version (bump up by one on visible feature change)
# set(VERSION_REVISION 1)       # REVISION number (leave undefined unless really needed, it's normally set to svn revision)

# ---------- common options --------------------------------------------

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-narrowing -Wno-sign-compare -Wno-unknown-pragmas")

# Turn-off strict aliasing for Python code
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GNUCXX_VERSION)
    if ((GNUCXX_VERSION VERSION_GREATER 4.7 OR GNUCXX_VERSION VERSION_EQUAL 4.7))
        set(no_strict_aliasing_flag "-fno-strict-aliasing")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(no_strict_aliasing_flag "-fno-strict-aliasing")
endif()

include_directories(.)          # solvers can include plask using <plask/plask.hpp>
include_directories(python)     # solvers can also include <plask/python.hpp>
include_directories(solvers)    # solvers can also include <providers/*.h>
include_directories(extlib)     # place for local external libraries
include_directories(${CMAKE_BINARY_DIR}/include) # for include config.h

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")
set(PLASK_PATH "${CMAKE_BINARY_DIR}/lib/plask")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (NOT DEFINED VERSION_REVISION)
    execute_process(COMMAND svnversion -n WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE SVN_REVISION)
    if(NOT ${SVN_REVISION} STREQUAL "")
        string(REGEX REPLACE "[0-9]+:|[A-Za-z]+" "" SVN_REVISION ${SVN_REVISION})
    endif()
    if(NOT ${SVN_REVISION} STREQUAL "")
        set(VERSION_REVISION "r${SVN_REVISION}")
    else()
        set(VERSION_REVISION 0)
    endif()
endif()
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}")
message(STATUS "PLaSK version ${VERSION}")

# ---------- Configurable options --------------------------------------------
include(CMakeDependentOption)

option(BUILD_GUI "Build GUI program (require QT)" OFF) # FIXME: install QT for windows users

option(BUILD_DOCUMENTATION "Build doxygen documentation." OFF)

option(BUILD_PYTHON "Build and install Python interface to PLaSK (recommended)." ON)

option(BUILD_TESTING "Build unit tests." ON)
cmake_dependent_option(RUN_TESTS "Run unit tests." ON "BUILD_TESTING" OFF)

option(USE_OMP "Use OpenMP" ON)

#option (PACK_EXE "Self-pack and strip executables." OFF)  # compress exe

set(PYTHON_VERSION "" CACHE STRING "Python version to use")

# ---------- External libraries --------------------------------------------
if(WIN32)
	get_filename_component(compliler_prefix_path ${CMAKE_CXX_COMPILER} PATH)
	get_filename_component(compliler_prefix_path ${compliler_prefix_path} PATH) # get rid of bin
	#TODO test how it should look like if we use MSVC
	list(APPEND CMAKE_SYSTEM_PREFIX_PATH ${compliler_prefix_path})
endif()

if(BUILD_PYTHON)
    set(PLASK_PYTHON_PATH "${PLASK_PATH}/python/plask")

    if(NOT "${PYTHON_VERSION}" STREQUAL "${previous_user_python_version}")
        unset(PYTHON_INCLUDE_DIR CACHE)
        unset(PYTHON_INCLUDE_PATH CACHE)
        set(previous_user_python_version ${PYTHON_VERSION} CACHE INTERNAL "Python version previously chosen by the user (to detect change)")
    endif()

    find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)
    find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" REQUIRED)
    find_package(numpy REQUIRED)

    # Require Python >= 2.6
    math(EXPR PYTHON_VERSION_NUM "100 * ${PYTHON_VERSION_MAJOR} + ${PYTHON_VERSION_MINOR}")
    if(${PYTHON_VERSION_NUM} LESS 206)
        message(FATAL_ERROR "Python 2.6 or newer required.")
    endif()
endif(BUILD_PYTHON)

if(USE_OMP)
    find_package(OpenMP 3.0)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

find_package(EXPAT REQUIRED)
include_directories(${EXPAT_INCLUDE_DIR})

# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
set(boost_components unit_test_framework system filesystem) #thread

if(BUILD_PYTHON)
    find_package(Boost QUIET COMPONENTS "python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
    unset(Boost_LIBRARIES)
    if(Boost_PYTHON-PY${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}_FOUND)
        list(APPEND boost_components "python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
        set(Boost_PYTHON_LIBRARIES ${Boost_PYTHON-PY${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}_LIBRARIES})
    else()
        list(APPEND boost_components "python")
    endif()
endif()

find_package(Boost REQUIRED COMPONENTS ${boost_components})
include_directories(${Boost_INCLUDE_DIRS})

if(MINGW)
    add_definitions(-Wno-attributes) # get rid of annoying warnings in Boost
endif()

## ---------- Fortran name mangling --------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/CheckFortran.cmake)
if (CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
	include(FortranCInterface)
	FortranCInterface_HEADER("${CMAKE_BINARY_DIR}/include/plask/f77.h" MACRO_NAMESPACE "F77_")
else ()
	message("Unable to determine Fortran name mangling scheme! Using the most probable version.")
endif()


# ----------===== Build and Install =====--------------------------------------------

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h plask/*.hpp solvers/providers/*.h solvers/providers/*.hpp)

file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp)

# Python sources and files
file(GLOB_RECURSE python_plask_src FOLLOW_SYMLINKS python/plask/*.cpp)
file(GLOB_RECURSE python_plask_py python/plask/*.py)

# GUI sources
file(GLOB_RECURSE plaskgui_src FOLLOW_SYMLINKS gui/*.cpp gui/*.h)


# ---------- external libraries built by us -----------------------------------------

# CAMOS
add_subdirectory(extlib/camos)

# irrXML
file(GLOB_RECURSE plask_extlib_irrxml_src FOLLOW_SYMLINKS extlib/irrxml/*.cpp)


# ---------- plask library --------------------------------------------

#add_library(libplask SHARED ${plask_src} ${plask_extlib_irrxml_src} ${plask_headers})
add_library(libplask SHARED ${plask_src} ${plask_extlib_irrxml_src})
target_link_libraries(libplask ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${EXPAT_LIBRARIES}) # ${Boost_THREAD_LIBRARY}
if(WIN32)
    target_link_libraries(libplask kernel32)
    install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core
                             ARCHIVE DESTINATION lib COMPONENT dev)
else(WIN32)
    target_link_libraries(libplask dl)
    install(TARGETS libplask LIBRARY DESTINATION lib COMPONENT core)
endif(WIN32)
set_target_properties(libplask PROPERTIES OUTPUT_NAME plask)

#if (NOT WIN32) #can be usefull if more threads libraries will be used
#    target_link_libraries(sdsolver "pthread")
#endif()

install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
install(DIRECTORY solvers/providers DESTINATION include/plask COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.h DESTINATION include/plask COMPONENT dev)

set(PLaSK_LIBRARIES libplask)

# ---------- default materials --------------------------------------------

add_subdirectory(materials)

# ---------- Python --------------------------------------------

if(BUILD_PYTHON)

    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

    add_executable(plask python/exe_main.cpp ${python_plask_src})
    if (WIN32)
            # For some reason MinGW wants to create import library for executable,
            # which overwrites libplask.dll.a. With this hack we name this useless
            # stuff libplask.exe.dll.a
            set_target_properties(plask PROPERTIES OUTPUT_NAME "plask.exe" SUFFIX "")
    endif()
    target_link_libraries(plask ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARIES} libplask)

    if (DEFINED no_strict_aliasing_flag)
        set_target_properties(plask PROPERTIES COMPILE_FLAGS ${no_strict_aliasing_flag}) # necessary for all code which includes "Python.h"
    endif()

    foreach(py_file ${python_plask_py})
        get_filename_component(py_file_base ${py_file} NAME)
        list(APPEND python_plask_py_copied ${PLASK_PYTHON_PATH}/${py_file_base})
        add_custom_command(OUTPUT ${PLASK_PYTHON_PATH}/${py_file_base}
                           COMMAND ${CMAKE_COMMAND} ARGS -E copy ${py_file} ${PLASK_PYTHON_PATH}/${py_file_base}
                           DEPENDS ${py_file})
    endforeach()
    add_custom_target(python-scripts ALL DEPENDS ${python_plask_py_copied})

    install(FILES ${python_plask_py_copied} DESTINATION lib/plask/python/plask COMPONENT core)
    install(TARGETS plask RUNTIME DESTINATION bin COMPONENT core)
    install(DIRECTORY python/plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "python*.h" PATTERN "python*.hpp" PATTERN ".*" EXCLUDE)

endif(BUILD_PYTHON)


# ---------- GUI program --------------------------------------------
if(BUILD_GUI)
    set(PLASK_GUI_MOC_HDRS gui/mainwindow.h gui/tree.h gui/utils/slots.h gui/view/objectview.h)
    #set(PLASK_GUI_UIS ./src/ui/Dialog1.ui)
    set(PLASK_GUI_RCS gui/plaskgui.qrc)

    find_package(Qt4 4.7 REQUIRED QtCore QtGui)
    include(${QT_USE_FILE})

    # this directory contains prop. brow. src + our CMakeLists.txt based on https://github.com/commontk/QtPropertyBrowser/blob/master/src/CMakeLists.txt
    add_subdirectory(extlib/qtpropertybrowser)  #this build qtpropertybrowser

    include_directories(extlib/qtpropertybrowser)   # allow for finding qtpropertybrowser headers

    qt4_add_resources(PLASK_GUI_RC_SRCS ${PLASK_GUI_RCS})
    #qt4_wrap_ui(PLASK_GUI_UI_HDRS ${PLASK_GUI_UIS})
    qt4_wrap_cpp(PLASK_GUI_MOC_SRCS ${PLASK_GUI_MOC_HDRS})
    add_executable(plaskgui ${plaskgui_src} ${property_browser_src} ${PLASK_GUI_MOC_SRCS} ${PLASK_GUI_RC_SRCS} ) # ${PLASK_GUI_UI_HDRS}
    target_link_libraries(plaskgui libplask ${QT_LIBRARIES} qtpropertybrowser)

    install(TARGETS plaskgui RUNTIME DESTINATION bin COMPONENT GUI)

endif(BUILD_GUI)


# ---------- macro for testing --------------------------------------------
if(WIN32)
    set(ENV_PATH $ENV{PATH})
    string(REPLACE ";" "\;" ENV_PATH "${ENV_PATH}")
    string(REPLACE "/" "\\" plask_bin_path "${CMAKE_BINARY_DIR}/bin")
endif()

macro(add_python_test test test_file)
    add_test(NAME python/${test}
             COMMAND plask ${CMAKE_SOURCE_DIR}/tests/python/main ${test_file})
    set_tests_properties(python/${test} PROPERTIES ENVIRONMENT "${plask_test_env}")
endmacro()

# ----------===== Solvers =====--------------------------------------------

macro(solv_option solv opt)
    string(REPLACE "/" "_" ${opt} ${solv})
    string(TOUPPER "BUILD_SOLVER_${${opt}}" ${opt})
endmacro()

# Automagically get list of available solvers and set options to turn on/off building particular solvers
file(GLOB_RECURSE solver_lists RELATIVE  ${CMAKE_SOURCE_DIR}/solvers solvers/*CMakeLists.txt)
list(REMOVE_ITEM solver_lists solvers/skel/CMakeLists.txt)
message(STATUS "Will build the following solvers (in case of error disable the offending solver and try again):")
foreach(solv ${solver_lists})
    string(REPLACE "/CMakeLists.txt" "" solv ${solv})
    if (NOT ${solv} STREQUAL skel)
        list(APPEND solvers ${solv})
        solv_option(${solv} opt)
        option(${opt} "Build solver '${solv}'" ON)
        if (${opt})
            message(STATUS "  ${solv}: YES")
        else()
            message(STATUS "  ${solv}: NO")
        endif()
    endif()
endforeach()

# Build solvers
set(solvers_test_targets "")
foreach(solv ${solvers})
    solv_option(${solv} opt)
    option(${opt} "Build solver '${solv}'" ON)
    if (${opt})
        if(BUILD_TESTING)
            string(REGEX REPLACE "/|_" "" solvtarget ${solv})
            list(APPEND solvers_test_targets "solver-${solvtarget}-test")
        endif()
        add_subdirectory(solvers/${solv})
    endif()
endforeach()

# ----------===== unit tests =====--------------------------------------------
if(BUILD_TESTING)

    set(plask_test_depends ${material_targets}) # we need materials for testing

    enable_testing()
    if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        file(GLOB_RECURSE plask-test_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.h)
        add_executable(plask-test ${plask-test_src})
        set_target_properties(plask-test PROPERTIES OUTPUT_NAME test_plask)
        target_link_libraries(plask-test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} libplask)
        set(plask_test_depends ${plask_test_depends} plask-test)
        file(GLOB_RECURSE plask_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/tests/plask FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.cxx)
        list(REMOVE_ITEM plask_tests main.cpp)
        foreach(test_file ${plask_tests})
            get_filename_component(test ${test_file} NAME_WE)
            add_test(plask/${test} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask -t ${test})
        endforeach()
    else(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        message("Boost Test Library not found. Building without C++ unit tests.")
    endif(Boost_UNIT_TEST_FRAMEWORK_FOUND)

    if(BUILD_PYTHON)
        file(GLOB_RECURSE python_tests_module_src FOLLOW_SYMLINKS tests/python/*.cpp)

        set(plask_test_depends ${plask_test_depends} plask)

        if(WIN32)
            add_library(plask-test-python SHARED ${python_tests_module_src})
            set_target_properties(plask-test-python PROPERTIES SUFFIX ".pyd" RUNTIME_OUTPUT_DIRECTORY "${PLASK_PATH}/python")
        else()
            add_library(plask-test-python MODULE ${python_tests_module_src})
            set_target_properties(plask-test-python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PLASK_PATH}/python")
        endif()
        set_target_properties(plask-test-python PROPERTIES OUTPUT_NAME plasktest PREFIX "")
        if (WIN32)
            target_link_libraries(plask-test-python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask)
        else()
            target_link_libraries(plask-test-python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
        endif()
        set(plask_test_depends ${plask_test_depends} plask-test-python)

        file(GLOB_RECURSE python_tests FOLLOW_SYMLINKS tests/python/*.py)
        foreach(test_file ${python_tests})
            get_filename_component(test ${test_file} NAME_WE)
            add_python_test(${test} ${test_file})
        endforeach()
    endif(BUILD_PYTHON)

    if(BUILD_GUI)
        set(plask_test_depends ${plask_test_depends} plaskgui qtpropertybrowser)
    endif()

    if (RUN_TESTS)
        add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure DEPENDS ${plask_test_depends} ${solvers_test_targets})
    else()
        add_custom_target(tests ${CMAKE_CTEST_COMMAND} --output-on-failure DEPENDS ${plask_test_depends} ${solvers_test_targets})
    endif()

endif(BUILD_TESTING)

# ----------===== documentation (api) =====--------------------------------------------
find_package(Doxygen)

if(BUILD_DOCUMENTATION)
    if(DOXYGEN_FOUND)
        file(COPY ${CMAKE_SOURCE_DIR}/doxygen.css DESTINATION ${CMAKE_BINARY_DIR})
        add_custom_target(doc ALL ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
        install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION share COMPONENT dev-doc
                PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
    else()
        message("Doxygen not found. Building without documentation.")
    endif()
else(BUILD_DOCUMENTATION)
    if(DOXYGEN_FOUND)
        add_custom_target(doc ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    endif()
endif(BUILD_DOCUMENTATION)


# ----------===== packaging =====--------------------------------------------

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Photonic LAser Simulation Kit")
set(CPACK_PACKAGE_VENDOR "Photonics Group, Technical University of Lodz")
#set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_SOURCE_DIR}/ABOUT)

set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_REVISION})

set(CPACK_GENERATOR "DEB;NSIS")
set(CPACK_PACKAGE_FILE_NAME "plask" CACHE STRING "Package basename")
mark_as_advanced(CPACK_PACKAGE_FILE_NAME)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Maciej Dems <maciej.dems@p.lodz.pl>")
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_SOURCE_PACKAGE_FILE_NAME "plask-${VERSION}" CACHE INTERNAL "tarball basename")
set(CPACK_SOURCE_GENERATOR "TBZ2")

set(CPACK_SOURCE_IGNORE_FILES
"~$"
"^${PROJECT_SOURCE_DIR}.*\\\\.kdev4"
"^${PROJECT_SOURCE_DIR}.*/\\\\..*"
"^${PROJECT_SOURCE_DIR}/debian/"
"^${PROJECT_SOURCE_DIR}/build*/"
"^${PROJECT_SOURCE_DIR}/debug/"
"^${PROJECT_SOURCE_DIR}/release/"
)

include(CPack)

# ----------===== config.h & version.h =====--------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/include/plask/config.h)
configure_file(${CMAKE_SOURCE_DIR}/version.h.in ${CMAKE_BINARY_DIR}/include/plask/version.h)
