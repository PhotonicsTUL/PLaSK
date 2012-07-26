# CMake file for project PLaSK

cmake_minimum_required(VERSION 2.8)
project (plask)

set(VERSION_MAJOR 0)            # MAJOR version (change to 1 on publication in Journal of Computer Physics)
set(VERSION_MINOR 0)            # MINOR version (change to 1 when anything works)
# set(VERSION_REVISION 1)       # REVISION number (leave undefined unless really needed, it's normally set to svn revision)

# ---------- common options --------------------------------------------

add_definitions(-Wall)         # all warnings
add_definitions(-std=c++0x)     # C++0x (C++11) standard

include_directories(.)          # solvers can include plask using <plask/plask.hpp>
include_directories(python)     # solvers can also include <plask/python.hpp>
include_directories(extlib)     # place for local external libraries
include_directories(${CMAKE_BINARY_DIR}/include) # for include config.h

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")


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

option(BUILD_SHARED_SOLVER_LIBS "Build shared libraries for solvers." OFF)

option(BUILD_GUI "Build GUI program (require QT)" OFF) # FIXME: install QT for windows users

option(BUILD_DOCUMENTATION "Build doxygen documentation." OFF)

option(BUILD_PYTHON "Build and install Python interface to PLaSK (recommended)." ON)
cmake_dependent_option(BUILD_PYTHON_MODULE "Build importable python module." OFF "BUILD_PYTHON" OFF)
cmake_dependent_option(BUILD_PYTHON_EXECUTABLE "Build executable with embedded Python interpreter." ON "BUILD_PYTHON" OFF)

option(BUILD_TESTING "Build unit tests." ON)
cmake_dependent_option(RUN_TESTS "Run unit tests." ON "BUILD_TESTING" OFF)

#option (PACK_EXE "Self-pack and strip executables." OFF)  # compress exe

set(PYTHON_VERSION "" CACHE STRING "Python version to use")

# ---------- External libraries --------------------------------------------
if(BUILD_PYTHON)
     if(NOT "${PYTHON_VERSION}" STREQUAL "${previous_user_python_version}")
        unset(PYTHON_EXECUTABLE CACHE)
        unset(PYTHON_LIBRARY CACHE)
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

# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
set(boost_components unit_test_framework)

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
#enable_language(Fortran OPTIONAL)
#if (CMAKE_Fortran_COMPILER_WORKS)
#   include(FortranCInterface)
#   FortranCInterface_HEADER("F77.h" MACRO_NAMESPACE "F77_")
#else ()
#   message(SEND_ERROR "Fortran compiler not found! Unable to automatically determine the name mangling scheme.")
#endif()


# ----------===== Build =====--------------------------------------------

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h plask/*.hpp)

file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp)

# default material database
file(GLOB_RECURSE materials_src FOLLOW_SYMLINKS materials/*.cpp materials/*.h)

# Python sources and files
file(GLOB_RECURSE python_plask_src FOLLOW_SYMLINKS python/plask/*.cpp)
file(GLOB_RECURSE python_plask_headers FOLLOW_SYMLINKS python/plask/*.h FOLLOW_SYMLINKS python/plask/*.hpp)
file(GLOB_RECURSE python_plask_py python/plask/*.py)

# GUI sources
file(GLOB_RECURSE plaskgui_src FOLLOW_SYMLINKS gui/*.cpp gui/*.h)

# external libs which are compiled by us:
#   irrXML
file(GLOB_RECURSE plask_extlib_irrxml_src FOLLOW_SYMLINKS extlib/irrxml/*.cpp)




# ---------- plask library --------------------------------------------

add_library(libplask SHARED ${plask_src} ${plask_extlib_irrxml_src} ${plask_headers})
if(WIN32)
    target_link_libraries(libplask kernel32)
else(WIN32)
    target_link_libraries(libplask dl)
endif(WIN32)
set_target_properties(libplask PROPERTIES OUTPUT_NAME plask)


# ---------- default materials --------------------------------------------

add_library(materials_default SHARED ${materials_src})
set_target_properties(materials_default PROPERTIES OUTPUT_NAME plask_materialsdefault)
target_link_libraries(materials_default libplask)


# ---------- Python --------------------------------------------

if(BUILD_PYTHON)
    set(plask_PYTHONPATH "${CMAKE_BINARY_DIR}/lib.python")

    # Determine directory to install *.py files to
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonInstallPath.py "from distutils import sysconfig\nprint(sysconfig.get_python_lib(False, prefix='${CMAKE_INSTALL_PREFIX}')+'/plask')")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonInstallPath.py" OUTPUT_VARIABLE PYTHON_INSTALL_DIR)
    string(STRIP ${PYTHON_INSTALL_DIR} PYTHON_INSTALL_DIR)
    # Determine directory to install binary Python modules to
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonModuleInstallPath.py "from distutils import sysconfig\nprint(sysconfig.get_python_lib(True, prefix='${CMAKE_INSTALL_PREFIX}')+'/plask')")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/DeterminePythonModuleInstallPath.py" OUTPUT_VARIABLE PYTHON_MODULE_INSTALL_DIR)
    string(STRIP ${PYTHON_MODULE_INSTALL_DIR} PYTHON_MODULE_INSTALL_DIR)

    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

    if (BUILD_PYTHON_MODULE)
        if(WIN32)
            add_library(plaskcore SHARED ${python_plask_src})
            set_target_properties(plaskcore PROPERTIES SUFFIX ".pyd" RUNTIME_OUTPUT_DIRECTORY "${plask_PYTHONPATH}/plask")
        else()
            add_library(plaskcore MODULE ${python_plask_src})
        endif()
        set_target_properties(plaskcore PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${plask_PYTHONPATH}/plask")
        target_link_libraries(plaskcore ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARIES} libplask)
    endif(BUILD_PYTHON_MODULE)

    if (BUILD_PYTHON_EXECUTABLE)
        add_executable(plask python/exe_main.cpp ${python_plask_src})
                if (WIN32)
                        # For some reason MinGW wants to create import library for executable,
                        # which overwrites libplask.dll.a. With this hack we name this useless
                        # stuff libplask.exe.dll.a
                        set_target_properties(plask PROPERTIES OUTPUT_NAME "plask.exe" SUFFIX "")
                endif()
        target_link_libraries(plask ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARIES} libplask)
    endif(BUILD_PYTHON_EXECUTABLE)

    foreach(py_file ${python_plask_py})
        get_filename_component(py_file_base ${py_file} NAME)
        list(APPEND python_plask_py_copied ${plask_PYTHONPATH}/plask/${py_file_base})
        add_custom_command(OUTPUT ${plask_PYTHONPATH}/plask/${py_file_base}
                           COMMAND ${CMAKE_COMMAND} ARGS -E copy ${py_file} ${plask_PYTHONPATH}/plask/${py_file_base}
                           DEPENDS ${py_file})
    endforeach()
    add_custom_target(python-scripts ALL DEPENDS ${python_plask_py_copied})

endif(BUILD_PYTHON)


# ---------- GUI program --------------------------------------------
if(BUILD_GUI)
    set(PLASK_GUI_MOC_HDRS gui/mainwindow.h gui/tree.h gui/utils/slots.h gui/view/elementview.h)
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
endif(BUILD_GUI)


# ---------- macro for testing --------------------------------------------
if(WIN32)
    set(ENV_PATH $ENV{PATH})
    string(REPLACE ";" "\;" ENV_PATH "${ENV_PATH}")
    string(REPLACE "/" "\\" plask_bin_path "${CMAKE_BINARY_DIR}/bin")
    set(plask_test_env "PYTHONPATH=${plask_PYTHONPATH};PATH=${ENV_PATH}\;${plask_bin_path}")
else()
    set(plask_test_env "PYTHONPATH=${plask_PYTHONPATH};LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib")
endif()

macro(add_python_test test test_file)
    if(BUILD_PYTHON_EXECUTABLE)
        add_test(NAME python-builtin/${test}
                 COMMAND plask ${CMAKE_SOURCE_DIR}/tests/python/main ${test_file})
        set_tests_properties(python-builtin/${test} PROPERTIES ENVIRONMENT "${plask_test_env}")
    endif()
    if(BUILD_PYTHON_MODULE)
        add_test(NAME python-module/${test}
                 COMMAND "${PYTHON_EXECUTABLE}" -B ${CMAKE_SOURCE_DIR}/tests/python/main ${test_file})
        set_tests_properties(python-module/${test} PROPERTIES ENVIRONMENT "${plask_test_env}")
    endif()
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

    set(plask_test_depends materials_default) # we need materials for testing

    enable_testing()
    if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        file(GLOB_RECURSE plask-test_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.h)
        add_executable(plask-test ${plask-test_src})
        set_target_properties(plask-test PROPERTIES OUTPUT_NAME test_plask)
        target_link_libraries(plask-test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} libplask materials_default)
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

        if(BUILD_PYTHON_EXECUTABLE)
            set(plask_test_depends ${plask_test_depends} plask)
        endif()
        if(BUILD_PYTHON_MODULE)
            set(plask_test_depends ${plask_test_depends} plaskcore)
        endif()

        if(WIN32)
            add_library(plask-test-python SHARED ${python_tests_module_src})
            set_target_properties(plask-test-python PROPERTIES SUFFIX ".pyd" RUNTIME_OUTPUT_DIRECTORY "${plask_PYTHONPATH}")
        else()
            add_library(plask-test-python MODULE ${python_tests_module_src})
        endif()
        set_target_properties(plask-test-python PROPERTIES OUTPUT_NAME plasktest PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${plask_PYTHONPATH}")
        if (BUILD_PYTHON_MODULE OR WIN32)
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
        file(COPY ${CMAKE_SOURCE_DIR}/doxygen.css DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        add_custom_target(doc ALL ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    else()
        message("Doxygen not found. Building without documentation.")
    endif()
else(BUILD_DOCUMENTATION)
    if(DOXYGEN_FOUND)
        add_custom_target(doc ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    endif()
endif(BUILD_DOCUMENTATION)

# ----------===== install =====--------------------------------------------
if(WIN32)
    install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core
                             ARCHIVE DESTINATION lib COMPONENT dev)
    install(TARGETS materials_default RUNTIME DESTINATION bin COMPONENT core
                                      ARCHIVE DESTINATION lib COMPONENT dev)
else()
    install(TARGETS libplask LIBRARY DESTINATION lib COMPONENT core)
    install(TARGETS materials_default LIBRARY DESTINATION lib COMPONENT core)
endif()

install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.h DESTINATION include/plask COMPONENT dev)

if(BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION share COMPONENT dev-doc
            PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
endif()

if(BUILD_PYTHON)
    install(FILES ${python_plask_py_copied} DESTINATION ${PYTHON_INSTALL_DIR} COMPONENT core)
    if(BUILD_PYTHON_EXECUTABLE)
        install(TARGETS plask RUNTIME DESTINATION bin COMPONENT core)
    endif()
    if(BUILD_PYTHON_MODULE)
        if(WIN32)
            install(TARGETS plaskcore RUNTIME DESTINATION ${PYTHON_MODULE_INSTALL_DIR} COMPONENT core)
        else()
            install(TARGETS plaskcore LIBRARY DESTINATION ${PYTHON_MODULE_INSTALL_DIR} COMPONENT core)
        endif()
    endif()
    install(DIRECTORY python/plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
endif()

if(BUILD_GUI)
    install(TARGETS plaskgui RUNTIME DESTINATION bin COMPONENT GUI)
endif()

# ----------===== Debug helpers =====--------------------------------------------
if (BUILD_PYTHON_EXECUTABLE)
    if(NOT WIN32)
        file(WRITE ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp/plask
            "PYTHONPATH=${plask_PYTHONPATH} ${CMAKE_BINARY_DIR}/bin/plask $@\n")
        file(COPY ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp/plask DESTINATION ${CMAKE_BINARY_DIR}
            FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE OWNER_WRITE)
    else()
        string(REPLACE "/" "\\" plask_exe "${CMAKE_BINARY_DIR}/bin/plask.exe")
        file(WRITE ${CMAKE_BINARY_DIR}/plask.bat
            "set PYTHONPATH=${plask_PYTHONPATH}\n${plask_exe} %*")
    endif()
endif()

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
