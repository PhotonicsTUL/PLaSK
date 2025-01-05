# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

cmake_minimum_required(VERSION 3.14)

# Silence unnesessary warning
cmake_policy(SET CMP0054 NEW)

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

project(plask)

# ---------- Common options --------------------------------------------

#set (CMAKE_CXX_STANDARD 17)    #TODO std should be set this way

option(SHOW_EXTENDED_WARNINGS "Show some additional warnings." ON)

if(MSVC)    # flags needed by Visual Studio:
    set(CMAKE_CXX_FLAGS "/bigobj ${CMAKE_CXX_FLAGS}")   # increase a number of sections in obj. file, required by plask
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")   # disable many warning (all stl & boost templates) about wrong dll import/export, TODO subject for future investigation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4018")   # disable warning about signed/unsigned generated mainly by workaround for OpenMP 2 (which can use only signed types)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")     # follow C++ standard for exception handling
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DHAVE_SNPRINTF")   # snprintf is not defined in MSVC
    add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS) # hide many warnings which are caused by boost and fmt under MSVC, https://msdn.microsoft.com/query/dev14.query?appId=Dev14IDEF1&l=EN-US&k=k(C4996)&rd=true
    add_definitions(-DBOOST_ALL_NO_LIB) # disable automatic linking with boost libraries, we link them manually
    #add_definitions(-DFMT_HEADER_ONLY)      # don't compile fmt, workaround for many linker errors
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")   # nothing is needed for C++11
else()
    set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
    if(SHOW_EXTENDED_WARNINGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion") # -Wconversion is enabled since it gives many same warnings as MSVC compiler
        if(CMAKE_COMPILER_IS_GNUCXX)
            # int-in-bool-context was turned off because of eigen, which shows them massively
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
        endif()
    endif()
endif()

if(WIN32 AND "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(WIN64 YES) # CMake cannot automatically detect that we compile for WIN64
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_WIN64")
endif()

# Turn-off strict aliasing for Python code and set some other specific compiler options
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GNUCXX_VERSION)
    if(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -Wno-unused-local-typedefs")
    else()
        if ((GNUCXX_VERSION VERSION_GREATER 4.7 OR GNUCXX_VERSION VERSION_EQUAL 4.7))
            set(no_strict_aliasing_flag "-fno-strict-aliasing")
        endif()
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(no_strict_aliasing_flag "-fno-strict-aliasing")
endif()

include_directories(.)          # solvers can include plask using <plask/plask.hpp>
include_directories(python)     # solvers can also include <plask/python.hpp>
include_directories(extlib)     # place for local external libraries
include_directories(${CMAKE_BINARY_DIR}/include) # for include plask/config.hpp

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")
foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # used by MSVC
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} ${CMAKE_CFG_INTDIR}/bin)
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} ${CMAKE_CFG_INTDIR}/lib)
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} ${CMAKE_CFG_INTDIR}/lib)
endforeach()
set(PLASK_PATH "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask")

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

if("$ENV{PLASK_VERSION}" STREQUAL "")
    execute_process(COMMAND git show -s --pretty=format:%ad.%h --date=short WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    RESULT_VARIABLE git_version_ok OUTPUT_VARIABLE PLASK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)  # old git does not have --date=format
    if("${git_version_ok}" STREQUAL "0")
        string(REPLACE "-" "." PLASK_VERSION "${PLASK_VERSION}")
    else()
        string(TIMESTAMP PLASK_VERSION "%Y.%m.%d")
    endif()
else()
    execute_process(COMMAND git show -s --pretty=format:%h WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    RESULT_VARIABLE git_version_ok OUTPUT_VARIABLE plask_version_suffix OUTPUT_STRIP_TRAILING_WHITESPACE)
    if("${git_version_ok}" STREQUAL "0")
        set(PLASK_VERSION "$ENV{PLASK_VERSION}.${plask_version_suffix}")
    endif()
endif()
string(TIMESTAMP CURRENT_YEAR "%Y")
message(STATUS "PLaSK version ${PLASK_VERSION}")

# ---------- Configurable options --------------------------------------------

include(CMakeDependentOption)

option(BUILD_DEVEL_DOC "Build doxygen documentation." OFF)

option(BUILD_PYTHON "Build and install Python interface to PLaSK (recommended)." ON)
cmake_dependent_option(BUILD_PYTHON_MODULE "Build Python interface as external module." OFF "BUILD_PYTHON" OFF)
cmake_dependent_option(INSTALL_PYTHON_PTH "Install 'plask.pth' into Python directory" OFF "BUILD_PYTHON_MODULE" OFF)
cmake_dependent_option(BUILD_GUI "Build and install GUI." ON "BUILD_PYTHON" OFF)

option(BUILD_TESTING "Build unit tests." ON)
cmake_dependent_option(BUILD_GUI_TESTING "Build unit tests for GUI." ON "BUILD_TESTING;BUILD_GUI" OFF)

option(USE_OMP "Use OpenMP" ON)

set(USE_PROFILER "" CACHE STRING "Type of the profiler to use. Leave empty for no profiling. Set to 'GNU' if you want to use gprof or to 'Google' for google perftools.")

if(WIN32 AND NOT MSVC)
    set(USE_MANIFEST "msvcr90" CACHE STRING "Manifest to include into Windows executables")
else()
    cmake_dependent_option(USE_SPLASH "Show splash screen using Xlib." ON "BUILD_GUI" OFF)
endif()

option(PRINT_STACKTRACE_ON_EXCEPTION "Print stack-trace on stderr when plask::Exception is thrown (works only in debug mode)." ON)

option(PLASK_OPTIONAL_STD "Use std optional (supported by C++17) instead of boost one." OFF)

#TODO if disabled, plask should find and use external fmt lib.
option(PLASK_EXPORTS_FMT "Export fmt symbols in libplask (affects only windows)" ON)

option(INSTALL_DEV "Install headers and libraries to system locations" OFF)

option(XML_STATIC "If enabled, XML_STATIC symbol will be defined, which is required to use static expat build in Windows." OFF)
if (XML_STATIC)
    add_definitions(-DXML_STATIC)
endif (XML_STATIC)


if(CMAKE_COMPILER_IS_GNUCXX)
    option(USE_GLIBCXX_PARALLEL "use the libstdc++ parallel mode (experimental in libstdc++v3)" OFF)
endif(CMAKE_COMPILER_IS_GNUCXX)

#option (PACK_EXE "Self-pack and strip executables." OFF)  # compress exe

set(PYTHON_VERSION "" CACHE STRING "Python version to use")

set(ECM_ENABLE_SANITIZERS "" CACHE STRING "semicolon-separated list of sanitizers to use: leak (requires address), undefined, (and up to one of:) address, memory, thread")
include(ECMEnableSanitizers)    # sanitizers, this module is also included in debian package: extra-cmake-modules

# ---------- Profiler --------------------------------------------
string(TOLOWER "${USE_PROFILER}" USE_PROFILER)
if(USE_PROFILER STREQUAL "gnu")
    set(CMAKE_C_FLAGS "-pg ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-pg ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "-pg ${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "-pg ${CMAKE_MODULE_LINKER_FLAGS}")
    message(STATUS "Using GNU profiler")
elseif(USE_PROFILER STREQUAL "google")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lprofiler")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lprofiler")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -lprofiler")
    message(STATUS "Using Google perftools profiler")
endif()

# ---------- External libraries --------------------------------------------
if(WIN32)
    get_filename_component(compiler_prefix_path ${CMAKE_CXX_COMPILER} PATH)
    get_filename_component(compiler_prefix_path ${compiler_prefix_path} PATH) # get rid of bin
    #TODO test how it should look like if we use MSVC
    list(APPEND CMAKE_SYSTEM_PREFIX_PATH ${compiler_prefix_path})
endif()

if(BUILD_PYTHON)
    set(PLASK_PYTHON_PATH "${PLASK_PATH}/python/plask")

    if(NOT PYTHON_VERSION)
        find_package(Python3 "3.8" COMPONENTS Interpreter Development NumPy REQUIRED)
    else()
        find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development NumPy REQUIRED)
    endif()
endif(BUILD_PYTHON)

find_package(EXPAT REQUIRED)
include_directories(${EXPAT_INCLUDE_DIR})

# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever

set(BOOST_MIN_VER 1.54.0)

if(WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost ${BOOST_MIN_VER} REQUIRED COMPONENTS filesystem)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if(WIN32)
    set(Boost_USE_STATIC_LIBS OFF)
endif()

if(BUILD_PYTHON)
    foreach(boost_python "python-py${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}"
                         "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}"
                         "python${Python3_VERSION_MAJOR}")
        find_package(Boost ${BOOST_MIN_VER} QUIET COMPONENTS "${boost_python}")
        unset(Boost_LIBRARIES)
        string(TOUPPER "${boost_python}" boost_python_name)
        if(Boost_${boost_python_name}_FOUND)
            set(Boost_Python3 Boost::${boost_python})
            string(REPLACE "-" "_" boost_python_define ${boost_python_name})
            add_definitions(-DBOOST_${boost_python_define}_DYN_LINK)
            break()
        endif()
    endforeach()
    if(NOT Boost_Python3)
        message(FATAL_ERROR "Boost Python not found")
    else()
        message(STATUS "Found ${Boost_Python3}")
    endif()
endif()

if(BUILD_TESTING)
    find_package(Boost ${BOOST_MIN_VER} REQUIRED COMPONENTS unit_test_framework)
    add_definitions(-DBOOST_UNIT_TEST_FRAMEWORK_DYN_LINK)
endif(BUILD_TESTING)

link_directories(${Boost_LIBRARY_DIRS})

# ---------- External libraries built by us -----------------------------------------

# libf2c for Windows
if(WIN32)
    add_subdirectory(extlib/libf2c)
endif()

# NSPCG
add_subdirectory(extlib/nspcg)

## ---------- OpenMP -----------------------------------------------------------

if(USE_OMP OR USE_GLIBCXX_PARALLEL) # compile with OpenMP
    find_package(OpenMP 3.0)
    if(OPENMP_FOUND)
        if(USE_GLIBCXX_PARALLEL)
            set(OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -march=native -D_GLIBCXX_PARALLEL")
            set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -march=native -D_GLIBCXX_PARALLEL")
        endif(USE_GLIBCXX_PARALLEL)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS_NO_OPENMP "${CMAKE_EXE_LINKER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS_NO_OPENMP "${CMAKE_SHARED_LINKER_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS_NO_OPENMP "${CMAKE_MODULE_LINKER_FLAGS}")
        if(NOT MSVC)    # TODO: czy na pewno trzeba podac flagi linkerowi, dokumentacja cmake nic o tym nie mowi
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
endif()

## ---------- Fortran name mangling --------------------------------------------
option(DETECT_FORTRAN_NAME_MANGLING "Detect Fortran name mangling scheme. When OFF, the most probable version will be used." OFF)
if(DETECT_FORTRAN_NAME_MANGLING)
    include(${CMAKE_SOURCE_DIR}/cmake/CheckFortran.cmake)
    if (CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
        include(FortranCInterface)
        FortranCInterface_HEADER("${CMAKE_BINARY_DIR}/include/plask/f77.h" MACRO_NAMESPACE "F77_")
    else ()
        message("Unable to determine Fortran name mangling scheme! Using the most probable version.")
    endif()
endif(DETECT_FORTRAN_NAME_MANGLING)

## ---------- Libraries for better stack printing ------------------------------

if(PRINT_STACKTRACE_ON_EXCEPTION)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        if(WIN32)
            add_definitions(-D_GLIBCXX_DEBUG)   # safe STL under linux, see https://gcc.gnu.org/onlinedocs/libstdc++/manual/debug_mode.html
            set(PRINT_STACKTRACE_LIBRARY "dbghelp")
        else()    # non-Windows:
            find_package(DW)
            if(DW_FOUND)
                set(PRINT_STACKTRACE_LIBRARY ${DW_LIBRARIES})
                include_directories(${DW_INCLUDE_DIRS})
                set(BACKWARD_HAS_DW 1)
            else()  # try find bfd library if dw is not found, see https://github.com/bombela/backward-cpp/blob/master/backward.cpp
                find_package(BFD)
                if(BFD_FOUND)
                    message(WARNING "PLaSK will be linked with GPLed libbfd.")  # https://sourceware.org/ml/binutils/2002-01/msg00581.html
                    set(PRINT_STACKTRACE_LIBRARY ${BFD_LIBRARY})
                    set(BACKWARD_HAS_BFD 1)
                else()
                    set(PRINT_STACKTRACE_LIBRARY "")
                endif()
            endif()
        endif()
    endif()
endif()

# ---------- Other external libraries -----------------------------------------

find_package(BLAS)
find_package(LAPACK)

# ----------===== Build and Install =====--------------------------------------------

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h plask/*.hpp solvers/providers/*.h solvers/providers/*.hpp)

file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp solvers/providers/*.cpp)

# Python sources
file(GLOB_RECURSE python_plask_src FOLLOW_SYMLINKS python/plask/*.cpp python/plask/*.hpp)

# ---------- Other external libraries build by us -----------------------------

# CAMOS
add_subdirectory(extlib/camos)

# FFTPACX
add_subdirectory(extlib/fftpacx)

# Triangle
add_subdirectory(extlib/triangle)

# fmt
if(PLASK_EXPORTS_FMT)
    list(APPEND plask_src extlib/fmt/format.cc extlib/fmt/posix.cc)
endif()

# ---------- PLaSK library --------------------------------------------

add_library(libplask SHARED ${plask_src} ${plask_headers})
target_link_libraries(libplask Boost::filesystem ${EXPAT_LIBRARIES} ${PRINT_STACKTRACE_LIBRARY} triangle) # ${Boost_THREAD_LIBRARY}
if(WIN32)
    target_link_libraries(libplask kernel32 iphlpapi)
    if(INSTALL_DEV)
        install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core
                                 ARCHIVE DESTINATION lib COMPONENT dev)
    else()
        install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core)
    endif()
    if(MSVC)    # default "plask.pdb" is overridden by file for plask.exe, so we use different name ("libplask.pdb")
        set_target_properties (libplask PROPERTIES PDB_NAME libplask)
    endif(MSVC)
else()
    target_link_libraries(libplask dl)
    install(TARGETS libplask LIBRARY DESTINATION lib COMPONENT core)
endif()
set_target_properties(libplask PROPERTIES OUTPUT_NAME plask COMPILE_FLAGS "-DPLASK_EXPORTS")
if(PLASK_EXPORTS_FMT)
    target_compile_definitions(libplask PRIVATE FMT_EXPORT INTERFACE FMT_SHARED)
endif()

#if (NOT WIN32) # can be useful if more thread libraries will be used
#    target_link_libraries(sdsolver "pthread")
#endif()

if(INSTALL_DEV)
    install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
    install(DIRECTORY solvers/providers DESTINATION include/plask COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
    install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.hpp DESTINATION include/plask COMPONENT dev)
    install(FILES cmake/plask-config.cmake DESTINATION share/plask/cmake COMPONENT dev)
endif()

if(INSTALL_DEV)
    install(FILES extlib/fmt/format.h extlib/fmt/core.h extlib/fmt/ostream.h DESTINATION include/fmt COMPONENT dev)
endif()


set(PLaSK_LIBRARIES libplask)
set(PLaSK_Python3_LIBRARIES libplask_python)

# ---------- Default materials --------------------------------------------

add_subdirectory(materials)

# ---------- Python --------------------------------------------

if(BUILD_PYTHON)

    # Python files
    foreach(py_dir "" "/util")
        set(python_plask_py_copied "")

        file(GLOB python_plask_py "python/plask${py_dir}/*.py")
        if("${py_dir}" STREQUAL "")
            list(REMOVE_ITEM python_plask_py "${CMAKE_SOURCE_DIR}/python/plask/_plask.py")
        else()
            file(MAKE_DIRECTORY ${PLASK_PYTHON_PATH}${py_dir})
        endif()

        foreach(py_file ${python_plask_py})
            get_filename_component(py_file_name ${py_file} NAME)
            list(APPEND python_plask_py_copied ${PLASK_PYTHON_PATH}${py_dir}/${py_file_name})
            list(APPEND python_plask_py_copied_all ${PLASK_PYTHON_PATH}${py_dir}/${py_file_name})
            add_custom_command(OUTPUT ${PLASK_PYTHON_PATH}${py_dir}/${py_file_name}
                               COMMAND ${CMAKE_COMMAND} ARGS -E copy ${py_file} ${PLASK_PYTHON_PATH}${py_dir}/${py_file_name}
                               DEPENDS ${py_file})
            if(WIN32)
                get_filename_component(py_file_base ${py_file} NAME_WE)
                list(APPEND python_plask_pyc ${PLASK_PYTHON_PATH}${py_dir}/__pycache__/${py_file_base}.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.pyc)
                list(APPEND python_plask_pyo ${PLASK_PYTHON_PATH}${py_dir}/__pycache__/${py_file_base}.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.opt-1.pyc)
            endif()
        endforeach()

        install(FILES ${python_plask_py_copied} DESTINATION lib/plask/python/plask${py_dir} COMPONENT core)
    endforeach()

    add_custom_target(python-scripts ALL DEPENDS ${python_plask_py_copied_all})

    if(WIN32)
        add_custom_command(TARGET python-scripts COMMAND ${Python3_EXECUTABLE} -m compileall -q ${PLASK_PYTHON_PATH})
        add_custom_command(TARGET python-scripts COMMAND ${Python3_EXECUTABLE} -O -m compileall -q ${PLASK_PYTHON_PATH})
        foreach(py_dir "" "/util")
            install(FILES ${python_plask_pyc} DESTINATION lib/plask/python/plask${py_dir}/__pycache__ COMPONENT core)
            install(FILES ${python_plask_pyo} DESTINATION lib/plask/python/plask${py_dir}/__pycache__ COMPONENT core)
        endforeach()
    endif()

    include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

    add_library(libplask_python SHARED ${python_plask_src})
    target_link_libraries(libplask_python ${Python3_LIBRARIES} ${Boost_Python3} libplask)
    set_target_properties(libplask_python PROPERTIES OUTPUT_NAME plask_python)
    if(DEFINED no_strict_aliasing_flag)
        set_target_properties(libplask_python PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flag} -DPLASK_PYTHON_EXPORTS")
    else()
        set_target_properties(libplask_python PROPERTIES COMPILE_FLAGS "-DPLASK_PYTHON_EXPORTS")
    endif()

    if(WIN32)
        if(INSTALL_DEV)
            install(TARGETS libplask_python RUNTIME DESTINATION bin COMPONENT core
                                            ARCHIVE DESTINATION lib COMPONENT dev)
        else()
            install(TARGETS libplask_python RUNTIME DESTINATION bin COMPONENT core)
        endif()
    else()
        install(TARGETS libplask_python LIBRARY DESTINATION lib COMPONENT core)
    endif()

    if(BUILD_PYTHON_MODULE)
        if(WIN32)
            add_library(plask_module SHARED ${python_plask_src})
        else()
            add_library(plask_module MODULE ${python_plask_src})
        endif()
        target_link_libraries(plask_module ${Python3_LIBRARIES} ${Boost_Python3} libplask)
        set_target_properties(plask_module PROPERTIES OUTPUT_NAME plask_python)
        if(DEFINED no_strict_aliasing_flag)
            set_target_properties(plask_module PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flag} -DPLASK_PYTHON_EXPORTS")
        else()
            set_target_properties(plask_module PROPERTIES COMPILE_FLAGS "-DPLASK_PYTHON_EXPORTS")
        endif()
        set_target_properties(plask_module PROPERTIES
                              LIBRARY_OUTPUT_DIRECTORY "${PLASK_PYTHON_PATH}"
                              OUTPUT_NAME "_plask"
                              INSTALL_RPATH "$ORIGIN"
                              PREFIX "")
        if(WIN32)
            set_target_properties(plask_module PROPERTIES
                                RUNTIME_OUTPUT_DIRECTORY "${PLASK_PYTHON_PATH}"
                                SUFFIX ".pyd")
        else()
            if(INSTALL_PYTHON_PTH)
                execute_process(
                    COMMAND "${Python3_EXECUTABLE}" -c
                    "from distutils.sysconfig import get_python_lib; print(get_python_lib(plat_specific=False,standard_lib=False))"
                    OUTPUT_VARIABLE PYTHON_SITE OUTPUT_STRIP_TRAILING_WHITESPACE)
                file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/plask.pth" "${CMAKE_INSTALL_PREFIX}/lib/plask/python\n")
                install(FILES "${CMAKE_BINARY_DIR}/CMakeFiles/plask.pth" DESTINATION "${PYTHON_SITE}" COMPONENT core)
            endif()
        endif()
        install(TARGETS plask_module DESTINATION lib/plask/python/plask COMPONENT core)

    endif()

    if(WIN32)
        add_library(plask SHARED python/exe_main.cpp)
        set_target_properties(plask PROPERTIES OUTPUT_NAME "plaskexe")
        set(plask_binary plask plask_exe)
    else()
        add_executable(plask python/exe_main.cpp)
        set(plask_binary plask)
    endif()
    target_link_libraries(plask ${Boost_Python3} ${Python3_LIBRARIES} libplask_python libplask)
    add_dependencies(plask python-scripts)

    if(DEFINED no_strict_aliasing_flag OR DEFINED EXE_FLAGS)
        set_target_properties(plask PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flag} ${EXE_FLAGS}") # necessary for all code which includes "Python.h"
    endif()
    if(DEFINED EXE_FLAGS)
        set_target_properties(plask PROPERTIES LINK_FLAGS "${EXE_FLAGS}")
    endif()

    set(plask_stubs _plask geometry mesh _material flow)
    foreach(stub ${plask_stubs})
        list(APPEND plask_stubs_files ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/python/plask/${stub}.pyi)
        list(APPEND plask_stubs_modules plask.${stub})
    endforeach()
    file(REMOVE ${plask_stubs_files}) # wrong stubs can make build fail
    if(BUILD_GUI)
        add_custom_command(OUTPUT ${plask_stubs_files}
                           COMMAND ${CMAKE_COMMAND} -E env PLASK_DEFAULT_MATERIALS= ${CMAKE_BINARY_DIR}/bin/plask -lwarning ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${plask_stubs_modules}
                           DEPENDS ${plask_binary} ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${PLASK_MATERIALS}
                           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/python
                          )
        install(FILES ${plask_stubs_files} DESTINATION lib/plask/python/plask COMPONENT gui)
    endif()

    install(TARGETS plask RUNTIME DESTINATION bin COMPONENT core)
    if (INSTALL_DEV)
        install(DIRECTORY python/plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "python*.h" PATTERN "python*.hpp" PATTERN ".*" EXCLUDE)
        install(DIRECTORY python/plask/python_util DESTINATION include/plask COMPONENT dev)
    endif()

    if(WIN32)
        add_executable(plask_exe python/win_main.cpp)
        set_target_properties(plask_exe PROPERTIES OUTPUT_NAME "plask")
        install(TARGETS plask_exe RUNTIME DESTINATION bin COMPONENT core)
    endif()

    # GUI
    if(BUILD_GUI)
        if(WIN32)
            foreach(bs 620 868 1116)
                add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash${bs}.bmp
                                   COMMAND ${Python3_EXECUTABLE} ARGS -c "from PIL import Image; Image.open('${CMAKE_SOURCE_DIR}/gui/splash${bs}.png').save('${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash${bs}.bmp')"
                                   DEPENDS ${CMAKE_SOURCE_DIR}/gui/splash${bs}.png)
            endforeach()
            add_library(plaskgui SHARED python/exe_gui.cpp)
        else()
            if(USE_SPLASH)
                find_package(X11)
                if(X11_FOUND)
                    set(SHOW_SPLASH TRUE)
                endif()
            endif()
            if(SHOW_SPLASH)
                foreach(bs 620 868 1116)
                    add_custom_command(OUTPUT include/plask/splash${bs}.h
                                    COMMAND ${Python3_EXECUTABLE} ARGS
                                        "${CMAKE_SOURCE_DIR}/toolset/make_splash_header.py"
                                        "${CMAKE_SOURCE_DIR}/gui/splash${bs}.png"
                                        "${CMAKE_BINARY_DIR}/include/plask"
                                    DEPENDS ${CMAKE_SOURCE_DIR}/gui/splash${bs}.png)
                endforeach()
                add_executable(plaskgui
                               python/exe_gui.cpp include/plask/splash620.h include/plask/splash868.h include/plask/splash1116.h)
                target_link_libraries(plaskgui ${X11_LIBRARIES})
            else()
                add_executable(plaskgui python/exe_gui.cpp)
            endif()
        endif()
        target_link_libraries(plaskgui ${Boost_Python3} ${Python3_LIBRARIES} libplask_python libplask)
        if(DEFINED no_strict_aliasing_flag OR DEFINED EXE_FLAGS)
            set_target_properties(plaskgui PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flags} ${EXE_FLAGS}") # necessary for all code which includes "Python.h"
        endif()
        if(DEFINED EXE_FLAGS)
            set_target_properties(plaskgui PROPERTIES LINK_FLAGS "${EXE_FLAGS}")
        endif()

        if(WIN32)
            file(GLOB_RECURSE gui_files RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gui/*.py)
            foreach(gui_file ${gui_files})
                add_custom_command(OUTPUT ${PLASK_PATH}/${gui_file}
                                   COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_SOURCE_DIR}/${gui_file} ${PLASK_PATH}/${gui_file}
                                   DEPENDS ${CMAKE_SOURCE_DIR}/${gui_file})
                list(APPEND gui_files_copied ${PLASK_PATH}/${gui_file})
                get_filename_component(gui_file_base ${gui_file} NAME_WE)
                get_filename_component(gui_file_dir ${gui_file} DIRECTORY)
                list(APPEND gui_files_pyc ${PLASK_PATH}/${gui_file_dir}/__pycache__/${gui_file_base}.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.pyc)
                list(APPEND gui_files_pyo ${PLASK_PATH}/${gui_file_dir}/__pycache__/${gui_file_base}.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.opt-1.pyc)
            endforeach()
            add_custom_command(OUTPUT ${gui_files_pyc} COMMAND ${Python3_EXECUTABLE} -m compileall -q ${PLASK_PATH}/gui DEPENDS ${gui_files_copied})
            add_custom_command(OUTPUT ${gui_files_pyo} COMMAND ${Python3_EXECUTABLE} -O -m compileall -q ${PLASK_PATH}/gui DEPENDS ${gui_files_copied})
            add_custom_target(gui-files ALL DEPENDS ${gui_files_pyc} ${gui_files_pyo})
            install(DIRECTORY ${PLASK_PATH}/gui DESTINATION lib/plask COMPONENT gui)
            install(DIRECTORY gui DESTINATION lib/plask COMPONENT gui
                    PATTERN .* EXCLUDE
                    PATTERN *.xpl EXCLUDE
                    PATTERN *.py EXCLUDE
                    PATTERN *.pyc EXCLUDE
                    PATTERN *.pyo EXCLUDE
                    PATTERN *.pyi EXCLUDE
                    PATTERN __pycache__ EXCLUDE
            )
        else()
            install(DIRECTORY gui DESTINATION lib/plask COMPONENT gui
                    PATTERN .* EXCLUDE
                    PATTERN *.xpl EXCLUDE
                    PATTERN *.pyc EXCLUDE
                    PATTERN *.pyo EXCLUDE
                    PATTERN *.pyi EXCLUDE
                    PATTERN __pycache__ EXCLUDE
            )
            # This will make plaskgui work without install
            file(MAKE_DIRECTORY ${PLASK_PATH})
            execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/gui ${PLASK_PATH}/gui)
        endif()

        install(TARGETS plaskgui DESTINATION bin COMPONENT gui)

        if(UNIX)
            install(FILES utils/application-x-plask.xml DESTINATION share/mime/packages COMPONENT gui)
            install(FILES utils/plaskgui.desktop DESTINATION share/applications COMPONENT gui)
            foreach(s 16 24 32 48)
                install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/${s}x${s}/apps/plask.png
                        DESTINATION share/icons/hicolor/${s}x${s}/apps COMPONENT core)
                install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/${s}x${s}/apps/plaskgui.png
                        DESTINATION share/icons/hicolor/${s}x${s}/apps COMPONENT gui)
                install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/${s}x${s}/mimetypes/application-x-plask.png
                        DESTINATION share/icons/hicolor/${s}x${s}/mimetypes COMPONENT gui)
            endforeach()
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/16x16/mimetypes/application-x-plask.png
                    DESTINATION share/icons/hicolor/16x16/mimetypes COMPONENT gui)
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/scalable/apps/plask.svg
                    DESTINATION share/icons/hicolor/scalable/apps COMPONENT gui)
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/scalable/apps/plaskgui.svg
                    DESTINATION share/icons/hicolor/scalable/apps COMPONENT gui)
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/scalable/mimetypes/application-x-plask.svg
                    DESTINATION share/icons/hicolor/scalable/mimetypes COMPONENT gui)
        endif()

        file(GLOB_RECURSE gui_pyc ${CMAKE_SOURCE_DIR}/gui/*.pyc)
        if(gui_pyc)
            file(REMOVE ${gui_pyc})
        endif()

        if(WIN32)
            file(WRITE ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/plaskgui.rc
                "${manifest}"
                "2 ICON \"${CMAKE_SOURCE_DIR}/utils/plask.ico\"\n"
                "200 ICON \"${CMAKE_SOURCE_DIR}/utils/xpl.ico\"\n"
                "201 BITMAP \"${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash620.bmp\"\n"
                "202 BITMAP \"${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash868.bmp\"\n"
                "203 BITMAP \"${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash1116.bmp\"\n")
            add_executable(plaskgui_exe WIN32 python/win_gui.cpp
                           ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/plaskgui.rc
                           ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash620.bmp
                           ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash868.bmp
                           ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.dir/splash1116.bmp)
            set_target_properties(plaskgui_exe PROPERTIES OUTPUT_NAME "plaskgui")
            install(TARGETS plaskgui_exe RUNTIME DESTINATION bin COMPONENT gui)
        endif()

    endif()

endif(BUILD_PYTHON)


# ---------- Macro for testing --------------------------------------------
if(WIN32)
    set(ENV_PATH $ENV{PATH})
    string(REPLACE ";" "\;" ENV_PATH "${ENV_PATH}")
    string(REPLACE "/" "\\" plask_bin_path "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
    #string(REPLACE "/" "\\" plask_bin_path "$<TARGET_FILE_DIR:plask>")
endif()

macro(add_python_test test test_file)
    add_test(NAME python/${test}
             COMMAND ${CMAKE_BINARY_DIR}/bin/plask -m runtest ${test_file})  # $<TARGET_FILE_DIR:plask>  $<TARGET_FILE:plask>
endmacro()

macro(add_gui_test test test_file)
    add_test(NAME gui/${test}
             COMMAND ${CMAKE_BINARY_DIR}/bin/plask -m runtest ${test_file} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/gui)  # $<TARGET_FILE_DIR:plask>  $<TARGET_FILE:plask>
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

# Scan solver categories and copy Python files
if(BUILD_PYTHON)
    file(GLOB solvers_cats RELATIVE ${CMAKE_SOURCE_DIR}/solvers ${CMAKE_SOURCE_DIR}/solvers/*)
    list(REMOVE_ITEM solvers_cats skel)
    foreach(cat ${solvers_cats})
        file(GLOB lst ${CMAKE_SOURCE_DIR}/solvers/${cat}/*/solvers.yml)
        file(MAKE_DIRECTORY ${PLASK_PATH}/solvers/${cat})
        add_custom_command(OUTPUT ${PLASK_PATH}/solvers/${cat}/__init__.py
                           COMMAND ${Python3_EXECUTABLE} ARGS ${CMAKE_SOURCE_DIR}/toolset/make_category_package.py ${CMAKE_SOURCE_DIR}/solvers/${cat} ${PLASK_PATH}/solvers/${cat}
                           DEPENDS ${lst})
        install(FILES ${PLASK_PATH}/solvers/${cat}/__init__.py DESTINATION lib/plask/solvers/${cat} COMPONENT solvers)
        list(APPEND solvers_packages ${PLASK_PATH}/solvers/${cat}/__init__.py)
        file(GLOB python_solvers_${cat} ${CMAKE_SOURCE_DIR}/solvers/${cat}/*.py)
        install(FILES ${python_solvers_${cat}} DESTINATION lib/plask/solvers/${cat} COMPONENT solvers)
    endforeach()
    add_custom_target(solvers-packages ALL DEPENDS ${solvers_packages})
endif()

# Build solvers
set(solvers_test_targets "")
set(validate_ymls_targets "")
set(SOLVERS_PYTHON_TARGETS "")
foreach(solv ${solvers})
    solv_option(${solv} opt)
    option(${opt} "Build solver '${solv}'" ON)
    if (${opt})
        string(REGEX REPLACE "/|_" "" solvtarget ${solv})
        if (BUILD_PYTHON)
            list(APPEND SOLVERS_PYTHON_TARGETS "solver-${solvtarget}-python")
        endif()
        if(BUILD_TESTING)
            list(APPEND solvers_test_targets "solver-${solvtarget}-test")
            list(APPEND validate_ymls_targets "validate-solver-${solvtarget}-yml")
        endif()
        add_subdirectory(solvers/${solv})
    endif()
endforeach()

add_custom_target(validate-yml DEPENDS ${validate_ymls_targets})

if(BUILD_PYTHON AND BUILD_GUI)
    add_custom_target(plask-stubs ALL DEPENDS ${plask_stubs_files} ${plask_solvers_stubs_files})

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share/plask/schemas/)
    file(GLOB common_yml FOLLOW_SYMLINKS ${CMAKE_SOURCE_DIR}/plask/common/*.yml)
    foreach(yml ${common_yml})
        get_filename_component(yml_name ${yml} NAME)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/share/plask/schemas/${yml_name}
                           COMMAND ${CMAKE_COMMAND} ARGS -E copy ${yml} ${CMAKE_BINARY_DIR}/share/plask/schemas/${yml_name}
                           DEPENDS ${yml})
        install(FILES ${CMAKE_BINARY_DIR}/share/plask/schemas/${yml_name} DESTINATION share/plask/schemas COMPONENT gui)
        list(APPEND common_schemas ${CMAKE_BINARY_DIR}/share/plask/schemas/${yml_name})
    endforeach()
    add_custom_target(common-schemas ALL DEPENDS ${common_schemas})
endif()

# ----------===== Unit tests =====--------------------------------------------
if(BUILD_TESTING)

    set(plask_test_depends ${PLASK_MATERIALS} solvers-packages) # we need materials for testing

    enable_testing()
    if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        file(GLOB_RECURSE plask-test_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.hpp)
        add_executable(plask-test ${plask-test_src})
        set_target_properties(plask-test PROPERTIES OUTPUT_NAME test_plask)
        target_link_libraries(plask-test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} libplask)
        set(plask_test_depends ${plask_test_depends} plask-test)
        file(GLOB_RECURSE plask_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/tests/plask FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.cxx)
        list(REMOVE_ITEM plask_tests main.cpp)
        foreach(test_file ${plask_tests})
            get_filename_component(test ${test_file} NAME_WE)
            add_test(plask/${test} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask -s no -t ${test})
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
        target_link_libraries(plask-test-python ${Python3_LIBRARIES} ${Boost_LIBRARIES} ${Boost_Python3} libplask libplask_python)
        set(plask_test_depends ${plask_test_depends} plask-test-python)

        add_custom_command(OUTPUT "${PLASK_PATH}/python/runtest.py"
                           COMMAND ${CMAKE_COMMAND} ARGS -E copy "${CMAKE_SOURCE_DIR}/tests/python/runtest.py" "${PLASK_PATH}/python"
                           DEPENDS "${CMAKE_SOURCE_DIR}/tests/python/runtest.py")
        set(plask_test_depends ${plask_test_depends} "${PLASK_PATH}/python/runtest.py")
        add_custom_target(python-test-runner ALL DEPENDS "${PLASK_PATH}/python/runtest.py")

        file(GLOB_RECURSE python_tests FOLLOW_SYMLINKS tests/python/*.py)
        list(REMOVE_ITEM python_tests "${CMAKE_SOURCE_DIR}/tests/python/runtest.py")
        foreach(test_file ${python_tests})
            get_filename_component(test ${test_file} NAME_WE)
            add_python_test(${test} ${test_file})
        endforeach()

        if(BUILD_GUI_TESTING)
            file(GLOB gui_tests FOLLOW_SYMLINKS tests/gui/*.py)
            foreach(test_file ${gui_tests})
                get_filename_component(test_ ${test_file} NAME_WE)
                if (NOT ${test_} STREQUAL gui_test_utils)
                    string(REPLACE "_" "/" test ${test_})
                    add_gui_test(${test} ${test_file})
                endif()
            endforeach()
        endif()
    endif()

    if (CTEST_CONFIGURATION_TYPE)
        set(CTEST_EXTRA_PARAMS "-C ${CTEST_CONFIGURATION_TYPE}")
    else()
        set(CTEST_EXTRA_PARAMS "")
    endif()

endif(BUILD_TESTING)

# ----------===== Documentation (api) =====--------------------------------------------
if(WIN32)
    set(doc_dest doc)
else()
    set(doc_dest share/doc/plask)
endif()

find_package(Doxygen OPTIONAL_COMPONENTS dot)
if(DOXYGEN_FOUND)
    if(DOXYGEN_DOT_EXECUTABLE)
        set(DOXYGEN_HAVE_DOT "YES")
    else()
        set(DOXYGEN_HAVE_DOT "NO")
    endif()
endif()
configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile)
if(BUILD_DEVEL_DOC)
    if(DOXYGEN_FOUND)
        add_custom_target(dox ALL ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile ${plask_headers} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        if(INSTALL_DEV)
            install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION ${doc_dest}/devel COMPONENT dev-doc
                    PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
        endif()
    else()
        message("Doxygen not found. Building without documentation.")
    endif()
else()
    if(DOXYGEN_FOUND)
        add_custom_target(dox ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile DEPENDS ${plask_headers} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()

file(GLOB examples ${CMAKE_SOURCE_DIR}/doc/*.xpl ${CMAKE_SOURCE_DIR}/doc/*.py)
list(REMOVE_ITEM examples ${CMAKE_SOURCE_DIR}/doc/conf.py)
install(FILES ${examples} DESTINATION ${doc_dest}/examples COMPONENT doc)

add_subdirectory(doc)




# ----------===== Optional tools =====--------------------------------------------

set(script_permissions "OWNER_WRITE;OWNER_READ;OWNER_EXECUTE;GROUP_READ;GROUP_EXECUTE;WORLD_READ;WORLD_EXECUTE")

if(WIN32)

    set(UTILS_BINARY_DIR "${CMAKE_BINARY_DIR}/CMakeFiles/utils")
    file(MAKE_DIRECTORY ${UTILS_BINARY_DIR})
    function(compile_python_script script_path)
        get_filename_component(script ${script_path} NAME)
        set(script_rc "${UTILS_BINARY_DIR}/${script}.rc")
        set(script_pyc "${UTILS_BINARY_DIR}/${script}.pyc")
        if(DEFINED ARGV1)
            file(WRITE ${script_rc} "${manifest}2 ICON \"${ARGV1}\"\n101 256 \"${script_pyc}\"\n")
            set(exe_type windows)
        else()
            file(WRITE ${script_rc} "${manifest}101 256 \"${script_pyc}\"\n")
            set(exe_type console)
        endif()
        if(CMAKE_HOST_UNIX)
            add_custom_command(OUTPUT ${script_pyc}
                               COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_SOURCE_DIR}/utils/${script_path}.py ${UTILS_BINARY_DIR}
                               COMMAND ${Python3_EXECUTABLE} -c "import compileall\\; compileall.compile_file\\(\\'${UTILS_BINARY_DIR}/${script}.py\\', \\'\\', quiet=True\\)"
                               DEPENDS ${CMAKE_SOURCE_DIR}/utils/${script_path}.py)
        else()
            add_custom_command(OUTPUT ${script_pyc}
                               COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_SOURCE_DIR}/utils/${script_path}.py ${UTILS_BINARY_DIR}
                               COMMAND ${Python3_EXECUTABLE} -c "import compileall; compileall.compile_file('${UTILS_BINARY_DIR}/${script}.py', '', quiet=True)"
                               DEPENDS ${CMAKE_SOURCE_DIR}/utils/${script_path}.py)
        endif()
        add_executable(${script} toolset/run_python.cpp ${script_rc})
        set_source_files_properties(${script_rc} PROPERTIES OBJECT_DEPENDS ${script_pyc})
        set_source_files_properties(toolset/run_python.cpp PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing")
        target_link_libraries(${script} ${Python3_LIBRARIES})
        set_target_properties(${script} PROPERTIES LINK_FLAGS "-m${exe_type} ${EXE_FLAGS}")
        install(TARGETS ${script} DESTINATION bin COMPONENT utils)
    endfunction()

else()

    find_package(bash-completion)
    if(NOT BASH_COMPLETION_FOUND)
        set(BASH_COMPLETION_COMPLETIONSDIR "/etc/bash_completion.d")
        message(STATUS "Using fallback bash completion dir ${BASH_COMPLETION_COMPLETIONSDIR}")
    endif()
    install(FILES utils/bash/plask DESTINATION ${BASH_COMPLETION_COMPLETIONSDIR} COMPONENT completions)

    set(ZSH_COMPLETIONS_PATH /usr/share/zsh/vendor-completions CACHE PATH "Location of ZSH completions file")
    install(FILES utils/zsh/_plask DESTINATION ${ZSH_COMPLETIONS_PATH} COMPONENT completions)

endif()


# ----------===== Packaging =====--------------------------------------------

# This is ugly! Although somehow useful for quick dirty package generation

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Photonic LAser Simulation Kit")
set(CPACK_PACKAGE_VENDOR "Photonics Group, Lodz University of Technology")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/ABOUT")
file(READ ${CMAKE_SOURCE_DIR}/ABOUT CPACK_PACKAGE_DESCRIPTION)

string(REGEX REPLACE "\\.[^.]*$" "" CPACK_PACKAGE_VERSION ${PLASK_VERSION})

set(CPACK_STRIP_FILES ON)

#set(CPACK_COMPONENTS_ALL "core;solvers;gui;utils;doc;dev")
set(CPACK_COMPONENTS_ALL "core;solvers;gui;doc;dev")
set(CPACK_COMPONENT_CORE_DISPLAY_NAME "PLaSK Core")
set(CPACK_COMPONENT_CORE_REQUIRED ON)
set(CPACK_COMPONENT_DOC_DISPLAY_NAME "User Manual and Examples")
set(CPACK_COMPONENT_SOLVERS_DISPLAY_NAME "Computational Solvers")
set(CPACK_COMPONENT_GUI_DISPLAY_NAME "Graphical User Interface")
set(CPACK_COMPONENT_UTILS_DISPLAY_NAME "Command-Line Utilities (e.g. dan2xpl)")
set(CPACK_COMPONENT_DEV_DISPLAY_NAME "Headers and Libraries")
set(CPACK_COMPONENT_DEV_DISABLED ON)

set(CPACK_COMPONENTS_GROUPING "ONE_PER_GROUP")
set(CPACK_COMPONENT_GROUP_APP_DISPLAY_NAME "PLaSK")
set(CPACK_COMPONENT_GROUP_APP_DESCRIPTION "Main PLaSK application")
set(CPACK_COMPONENT_GROUP_APP_BOLD_TITLE ON)
set(CPACK_COMPONENT_GROUP_DEVEL_DISPLAY_NAME "Development")
set(CPACK_COMPONENT_GROUP_DEVEL_DESCRIPTION "Development headers and libraries")

set(CPACK_COMPONENT_CORE_GROUP app)
set(CPACK_COMPONENT_DOC_GROUP app)
set(CPACK_COMPONENT_SOLVERS_GROUP app)
set(CPACK_COMPONENT_GUI_GROUP app)
set(CPACK_COMPONENT_UTILS_GROUP app)
set(CPACK_COMPONENT_DEV_GROUP devel)

# Settings for NSIS generator
if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    if(DLLS_DIR)
        file(GLOB dlls "${DLLS_DIR}/*")
        install(FILES ${dlls} DESTINATION bin COMPONENT core)
    endif()
    set(CPACK_PACKAGE_NAME "PLaSK")
    set(CPACK_NSIS_PACKAGE_NAME "PLaSK")
    list(APPEND CPACK_PACKAGE_EXECUTABLES "plaskgui;PLaSK GUI")
    list(APPEND CPACK_PACKAGE_EXECUTABLES "plask;PLaSK Console")
    set(CPACK_CREATE_DESKTOP_LINKS "plaskgui;PLaSK GUI")
    if(WIN64)
        set(CPACK_SYSTEM_NAME "win64")
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    endif()
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "PLaSK")
    set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "plask")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/ABOUT")
    set(CPACK_NSIS_DISPLAY_NAME "PLaSK ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "plaskgui.exe")
    # File types association:
    string(REPLACE "/" "\\\\" CMAKE_SOURCE_DIR_WINDOWS "${CMAKE_SOURCE_DIR}")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\plaskgui.exe")
    # Very ugly hack!
    set(CPACK_NSIS_COMPRESSOR "lzma
        !define MUI_STARTMENUPAGE_DEFAULTFOLDER \\\"PLaSK\\\"
        !include \\\"${CMAKE_SOURCE_DIR_WINDOWS}\\\\cmake\\\\fileassoc.nsh\\\"
        !include \\\"LogicLib.nsh\\\"
        !include \\\"FileFunc.nsh\\\"")
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
        Var /GLOBAL PYROOTM
        Var /GLOBAL PYROOTU
        Var /GLOBAL cmdLineParams
        SetRegView 64
        !insertmacro APP_ASSOCIATE \\\"xpl\\\" \\\"PLaSK.xpl\\\" \\\"PLaSK structure data\\\" \\\"$INSTDIR\\\\bin\\\\plaskgui.exe,1\\\" \\\"Open in PLaSK GUI\\\" \\\"$INSTDIR\\\\bin\\\\plaskgui.exe \\\$\\\\\\\"%1\\\$\\\\\\\"\\\"
        WriteRegStr HKLM \\\"SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\App Paths\\\\plask.exe\\\" \\\"\\\" \\\"$INSTDIR\\\\bin\\\\plask.exe\\\"
        WriteRegStr HKLM \\\"SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\App Paths\\\\plaskgui.exe\\\" \\\"\\\" \\\"$INSTDIR\\\\bin\\\\plaskgui.exe\\\"
        ReadRegStr $PYROOTU HKCU \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}\\\\InstallPath\\\" \\\"\\\"
        StrCmp $PYROOTU \\\"\\\" plask_global_python
        FileOpen $9 \\\"$PYROOTU\\\\Lib\\\\site-packages\\\\plask.pth\\\" w
        FileWrite $9 \\\"$INSTDIR\\\\lib\\\\plask\\\\python$\\\\r$\\\\n\\\"
        FileClose $9
        plask_global_python:
        ReadRegStr $PYROOTM HKLM \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}\\\\InstallPath\\\" \\\"\\\"
        StrCmp $PYROOTM \\\"\\\" plask_no_python
        FileOpen $9 \\\"$PYROOTM\\\\Lib\\\\site-packages\\\\plask.pth\\\" w
        FileWrite $9 \\\"$INSTDIR\\\\lib\\\\plask\\\\python$\\\\r$\\\\n\\\"
        FileClose $9
        plask_no_python:
        !insertmacro UPDATEFILEASSOC
        IntCmp $dev_selected 0 plask_no_define_root
        WriteRegStr HKLM \\\"SYSTEM\\\\CurrentControlSet\\\\Control\\\\Session Manager\\\\Environment\\\" \\\"PLASK_ROOT\\\" \\\"$INSTDIR\\\"
        SendMessage \\\${HWND_BROADCAST} \\\${WM_WININICHANGE} 0 \\\"STR:Environment\\\" /TIMEOUT=5000
        plask_no_define_root:
        SetRegView 32
        Push $R0
        \\\${GetParameters} $cmdLineParams
        \\\${GetOptions} $cmdLineParams '/DeleteInstaller' $R0
        IfErrors nodelete
        IfRebootFlag 0 noreboot
        Delete /rebootok \\\"$EXEPATH\\\"
        Goto nodelete
        noreboot:
        Delete /rebootok \\\"$EXEPATH\\\"
        SetRebootFlag false
        nodelete:
        Pop $R0
    ")
    set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
        SetRegView 64
        DeleteRegKey HKLM \\\"SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\App Paths\\\\plask.exe\\\"
        DeleteRegKey HKLM \\\"SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\App Paths\\\\plaskgui.exe\\\"
        ReadRegStr $1 HKCU \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}\\\\InstallPath\\\" \\\"\\\"
        Delete \\\"$1\\\\Lib\\\\site-packages\\\\plask.pth\\\"
        ReadRegStr $1 HKLM \\\"SOFTWARE\\\\Python\\\\PythonCore\\\\${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}\\\\InstallPath\\\" \\\"\\\"
        Delete \\\"$1\\\\Lib\\\\site-packages\\\\plask.pth\\\"
        !insertmacro APP_UNASSOCIATE \\\"xpl\\\" \\\"PLaSK.xpl\\\"
        !insertmacro UPDATEFILEASSOC
        IntCmp $dev_was_installed 0 plask_no_delete_root
        DeleteRegValue HKLM \\\"SYSTEM\\\\CurrentControlSet\\\\Control\\\\Session Manager\\\\Environment\\\" \\\"PLASK_ROOT\\\"
        SendMessage \\\${HWND_BROADCAST} \\\${WM_WININICHANGE} 0 \\\"STR:Environment\\\" /TIMEOUT=5000
        plask_no_delete_root:
        SetRegView 32
    ")
endif()

# Settings for DEB and RPM generator
if(UNIX)
    set(CPACK_GENERATOR "DEB;RPM;TBZ2")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0)

    set(CPACK_SET_DESTDIR "ON")

    if(NOT DEFINED CPACK_PACKAGE_NAME)
        set(CPACK_PACKAGE_NAME plask)
    endif()

    file(COPY ${CMAKE_SOURCE_DIR}/utils/hooks/postinst DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_SOURCE_DIR}/utils/hooks/prerm DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_SOURCE_DIR}/utils/hooks/rpmpostin DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_SOURCE_DIR}/utils/hooks/rpmpreun DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_SOURCE_DIR}/utils/hooks/rpmpostun DESTINATION ${CMAKE_BINARY_DIR})

    # execute_process(COMMAND chmod 0755 ${CMAKE_BINARY_DIR}/postinst ${CMAKE_BINARY_DIR}/prerm)

    install(FILES "${CMAKE_SOURCE_DIR}/utils/binfmts/plask" DESTINATION "share/binfmts")

    # execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sys; v=${Boost_VERSION}; sys.stdout.write('.'.join(str(s) for s in (v//100000,v//100%1000,v%100)))" OUTPUT_VARIABLE boostver)
    set(boostver "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")

    set(CPACK_DEB_COMPONENT_INSTALL ON)
    set(CPACK_DEBIAN_APP_PACKAGE_NAME plask)
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Maciej Dems <maciej.dems@p.lodz.pl>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
    file(READ ${CMAKE_SOURCE_DIR}/ABOUT plask_about)
    set(CPACK_DEBIAN_APP_DESCRIPTION "${plask_about}\nTo compile your own materials, install the 'plask-dev' package.")
    set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "libc6, libexpat1, libboost-python${boostver}, libboost-filesystem${boostver}, binfmt-support")
    if("${OpenMP_CXX_FLAGS}" MATCHES ".*libiomp5*")
        set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, libomp5")
    endif()
    if("${BLAS_LIBRARIES}" MATCHES ".*libopenblas.so.*")
        set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, libopenblas-base | libopenblas0-pthread | libopenblas0-openmp | libopenblas0-serial")
    elseif("${BLAS_LIBRARIES}" MATCHES ".*libmkl.*")
        foreach(libmkl rt core sequential intel_lp64 gf_lp64 gnu_thread intel_thread)
            if("${BLAS_LIBRARIES};${LAPACK_LIBRARIES}" MATCHES ".*libmkl_${libmkl}.so.*")
                string(REPLACE "_" "-" mklpkg "${libmkl}")
                set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, libmkl-${mklpkg}")
            endif()
        endforeach()
        set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, libmkl-def, libmkl-avx2")
    else()
        set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, libblas.so.3, liblapack.so.3")
    endif()
    set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, python3 (>= 3.${Python3_VERSION_MINOR}.0), python3-numpy, python3-matplotlib, python3-h5py, python3-scipy")
    set(CPACK_DEBIAN_APP_PACKAGE_RECOMMENDS "ipython3 (>= 0.13)")
    if(BUILD_GUI)
        set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "${CPACK_DEBIAN_APP_PACKAGE_DEPENDS}, python3-pyqt5 | python3-pyside2.qthelp, python3-lxml, python3-yaml")
        set(CPACK_DEBIAN_APP_PACKAGE_RECOMMENDS "${CPACK_DEBIAN_APP_PACKAGE_RECOMMENDS}, python3-jedi, python3-paramiko")
    endif()
    set(CPACK_DEBIAN_APP_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/postinst;${CMAKE_BINARY_DIR}/prerm")
    set(CPACK_DEBIAN_DEVEL_PACKAGE_NAME plask-dev)
    set(CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS "plask (= ${CPACK_PACKAGE_VERSION})")
    set(CPACK_DEBIAN_DEVEL_DESCRIPTION "${plask_about}\nThis package contains headers and libraries needed to compile custom materials\n for PLaSK.")

    set(CPACK_RPM_COMPONENT_INSTALL ON)
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    set(CPACK_RPM_APP_PACKAGE_NAME plask)
    set(CPACK_RPM_PACKAGE_GROUP Applications/Engineering)
    set(CPACK_RPM_APP_PACKAGE_REQUIRES "expat boost-filesystem openblas boost-python3 python3-numpy python3-pip")
    set(CPACK_RPM_APP_PACKAGE_SUGGESTS "scipy python-matplotlib h5py")
    if(BUILD_GUI)
        set(CPACK_RPM_APP_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES} python3-pyqt5-sip python3-pyyaml")
    endif()
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST "/usr;/usr/bin;/usr/include;/usr/lib;/usr/share;/usr/share/applications;/usr/share/doc;/usr/share/icons;/usr/share/icons/hicolor;/usr/share/icons/hicolor/16x16;/usr/share/icons/hicolor/16x16/apps;/usr/share/icons/hicolor/16x16/mimetypes;/usr/share/icons/hicolor/24x24;/usr/share/icons/hicolor/24x24/apps;/usr/share/icons/hicolor/24x24/mimetypes;/usr/share/icons/hicolor/32x32;/usr/share/icons/hicolor/32x32/apps;/usr/share/icons/hicolor/32x32/mimetypes;/usr/share/icons/hicolor/48x48;/usr/share/icons/hicolor/48x48/apps;/usr/share/icons/hicolor/48x48/mimetypes;/usr/share/icons/hicolor/scalable;/usr/share/icons/hicolor/scalable/apps;/usr/share/icons/hicolor/scalable/mimetypes;/usr/share/icons;/usr/share/icons/breeze;/usr/share/icons/breeze/16x16;/usr/share/icons/breeze/16x16/apps;/usr/share/icons/breeze/16x16/mimetypes;/usr/share/icons/breeze/24x24;/usr/share/icons/breeze/24x24/apps;/usr/share/icons/breeze/24x24/mimetypes;/usr/share/icons/breeze/32x32;/usr/share/icons/breeze/32x32/apps;/usr/share/icons/breeze/32x32/mimetypes;/usr/share/icons/breeze/48x48;/usr/share/icons/breeze/48x48/apps;/usr/share/icons/breeze/48x48/mimetypes;/usr/share/icons/breeze/scalable;/usr/share/icons/breeze/scalable/apps;/usr/share/icons/breeze/scalable/mimetypes;/usr/share/mime;/usr/share/mime/packages;/usr/share/mime/packages")
    set(CPACK_RPM_APP_POST_INSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/rpmpostin")
    set(CPACK_RPM_APP_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/rpmpreun")
    set(CPACK_RPM_APP_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/rpmpostun")
    set(CPACK_RPM_DEVEL_PACKAGE_NAME plask-devel)
    set(CPACK_RPM_DEVEL_PACKAGE_REQUIRES "plask = ${CPACK_PACKAGE_VERSION}")

    install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION "share/doc/${CPACK_PACKAGE_NAME}" RENAME copyright
            PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)

    find_program(LSB_RELEASE_EXECUTABLE lsb_release)
    if(LSB_RELEASE_EXECUTABLE)
        execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -sc OUTPUT_VARIABLE LSB_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -sr OUTPUT_VARIABLE LSB_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -si OUTPUT_VARIABLE LSB_DISTRIBUTOR_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        set(LSB_DISTRIBUTOR_ID "unknown")
        set(LSB_RELEASE "unknown")
        set(LSB_CODENAME "unknown")
    endif()

    EXECUTE_PROCESS(COMMAND date -R OUTPUT_VARIABLE CURRENT_DATE)

    configure_file(${CMAKE_SOURCE_DIR}/changelog.Debian.in ${CMAKE_BINARY_DIR}/changelog.Debian)
    execute_process(COMMAND gzip -c -9 ${CMAKE_BINARY_DIR}/changelog.Debian WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    OUTPUT_FILE "${CMAKE_BINARY_DIR}/changelog.Debian.gz")
    install(FILES "${CMAKE_BINARY_DIR}/changelog.Debian.gz" DESTINATION "share/doc/${CPACK_PACKAGE_NAME}")
endif()

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

# ----------===== config.hpp & version.hpp =====--------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/config.hpp.in ${CMAKE_BINARY_DIR}/include/plask/config.hpp)
configure_file(${CMAKE_SOURCE_DIR}/version.hpp.in ${CMAKE_BINARY_DIR}/include/plask/version.hpp)
