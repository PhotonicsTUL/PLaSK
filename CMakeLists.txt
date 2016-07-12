# CMake file for project PLaSK

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0017 OLD)       # ensure our CMake modules are loaded first
if(POLICY CMP0058)
    cmake_policy(SET CMP0058 OLD)   # for Ninja
endif()

project(plask)

set(VERSION_MAJOR 0)            # MAJOR version (change to 1 on publication in Journal of Computer Physics)
set(VERSION_MINOR 2)            # MINOR version (bump up by one on visible feature change)
# set(VERSION_REVISION 1)       # REVISION number (leave undefined unless really needed, it's normally set to svn revision)

# ---------- Common options --------------------------------------------

if(MSVC)    # flags needed by Visual Studio:
    set(CMAKE_CXX_FLAGS "/bigobj ${CMAKE_CXX_FLAGS}")   #increase a number of sections in obj. file, required by plask
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")   #disable many warning (all stl & boost tamplates) about wrong dll import/export, TODO subject for future investigation
    add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
    # _D_USE_MATH_DEFINES gives M_PI, etc. in <cmath>, maybe it is better idea to define this just before including <cmath>
    # NOMINMAX prevents windows header from defining min, max macros, see http://stackoverflow.com/questions/1904635/warning-c4003-and-errors-c2589-and-c2059-on-x-stdnumeric-limitsintmax
    # _SCL_SECURE_NO_WARNINGS and D_CRT_SECURE_NO_WARNINGS hide many warnings which are coused by boost and cppforamt under MSVC, https://msdn.microsoft.com/query/dev14.query?appId=Dev14IDEF1&l=EN-US&k=k(C4996)&rd=true
    add_definitions(-DBOOST_ALL_DYN_LINK)   #MSVC+cmake require this to properly detect dynamic boost libraries
    add_definitions(-DXML_STATIC)           #We use static expat - TODO: this is needed only for libplask, static/dynamic should be detected
    add_definitions(-DFMT_HEADER_ONLY)  #don't compile cppformat, workaround many linker errors
else(MSVC)
    if(MINGW)
        set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-narrowing -Wno-sign-compare -Wno-unknown-pragmas -Wno-ignored-qualifiers")   # -Wsuggest-override - older gcc does not support this
endif(MSVC)

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
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(no_strict_aliasing_flag "-fno-strict-aliasing")
endif()

include_directories(.)          # solvers can include plask using <plask/plask.hpp>
include_directories(python)     # solvers can also include <plask/python.hpp>
include_directories(extlib)     # place for local external libraries
include_directories(${CMAKE_BINARY_DIR}/include) # for include plask/config.h

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

option(BUILD_DEVEL_DOC "Build doxygen documentation." OFF)

option(BUILD_PYTHON "Build and install Python interface to PLaSK (recommended)." ON)
cmake_dependent_option(BUILD_PYTHON_MODULE "Build Python interface as external module." OFF "BUILD_PYTHON" OFF)
cmake_dependent_option(BUILD_GUI "Build and install GUI." ON "BUILD_PYTHON" OFF)

option(BUILD_TESTING "Build unit tests." ON)
cmake_dependent_option(RUN_TESTS "Run unit tests." ON "BUILD_TESTING" OFF)

option(USE_OMP "Use OpenMP" ON)

set(USE_PROFILER "" CACHE STRING "Type of the profiler to use. Leave empty for no profiling. Set to 'GNU' if you want to use gprof or to 'Google' for google perftools.")

if(WIN32)
    set(USE_MANIFEST "mscvr90" CACHE STRING "Manifest to include into Windows executables")
endif()

option(LICENSE_CHECKING "Perform license check" OFF)

option(PRINT_STACKTRACE_ON_EXCEPTION "Print stack-trace on stderr when plask::Exception is throwed (works only in debug mode)." ON)

#TODO if disabled, plask should find and use external cppformat lib.
#option(PLASK_EXPORTS_CPPFORMAT "Export cppformat symbols in libplask (affects only windows)" ON)

option(INSTALL_DEV "Install headers to system locations" OFF)
option(INSTALL_MATERIALS_DEV "Install headers for compuling materials" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    option(USE_GLIBCXX_PARALLEL "use the libstdc++ parallel mode (experimental in libstdc++v3)" OFF)
endif(CMAKE_COMPILER_IS_GNUCXX)

#option (PACK_EXE "Self-pack and strip executables." OFF)  # compress exe

set(PYTHON_VERSION "" CACHE STRING "Python version to use")

set(ECM_ENABLE_SANITIZERS "" CACHE STRING "semicolon-separated list of sanitizers to use: leak (requires address), undefined, (and up to one of:) address, memory, thread")
include(ECMEnableSanitizers)    # Sanitizers, this module is also included in debian package: extra-cmake-modules

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

# ---------- NuGet paths ---------------------------------------------------
if(NUGET_PACKAGES_PATH)
    # TODO use the highest number not higher than compiler version intead of constants like "v110"
    file(GLOB nuget_expat_includedir "${NUGET_PACKAGES_PATH}/expat*/build/native/include")
    file(GLOB nuget_expat_libdir "${NUGET_PACKAGES_PATH}/expat*/build/native/lib/v110/x64/Release/static/utf8")
    list(APPEND CMAKE_INCLUDE_PATH ${nuget_expat_includedir})
    list(APPEND CMAKE_LIBRARY_PATH ${nuget_expat_libdir})
    file(GLOB nuget_eigen_includedir "${NUGET_PACKAGES_PATH}/Eigen*/build/native/include")
        list(APPEND CMAKE_INCLUDE_PATH ${nuget_eigen_includedir})
    #file(GLOB nuget_blas_includedir "${NUGET_PACKAGES_PATH}/OpenBLAS*/lib/native/include")
    #file(GLOB nuget_blas_libdir "${NUGET_PACKAGES_PATH}/OpenBLAS*/lib/native/lib/x64" "${NUGET_PACKAGES_PATH}/lib/native/bin/x64")
    #list(APPEND CMAKE_INCLUDE_PATH ${nuget_blas_includedir})
    #list(APPEND CMAKE_LIBRARY_PATH ${nuget_blas_libdir})
    #file(GLOB nuget_boost_includedir "${NUGET_PACKAGES_PATH}/boost*/lib/native/include")
    #list(APPEND CMAKE_INCLUDE_PATH ${nuget_boost_includedir})
    #message("CMAKE_INCLUDE_PATH is: ${CMAKE_INCLUDE_PATH}")
    #message("CMAKE_LIBRARY_PATH is: ${CMAKE_LIBRARY_PATH}")
endif(NUGET_PACKAGES_PATH)

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

find_package(EXPAT REQUIRED)
include_directories(${EXPAT_INCLUDE_DIR})

# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
set(boost_components system filesystem) # thread
if(BUILD_TESTING)
    set(boost_components ${boost_components} unit_test_framework)
endif(BUILD_TESTING)

option(USE_STATIC_BOOST "use static boost (not recommended)" OFF)
if(USE_STATIC_BOOST)
    add_definitions("-DBOOST_PYTHON_STATIC_LIB")  # fix linker errors to boost python
    set(Boost_USE_STATIC_LIBS ON)
endif(USE_STATIC_BOOST)

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
link_directories(${Boost_LIBRARY_DIRS})

if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes") # get rid of annoying warnings in Boost
endif()

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
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
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

## ---------- libraries for better stack printing ------------------------------
if(PRINT_STACKTRACE_ON_EXCEPTION)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(NOT MSVC)
        add_definitions(-D_GLIBCXX_DEBUG)   # safe STL under linux, see https://gcc.gnu.org/onlinedocs/libstdc++/manual/debug_mode.html
    endif(NOT MSVC)
    if(WIN32)
        add_subdirectory(extlib/PEparser)    # PE parser (for proper resolving some symbols in stack printing under windows mingw)
        set(PRINT_STACKTRACE_LIBRARY "dbghelp" PEparser)
    else()
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


# ----------===== Build and Install =====--------------------------------------------

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h plask/*.hpp solvers/providers/*.h solvers/providers/*.hpp)

file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp solvers/providers/*.cpp)

# Python sources and files
file(GLOB_RECURSE python_plask_src FOLLOW_SYMLINKS python/plask/*.cpp python/plask/*.h)
file(GLOB_RECURSE python_plask_py python/plask/*.py)

if(WIN32 AND NOT MSVC AND USE_MANIFEST)
    set(manifest "1 24 \"${CMAKE_SOURCE_DIR}/toolset/mingw/${USE_MANIFEST}.manifest\"\n")
else()
    set(manifest "")
endif()

# ---------- External libraries built by us -----------------------------------------

# CAMOS
add_subdirectory(extlib/camos)
add_subdirectory(extlib/fftpacx)

# cppformat
if (NOT MSVC)
    list(APPEND plask_src extlib/cppformat/format.cc)
endif (NOT MSVC)

# ---------- PLaSK library --------------------------------------------

add_library(libplask SHARED ${plask_src} ${plask_headers})
#add_library(libplask SHARED ${plask_src})
target_link_libraries(libplask ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${EXPAT_LIBRARIES} ${PRINT_STACKTRACE_LIBRARY}) # ${Boost_THREAD_LIBRARY}
if(WIN32)
    target_link_libraries(libplask kernel32 iphlpapi)
    if(INSTALL_DEV)
        install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core
                                 ARCHIVE DESTINATION lib COMPONENT dev)
    else()
        install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core)
    endif()
    if(MSVC)    # defualt "plask.pdb" is overriden by file for plask.exe, so we use different name ("libplask.pdb")
        set_target_properties (libplask PROPERTIES PDB_NAME libplask)
    endif(MSVC)
else()
    target_link_libraries(libplask dl)
    install(TARGETS libplask LIBRARY DESTINATION lib COMPONENT core)
endif()
set_target_properties(libplask PROPERTIES OUTPUT_NAME plask COMPILE_FLAGS "-DPLASK_EXPORTS -DFMT_EXPORT")

#if (NOT WIN32) #can be usefull if more threads libraries will be used
#    target_link_libraries(sdsolver "pthread")
#endif()

if(INSTALL_DEV)
    install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
    install(DIRECTORY solvers/providers DESTINATION include/plask COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
    install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.h DESTINATION include/plask COMPONENT dev)
endif()

set(PLaSK_LIBRARIES libplask)

# ---------- Default materials --------------------------------------------

add_subdirectory(materials)

# ---------- Program to sign license file ----------------------
file(GLOB_RECURSE license_sign_src FOLLOW_SYMLINKS license_sign/*.cpp license_sign/*.h)
add_executable(plask-license-sign ${license_sign_src})
target_link_libraries(plask-license-sign libplask)

# Copy example license file for easier development
if(CMAKE_CONFIGURATION_TYPES)
    foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # under multi-target env. (like MSVC) install licence for all configurations
        configure_file(${CMAKE_SOURCE_DIR}/plask_license.xml ${CMAKE_BINARY_DIR}/${CONF}/plask_license.xml COPYONLY)
    endforeach()
else()
    configure_file(${CMAKE_SOURCE_DIR}/plask_license.xml ${CMAKE_BINARY_DIR}/plask_license.xml COPYONLY)
endif()


# ---------- Python --------------------------------------------

if(BUILD_PYTHON)

    foreach(py_file ${python_plask_py})
        get_filename_component(py_file_base ${py_file} NAME)
        list(APPEND python_plask_py_copied ${PLASK_PYTHON_PATH}/${py_file_base})
        add_custom_command(OUTPUT ${PLASK_PYTHON_PATH}/${py_file_base}
                           COMMAND ${CMAKE_COMMAND} ARGS -E copy ${py_file} ${PLASK_PYTHON_PATH}/${py_file_base}
                           DEPENDS ${py_file})
        list(APPEND python_plask_pyc ${PLASK_PYTHON_PATH}/${py_file_base}c)
        list(APPEND python_plask_pyo ${PLASK_PYTHON_PATH}/${py_file_base}o)
    endforeach()

    add_custom_target(python-scripts ALL DEPENDS ${python_plask_py_copied})

    if(WIN32)
        add_custom_command(TARGET python-scripts COMMAND ${PYTHON_EXECUTABLE} -m compileall -q ${PLASK_PYTHON_PATH})
        add_custom_command(TARGET python-scripts COMMAND ${PYTHON_EXECUTABLE} -O -m compileall -q ${PLASK_PYTHON_PATH})
        install(FILES ${python_plask_pyc} DESTINATION lib/plask/python/plask COMPONENT core)
        install(FILES ${python_plask_pyo} DESTINATION lib/plask/python/plask COMPONENT core)
    endif()

    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

    add_library(libplask_python SHARED ${python_plask_src})
    target_link_libraries(libplask_python ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARIES} libplask)
    set_target_properties(libplask_python PROPERTIES OUTPUT_NAME plask_python)
    if(DEFINED no_strict_aliasing_flag)
        set_target_properties(libplask_python PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flag} -DPLASK_PYTHON_EXPORTS")
    else()
        set_target_properties(libplask_python PROPERTIES COMPILE_FLAGS -DPLASK_PYTHON_EXPORTS)
    endif()
    #if(WIN32)
    #    set_target_properties(libplask_python PROPERTIES LINK_FLAGS -Wl,--export-all-symbols)
    #endif()
    # if(BUILD_PYTHON_MODULE)
    # endif()

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

    if(WIN32 AND NOT MSVC)
        file(WRITE ${CMAKE_BINARY_DIR}/CMakeFiles/plask.rc "${manifest}")
        add_executable(plask python/exe_main.cpp ${CMAKE_BINARY_DIR}/CMakeFiles/plask.rc)
        # For some reason MinGW wants to create import library for executable,
        # which overwrites libplask.dll.a. With this hack we name this useless
        # stuff libplask.exe.dll.a
        set_target_properties(plask PROPERTIES OUTPUT_NAME "plask.exe" SUFFIX "")
    else()
        add_executable(plask python/exe_main.cpp)
    endif()
    target_link_libraries(plask ${Boost_PYTHON_LIBRARIES} ${PYTHON_LIBRARIES} libplask_python libplask)
    add_dependencies(plask python-scripts)

    if(DEFINED no_strict_aliasing_flag OR DEFINED EXE_FLAGS)
        set_target_properties(plask PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flag} ${EXE_FLAGS}") # necessary for all code which includes "Python.h"
    endif()
    if(DEFINED EXE_FLAGS)
        set_target_properties(plask PROPERTIES LINK_FLAGS "${EXE_FLAGS}")
    endif()

    set(plask_stubs _plask geometry mesh material flow)
    foreach(stub ${plask_stubs})
        list(APPEND plask_stubs_files ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/python/plask/${stub}.py)
        list(APPEND plask_stubs_modules plask.${stub})
    endforeach()
    file(REMOVE ${plask_stubs_files}) # wrong stubs can make build fail
    if(BUILD_GUI)
        if(CMAKE_CONFIGURATION_TYPES)
            foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # under multi-target env. (like MSVC) install licence for all configurations
                file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CONF}/share/plask/stubs)
            endforeach()
        else()
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/share/plask/stubs)
        endif()
        add_custom_command(OUTPUT ${plask_stubs_files}
                           COMMAND plask -lwarning ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${plask_stubs_modules}
                           DEPENDS plask ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${material_targets}
                           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/python
                          )
        install(FILES ${plask_stubs_files} DESTINATION lib/plask/python/plask COMPONENT gui)
    endif()

    install(FILES ${python_plask_py_copied} DESTINATION lib/plask/python/plask COMPONENT core)
    install(TARGETS plask RUNTIME DESTINATION bin COMPONENT core)
    if (INSTALL_DEV)
        install(DIRECTORY python/plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "python*.h" PATTERN "python*.hpp" PATTERN ".*" EXCLUDE)
    endif()

    # GUI
    if(BUILD_GUI)
        if(WIN32)
            file(WRITE ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.rc "${manifest}2 ICON \"${CMAKE_SOURCE_DIR}/utils/plask.ico\"\n200 ICON \"${CMAKE_SOURCE_DIR}/utils/xpl.ico\"\n")
            add_executable(plaskgui WIN32 python/exe_gui.cpp ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.rc)
        else()
            add_executable(plaskgui WIN32 python/exe_gui.cpp)
        endif()
        target_link_libraries(plaskgui ${Boost_PYTHON_LIBRARIES} ${PYTHON_LIBRARIES} libplask_python libplask)
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
                list(APPEND gui_files_pyc ${PLASK_PATH}/${gui_file}c)
                list(APPEND gui_files_pyo ${PLASK_PATH}/${gui_file}o)
            endforeach()
            add_custom_command(OUTPUT ${gui_files_pyc} COMMAND ${PYTHON_EXECUTABLE} -m compileall -q ${PLASK_PATH}/gui DEPENDS ${gui_files_copied})
            add_custom_command(OUTPUT ${gui_files_pyo}  COMMAND ${PYTHON_EXECUTABLE} -O -m compileall -q ${PLASK_PATH}/gui)
            add_custom_target(gui-files ALL DEPENDS ${gui_files_pyc} ${gui_files_pyo})
            install(DIRECTORY ${PLASK_PATH}/gui DESTINATION lib/plask COMPONENT GUI)
            install(DIRECTORY gui DESTINATION lib/plask COMPONENT GUI
                    PATTERN .* EXCLUDE
                    PATTERN *.xpl EXCLUDE
                    PATTERN *.py EXCLUDE
                    PATTERN *.pyc EXCLUDE
                    PATTERN *.pyo EXCLUDE
            )
        else()
            install(DIRECTORY gui DESTINATION lib/plask COMPONENT GUI
                    PATTERN .* EXCLUDE
                    PATTERN *.xpl EXCLUDE
                    PATTERN *.pyc EXCLUDE
                    PATTERN *.pyo EXCLUDE
            )
        endif()

        install(TARGETS plaskgui DESTINATION bin COMPONENT GUI)

        if(UNIX)
            install(FILES utils/application-x-plask.xml DESTINATION share/mime/packages COMPONENT GUI)
            install(FILES utils/plaskgui.desktop DESTINATION share/applications COMPONENT GUI)
            foreach(s 16 24 32 48)
                install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/${s}x${s}/apps/plask.png
                        DESTINATION share/icons/hicolor/${s}x${s}/apps COMPONENT core)
                install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/${s}x${s}/apps/plaskgui.png
                        DESTINATION share/icons/hicolor/${s}x${s}/apps COMPONENT GUI)
                install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/${s}x${s}/mimetypes/application-x-plask.png
                        DESTINATION share/icons/hicolor/${s}x${s}/mimetypes COMPONENT GUI)
            endforeach()
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/16x16/mimetypes/application-x-plask.png
                    DESTINATION share/icons/hicolor/16x16/mimetypes COMPONENT GUI)
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/scalable/apps/plask.svg
                    DESTINATION share/icons/hicolor/scalable/apps COMPONENT GUI)
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/scalable/apps/plaskgui.svg
                    DESTINATION share/icons/hicolor/scalable/apps COMPONENT GUI)
            install(FILES ${CMAKE_SOURCE_DIR}/gui/icons/hicolor/scalable/mimetypes/application-x-plask.svg
                    DESTINATION share/icons/hicolor/scalable/mimetypes COMPONENT GUI)
        endif()

        file(GLOB_RECURSE gui_pyc ${CMAKE_SOURCE_DIR}/gui/*.pyc)
        if(gui_pyc)
            file(REMOVE ${gui_pyc})
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
             COMMAND plask ${CMAKE_SOURCE_DIR}/tests/python/main ${test_file})  # $<TARGET_FILE_DIR:plask>  $<TARGET_FILE:plask>
endmacro()

macro(add_xpl_test test test_file)
    add_test(NAME xpl/${test}
             COMMAND plask ${CMAKE_SOURCE_DIR}/tests/python/xpl ${test_file})
endmacro()

macro(add_solver_test test test_file)
    add_test(NAME ${test} COMMAND ${PLASK_SOLVER_PATH}/${test_file})
    if(WIN32)
        set_tests_properties(${test} PROPERTIES ENVIRONMENT "PATH=${plask_bin_path}\;${ENV_PATH}")
    endif()
    set(SOLVER_TEST_DEPENDS ${test_file})
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
        file(GLOB lst ${CMAKE_SOURCE_DIR}/solvers/${cat}/*/solvers.xml)
        add_custom_command(OUTPUT ${PLASK_PATH}/solvers/${cat}/__init__.py
                           COMMAND ${PYTHON_EXECUTABLE} ARGS ${CMAKE_SOURCE_DIR}/toolset/make_category_package.py ${CMAKE_SOURCE_DIR}/solvers/${cat} ${PLASK_PATH}/solvers/${cat}
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

if(BUILD_PYTHON AND BUILD_GUI)
    add_custom_target(plask-stubs ALL DEPENDS ${plask_stubs_files} ${plask_solvers_stubs_files})
endif()

# ----------===== Unit tests =====--------------------------------------------
if(BUILD_TESTING)

    set(plask_test_depends ${material_targets} solvers-packages) # we need materials for testing

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
        if (WIN32)
            target_link_libraries(plask-test-python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask libplask_python)
        else()
            target_link_libraries(plask-test-python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask)
        endif()
        set(plask_test_depends ${plask_test_depends} plask-test-python)

        file(GLOB_RECURSE python_tests FOLLOW_SYMLINKS tests/python/*.py)
        foreach(test_file ${python_tests})
            get_filename_component(test ${test_file} NAME_WE)
            add_python_test(${test} ${test_file})
        endforeach()
    endif(BUILD_PYTHON)

    if (NOT CTEST_CONFIGURATION_TYPE)
       set(CTEST_CONFIGURATION_TYPE ${CMAKE_BUILD_TYPE})
       message(STATUS "Setting CTEST Configuration Type to ${CTEST_CONFIGURATION_TYPE}")
    endif()

    if (RUN_TESTS)
        add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 60 DEPENDS ${plask_test_depends} ${solvers_test_targets})  #-C ${CTEST_CONFIGURATION_TYPE}
    else()
        add_custom_target(tests ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 60 DEPENDS ${plask_test_depends} ${solvers_test_targets})
    endif()

endif(BUILD_TESTING)

# ----------===== Documentation (api) =====--------------------------------------------
if(WIN32)
    set(doc_dest doc)
else()
    set(doc_dest share/doc/plask)
endif()

find_package(Doxygen)
configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile)
if(BUILD_DEVEL_DOC)
    if(DOXYGEN_FOUND)
        add_custom_target(dox ALL ${DOXYGEN} ${CMAKE_BINARY_DIR}/Doxyfile ${plask_headers} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        if(INSTALL_DEV)
            install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION ${doc_dest}/devel COMPONENT dev-doc
                    PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
        endif()
    else()
        message("Doxygen not found. Building without documentation.")
    endif()
else()
    if(DOXYGEN_FOUND)
        add_custom_target(dox ${DOXYGEN} ${CMAKE_BINARY_DIR}/Doxyfile DEPENDS ${plask_headers} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
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
                               COMMAND ${PYTHON_EXECUTABLE} -c "import compileall\\; compileall.compile_file\\(\\'${UTILS_BINARY_DIR}/${script}.py\\', \\'\\', quiet=True\\)"
                               DEPENDS ${CMAKE_SOURCE_DIR}/utils/${script_path}.py)
        else()
            add_custom_command(OUTPUT ${script_pyc}
                               COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_SOURCE_DIR}/utils/${script_path}.py ${UTILS_BINARY_DIR}
                               COMMAND ${PYTHON_EXECUTABLE} -c "import compileall; compileall.compile_file('${UTILS_BINARY_DIR}/${script}.py', '', quiet=True)"
                               DEPENDS ${CMAKE_SOURCE_DIR}/utils/${script_path}.py)
        endif()
        add_executable(${script} toolset/run_python.cpp ${script_rc})
        set_source_files_properties(${script_rc} PROPERTIES OBJECT_DEPENDS ${script_pyc})
        set_source_files_properties(toolset/run_python.cpp PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing")
        target_link_libraries(${script} ${PYTHON_LIBRARIES})
        set_target_properties(${script} PROPERTIES LINK_FLAGS "-m${exe_type} ${EXE_FLAGS}")
        install(TARGETS ${script} DESTINATION bin COMPONENT utils)
    endfunction()

    #compile_python_script(dan2xpl)
    #install(FILES utils/dan2xpl.py DESTINATION bin COMPONENT utils)
    #install(FILES utils/xpl2dan.py DESTINATION bin COMPONENT utils)

#else()

    #install(FILES utils/dan2xpl.py DESTINATION bin PERMISSIONS ${script_permissions} COMPONENT utils RENAME dan2xpl)
    #install(FILES utils/xpl2dan.py DESTINATION bin PERMISSIONS ${script_permissions} COMPONENT utils RENAME xpl2dan)

endif()


# ----------===== Public headers for writing material databases =====--------

if(NOT INSTALL_DEV AND INSTALL_MATERIALS_DEV)
    install(FILES plask/material.hpp plask/math.h plask/memory.h plask/exceptions.h plask/parallel.h DESTINATION include/plask COMPONENT dev)
    install(FILES plask/material/material.h plask/material/db.h  DESTINATION include/plask/material COMPONENT dev)
    install(FILES plask/utils/format.h plask/utils/string.h plask/utils/system.h DESTINATION include/plask/utils COMPONENT dev)
    install(FILES plask/vector/tensor2.h plask/vector/tensor3.h DESTINATION include/plask/vector COMPONENT dev)
    install(FILES plask/phys/constants.h plask/phys/functions.h DESTINATION include/plask/phys COMPONENT dev)
    install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.h DESTINATION include/plask COMPONENT dev)
endif()

# ----------===== Packaging =====--------------------------------------------

# This is ugly! Although somehow useful for quick dirty package generation

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Photonic LAser Simulation Kit")
set(CPACK_PACKAGE_VENDOR "Photonics Group, Lodz University of Technology")
set(CPACK_PACKAGE_DESCRIPTION_FILE  "${CMAKE_SOURCE_DIR}/ABOUT")
file(READ ${CMAKE_SOURCE_DIR}/ABOUT CPACK_PACKAGE_DESCRIPTION)

set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_REVISION})

set(CPACK_STRIP_FILES ON)

#set(CPACK_COMPONENTS_ALL "core;solvers;GUI;utils;doc;dev")
set(CPACK_COMPONENTS_ALL "core;solvers;GUI;doc;dev")
set(CPACK_COMPONENT_CORE_DISPLAY_NAME "PLaSK core")
set(CPACK_COMPONENT_CORE_REQUIRED ON)
set(CPACK_COMPONENT_DOC_DISPLAY_NAME "User Manual and Examples")
set(CPACK_COMPONENT_SOLVERS_DISPLAY_NAME "Computational Solvers")
set(CPACK_COMPONENT_GUI_DISPLAY_NAME "Graphical User Interface")
set(CPACK_COMPONENT_UTILS_DISPLAY_NAME "Command-Line Utilities (e.g. dan2xpl)")
set(CPACK_COMPONENT_DEV_DISPLAY_NAME "Custom Materials Headers")

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
    set(CPACK_NSIS_DISPLAY_NAME "PLaSK ${VERSION}")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH ON)
    # File types association:
    string(REPLACE "/" "\\\\" CMAKE_SOURCE_DIR_WINDOWS "${CMAKE_SOURCE_DIR}")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\plaskgui.exe")
    # Very ugly hack!
    set(CPACK_NSIS_COMPRESSOR "lzma
        !define MUI_STARTMENUPAGE_DEFAULTFOLDER \\\"PLaSK\\\"
        !include ${CMAKE_SOURCE_DIR_WINDOWS}\\\\cmake\\\\fileassoc.nsh")
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
        !insertmacro APP_ASSOCIATE \\\"xpl\\\" \\\"PLaSK.xpl\\\" \\\"PLaSK structure data\\\" \\\"$INSTDIR\\\\bin\\\\plaskgui.exe,1\\\" \\\"Open in PLaSK GUI\\\" \\\"$INSTDIR\\\\bin\\\\plaskgui.exe \\\$\\\\\\\"%1\\\$\\\\\\\"\\\"
        !insertmacro UPDATEFILEASSOC
    ")
    set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
        !insertmacro APP_UNASSOCIATE \\\"xpl\\\" \\\"PLaSK.xpl\\\"
        !insertmacro UPDATEFILEASSOC
    ")
    set(CPACK_NSIS_EXTRA_PREINSTALL_COMMANDS "
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\launch.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\plugins\\\\launcher\\\\local.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\source.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\solvers.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\geometry\\\\copy.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\model\\\\geometry\\\\copy.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\geometry\\\\plot.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\_resources.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\_resources.pyside.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\script\\\\editor.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\model\\\\script.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\script\\\\pycode.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\material_plot.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\geometry\\\\section.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\grids\\\\section.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\model\\\\grids\\\\section.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\base.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\model\\\\base.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\model\\\\grids\\\\grid.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\utils\\\\gui.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\controller\\\\script.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\pyeditor.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\external\\\\pyqtfrontend.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\external\\\\highlighter\\\\js.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\resources.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\main.py*\\\"
        Delete \\\"$INSTDIR\\\\lib\\\\plask\\\\gui\\\\widgets\\\    extedit.py*\\\"
    ") #HACK Temporary fix to remove files from old versions
endif()

# Settings for DEB and RPM generator
if(UNIX)
    set(CPACK_GENERATOR "DEB;RPM")

    if(NOT DEFINED CPACK_PACKAGE_NAME)
        set(CPACK_PACKAGE_NAME plask)
    endif()

    file(WRITE ${CMAKE_BINARY_DIR}/postinst "#!/bin/sh\nset -e\n${PYTHON_EXECUTABLE} -m compileall ${CMAKE_INSTALL_PREFIX}/lib/plask\n/sbin/ldconfig\n")
    file(WRITE ${CMAKE_BINARY_DIR}/prerm "#!/bin/sh\nset -e\nfind ${CMAKE_INSTALL_PREFIX}/lib/plask -type f -name \"*.pyc\" -delete\nexit 0\n")
    file(WRITE ${CMAKE_BINARY_DIR}/rpmpostin "#!/bin/sh\nset -e\n${PYTHON_EXECUTABLE} -m compileall \$RPM_INSTALL_PREFIX/lib/plask\n/sbin/ldconfig\n")
    file(WRITE ${CMAKE_BINARY_DIR}/rpmpreun "#!/bin/sh\nset -e\nfind \$RPM_INSTALL_PREFIX/lib/plask -type f -name \"*.pyc\" -delete\nexit 0\n")
    file(WRITE ${CMAKE_BINARY_DIR}/rpmpostun "#!/bin/sh\nset -e\n/sbin/ldconfig\n")

    execute_process(COMMAND chmod 0755 ${CMAKE_BINARY_DIR}/postinst ${CMAKE_BINARY_DIR}/prerm)

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; v=${Boost_VERSION}; sys.stdout.write('.'.join(str(s) for s in [v//100000,v//100%1000,v//10%10]))"
                    OUTPUT_VARIABLE boostver)

    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Maciej Dems <maciej.dems@p.lodz.pl>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
    file(READ ${CMAKE_SOURCE_DIR}/ABOUT CPACK_DEBIAN_PACKAGE_DESCRIPTION)
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Photonic LAser Simulation Kit\n${CPACK_DEBIAN_PACKAGE_DESCRIPTION}")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libexpat1, libboost-python${boostver}, libboost-system${boostver}, libboost-filesystem${boostver}, libblas.so.3, liblapack.so.3")
    if (${PYTHON_VERSION_STRING} VERSION_LESS 3.0.0)
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, python (>= 2.${PYTHON_VERSION_MINOR}.0), python-numpy")
        if(BUILD_GUI)
            set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, python-pyside | python-qt4")
        endif()
        set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "ipython (>= 0.13), python-matplotlib, python-h5py, python-scipy")
    else()
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, python3 (>= 3.${PYTHON_VERSION_MINOR}.0), python3-numpy")
        if(BUILD_GUI)
            set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, python3-pyside | python3-qt4")
        endif()
        set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "ipython3 (>= 0.13), python3-matplotlib, python3-h5py, python3-scipy")
    endif()
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/postinst;${CMAKE_BINARY_DIR}/prerm")
    # set(CPACK_DEB_COMPONENT_INSTALL ON) # no postinst/prerm scripts are added if this is turned on
    set(CPACK_RPM_PACKAGE_GROUP Applications/Engineering)
    # set(CPACK_RPM_PACKAGE_REQUIRES "expat boost-python boost-system boost-filesystem openblas lapack scipy python-matplotlib h5py")
    set(CPACK_RPM_PACKAGE_REQUIRES "scipy python-matplotlib h5py")
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST "/usr;/usr/bin;/usr/include;/usr/lib;/usr/share;/usr/share/applications;/usr/share/doc;/usr/share/icons;/usr/share/icons/hicolor;/usr/share/icons/hicolor/16x16;/usr/share/icons/hicolor/16x16/apps;/usr/share/icons/hicolor/16x16/mimetypes;/usr/share/icons/hicolor/24x24;/usr/share/icons/hicolor/24x24/apps;/usr/share/icons/hicolor/24x24/mimetypes;/usr/share/icons/hicolor/32x32;/usr/share/icons/hicolor/32x32/apps;/usr/share/icons/hicolor/32x32/mimetypes;/usr/share/icons/hicolor/48x48;/usr/share/icons/hicolor/48x48/apps;/usr/share/icons/hicolor/48x48/mimetypes;/usr/share/icons/hicolor/scalable;/usr/share/icons/hicolor/scalable/apps;/usr/share/icons/hicolor/scalable/mimetypes;/usr/share/icons;/usr/share/icons/breeze;/usr/share/icons/breeze/16x16;/usr/share/icons/breeze/16x16/apps;/usr/share/icons/breeze/16x16/mimetypes;/usr/share/icons/breeze/24x24;/usr/share/icons/breeze/24x24/apps;/usr/share/icons/breeze/24x24/mimetypes;/usr/share/icons/breeze/32x32;/usr/share/icons/breeze/32x32/apps;/usr/share/icons/breeze/32x32/mimetypes;/usr/share/icons/breeze/48x48;/usr/share/icons/breeze/48x48/apps;/usr/share/icons/breeze/48x48/mimetypes;/usr/share/icons/breeze/scalable;/usr/share/icons/breeze/scalable/apps;/usr/share/icons/breeze/scalable/mimetypes;/usr/share/mime;/usr/share/mime/packages;/usr/share/mime/packages")
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/rpmpostin")
    set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/rpmpreun")
    set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/rpmpostun")

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

# ----------===== config.h & version.h =====--------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/include/plask/config.h)
configure_file(${CMAKE_SOURCE_DIR}/version.h.in ${CMAKE_BINARY_DIR}/include/plask/version.h)
