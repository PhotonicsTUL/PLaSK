# CMake file for project PLaSK

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0017 OLD)   # ensure our CMake modules are loaded first

project(plask)

set(VERSION_MAJOR 0)            # MAJOR version (change to 1 on publication in Journal of Computer Physics)
set(VERSION_MINOR 2)            # MINOR version (bump up by one on visible feature change)
# set(VERSION_REVISION 1)       # REVISION number (leave undefined unless really needed, it's normally set to svn revision)

# ---------- Common options --------------------------------------------

if(MINGW)
    set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-narrowing -Wno-sign-compare -Wno-unknown-pragmas -Wno-ignored-qualifiers")

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

option(BUILD_DEVEL_DOC "Build doxygen documentation." OFF)

option(BUILD_PYTHON "Build and install Python interface to PLaSK (recommended)." ON)
cmake_dependent_option(BUILD_PYTHON_MODULE "Build Python interface as external module." OFF "BUILD_PYTHON" OFF)
cmake_dependent_option(BUILD_GUI "Build and install GUI." ON "BUILD_PYTHON" OFF)

option(BUILD_TESTING "Build unit tests." ON)
cmake_dependent_option(RUN_TESTS "Run unit tests." ON "BUILD_TESTING" OFF)

option(USE_OMP "Use OpenMP" ON)

option(PRINT_STACKTRACE_ON_EXCEPTION "Print stack-trace on stderr when plask::Exception is throwed (works only in debug mode)." ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    option(USE_GLIBCXX_PARALLEL "use the libstdc++ parallel mode (experimental in libstdc++v3)" OFF)
endif(CMAKE_COMPILER_IS_GNUCXX)

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

if(USE_OMP OR USE_GLIBCXX_PARALLEL) # compile with OpenMP
    find_package(OpenMP 3.0)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        if(USE_GLIBCXX_PARALLEL)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            add_definitions("-D_GLIBCXX_PARALLEL")
        endif(USE_GLIBCXX_PARALLEL)
    endif()
endif()

find_package(EXPAT REQUIRED)
include_directories(${EXPAT_INCLUDE_DIR})

# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
set(boost_components system filesystem) #thread
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

if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes") # get rid of annoying warnings in Boost
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
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	if(win32)
		set(PRINT_STACKTRACE_LIBRARY "dbghelp")
	else(win32)
		find_package(DW)
		if(DW_FOUND)
			set(PRINT_STACKTRACE_LIBRARY ${DW_LIBRARIES})
			include_directories(${DW_INCLUDE_DIRS})
			set(BACKWARD_HAS_DW 1)
		else(DW_FOUND)  # try find bfd library if dw is not found, see https://github.com/bombela/backward-cpp/blob/master/backward.cpp
			find_package(BFD)
			if(BFD_FOUND)
				set(PRINT_STACKTRACE_LIBRARY ${BFD_LIBRARY})
				set(BACKWARD_HAS_BFD 1)
			else(BFD_FOUND)
				set(PRINT_STACKTRACE_LIBRARY "")
			endif(BFD_FOUND)
		endif(DW_FOUND)
	endif(win32)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(PRINT_STACKTRACE_ON_EXCEPTION)


# ----------===== Build and Install =====--------------------------------------------

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h plask/*.hpp solvers/providers/*.h solvers/providers/*.hpp)

file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp solvers/providers/*.cpp)

# Python sources and files
file(GLOB_RECURSE python_plask_src FOLLOW_SYMLINKS python/plask/*.cpp python/plask/*.h)
file(GLOB_RECURSE python_plask_py python/plask/*.py)

if(WIN32)
    set(manifest "1 24 \"${CMAKE_SOURCE_DIR}/toolset/mingw/msvcr90.manifest\"\n")
else()
    set(manifest "")
endif()

# ---------- External libraries built by us -----------------------------------------

# CAMOS
add_subdirectory(extlib/camos)
add_subdirectory(extlib/fftpacx)


# ---------- PLaSK library --------------------------------------------

add_library(libplask SHARED ${plask_src} ${plask_headers})
#add_library(libplask SHARED ${plask_src})
target_link_libraries(libplask ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${EXPAT_LIBRARIES} ${PRINT_STACKTRACE_LIBRARY}) # ${Boost_THREAD_LIBRARY}
if(WIN32)
    target_link_libraries(libplask kernel32)
    install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core
                             ARCHIVE DESTINATION lib COMPONENT dev)
else()
    target_link_libraries(libplask dl)
    install(TARGETS libplask LIBRARY DESTINATION lib COMPONENT core)
endif()
set_target_properties(libplask PROPERTIES OUTPUT_NAME plask COMPILE_FLAGS -DPLASK_EXPORTS)

#if (NOT WIN32) #can be usefull if more threads libraries will be used
#    target_link_libraries(sdsolver "pthread")
#endif()

install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
install(DIRECTORY solvers/providers DESTINATION include/plask COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.h DESTINATION include/plask COMPONENT dev)

set(PLaSK_LIBRARIES libplask)

# ---------- Default materials --------------------------------------------

add_subdirectory(materials)

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
        install(TARGETS libplask_python RUNTIME DESTINATION bin COMPONENT core
                                        ARCHIVE DESTINATION lib COMPONENT dev)
    else()
        install(TARGETS libplask_python LIBRARY DESTINATION lib COMPONENT core)
    endif()

    if(WIN32)
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

    install(FILES ${python_plask_py_copied} DESTINATION lib/plask/python/plask COMPONENT core)
    install(TARGETS plask RUNTIME DESTINATION bin COMPONENT core)
    install(DIRECTORY python/plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "python*.h" PATTERN "python*.hpp" PATTERN ".*" EXCLUDE)

    # GUI
    if(BUILD_GUI)
        if(WIN32)
            file(WRITE ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.rc "${manifest}2 ICON \"${CMAKE_SOURCE_DIR}/utils/qtplask.ico\"\n")
            add_executable(plaskgui python/exe_gui.cpp ${CMAKE_BINARY_DIR}/CMakeFiles/plaskgui.rc)
        else()
            add_executable(plaskgui python/exe_gui.cpp)
        endif()
        target_link_libraries(plaskgui ${Boost_PYTHON_LIBRARIES} ${PYTHON_LIBRARIES} libplask_python libplask)
        if(DEFINED no_strict_aliasing_flag OR DEFINED EXE_FLAGS)
            set_target_properties(plaskgui PROPERTIES COMPILE_FLAGS "${no_strict_aliasing_flags} ${EXE_FLAGS}") # necessary for all code which includes "Python.h"
        endif()
        if(WIN32)
            set_target_properties(plaskgui PROPERTIES LINK_FLAGS "-mwindows ${EXE_FLAGS}")
        elseif(DEFINED EXE_FLAGS)
            set_target_properties(plaskgui PROPERTIES LINK_FLAGS "${EXE_FLAGS}")
        endif()

        #TODO compile pyc files
        install(DIRECTORY gui DESTINATION lib/plask COMPONENT GUI
                PATTERN .* EXCLUDE
                PATTERN *.xpl EXCLUDE
                PATTERN *.qrc EXCLUDE
                PATTERN *.pyc EXCLUDE
                PATTERN gui/qt.py* EXCLUDE
                PATTERN gui/_resources.py* EXCLUDE
        )
        install(FILES gui/qt.pyside.py DESTINATION lib/plask/gui COMPONENT GUI RENAME qt.py)
        install(FILES gui/_resources.pyside.py DESTINATION lib/plask/gui COMPONENT GUI RENAME _resources.py)
        install(TARGETS plaskgui DESTINATION bin COMPONENT GUI)

        if(UNIX)
            install(FILES utils/application-x-plask.xml DESTINATION share/mime/packages COMPONENT GUI)
            install(FILES utils/plaskgui.desktop DESTINATION share/applications COMPONENT GUI)
        endif()
    endif()

endif(BUILD_PYTHON)


# ---------- Macro for testing --------------------------------------------
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

# Scan solver categories for solvers.lst and process them
file(GLOB_RECURSE lst_lists RELATIVE ${CMAKE_SOURCE_DIR} solvers/*solvers.lst)
foreach(lst ${lst_lists})
    add_custom_command(OUTPUT ${PLASK_PATH}/${lst}
                       COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_SOURCE_DIR}/${lst} ${PLASK_PATH}/${lst}
                       DEPENDS ${CMAKE_SOURCE_DIR}/${lst})
    list(APPEND solvers_lst_files ${PLASK_PATH}/${lst})
    get_filename_component(lst_dir ${lst} PATH)
    install(FILES ${PLASK_PATH}/${lst} DESTINATION lib/plask/${lst_dir} COMPONENT solvers)
    if(BUILD_PYTHON)
        add_custom_command(OUTPUT ${PLASK_PATH}/${lst_dir}/__init__.py
                           COMMAND ${PYTHON_EXECUTABLE} ARGS ${CMAKE_SOURCE_DIR}/toolset/solvers_lst.py ${PLASK_PATH} ${lst_dir}
                           DEPENDS ${PLASK_PATH}/${lst})
        list(APPEND solvers_lst_files ${PLASK_PATH}/${lst_dir}/__init__.py)
    endif()
endforeach()
add_custom_target(solvers-lst ALL DEPENDS ${solvers_lst_files})

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

# ----------===== Unit tests =====--------------------------------------------
if(BUILD_TESTING)

    set(plask_test_depends ${material_targets} solvers-lst) # we need materials for testing

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
            target_link_libraries(plask-test-python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask)
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

    if (RUN_TESTS)
        add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 60 DEPENDS ${plask_test_depends} ${solvers_test_targets})
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
        install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION ${doc_dest}/devel COMPONENT dev-doc
                PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
    else()
        message("Doxygen not found. Building without documentation.")
    endif()
else()
    if(DOXYGEN_FOUND)
        add_custom_target(dox ${DOXYGEN} ${CMAKE_BINARY_DIR}/Doxyfile DEPENDS ${plask_headers} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()

file(GLOB examples ${CMAKE_SOURCE_DIR}/doc/*.xpl ${CMAKE_SOURCE_DIR}/doc/*.py)
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

    compile_python_script(qtplask ${CMAKE_SOURCE_DIR}/utils/qtplask.ico)
    compile_python_script(dan2xpl)
    install(FILES utils/dan2xpl.py DESTINATION bin COMPONENT utils)
    install(FILES utils/xpl2dan.py DESTINATION bin COMPONENT utils)

else()

    install(FILES utils/qtplask.py DESTINATION bin PERMISSIONS ${script_permissions} COMPONENT utils RENAME qtplask)
    install(FILES utils/dan2xpl.py DESTINATION bin PERMISSIONS ${script_permissions} COMPONENT utils RENAME dan2xpl)
    install(FILES utils/xpl2dan.py DESTINATION bin PERMISSIONS ${script_permissions} COMPONENT utils RENAME xpl2dan)

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

set(CPACK_COMPONENTS_ALL "core;solvers;GUI;utils;doc")
set(CPACK_COMPONENT_CORE_DISPLAY_NAME "PLaSK core")
set(CPACK_COMPONENT_CORE_REQUIRED ON)
set(CPACK_COMPONENT_DOC_DISPLAY_NAME "User manual and examples")
set(CPACK_COMPONENT_SOLVERS_DISPLAY_NAME "Computational solvers")
set(CPACK_COMPONENT_GUI_DISPLAY_NAME "Graphical User Interface")
set(CPACK_COMPONENT_UTILS_DISPLAY_NAME "Command-line utilities (e.g. dan2xpl)")

# Settings for NSIS generator
if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    if(DLLS_DIR)
        file(GLOB dlls "${DLLS_DIR}/*")
        install(FILES ${dlls} DESTINATION bin COMPONENT core)
    endif()
    set(CPACK_PACKAGE_NAME "PLaSK")
    list(APPEND CPACK_PACKAGE_EXECUTABLES "plaskgui;PLaSK GUI")
    list(APPEND CPACK_PACKAGE_EXECUTABLES "qtplask;PLaSK Runner")
    list(APPEND CPACK_PACKAGE_EXECUTABLES "plask;PLaSK Console")
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
    set(CPACK_NSIS_COMPRESSOR "lzma\n\n  ;Default start menu folder\n  !define MUI_STARTMENUPAGE_DEFAULTFOLDER \\\"PLaSK\\\"") # very ugly hack!
endif()

# Settings for DEB generator
if(UNIX)
    set(CPACK_GENERATOR "DEB")
    file(WRITE ${CMAKE_BINARY_DIR}/postinst "#!/bin/sh\nset -e\n${PYTHON_EXECUTABLE} -m compileall ${CMAKE_INSTALL_PREFIX}/lib/plask\n")
    file(WRITE ${CMAKE_BINARY_DIR}/prerm "#!/bin/sh\nset -e\nfind ${CMAKE_INSTALL_PREFIX}/lib/plask -type f -name \"*.pyc\" -delete\nexit 0\n")

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; v=${Boost_VERSION}; sys.stdout.write('.'.join(str(s) for s in [v/100000,v/100%1000,v/10%10]))"
                    OUTPUT_VARIABLE boostver)

    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Maciej Dems <maciej.dems@p.lodz.pl>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
    file(READ ${CMAKE_SOURCE_DIR}/ABOUT CPACK_DEBIAN_PACKAGE_DESCRIPTION)
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Photonic LAser Simulation Kit\n${CPACK_DEBIAN_PACKAGE_DESCRIPTION}")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libexpat1, libboost-python${boostver}, libboost-system${boostver}, libboost-filesystem${boostver}, libblas.so.3, liblapack.so.3, python-numpy")
    set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "ipython (>= 0.13), python-matplotlib, python-h5py")
    set(CPACK_DEBIAN_PACKAGE_SUGGESTS "ipython-qtconsole (>= 0.13)")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/postinst;${CMAKE_BINARY_DIR}/prerm")
    # set(CPACK_DEB_COMPONENT_INSTALL ON) # no postinst/prerm scripts are added if this is turned on
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
