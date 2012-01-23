# CMake file for project PLaSK

cmake_minimum_required (VERSION 2.8)
project (plask)

set(VERSION_MAJOR 0)            # MAJOR version (change to 1 on publication in Journal of Computer Physics)
set(VERSION_MINOR 0)            # MINOR version (change to 1 when anything works)
# set(VERSION_REVISION 1)       # REVISION number (leave undefined unless really needed)

# ---------- common options --------------------------------------------

#add_definitions(-Wall)         # all warnings
add_definitions(-std=c++0x)     # C++0x (C++11) standard

include_directories(.)          # external modules can include plask using <plask/....h>
include_directories(extlib)     # place for local external libraries
include_directories(${CMAKE_BINARY_DIR}/include ${CMAKE_BINARY_DIR}/include/plask) # for include config.h

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (NOT DEFINED VERSION_REVISION)
    execute_process(COMMAND svnversion -n WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE SVN_REVISION)
    if(DEFINED SVN_REVISION)
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

option(BUILD_SHARED_LIBS "Build shared library." OFF)

option (BUILD_DOCUMENTATION "Build doxygen documentation." OFF)

option (BUILD_PYTHON "Build and install Python interface to PLaSK (recommended)." ON)
cmake_dependent_option(BUILD_PYTHON_MODULE "Build importable python module." OFF "BUILD_PYTHON;BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(BUILD_PYTHON_EXECUTABLE "Build executable with embedded Python interpreter." ON "BUILD_PYTHON" OFF)

option (BUILD_TESTS "Build unit tests." ON)
cmake_dependent_option(RUN_TESTS "Run unit tests." ON "BUILD_TESTS" OFF)
option (PACK_EXE "Self-pack and strip executables." OFF)  # compress exe


# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
set(boost_components unit_test_framework)

if(BUILD_PYTHON)
    list(APPEND boost_components python)
endif()

find_package(Boost REQUIRED COMPONENTS ${boost_components})


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

#external libs which are compiled by us
file(GLOB_RECURSE plask_extlib_src FOLLOW_SYMLINKS extlib/irrxml/*.cpp)

file(GLOB_RECURSE python_plask_src FOLLOW_SYMLINKS python/plask/*.cpp)


# ---------- plask library --------------------------------------------

if (BUILD_SHARED_LIBS)
    add_library(libplask SHARED ${plask_src} ${plask_extlib_src} ${plask_headers})  # QtCreator needs have ${plask_headers} to show this files in IDE
else()
    add_library(libplask STATIC ${plask_src} ${plask_extlib_src} ${plask_headers})
endif()
set_target_properties(libplask PROPERTIES OUTPUT_NAME plask)

# ---------- Python --------------------------------------------

if (BUILD_PYTHON)
    # Find Python
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    find_package(numpy REQUIRED)

    # Require Python >= 2.6
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckPythonversion.py "from sys import version\nif version >= '2.6': print 'YES'\nelse: print 'NO'")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckPythonversion.py" OUTPUT_VARIABLE PYTHON_SUFFICIENT_VERSION)
    string(STRIP ${PYTHON_SUFFICIENT_VERSION} PYTHON_SUFFICIENT_VERSION)
    if (NOT PYTHON_SUFFICIENT_VERSION)
        message(FATAL_ERROR "Python 2.6 or newer required.")
    endif()

    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

    if (BUILD_PYTHON_MODULE)
        add_library(modplask MODULE ${python_plask_src})
        set_target_properties(modplask PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "lib/plask")
        if(WIN32)
            set_target_properties(modplask PROPERTIES SUFFIX ".pyd")
        endif()
        target_link_libraries(modplask ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask)
    endif(BUILD_PYTHON_MODULE)

    if (BUILD_PYTHON_EXECUTABLE)
        add_executable(plask python/exe_main.cpp ${python_plask_src})
		if (WIN32)
			# For some reason MinGW wants to create import library for executable,
			# which overwrites libplask.dll.a. With this hack we name this useless
			# stuff libplask.exe.dll.a
			set_target_properties(plask PROPERTIES OUTPUT_NAME "plask.exe" SUFFIX "")
		endif()
        target_link_libraries(plask ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask)
    endif(BUILD_PYTHON_EXECUTABLE)

endif(BUILD_PYTHON)

# ---------- Compress and strip executable (target) --------------------------------------------
macro(compress targetname)
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
endmacro(compress targetname)

if (PACK_EXE)
    find_package(SelfPackers)
    find_program(CMAKE_STRIP NAMES strip)
endif(PACK_EXE)

# ----------===== unit tests =====--------------------------------------------
if(BUILD_TESTS)

    enable_testing()
    if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        file(GLOB_RECURSE test-plask_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.h)
        add_executable(test-plask ${test-plask_src})
        set_target_properties(test-plask PROPERTIES OUTPUT_NAME test_plask)
        target_link_libraries(test-plask ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} libplask)
        set(plask_test_depends "DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask")
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

        add_library(plasktest MODULE ${python_tests_module_src})
        set_target_properties(plasktest PROPERTIES PREFIX "")
        if (BUILD_PYTHON_MODULE OR WIN32)
            target_link_libraries(plasktest ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} libplask)
            if(WIN32)
                set_target_properties(plasktest PROPERTIES SUFFIX ".pyd")
            endif()
        else()
            target_link_libraries(plasktest ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
        endif()

        if(WIN32)
            set(PATH $ENV{PATH})
            string(REPLACE ";" "\;" PATH "${PATH}")
        endif()

        file(GLOB_RECURSE python_tests FOLLOW_SYMLINKS tests/python/*.py)
        foreach(test_file ${python_tests})
            get_filename_component(test ${test_file} NAME_WE)
            if(BUILD_PYTHON_EXECUTABLE)
                add_test(NAME python-builtin/${test}
                         COMMAND plask ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/main ${test_file})
                if(WIN32)
                    set_tests_properties(python-builtin/${test} PROPERTIES
                                         ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}\\python\;${CMAKE_BINARY_DIR}\\lib;PYTHONDONTWRITEBYTECODE=YES;PATH=${PATH}\;${CMAKE_BINARY_DIR}\\bin")
                else()
                    set_tests_properties(python-builtin/${test} PROPERTIES
                                         ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_BINARY_DIR}/lib;PYTHONDONTWRITEBYTECODE=YES")
                endif()
            endif()
            if(BUILD_PYTHON_MODULE)
                add_test(NAME python-module/${test}
                         COMMAND "${PYTHON_EXECUTABLE}" -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/main ${test_file})
                if(WIN32)
                    set_tests_properties(python-module/${test} PROPERTIES
                                         ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}\\python\;${CMAKE_BINARY_DIR}\\lib\\plask\;${CMAKE_BINARY_DIR}\\lib;PYTHONDONTWRITEBYTECODE=YES;PATH=${PATH}\;${CMAKE_BINARY_DIR}\\bin")
                else()
                    set_tests_properties(python-module/${test} PROPERTIES
                                         ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_BINARY_DIR}/lib/plask:${CMAKE_BINARY_DIR}/lib;PYTHONDONTWRITEBYTECODE=YES")
                endif()
            endif()
        endforeach()
    endif(BUILD_PYTHON)

    if (RUN_TESTS)
        add_custom_target(tests ALL ${CMAKE_CTEST_COMMAND} --output-on-failure ${plask_test_depends})
    else()
        add_custom_target(tests ${CMAKE_CTEST_COMMAND} --output-on-failure ${plask_test_depends})
    endif()

endif(BUILD_TESTS)

# ----------===== documentation (api) =====--------------------------------------------
find_package(Doxygen)

if(BUILD_DOCUMENTATION)
    if(DOXYGEN_FOUND)
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
if(BUILD_SHARED_LIBS)
    if(WIN32)
        install(TARGETS libplask RUNTIME DESTINATION bin COMPONENT core
                                 LIBRARY DESTINATION lib COMPONENT core
                                 ARCHIVE DESTINATION lib COMPONENT dev)
    else()
        install(TARGETS libplask LIBRARY DESTINATION lib COMPONENT core)
    endif()
else()
    install(TARGETS libplask ARCHIVE DESTINATION lib COMPONENT dev)
endif()

install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN ".*" EXCLUDE)
install(FILES ${CMAKE_BINARY_DIR}/include/plask/config.h DESTINATION include/plask COMPONENT dev)

if(BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc DESTINATION share COMPONENT dev-doc
            PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
endif()

if(BUILD_PYTHON_EXECUTABLE)
    install(TARGETS plask RUNTIME DESTINATION bin COMPONENT core)
endif()

# TODO: install python modules

# ----------===== packaging =====--------------------------------------------

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Photonic LAser Simulation Kit")
set(CPACK_PACKAGE_VENDOR "Photonics Group, Technical University of Lodz")
#set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README)

set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_REVISION})

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_FILE_NAME "plask" CACHE STRING "deb package basename")
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

# ----------===== Debug helpers =====--------------------------------------------
if (BUILD_PYTHON_EXECUTABLE)
    if(NOT WIN32)
        file(WRITE ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp/plask
            "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_BINARY_DIR}/lib PYTHONDONTWRITEBYTECODE=YES ${CMAKE_BINARY_DIR}/bin/plask $@\n")
        file(COPY ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp/plask DESTINATION ${CMAKE_BINARY_DIR}
            FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE OWNER_WRITE)
    else()
        file(WRITE ${CMAKE_BINARY_DIR}/plask.bat
            "set PYTHONPATH=${CMAKE_SOURCE_DIR}\\python\;${CMAKE_BINARY_DIR}\\lib\nset PYTHONDONTWRITEBYTECODE=YES\n${CMAKE_BINARY_DIR}\\bin\\plask.exe %*")
    endif()
endif()

# ----------===== config.h =====--------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/include/plask/config.h)
