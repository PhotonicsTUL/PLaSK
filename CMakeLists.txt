# CMake file for project PLaSK

cmake_minimum_required (VERSION 2.8)
project (plask)

set(VERSION_MAJOR 0)            # MAJOR version (change to 1 on publication in Journal of Computer PHysics)
set(VERSION_MINOR 0)            # MINOR version (change to 1 when anything works)


# ---------- common options --------------------------------------------

add_definitions(-Wall)          # all warnings
add_definitions(-std=c++0x)     # C++0x (C++11) standard

include_directories(.)          # external modules can include plask using <plask/....h>
include_directories(extlibs)    # place for local external libraries

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

execute_process(COMMAND svnversion -n WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE VERSION_REVISION)
string(REGEX REPLACE "[0-9]+:|[A-Za-z]+" "" VERSION_REVISION ${VERSION_REVISION})
if(${VERSION_REVISION})
    set(VERSION_REVISION "r${VERSION_REVISION}")
else(${VERSION_REVISION})
    set(VERSION_REVISION 0)
endif(${VERSION_REVISION})
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}")
message(STATUS "PLaSK version ${VERSION}")

# ---------- configurable options --------------------------------------------
option (BUILD_SHARED_LIBS "build also shared library" OFF)
option (BUILD_PYTHON_MODULE "build and install Python module plask" ON)
option(BUILD_DOCUMENTATION "build doxygen documentation" OFF)
option (MAKE_TESTS "build and run unit tests" ON)
option(PACK_EXE "self-pack and strip executables" OFF)  #compress exe


# ---------- Boost components --------------------------------------------
# Add any boost component here that any part of the project can use ever
find_package(Boost COMPONENTS unit_test_framework python)


## ---------- Fortran name mangling --------------------------------------------
#enable_language(Fortran OPTIONAL)
#if (CMAKE_Fortran_COMPILER_WORKS)
#   include(FortranCInterface)
#   FortranCInterface_HEADER("F77.h" MACRO_NAMESPACE "F77_")
#else ()
#   message(SEND_ERROR "Fortran compiler not found! Unable to automatically determine the name mangling scheme.")
#endif()


# ----------===== plask library =====--------------------------------------------
# creating configuration file for plask library
#CONFIGURE_FILE(plask/config.h.cmake plask/config.h)

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h)
file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp plask/*.cxx plask/*.h)


if (BUILD_SHARED_LIBS)
    add_library(plask SHARED ${plask_src})
    set_target_properties(plask PROPERTIES COMPILE_FLAGS -fPIC)
else()
    add_library(plask STATIC ${plask_src})
endif()

# ----------===== Python =====--------------------------------------------

if (BUILD_PYTHON_MODULE)
    # Find Python
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    find_package(numpy REQUIRED)

    # --- Here build the modules ---


endif(BUILD_PYTHON_MODULE)

# ---------- compress and strip executable (target) --------------------------------------------
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

if (PACK_EXE)
    find_package(SelfPackers)
    find_program(CMAKE_STRIP NAMES strip)
endif(PACK_EXE)

# ----------===== unit tests =====--------------------------------------------
if(MAKE_TESTS)

    enable_testing()
    if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        file(GLOB_RECURSE test_plask_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.cxx tests/plask/*.h)
        add_executable(test_plask ${test_plask_src})
        target_link_libraries(test_plask ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} plask)
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
find_package(Doxygen)

if(BUILD_DOCUMENTATION)
    if(DOXYGEN_FOUND)
        add_custom_target(doc ALL ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    else()
        message("Doxygen not found. Building without documentation.")
    endif()
else()
    if(DOXYGEN_FOUND)
        add_custom_target(doc ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    endif()
endif()

# ----------===== install =====--------------------------------------------
if(BUILD_SHARED_LIBS)
    install(TARGETS plask LIBRARY DESTINATION lib COMPONENT core)
else()
    install(TARGETS plask ARCHIVE DESTINATION lib COMPONENT dev)
endif()

install(DIRECTORY plask DESTINATION include COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN ".*" EXCLUDE)

if(BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc DESTINATION share COMPONENT dev-doc
            PATTERN "html/*.md5" EXCLUDE PATTERN "html/*.map" EXCLUDE)
endif()

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
"^${PROJECT_SOURCE_DIR}/build/"
"^${PROJECT_SOURCE_DIR}/debug/"
"^${PROJECT_SOURCE_DIR}/release/"
)
include(CPack)

