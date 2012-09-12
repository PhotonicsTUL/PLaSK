#
# Helpers for cmake files in solvers subdirectories
#

cmake_minimum_required(VERSION 2.8)

# Obtain relative path name
string(REPLACE "${CMAKE_SOURCE_DIR}/solvers/" "" SOLVER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Obtain solver library name (it can actually contain several solvers)
string(REGEX REPLACE "/|_" "" TARGET_NAME ${SOLVER_DIR})

# Add prefix to target name
set(TARGET_NAME solver-${TARGET_NAME})

# Obtain default canonical solver name
get_filename_component(SOLVER_NAME ${SOLVER_DIR} NAME)

# Construct solver library name (category_solvername)
get_filename_component(SOLVER_LIB_NAME ${SOLVER_DIR} PATH)
set(SOLVER_LIB_NAME "${SOLVER_LIB_NAME}_${SOLVER_NAME}")

# Obtain intermediate path list and to create necessary __init__.py files to mark packages
get_filename_component(SOLVER_PATH ${SOLVER_DIR} PATH)
if(NOT SOLVER_PATH STREQUAL "")
    string(REPLACE "/" ";" SOLVER_DIRECTORIES ${SOLVER_PATH})
endif()
set(PLASK_SOLVER_PATH "${PLASK_PATH}/solvers/${SOLVER_PATH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH})

set(SOLVER_INSTALL_PATH lib/plask/solvers/${SOLVER_PATH})

# Automatically set sources from the current directory
file(GLOB solver_src *.cpp *.hpp *.h)
file(GLOB interface_src python/*.cpp)


# Turn-off strict aliasing for Python code
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GNUCXX_VERSION)
    if ((GNUCXX_VERSION VERSION_GREATER 4.7 OR GNUCXX_VERSION VERSION_EQUAL 4.7))
        set(no_strict_aliasing_flag "-fno-strict-aliasing")
    endif()
endif()


# This is macro that sets all the targets automagically
macro(make_default)

    # Build solver library
    add_library(${TARGET_NAME} SHARED ${solver_src})
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${SOLVER_LIB_NAME})
    target_link_libraries(${TARGET_NAME} libplask ${SOLVER_LINK_LIBRARIES})
    include_directories(${SOLVER_INCLUDE_DIRECTORIES})
    if (DEFINED SOLVER_LINK_FLAGS)
        set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS ${SOLVER_LINK_FLAGS})
    endif()
    if (DEFINED SOLVER_COMPILE_FLAGS)
        set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS ${SOLVER_COMPILE_FLAGS})
    endif()

    if(WIN32)
        install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers
                                       ARCHIVE DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers-dev)
    else()
        install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
    endif()

    if(BUILD_PYTHON)
        set(PYTHON_TARGET_NAME ${TARGET_NAME}-python)
        # Make package hierarchy
        set(curr_path "${PLASK_PATH}/solvers")
        foreach(dir ${SOLVER_DIRECTORIES})
            set(curr_path "${curr_path}/${dir}")
            file(MAKE_DIRECTORY ${curr_path})
            file(WRITE "${curr_path}/__init__.py" "")
        endforeach()
        # Build Python interface
        if(WIN32)
            add_library(${PYTHON_TARGET_NAME} SHARED ${interface_src})
        else()
            add_library(${PYTHON_TARGET_NAME} MODULE ${interface_src})
        endif()
        target_link_libraries(${PYTHON_TARGET_NAME} ${TARGET_NAME} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
        set_target_properties(${PYTHON_TARGET_NAME} PROPERTIES
                              LIBRARY_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH}
                              PREFIX "")
        if (DEFINED no_strict_aliasing_flag)
            set_target_properties(plask PROPERTIES COMPILE_FLAGS ${no_strict_aliasing_flag}) # necessary for all code which includes "Python.h"
        endif()
        if(WIN32)
            set_target_properties(${PYTHON_TARGET_NAME} PROPERTIES
                                  RUNTIME_OUTPUT_DIRECTORY "${PLASK_SOLVER_PATH}"
                                  SUFFIX ".pyd")
            install(TARGETS ${PYTHON_TARGET_NAME} RUNTIME DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers
                                                  LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers-dev)
        else()
            set_target_properties(${PYTHON_TARGET_NAME} PROPERTIES
                                  OUTPUT_NAME ${SOLVER_NAME})
            install(TARGETS ${PYTHON_TARGET_NAME} LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
        endif()
    endif()

    if(BUILD_TESTING)
        add_custom_target(${TARGET_NAME}-test DEPENDS ${TARGET_NAME} ${PYTHON_TARGET_NAME} ${SOLVER_TEST_DEPENDS})
    endif()

endmacro()
