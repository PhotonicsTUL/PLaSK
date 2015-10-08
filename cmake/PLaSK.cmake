#
# Helpers for cmake files in solvers subdirectories
#

cmake_minimum_required(VERSION 2.8)

# Obtain relative path name
string(REPLACE "${CMAKE_SOURCE_DIR}/solvers/" "" SOLVER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Obtain solver library name (it can actually contain several solvers)
string(REGEX REPLACE "/|_" "" SOLVER_LIBRARY ${SOLVER_DIR})

# Add prefix to target name
set(SOLVER_LIBRARY solver-${SOLVER_LIBRARY})

# Obtain default canonical solver name
get_filename_component(SOLVER_NAME ${SOLVER_DIR} NAME)

# Construct solver library name (category_solvername)
get_filename_component(SOLVER_CATEGORY_NAME ${SOLVER_DIR} PATH)
set(SOLVER_LIB_NAME "${SOLVER_CATEGORY_NAME}_${SOLVER_NAME}")

# Obtain intermediate path list and to create necessary __init__.py files to mark packages
get_filename_component(SOLVER_PATH ${SOLVER_DIR} PATH)
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
    add_library(${SOLVER_LIBRARY} SHARED ${solver_src})
    set_target_properties(${SOLVER_LIBRARY} PROPERTIES OUTPUT_NAME ${SOLVER_LIB_NAME})
    target_link_libraries(${SOLVER_LIBRARY} libplask ${SOLVER_LINK_LIBRARIES})
    include_directories(${SOLVER_INCLUDE_DIRECTORIES})
    if (DEFINED SOLVER_LINK_FLAGS)
        set_target_properties(${SOLVER_LIBRARY} PROPERTIES LINK_FLAGS ${SOLVER_LINK_FLAGS})
    endif()
    if (DEFINED SOLVER_COMPILE_FLAGS)
        set_target_properties(${SOLVER_LIBRARY} PROPERTIES COMPILE_FLAGS "${SOLVER_COMPILE_FLAGS} -DPLASK_SOLVERS_EXPORTS")
    else()
        set_target_properties(${SOLVER_LIBRARY} PROPERTIES COMPILE_FLAGS -DPLASK_SOLVERS_EXPORTS)
    endif()

    if(WIN32)
        install(TARGETS ${SOLVER_LIBRARY} RUNTIME DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
    else()
        install(TARGETS ${SOLVER_LIBRARY} LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
    endif()

    if(BUILD_PYTHON)
        set(SOLVER_PYTHON_MODULE ${SOLVER_LIBRARY}-python)
        # Make package hierarchy
        set(curr_path "${PLASK_PATH}/solvers")
        set(install_path "lib/plask/solvers")
        if(WIN32)
            add_library(${SOLVER_PYTHON_MODULE} SHARED ${interface_src})
        else()
            add_library(${SOLVER_PYTHON_MODULE} MODULE ${interface_src})
        endif()
        target_link_libraries(${SOLVER_PYTHON_MODULE} ${SOLVER_LIBRARY} ${Boost_PYTHON_LIBRARIES} ${PYTHON_LIBRARIES} libplask_python ${SOLVER_PYTHON_LINK_LIBRARIES})
        set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES
                              LIBRARY_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH}
                              OUTPUT_NAME ${SOLVER_NAME}
                              INSTALL_RPATH "$ORIGIN"
                              PREFIX "")
        if (DEFINED no_strict_aliasing_flag)
            set_target_properties(plask PROPERTIES COMPILE_FLAGS ${no_strict_aliasing_flag}) # necessary for all code which includes "Python.h"
        endif()
        if(WIN32)
            set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES
                                  RUNTIME_OUTPUT_DIRECTORY "${PLASK_SOLVER_PATH}"
                                  SUFFIX ".pyd")
            install(TARGETS ${SOLVER_PYTHON_MODULE} RUNTIME DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers
                                                  LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers-dev)
        else()
            install(TARGETS ${SOLVER_PYTHON_MODULE} LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
        endif()
        if(BUILD_GUI)
            string(REPLACE "/" "." SOLVER_MODULE ${SOLVER_DIR})
            set(SOLVER_STUB ${CMAKE_BINARY_DIR}/share/plask/stubs/${SOLVER_DIR}.py)
            add_custom_command(OUTPUT ${SOLVER_STUB}
                               COMMAND plask -lwarning ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${SOLVER_MODULE}
                               DEPENDS ${SOLVER_PYTHON_MODULE} ${CMAKE_SOURCE_DIR}/toolset/makestub.py
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/share/plask/stubs
                              )
            install(FILES ${SOLVER_STUB} DESTINATION share/plask/stubs/${SOLVER_CATEGORY_NAME} COMPONENT gui)
            add_custom_target(${SOLVER_LIBRARY}-stub ALL DEPENDS ${SOLVER_LIBRARY} ${SOLVER_PYTHON_MODULE} ${SOLVER_STUB})
        endif()
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/solvers.xml)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/plask/solvers/${SOLVER_DIR}.xml
                            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/solvers.xml
                            COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/solvers.xml ${CMAKE_BINARY_DIR}/lib/plask/solvers/${SOLVER_DIR}.xml
                            )
        string(REPLACE "/" "_" SOLVER_MODULE ${SOLVER_DIR})
        add_custom_target(${SOLVER_LIBRARY}-xml ALL DEPENDS ${CMAKE_BINARY_DIR}/lib/plask/solvers/${SOLVER_DIR}.xml)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/solvers.xml DESTINATION lib/plask/solvers/${SOLVER_CATEGORY_NAME} RENAME ${SOLVER_NAME}.xml COMPONENT GUI)
    endif()

    if(BUILD_TESTING)
        add_custom_target(${SOLVER_LIBRARY}-test DEPENDS ${SOLVER_LIBRARY} ${SOLVER_PYTHON_MODULE} ${SOLVER_TEST_DEPENDS})
    endif()

endmacro()
