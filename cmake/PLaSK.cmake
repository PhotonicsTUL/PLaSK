#
# Helpers for cmake files in solvers subdirectories
#
cmake_minimum_required(VERSION 3.14)

if(POLICY CMP0046)
    cmake_policy(SET CMP0046 NEW)       # ensure add_dependencies raises error if target does not exist
endif()

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

# Obtain intermediate path to create necessary __init__.py files to mark packages
get_filename_component(SOLVER_PATH ${SOLVER_DIR} PATH)
set(PLASK_SOLVER_PATH "${PLASK_PATH}/solvers/${SOLVER_PATH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH})

foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # used by MSVC
    string(TOUPPER "${CONF}" CONF)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} ${PLASK_SOLVER_PATH})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} ${PLASK_SOLVER_PATH})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} ${PLASK_SOLVER_PATH})    # <- needed??
endforeach()

set(SOLVER_INSTALL_PATH lib/plask/solvers/${SOLVER_PATH})

# Automatically set sources from the current directory
file(GLOB solver_src *.cpp)
file(GLOB interface_src python/*.cpp)


# Turn-off strict aliasing for Python code
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GNUCXX_VERSION)
    if ((GNUCXX_VERSION VERSION_GREATER 4.7 OR GNUCXX_VERSION VERSION_EQUAL 4.7))
        set(no_strict_aliasing_flag "-fno-strict-aliasing")
    endif()
endif()

# This macro should be used to add all tests from solvers
macro(add_solver_test test_name test_file)
    if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
        string(FIND "${test_file}" "." _extidx)
        if(${_extidx} EQUAL -1)
            set(ext "")
        else()
            string(SUBSTRING "${test_file}" ${_extidx} -1 ext)
        endif()
    else()
        get_filename_component(ext ${test_file} LAST_EXT)
    endif()
    set(test solvers/${SOLVER_CATEGORY_NAME}/${SOLVER_NAME}/${test_name})
    if(("${ext}" STREQUAL ".py") OR ("${ext}" STREQUAL ".PY") OR ("${ext}" STREQUAL ".xpl") OR ("${ext}" STREQUAL ".XPL"))
        add_test(NAME ${test}
                COMMAND ${CMAKE_BINARY_DIR}/bin/plask -m runtest ${test_file})
    else()
        add_test(NAME ${test} COMMAND ${PLASK_SOLVER_PATH}/${test_file})
        if(WIN32)
            set_tests_properties(${test} PROPERTIES ENVIRONMENT "PATH=${plask_bin_path}\;${ENV_PATH}")
        endif()
        set(SOLVER_TEST_DEPENDS ${test_file})
    endif()
endmacro()



# This is macro that sets all the targets automagically
macro(make_default)

    if(NOT PLaSK_LIBRARIES)
        set(PLaSK_LIBRARIES plask)
    endif()

    if(NOT PLaSK_Python3_LIBRARIES)
        set(PLaSK_Python3_LIBRARIES plask_python)
    endif()

    # Build solver library
    add_library(${SOLVER_LIBRARY} SHARED ${solver_src})
    set_target_properties(${SOLVER_LIBRARY} PROPERTIES OUTPUT_NAME ${SOLVER_LIB_NAME})
    target_link_libraries(${SOLVER_LIBRARY} ${PLaSK_LIBRARIES} ${SOLVER_LINK_LIBRARIES})
    include_directories(${SOLVER_INCLUDE_DIRECTORIES})
    if (DEFINED SOLVER_LINK_FLAGS)
        set_target_properties(${SOLVER_LIBRARY} PROPERTIES LINK_FLAGS ${SOLVER_LINK_FLAGS})
    endif()
    if (DEFINED SOLVER_COMPILE_FLAGS)
        set_target_properties(${SOLVER_LIBRARY} PROPERTIES COMPILE_FLAGS "${SOLVER_COMPILE_FLAGS} -DPLASK_SOLVERS_EXPORTS")
    else()
        set_target_properties(${SOLVER_LIBRARY} PROPERTIES COMPILE_FLAGS -DPLASK_SOLVERS_EXPORTS)
    endif()

    if (NOT "${SOLVER_DEPENDS}" STREQUAL "")
        add_dependencies(${SOLVER_LIBRARY} ${SOLVER_DEPENDS})
    endif()

    if(WIN32)
        install(TARGETS ${SOLVER_LIBRARY} RUNTIME DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
    else()
        install(TARGETS ${SOLVER_LIBRARY} LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
    endif()

    if(BUILD_PYTHON AND interface_src)
        set(SOLVER_PYTHON_MODULE ${SOLVER_LIBRARY}-python)
        # Make package hierarchy
        set(curr_path "${PLASK_PATH}/solvers")
        set(install_path "lib/plask/solvers")
        if(WIN32)
            add_library(${SOLVER_PYTHON_MODULE} SHARED ${interface_src})
        else()
            add_library(${SOLVER_PYTHON_MODULE} MODULE ${interface_src})
        endif()
        target_link_libraries(${SOLVER_PYTHON_MODULE} ${SOLVER_LIBRARY} ${Boost_Python3} ${Python3_LIBRARIES} ${PLaSK_Python3_LIBRARIES} ${PLaSK_LIBRARIES} ${SOLVER_PYTHON_LINK_LIBRARIES})
        set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES
                              LIBRARY_OUTPUT_DIRECTORY ${PLASK_SOLVER_PATH}
                              OUTPUT_NAME ${SOLVER_NAME}
                              INSTALL_RPATH "$ORIGIN"
                              PREFIX "")
        foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # used by MSVC
            STRING(TOUPPER "${CONF}" CONF)
            set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES
                                  LIBRARY_OUTPUT_DIRECTORY_${CONF} ${PLASK_SOLVER_PATH})
        endforeach()
        if (DEFINED no_strict_aliasing_flag)
            set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES COMPILE_FLAGS ${no_strict_aliasing_flag}) # necessary for all code which includes "Python.h"
        endif()
        if(WIN32)
            set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES
                                  RUNTIME_OUTPUT_DIRECTORY "${PLASK_SOLVER_PATH}"
                                  SUFFIX ".pyd")
            foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # used by MSVC
                STRING(TOUPPER "${CONF}" CONF)
                set_target_properties(${SOLVER_PYTHON_MODULE} PROPERTIES
                                      RUNTIME_OUTPUT_DIRECTORY_${CONF} ${PLASK_SOLVER_PATH})
            endforeach()
            install(TARGETS ${SOLVER_PYTHON_MODULE} RUNTIME DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers
                                                    LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers-dev)
        else()
            install(TARGETS ${SOLVER_PYTHON_MODULE} LIBRARY DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT solvers)
        endif()
        if(BUILD_GUI)
            string(REPLACE "/" "." SOLVER_MODULE ${SOLVER_DIR})
            set(SOLVER_STUB ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/share/plask/stubs/${SOLVER_DIR}.py)
            add_custom_command(OUTPUT ${SOLVER_STUB}
                               COMMAND ${CMAKE_BINARY_DIR}/bin/plask -lwarning ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${SOLVER_MODULE}
                               DEPENDS ${plask_binary} ${SOLVER_PYTHON_MODULE} ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${PLASK_MATERIALS}
                               WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/share/plask/stubs
                              )
            install(FILES ${SOLVER_STUB} DESTINATION share/plask/stubs/${SOLVER_CATEGORY_NAME} COMPONENT gui)
            add_custom_target(${SOLVER_LIBRARY}-stub ALL DEPENDS ${SOLVER_LIBRARY} ${SOLVER_PYTHON_MODULE} ${SOLVER_STUB})
        endif()
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/solvers.yml)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR}.yml
                           DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/solvers.yml
                           COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/solvers.yml ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR}.yml
                          )
        add_custom_target(validate-${SOLVER_LIBRARY}-yml COMMAND ${CMAKE_SOURCE_DIR}/toolset/validate_solvers_yaml.py ${CMAKE_CURRENT_SOURCE_DIR}/solvers.yml)
        add_custom_target(${SOLVER_LIBRARY}-yml ALL DEPENDS ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR}.yml)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/solvers.yml DESTINATION lib/plask/solvers/${SOLVER_CATEGORY_NAME} RENAME ${SOLVER_NAME}.yml COMPONENT solvers)
    endif()

    if(BUILD_TESTING)
        add_custom_target(${SOLVER_LIBRARY}-test DEPENDS ${SOLVER_LIBRARY} ${SOLVER_PYTHON_MODULE} ${SOLVER_TEST_DEPENDS})
    endif()

endmacro()


# This is macro that sets all python targets automagically
macro(make_pure_python)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR})
    file(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py solvers.yml)
    foreach(file ${sources})
        list(APPEND python_targets ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR}/${file})
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR}/${file}
                           COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/plask/solvers/${SOLVER_DIR}/${file}
                           DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
        install(FILES ${file} DESTINATION lib/plask/solvers/${SOLVER_DIR} COMPONENT solvers)
    endforeach()
    add_custom_target(${SOLVER_LIBRARY}-python ALL DEPENDS ${python_targets} ${SOLVER_DEPENDS})
    add_custom_target(validate-${SOLVER_LIBRARY}-yml COMMAND ${CMAKE_SOURCE_DIR}/toolset/validate_solvers_yaml.py ${CMAKE_CURRENT_SOURCE_DIR}/solvers.yml)

    if(BUILD_GUI)
        string(REPLACE "/" "." SOLVER_MODULE ${SOLVER_DIR})
        set(SOLVER_STUB ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/share/plask/stubs/${SOLVER_DIR}.py)
        add_custom_command(OUTPUT ${SOLVER_STUB}
                            COMMAND ${CMAKE_BINARY_DIR}/bin/plask -lwarning ${CMAKE_SOURCE_DIR}/toolset/makestub.py ${SOLVER_MODULE}
                            DEPENDS ${plask_binary} ${SOLVER_PYTHON_MODULE} ${CMAKE_SOURCE_DIR}/toolset/makestub.py
                            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/share/plask/stubs
                            )
        install(FILES ${SOLVER_STUB} DESTINATION share/plask/stubs/${SOLVER_CATEGORY_NAME} COMPONENT gui)
        add_custom_target(${SOLVER_LIBRARY}-stub ALL DEPENDS ${SOLVER_LIBRARY}-python ${SOLVER_PYTHON_MODULE} ${SOLVER_STUB})
        if(DEFINED SOLVER_GUI_INSTALL_FILES)
            install(FILES ${SOLVER_GUI_INSTALL_FILES} DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT gui)
        endif()
        if(DEFINED SOLVER_GUI_INSTALL_DIRECTORIES)
            install(DIRECTORIES ${SOLVER_GUI_INSTALL_DIRECTORIES} DESTINATION ${SOLVER_INSTALL_PATH} COMPONENT gui
                    PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE PATTERN "*.pyo" EXCLUDE)
        endif()
    endif()

    if(BUILD_TESTING)
        add_custom_target(${SOLVER_LIBRARY}-test DEPENDS ${SOLVER_LIBRARY}-python ${SOLVER_TEST_DEPENDS})
    endif()

endmacro()
