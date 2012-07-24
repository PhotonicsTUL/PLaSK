#
# Helpers for cmake files in solvers subdirectories
#

# Obtain relative path name
string(REPLACE "${CMAKE_SOURCE_DIR}/solvers/" "" SOLVER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Obtain solver library name (it can actually contain several solvers)
string(REGEX REPLACE "/|_" "" TARGET_NAME ${SOLVER_DIR})
set(SOLVER_LIBRARY_NAME "plask_${TARGET_NAME}")

# Add prefix to target name
set(TARGET_NAME solver-${TARGET_NAME})

# Obtain default canonical solver name
get_filename_component(SOLVER_NAME ${SOLVER_DIR} NAME)

# Obtain name of the variable indicating successful configuration of the solver
string(REPLACE "/" "_" BUILD_SOLVER_OK ${SOLVER_DIR})
string(TOUPPER "BUILD_SOLVER_${BUILD_SOLVER_OK}_OK" BUILD_SOLVER_OK)

# Obtain intermediate path list and to create necessary __init__.py files to mark packages
get_filename_component(SOLVER_PATH ${SOLVER_DIR} PATH)
if(NOT SOLVER_PATH STREQUAL "")
    string(REPLACE "/" ";" SOLVER_DIRECTORIES ${SOLVER_PATH})
endif()
set(PYTHON_SOLVER_PATH "${plask_PYTHONPATH}/plask/${SOLVER_PATH}")

# Automatically set sources from the current directory
file(GLOB solver_src *.cpp *.hpp *.h)
file(GLOB interface_src python/*.cpp)

if(BUILD_SHARED_SOLVER_LIBS)
    set(BUILD_SHARED_LIBS YES)
else()
    set(BUILD_SHARED_LIBS NO)
    add_definitions(-fPIC)
endif()


# This is macro that sets all the targets automagically
macro(make_default)

    # Build solver library
    add_library(${TARGET_NAME} ${solver_src})
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${SOLVER_LIBRARY_NAME})
    target_link_libraries(${TARGET_NAME} libplask ${SOLVER_LINK_LIBRARIES})
    include_directories(${SOLVER_INCLUDE_DIRECTORIES})
    if (DEFINED SOLVER_LINK_FLAGS)
        set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS ${SOLVER_LINK_FLAGS})
    endif()
    if (DEFINED SOLVER_COMPILE_FLAGS)
        set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS ${SOLVER_COMPILE_FLAGS})
    endif()

    if(BUILD_SHARED_SOLVER_LIBS)
        if(WIN32)
            install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin COMPONENT solvers
                                           ARCHIVE DESTINATION lib COMPONENT solvers-dev)
        else()
            install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib COMPONENT solvers)
        endif()
    else()
        install(TARGETS ${TARGET_NAME} ARCHIVE DESTINATION lib COMPONENT solvers-dev)
    endif()

    if(BUILD_PYTHON)
        set(PYTHON_TARGET_NAME ${TARGET_NAME}-python)
        # Make package hierarchy
        set(curr_path "${plask_PYTHONPATH}/plask")
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
                              LIBRARY_OUTPUT_DIRECTORY ${PYTHON_SOLVER_PATH}
                              OUTPUT_NAME ${SOLVER_NAME}
                              PREFIX "")
        if(WIN32)
            set_target_properties(${PYTHON_TARGET_NAME} PROPERTIES
                                  RUNTIME_OUTPUT_DIRECTORY "${PYTHON_SOLVER_PATH}"
                                  SUFFIX ".pyd")
            install(TARGETS ${PYTHON_TARGET_NAME} RUNTIME DESTINATION ${PYTHON_SOLVER_INSTALL_DIR}/${SOLVER_PATH} COMPONENT solvers)
        else()
            install(TARGETS ${PYTHON_TARGET_NAME} LIBRARY DESTINATION ${PYTHON_SOLVER_INSTALL_DIR}/${SOLVER_PATH} COMPONENT solvers)
        endif()
    endif()

    if(BUILD_TESTING)
        add_custom_target(${TARGET_NAME}-test DEPENDS ${TARGET_NAME} ${PYTHON_TARGET_NAME} ${SOLVER_TEST_DEPENDS})
    endif()

endmacro()
