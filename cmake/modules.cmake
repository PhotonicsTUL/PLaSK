#
# Helpers for cmake files in modules subdirectories
#

# Obtain relative path name
string(REPLACE "${CMAKE_SOURCE_DIR}/modules/" "" MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Obtain module library name (it can actually contain several modules)
string(REGEX REPLACE "/|_" "" TARGET_NAME ${MODULE_DIR})
set(MODULE_LIBRARY_NAME "plask_${TARGET_NAME}")

# Add prefix to target name
set(TARGET_NAME module-${TARGET_NAME})

# Obtain default canonical module name
get_filename_component(MODULE_NAME ${MODULE_DIR} NAME)

# Obtain name of the variable indicating successful configuration of the module
string(REPLACE "/" "_" BUILD_MODULE_OK ${MODULE_DIR})
string(TOUPPER "BUILD_MODULE_${BUILD_MODULE_OK}_OK" BUILD_MODULE_OK)

# Obtain intermediate path list and to create necessary __init__.py files to mark packages
get_filename_component(MODULE_PATH ${MODULE_DIR} PATH)
if(NOT MODULE_PATH STREQUAL "")
    string(REPLACE "/" ";" MODULE_DIRECTORIES ${MODULE_PATH})
endif()
set(PYTHON_MODULE_PATH "${plask_PYTHONPATH}/plask/${MODULE_PATH}")

# Automatically set sources from the current directory
file(GLOB module_src *.cpp *.hpp *.h)
file(GLOB interface_src python/*.cpp)

if(BUILD_SHARED_MODULE_LIBS)
    set(BUILD_SHARED_LIBS YES)
else()
    set(BUILD_SHARED_LIBS NO)
    add_definitions(-fPIC)
endif()


# This is macro that sets all the targets automagically
macro(make_default)

    # Build module library
    add_library(${TARGET_NAME} ${module_src})
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${MODULE_LIBRARY_NAME})
    target_link_libraries(${TARGET_NAME} libplask ${MODULE_LINK_LIBRARIES})
    include_directories(${MODULE_INCLUDE_DIRECTORIES})
    if (DEFINED MODULE_LINK_FLAGS)
        set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS ${MODULE_LINK_FLAGS})
    endif()
    if (DEFINED MODULE_COMPILE_FLAGS)
        set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS ${MODULE_COMPILE_FLAGS})
    endif()

    if(BUILD_SHARED_MODULE_LIBS)
        if(WIN32)
            install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin COMPONENT modules
                                           ARCHIVE DESTINATION lib COMPONENT modules-dev)
        else()
            install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib COMPONENT modules)
        endif()
    else()
        install(TARGETS ${TARGET_NAME} ARCHIVE DESTINATION lib COMPONENT modules-dev)
    endif()

    if(BUILD_PYTHON)
        set(PYTHON_TARGET_NAME ${TARGET_NAME}-python)
        # Make package hierarchy
        set(curr_path "${plask_PYTHONPATH}/plask")
        foreach(dir ${MODULE_DIRECTORIES})
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
                              LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_PATH}
                              OUTPUT_NAME ${MODULE_NAME}
                              PREFIX "")
        if(WIN32)
            set_target_properties(${PYTHON_TARGET_NAME} PROPERTIES
                                  RUNTIME_OUTPUT_DIRECTORY "${PYTHON_MODULE_PATH}"
                                  SUFFIX ".pyd")
            install(TARGETS ${PYTHON_TARGET_NAME} RUNTIME DESTINATION ${PYTHON_MODULE_INSTALL_DIR}/${MODULE_PATH} COMPONENT modules)
        else()
            install(TARGETS ${PYTHON_TARGET_NAME} LIBRARY DESTINATION ${PYTHON_MODULE_INSTALL_DIR}/${MODULE_PATH} COMPONENT modules)
        endif()
    endif()

    if(BUILD_TESTING)
        add_custom_target(${TARGET_NAME}-test DEPENDS ${TARGET_NAME} ${PYTHON_TARGET_NAME} ${MODULE_TEST_DEPENDS})
    endif()

endmacro()
