#
# Helpers for cmake files in modules subdirectories
#

# Obtain relative path name
string(REPLACE "${CMAKE_SOURCE_DIR}/modules/" "" MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Obtain module library name (it can actually contain several modules)
string(REPLACE "/" "" TARGET_NAME ${MODULE_DIR})
set(MODULE_LIBRARY_NAME "plask_${TARGET_NAME}")

# Obtain default canonical module name
get_filename_component(MODULE_NAME ${MODULE_DIR} NAME)

# Obtain intermediate path list and to create necessary __init__.py files to mark packages
get_filename_component(MODULE_PATH ${MODULE_DIR} PATH)
if(NOT MODULE_PATH STREQUAL "")
    string(REPLACE "/" ";" MODULE_DIRECTORIES ${MODULE_PATH})
endif()
set(MODULE_PATH "${CMAKE_BINARY_DIR}/lib.python/plask/${MODULE_PATH}")

# Automatically set sources from the current directory
file(GLOB module_src *.cpp *.hpp *.h)
file(GLOB interface_src python/*.cpp)



# This is macro that sets all the targets autmagically
macro(make_default)

    # Build module library
    add_library(module-${TARGET_NAME} ${module_src})
    set_target_properties(module-${TARGET_NAME} PROPERTIES OUTPUT_NAME ${MODULE_LIBRARY_NAME})
    target_link_libraries(module-${TARGET_NAME} libplask ${MODULE_LINK_LIBRARIES})
    include_directories(${MODULE_INCLUDE_DIRECTORIES})
    if (DEFINED MODULE_LINK_FLAGS)
        set_target_properties(module-${TARGET_NAME} PROPERTIES LINK_FLAGS ${MODULE_LINK_FLAGS})
    endif()
    if (DEFINED MODULE_COMPILE_FLAGS)
        set_target_properties(module-${TARGET_NAME} PROPERTIES COMPILE_FLAGS ${MODULE_COMPILE_FLAGS})
    endif()

    if(BUILD_PYTHON)

        # Make package hierarchy
        set(curr_path "${CMAKE_BINARY_DIR}/lib.python/plask")
        foreach(dir ${MODULE_DIRECTORIES})
            set(curr_path "${curr_path}/${dir}")
            file(MAKE_DIRECTORY ${curr_path})
            file(WRITE "${curr_path}/__init__.py" "")
        endforeach()

        # Build Python interface
        add_library(module-${TARGET_NAME}-python MODULE ${interface_src})
        target_link_libraries(module-${TARGET_NAME}-python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
        set_target_properties(module-${TARGET_NAME}-python PROPERTIES OUTPUT_NAME ${MODULE_NAME} PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${MODULE_PATH})
        if(WIN32)
            set_target_properties(module-${TARGET_NAME}-python PROPERTIES SUFFIX ".pyd")
        endif()

    endif()

endmacro()
