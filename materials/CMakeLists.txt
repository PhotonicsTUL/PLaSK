# Build default material database

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PLASK_PATH}/materials)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PLASK_PATH}/materials)

# List of all material groups
file(GLOB materials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *)

# We will remove old libraries from buid path (therwise the debugging may fail)
if(WIN32)
    set(lib_suffix dll)
else()
    set(lib_suffix so)
endif()
file(GLOB spurious_libs ${PLASK_PATH}/materials/*.${lib_suffix})

foreach(material ${materials})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${material})
        file(GLOB_RECURSE material_src FOLLOW_SYMLINKS ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.h)
        if(material_src)
            if (WIN32)
                add_library(materials-${material} SHARED ${material_src})
                install(TARGETS materials-${material} RUNTIME DESTINATION lib/plask/materials COMPONENT core)
            else()
                add_library(materials-${material} MODULE ${material_src})
                install(TARGETS materials-${material} LIBRARY DESTINATION lib/plask/materials COMPONENT core)
            endif()

            set_target_properties(materials-${material} PROPERTIES OUTPUT_NAME ${material} PREFIX "")
            target_link_libraries(materials-${material} ${PLaSK_LIBRARIES})

            list(REMOVE_ITEM spurious_libs ${PLASK_PATH}/materials/${material}.${lib_suffix})

            list(APPEND targets materials-${material})
        endif()
    endif()
endforeach()

# Now delete the old libs
foreach(lib ${spurious_libs})
    file(REMOVE ${lib})
endforeach()

set(material_targets ${targets} PARENT_SCOPE)
