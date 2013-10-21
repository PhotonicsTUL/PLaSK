# Build default material database

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PLASK_PATH}/materials)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PLASK_PATH}/materials)

# List of all material groups
file(GLOB materials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *)

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

            list(APPEND targets materials-${material})
        endif()
    endif()
endforeach()

set(material_targets ${targets} PARENT_SCOPE)
