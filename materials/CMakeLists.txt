# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# Build default material database

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PLASK_PATH}/materials)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PLASK_PATH}/materials)

foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # does not use double Debug, Release, etc. in path (on is already included in ${PLASK_PATH})
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} ${PLASK_PATH}/materials)
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} ${PLASK_PATH}/materials)
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} ${PLASK_PATH}/materials)
endforeach()

# List of all material groups
file(GLOB materials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *)

# We will remove old libraries from build path (otherwise the debugging may fail)
if(WIN32)
    set(lib_suffix dll)
else()
    set(lib_suffix so)
endif()
file(GLOB spurious_libs ${PLASK_PATH}/materials/*.${lib_suffix})

foreach(material ${materials})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${material})
        file(GLOB_RECURSE material_src FOLLOW_SYMLINKS
             ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.hh ${CMAKE_CURRENT_SOURCE_DIR}/${material}/*.h)
        if(material_src)
            add_library(materials-${material} MODULE ${material_src})
            install(TARGETS materials-${material} LIBRARY DESTINATION lib/plask/materials COMPONENT core)

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

set(PLASK_MATERIALS ${targets} PARENT_SCOPE)
