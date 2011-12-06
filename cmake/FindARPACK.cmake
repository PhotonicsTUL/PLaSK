# - Try to find ARPACK
# Once done this will define
#  ARPACK_FOUND - System has ARPACK
#  ARPACK_LIBRARY - The ARPACK library (for default component arpack)
#  PARPACK_LIBRARY - The PARPACK library (for component parpack)
#  ARPACK_LIBRARIES - Libraries for all the ARPACK components

if(ARPACK_FIND_COMPONENTS MATCHES "^$")
  set(_components arpack)
else()
  set(_components ${ARPACK_FIND_COMPONENTS})
endif()

set(ARPACK_LIBRARIES)
foreach(_lib ${_components})
    if (NOT _lib STREQUAL "arpack" AND NOT _lib STREQUAL "parpack")
        message(FATAL_ERROR "FindARPACK: unknown component '${_comp}' specified. "
                            "Valid components are 'arpack' and 'parpack'.")
    endif()

    string(TOUPPER ${_lib} _LIB)
    find_library(${_LIB}_LIBRARY ${_lib})
    if (${_LIB}_LIBRARY)
        list(APPEND ARPACK_LIBRARIES ${${_LIB}_LIBRARY})
        mark_as_advanced(${_LIB}_LIBRARY)
    endif()
endforeach()

# handle the QUIETLY and REQUIRED arguments and set ARPACK_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(arpack DEFAULT_MSG ARPACK_LIBRARIES)
