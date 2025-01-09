# - Try to find PLaSK
#
# Usage: find_package(PLaSK [REQUIRED])
#
# This file is a CMake configuration file for the PLaSK project.
#
# The optional argument `REQUIRED` can be used to generate an error and stop
# the configuration process if PLaSK is not found.
#
# Variables used by this module:
#  PLASK_ROOT                 - PLaSK root directory
#
# Variables defined by this module:
#  PLASK_FOUND                - system has PLaSK
#  PLASK_INCLUDE_DIR          - the PLaSK include directory (cached)
#  PLASK_INCLUDE_DIRS         - the PLaSK include directories
#  PLASK_LIBRARY              - the PLaSK library (cached)
#  PLASK_LIBRARIES            - list of all PLaSK libraries found
#
# Functions defined by this module:
#  add_plask_material_library(name sources...) - add a PLaSK material library

if(NOT DEFINED PLASK_ROOT AND DEFINED ENV{PLASK_ROOT})
    set(PLASK_ROOT "$ENV{PLASK_ROOT}")
endif()

find_library(PLASK_LIBRARY NAMES plask libplask HINTS "${PLASK_ROOT}/lib")
find_path(PLASK_INCLUDE_DIR plask/plask.hpp HINTS "${PLASK_ROOT}/include")

find_package(Boost CONFIG REQUIRED)

list(APPEND PLASK_INCLUDE_DIRS "${PLASK_INCLUDE_DIR}" "${Boost_INCLUDE_DIRS}")
list(APPEND PLASK_LIBRARIES "${PLASK_LIBRARY}")

mark_as_advanced(PLASK_LIBRARY PLASK_LIBRARIES PLASK_INCLUDE_DIR PLASK_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PLaSK DEFAULT_MSG PLASK_LIBRARY)

function(add_plask_material_library libname)
    add_library(${libname} MODULE ${ARGN})
    target_link_libraries(${libname} ${PLASK_LIBRARIES})
    target_include_directories(${libname} PRIVATE ${PLASK_INCLUDE_DIRS})
    set_target_properties(${libname} PROPERTIES PREFIX "")
endfunction()
