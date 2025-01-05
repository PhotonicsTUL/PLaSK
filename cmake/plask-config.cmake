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
#                               (identical to PLASK_INCLUDE_DIR)
#  PLASK_LIBRARY              - the PLaSK library (cached)
#  PLASK_LIBRARIES            - list of all PLaSK libraries found

if(NOT DEFINED PLASK_ROOT AND DEFINED ENV{PLASK_ROOT})
    set(PLASK_ROOT "$ENV{PLASK_ROOT}")
endif()

if(DEFINED PLASK_ROOT)
    find_library(PLASK_LIBRARY NAMES plask libplask HINTS "${PLASK_ROOT}/lib")
    set(PLASK_INCLUDE_DIR "${PLASK_ROOT}/include")
    set(PLASK_INCLUDE_DIRS "${PLASK_ROOT}/include")
else()
    find_library(PLASK_LIBRARY NAMES plask libplask)
    set(PLASK_INCLUDE_DIR "")
    set(PLASK_INCLUDE_DIRS "")
endif()

list(APPEND PLASK_LIBRARIES "${PLASK_LIBRARY}")

mark_as_advanced(PLASK_LIBRARY PLASK_LIBRARIES PLASK_INCLUDE_DIR PLASK_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PLaSK DEFAULT_MSG PLASK_LIBRARY)
