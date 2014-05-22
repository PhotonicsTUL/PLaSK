# - Try to find DW
# Once done this will define
#  DW_FOUND - System has DW
#  DW_INCLUDE_DIRS - The DW include directories
#  DW_LIBRARIES - The libraries needed to use DW
#  DW_DEFINITIONS - Compiler switches required for using DW
#
# Piotr Beling (2014)

find_package(PkgConfig)
pkg_check_modules(PC_LIBDW QUIET dw)
set(DW_DEFINITIONS ${PC_LIBDW_CFLAGS_OTHER})

find_path(DW_INCLUDE_DIR elfutils/libdw.h
          HINTS ${PC_LIBDW_INCLUDEDIR} ${PC_LIBDW_INCLUDE_DIRS}
          PATH_SUFFIXES DW )

find_library(DW_LIBRARY NAMES dw dwarf
             HINTS ${PC_LIBDW_LIBDIR} ${PC_LIBDW_LIBRARY_DIRS} )

set(DW_LIBRARIES ${DW_LIBRARY} )
set(DW_INCLUDE_DIRS ${DW_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set DW_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(DW  DEFAULT_MSG
                                  DW_LIBRARY DW_INCLUDE_DIR)

mark_as_advanced(DW_INCLUDE_DIR DW_LIBRARY )
