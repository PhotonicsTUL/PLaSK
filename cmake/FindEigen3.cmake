# - Try to find Eigen3 lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Eigen3 3.0.5)
# to require version 2.0.5 to newer of Eigen3.
#
# Once done this will define
#
#  EIGEN3_FOUND - system has eigen lib with correct version
#  EIGEN3_INCLUDE_DIR - the eigen include directory
#  EIGEN3_VERSION - eigen version

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Redistribution and use is allowed according to the terms of the BSD license.

if(NOT Eigen3_FIND_VERSION)
  set(Eigen3_FIND_VERSION_MAJOR 3)
  set(Eigen3_FIND_VERSION_MINOR 0)
  set(Eigen3_FIND_VERSION_PATCH 0)

  set(Eigen3_FIND_VERSION "${Eigen3_FIND_VERSION_MAJOR}.${Eigen3_FIND_VERSION_MINOR}.${Eigen3_FIND_VERSION_PATCH}")
endif(NOT Eigen3_FIND_VERSION)

macro(_eigen3_get_version)
  file(READ "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen3_version_header LIMIT 5000 OFFSET 1000)

  string(REGEX MATCH "define *EIGEN_WORLD_VERSION ([0-9]*)" _eigen3_world_version_match "${_eigen3_version_header}")
  set(EIGEN3_WORLD_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define *EIGEN_MAJOR_VERSION ([0-9]*)" _eigen3_major_version_match "${_eigen3_version_header}")
  set(EIGEN3_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define *EIGEN_MINOR_VERSION ([0-9]*)" _eigen3_minor_version_match "${_eigen3_version_header}")
  set(EIGEN3_MINOR_VERSION "${CMAKE_MATCH_1}")

  set(EIGEN3_VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})
endmacro(_eigen3_get_version)

find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
     PATHS
     ${INCLUDE_INSTALL_DIR}
     PATH_SUFFIXES eigen3
   )

if(EIGEN3_INCLUDE_DIR)
  _eigen3_get_version()
endif(EIGEN3_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3 REQUIRED_VARS EIGEN3_INCLUDE_DIR
                                         VERSION_VAR EIGEN3_VERSION)

mark_as_advanced(EIGEN3_INCLUDE_DIR)
