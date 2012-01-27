#
# This is a default module CMake file.
# In most cases you should be able to use it without modifications.
#

# Setup everything necessary
include(${CMAKE_SOURCE_DIR}/cmake/modules.cmake)


# Uncomment and edit the line below if you need to link some external libraries.
# Note that mose probably you will need to use CMake command find_package or find_library
# to find the libray in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB
#
# Example:
#  find_package(GSL)
#  set(MODULE_LINK_LIBRARIES ${GSL_LIBRARIES})
#
#set(MODULE_LINK_LIBRARIES mylib yourlib)


# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.
#
#set(MODULE_INCLUDE_DIRECTORIES ${GSL_INCLUDE_DIRS})


# Uncomment and edit the line below if you need some special linker flags.
#
# Example:
#  set(MODULE_LINK_FLAGS ${GSL_EXE_LINKER_FLAGS})
#
#set(MODULE_LINK_FLAGS myflags)


# Uncomment and edit the line below if you need some special compiler flags.
#
# Example:
#  set(MODULE_COMPILE_FLAGS ${CMAKE_GSL_CXX_FLAGS})
#
#set(MODULE_COMPILE_FLAGS myflags)


# If you want, you can manually specify your sources. Just uncomment and edit
# the following lines. If you don't do it, your module directory will be scanned
# automatically for module_src and the 'python' subdirectory for interface_src.
#
#set(module_src file1.cpp file2.cpp file3.h file4.hpp)
#set(interface_src python/file1.cpp python/file2.cpp python/file3.h)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()
