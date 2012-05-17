## Setup everything necessary
include(${CMAKE_SOURCE_DIR}/cmake/modules.cmake)


# Uncomment and edit the line below if you need to link some external libraries.
# Note that mose probably you will need to use CMake command find_package or find_library
# to find the libray in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB, Eigen3
#
#find_package(GSL)
#set(MODULE_LINK_LIBRARIES ${GSL_LIBRARIES})

# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.
#
find_package(Eigen3)
set(MODULE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})

# Uncomment and edit the line below if you need some special linker flags.
#
#set(MODULE_LINK_FLAGS ${GSL_EXE_LINKER_FLAGS})


# Uncomment and edit the line below if you need some special compiler flags.
#
#set(MODULE_COMPILE_FLAGS ${CMAKE_GSL_CXX_FLAGS})


# If you want, you can manually specify your sources. Just uncomment and edit
# the following lines. If you don't do it, your module directory will be scanned
# automatically for module_src and the 'python' subdirectory for interface_src.
#
#set(module_src file1.cpp file2.cpp file3.h file4.hpp)
#set(interface_src python/file1.cpp python/file2.cpp python/file3.h)


# To add unit tests to your project create appropriate targets (binary executables
# or Python scripts) and register them using CMake command add_test for binaries
# and macro add_python_test(name file.py) for Python scripts. You should name
# the test as 'modules/your_module/test_name'. Then uncomment and edit the line
# below.
#
#add_executable(mytest test/source1.cpp test/source2.cpp)
#set(MODULE_TEST_DEPENDS mytest)
enable_testing()
#add_test(modules/foo/bar mytest)
add_python_test(modules/optical/effective ${CMAKE_CURRENT_SOURCE_DIR}/tests/effective.py)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()
