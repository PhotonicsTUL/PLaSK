#/// \file
#///  This is a default solver CMake file.
#///
#/// In most cases you should be able to use it only setting the project name

# Set the project name in the style: plask/solvergroup/solverlib
project(plask/electrical/diffusion)


# Do not change thew following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)


# Uncomment and edit the line below if you need to link some external libraries.
# Note that most probably you will need to use CMake command find_package or find_library
# to find the library in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB
#
find_package(LAPACK)
set(SOLVER_LINK_LIBRARIES ${LAPACK_LIBRARIES} nspcg)


# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.
#
#set(SOLVER_INCLUDE_DIRECTORIES ${GSL_INCLUDE_DIRS})


# Uncomment and edit the line below if you need some special linker flags.
#
#set(SOLVER_LINK_FLAGS ${GSL_EXE_LINKER_FLAGS})


# Uncomment and edit the line below if you need some special compiler flags.
#
#set(SOLVER_COMPILE_FLAGS ${CMAKE_GSL_CXX_FLAGS})


# If you want, you can manually specify your sources. Just uncomment and edit
# the following lines. If you don't do it, your solver directory will be scanned
# automatically for solver_src and the 'python' subdirectory for interface_src.
#
# set(solver_src diffusion1d.cpp)
#set(interface_src python/file1.cpp python/file2.cpp python/file3.h)

# To add unit tests to your project create appropriate targets (binary executables,
# Python scripts, or XPL files) and register them using CMake command add_solver_test.
# Then uncomment and edit the line below.
#
enable_testing()
#add_executable(mytest test/your_test.cpp test/other_source.cpp)
#set(SOLVER_TEST_DEPENDS mytest)
#add_solvers_test(foo ${PLASK_SOLVER_PATH}/mytest)
#add_solver_test(bar ${CMAKE_CURRENT_SOURCE_DIR}/tests/mytest.py)
add_solver_test(uniform ${CMAKE_CURRENT_SOURCE_DIR}/tests/uniform.py)
add_solver_test(gaussian ${CMAKE_CURRENT_SOURCE_DIR}/tests/gaussian.py)
add_solver_test(diffusion3d ${CMAKE_CURRENT_SOURCE_DIR}/tests/diffusion3d.py)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()
