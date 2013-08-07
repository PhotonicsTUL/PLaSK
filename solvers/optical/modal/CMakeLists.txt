#/// \file
#///  This is a default solver CMake file.
#///
#/// In most cases you should be able to use it only setting the project name

# Set the project name in the style: plask/solvergroup/solverlib
project(plask/optical/modal)


# Do not change thew following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
find_package(PLaSK)


# Uncomment and edit the line below if you need to link some external libraries.
# Note that most probably you will need to use CMake command find_package or find_library
# to find the library in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB

find_package(LAPACK)

set(SOLVER_LINK_LIBRARIES ${LAPACK_LIBRARIES})

# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.

find_package(Eigen3)

set(SOLVER_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})

option(USE_FFTW "Use FFTW library (impossible for published app)" OFF)
if(USE_FFTW)
    find_package(FFTW3)
    list(APPEND SOLVER_LINK_LIBRARIES ${FFTW3_THREADS_LIBRARY})
    list(APPEND SOLVER_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR})
else()
    # TODO
endif()


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
#set(solver_src file1.cpp file2.cpp file3.h file4.hpp)
#set(interface_src python/file1.cpp python/file2.cpp python/file3.h)


# To add unit tests to your project create appropriate targets (binary executables
# or Python scripts) and register them using CMake command add_test for binaries
# and macro add_python_test(name file.py) for Python scripts. You should name
# the test as 'solvers/your_solver/test_name'. Then uncomment and edit the line
# below.
#
enable_testing()
#add_executable(mytest test/source1.cpp test/source2.cpp)
#set(SOLVER_TEST_DEPENDS mytest)
#add_test(solvers/foo/bar ${PLASK_SOLVER_PATH}/mytest)
#add_python_test(python/solvers/foo/bar ${CMAKE_CURRENT_SOURCE_DIR}/tests/mytest.py)
add_python_test(solvers/optical/modal ${CMAKE_CURRENT_SOURCE_DIR}/tests/modal.py)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()
