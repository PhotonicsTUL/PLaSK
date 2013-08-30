#/// \file
#///  This is a default solver CMake file.
#///
#/// In most cases you should be able to use it only setting the project name

# Set the project name in the style: plask/solvergroup/solverlib
project(plask/optical/slab)


# Do not change thew following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
find_package(PLaSK)


# Uncomment and edit the line below if you need to link some external libraries.
# Note that most probably you will need to use CMake command find_package or find_library
# to find the library in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB

find_package(LAPACK REQUIRED)

set(SOLVER_LINK_LIBRARIES ${LAPACK_LIBRARIES})

# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.

#find_package(Eigen3)

option(USE_FFTW "Use FFTW library (impossible for published app)" ON)
if(USE_FFTW)
    if (OPENMP_FOUND)
        find_package(FFTW3 COMPONENTS double openmp)
        set(USE_PARALLEL_FFT YES)
    else()
        find_package(FFTW3)
    endif()
    list(APPEND SOLVER_LINK_LIBRARIES ${FFTW3_LIBRARIES})
    list(APPEND SOLVER_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR})
else()
    # TODO
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/include/plask/optical/slab/config.h)



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

if(BUILD_TESTING)
    add_executable(fft_test tests/fft_test.cpp)
    target_link_libraries(fft_test libplask ${SOLVER_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES})
    set(SOLVER_TEST_DEPENDS fft_test)
    add_test(solvers/optical/slab/fft ${PLASK_SOLVER_PATH}/fft_test)
endif()

add_python_test(solvers/optical/slab ${CMAKE_CURRENT_SOURCE_DIR}/tests/slab.py)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()
