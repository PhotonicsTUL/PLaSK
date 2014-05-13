# Set the project name in the style: plask/solvergroup/solverlib
project(plask/optical/effective)

# Do not change thew following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
find_package(PLaSK)


# Uncomment and edit the line below if you need to link some external libraries.
# Note that mose probably you will need to use CMake command find_package or find_library
# to find the libray in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB, Eigen3
#
#find_package(Boost REQUIRED COMPONENTS math_tr1)
#set(SOLVER_LINK_LIBRARIES ${Boost_MATH_TR1_LIBRARIES})
#find_package(BLAS)
#set(SOLVER_LINK_LIBRARIES ${BLAS_LIBRARIES})
set(SOLVER_LINK_LIBRARIES camos)

# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.
#
find_package(Eigen3)
set(SOLVER_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})


# Uncomment and edit the line below if you need some special linker flags.
#
#set(SOLVER_LINK_FLAGS ${GSL_EXE_LINKER_FLAGS})


# Uncomment and edit the line below if you need some special compiler flags.
#
#set(SOLVER_COMPILE_FLAGS ${CMAKE_GSL_CXX_FLAGS})

# Fix a bug with alignment for Eigen matrices
#if(WIN32 AND CMAKE_COMPILER_IS_GNUCXX)
#    set(SOLVER_COMPILE_FLAGS -mincoming-stack-boundary=2)
#endif()

# If you want, you can manually specify your sources. Just uncomment and edit
# the following lines. If you don't do it, your solver directory will be scanned
# automatically for solver_src and the 'python' subdirectory for interface_src.
#
#set(solver_src eim.cpp eim.h efm.cpp efm.h broyden.cpp broyden.h bessel.h)
#set(interface_src python/effective.cpp)
file(GLOB solver_src *.cpp *.hpp *.h)

# To add unit tests to your project create appropriate targets (binary executables
# or Python scripts) and register them using CMake command add_test for binaries
# and macro add_python_test(name file.py) for Python scripts. You should name
# the test as 'solvers/your_solver/test_name'. Then uncomment and edit the line
# below.
#
enable_testing()

#add_executable(eimtest tests/eimtest.cpp)
#target_link_libraries(eimtest libplask ${SOLVER_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES})
#set(SOLVER_TEST_DEPENDS eimtest)
#add_test(solvers/optical/effective ${PLASK_SOLVER_PATH}/eimtest)

add_python_test(solvers/loading ${CMAKE_CURRENT_SOURCE_DIR}/tests/loading.py)
add_python_test(solvers/optical/effective ${CMAKE_CURRENT_SOURCE_DIR}/tests/effective.py)

# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()

