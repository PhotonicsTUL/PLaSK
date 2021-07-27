#/// \file
#///  This is a default solver CMake file.
#///
#/// In most cases you should be able to use it only setting the project name

# Set the project name in the style: plask/solvergroup/solverlib
project(plask/optical/slab)


# Do not change thew following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)


# Custom settings
option(USE_GSL "Use GSL library instead of Boost")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in ${CMAKE_BINARY_DIR}/include/plask/optical/slab/config.hpp)




# Uncomment and edit the line below if you need to link some external libraries.
# Note that most probably you will need to use CMake command find_package or find_library
# to find the library in the system.
#
# Some libraries, that you can find using find_package command are:
#  BLAS, LAPACK, Matlab, ARPACK, FFTW3, GSL, ZLIB

find_package(LAPACK REQUIRED)

set(SOLVER_LINK_LIBRARIES ${LAPACK_LIBRARIES} fftpacx)

if(USE_GSL)
    find_package(GSL REQUIRED)
    set(SOLVER_LINK_LIBRARIES ${SOLVER_LINK_LIBRARIES} ${GSL_LIBRARIES})
endif()

# Uncomment and edit the line below if you need some special include directories.
# If you use external libraries, you can use the variables returned by find_package.
# Don't include external directories with your own headers. Just copy them here and
# commit to the repository.
#set(SOLVER_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/xxxx)

#find_package(Eigen3)

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

file(GLOB solver_src *.cpp fourier/*.cpp bessel/*.cpp)


# To add unit tests to your project create appropriate targets (binary executables,
# Python scripts, or XPL files) and register them using CMake command add_solver_test.
# Then uncomment and edit the line below.
#
enable_testing()

if(BUILD_TESTING)
    foreach(slab_test fft toeplitz)
        add_executable(${slab_test}_test tests/${slab_test}_test.cpp)
        target_link_libraries(${slab_test}_test libplask ${SOLVER_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES})
        add_solver_test(${slab_test} ${slab_test}_test)
    endforeach()
    add_executable(invmult_benchmark tests/invmult_benchmark.cpp)
    target_link_libraries(invmult_benchmark libplask ${SOLVER_LIBRARY})
endif()

add_solver_test(layers ${CMAKE_CURRENT_SOURCE_DIR}/tests/slab.py)
add_solver_test(temperature ${CMAKE_CURRENT_SOURCE_DIR}/tests/temp.py)
# add_solver_test(bessel ${CMAKE_CURRENT_SOURCE_DIR}/tests/bessel.py)
add_solver_test(refl2d ${CMAKE_CURRENT_SOURCE_DIR}/tests/reflection2d.py)
add_solver_test(refl3d ${CMAKE_CURRENT_SOURCE_DIR}/tests/reflection3d.py)
add_solver_test(grating ${CMAKE_CURRENT_SOURCE_DIR}/tests/grating.py)
add_solver_test(fresnel ${CMAKE_CURRENT_SOURCE_DIR}/tests/fresnel.py)
add_solver_test(wire2d ${CMAKE_CURRENT_SOURCE_DIR}/tests/wire2d.py)
add_solver_test(vcsel ${CMAKE_CURRENT_SOURCE_DIR}/tests/vcsel.py)
add_solver_test(vcsel3d ${CMAKE_CURRENT_SOURCE_DIR}/tests/vcsel3d.xpl)
add_solver_test(huang ${CMAKE_CURRENT_SOURCE_DIR}/tests/huang.xpl)
add_solver_test(tilted ${CMAKE_CURRENT_SOURCE_DIR}/tests/tilted.xpl)
add_solver_test(carriers ${CMAKE_CURRENT_SOURCE_DIR}/tests/carriers.py)


# Uncomment and edit the line below if you need to link some external libraries
# to Python wrapper.
# set(SOLVER_PYTHON_LINK_LIBRARIES)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_default()

### Last revision with FFTW: r1879
