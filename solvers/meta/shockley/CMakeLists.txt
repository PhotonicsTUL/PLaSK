# Set the project name in the style: plask/solvergroup/solverlib
project(plask/meta/shockley)


# Do not change the following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)


# Uncomment and edit the line below if your solver should be built after any
# other targets. This is most useful for meta-solvers.
set(SOLVER_DEPENDS solver-thermalstatic-python solver-electricalshockley-python
                   solver-opticaleffective-python solver-opticalslab-python)


# To add unit tests to your project create appropriate targets (binary executables
# or Python scripts) and register them using CMake command add_test for binaries
# and macro add_python_test(name file.py) for Python scripts. You should name
# the test as 'solvers/your_solver/test_name'. Then uncomment and edit the line
# below.
#
enable_testing()
#add_executable(mytest test/your_test.cpp test/other_source.cpp)
#set(SOLVER_TEST_DEPENDS mytest)
#add_test(solvers/foo/bar ${PLASK_SOLVER_PATH}/mytest)
#add_python_test(python/solvers/foo/bar ${CMAKE_CURRENT_SOURCE_DIR}/tests/mytest.py)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_pure_python()
