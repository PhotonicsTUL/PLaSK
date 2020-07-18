# Set the project name in the style: plask/solvergroup/solverlib
project(plask/meta/shockley)


# Do not change the following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)


# Uncomment and edit the line below if your solver should be built after any
# other targets. This is most useful for meta-solvers.
set(SOLVER_DEPENDS solver-thermalstatic-python solver-electricalshockley-python
                   solver-opticaleffective-python solver-opticalslab-python)


# To add unit tests to your project create appropriate targets (binary executables,
# Python scripts, or XPL files) and register them using CMake command add_solver_test.
# Then uncomment and edit the line below.
#
enable_testing()
#add_executable(mytest test/your_test.cpp test/other_source.cpp)
#set(SOLVER_TEST_DEPENDS mytest)
#add_solvers_test(foo ${PLASK_SOLVER_PATH}/mytest)
#add_solver_test(bar ${CMAKE_CURRENT_SOURCE_DIR}/tests/mytest.py)
add_solver_test(thermoelectric ${CMAKE_CURRENT_SOURCE_DIR}/tests/thermoelectric.xpl)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_pure_python()
