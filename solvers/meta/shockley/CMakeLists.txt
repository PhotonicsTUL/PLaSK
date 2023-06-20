# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# Set the project name in the style: plask/solvergroup/solverlib
project(plask/meta/shockley)


# Do not change the following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)


# Uncomment and edit the line below if your solver should be built after any
# other targets. This is most useful for meta-solvers.
set(SOLVER_DEPENDS solver-thermalstatic-python solver-electricalshockley-python
                   solver-opticaleffective-python solver-opticalslab-python solver-gainfreecarrier-python)


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
add_solver_test(iterative ${CMAKE_CURRENT_SOURCE_DIR}/tests/iterative.xpl)


# Build everything the default way.
# Call this macro unless you really know what you are doing!
make_pure_python()
