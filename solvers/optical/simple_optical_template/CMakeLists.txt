# Set the project name in the style: plask/solvergroup/solverlib
project(plask/optical/simple_optical_template)

# Do not change the following two lines
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)

file(GLOB solver_src *.cpp *.hpp *.h)

enable_testing()

make_default()