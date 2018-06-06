
project(plask/optical/semivectorial)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
include(PLaSK)

file(GLOB solver_src *.cpp *.hpp *.h)

enable_testing()

make_default()
