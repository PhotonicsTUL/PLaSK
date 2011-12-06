# - Try to find numpy include directories
# Once done it will define
#  NUMPY_FOUND - System has numpy
#  NUMPY_INCLUDE_DIRS - Include directories when compiling against numpy

find_package(PythonInterp)

if(PYTHON_EXECUTABLE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindNumpy.py "try: import numpy; print numpy.get_include()\nexcept: pass\n")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindNumpy.py" OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS)
    string(STRIP ${NUMPY_INCLUDE_DIRS} NUMPY_INCLUDE_DIRS)
endif(PYTHON_EXECUTABLE)

# handle the QUIETLY and REQUIRED arguments and set ARPACK_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(numpy DEFAULT_MSG NUMPY_INCLUDE_DIRS)
