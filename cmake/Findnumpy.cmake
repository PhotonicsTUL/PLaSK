# - Try to find numpy include directories
# Once done it will define
#  NUMPY_FOUND - System has numpy
#  NUMPY_INCLUDE_DIRS - Include directories when compiling against numpy

find_package(PythonInterp QUIET)

if(PYTHON_EXECUTABLE AND NOT NUMPY_INCLUDE_DIRS)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "try: import numpy; print(numpy.get_include())\nexcept: pass\n"
                    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)
    string(STRIP "${NUMPY_INCLUDE_DIR}" NUMPY_INCLUDE_DIR)
    set(NUMPY_INCLUDE_DIRS "${NUMPY_INCLUDE_DIR}" CACHE PATH "numpy include directories")
else()
    set(NUMPY_INCLUDE_DIR "${NUMPY_INCLUDE_DIRS}")
endif()

# handle the QUIETLY and REQUIRED arguments and set NUMPY_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(numpy DEFAULT_MSG NUMPY_INCLUDE_DIRS)
