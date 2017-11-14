#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../simple_optical.h"
using namespace plask::solvers::simple_optical;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(simple_optical)
{
    if (!plask_import_array()) throw(py::error_already_set());
    {CLASS(SimpleOptical, "SimpleOpticalCyl", "Short solver description.")
     METHOD(say_hello, say_hello, "This is demo function, which say hello");
     METHOD(simpleVerticalSolver, simpleVerticalSolver, "This is method to calcurate field in 1D");
      
        //METHOD(python_method_name, method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
 //METHOD(say_hello, say_hello, "This is demo function, which say hello");
	
    }

}