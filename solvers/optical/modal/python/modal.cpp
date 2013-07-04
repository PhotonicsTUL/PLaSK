/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../reflection_solver.h"
using namespace plask::solvers::modal;

BOOST_PYTHON_MODULE(modal)
{
//     {CLASS(Class_Name, "YourSolver", "Short solver description.")
//         METHOD(python_method_name, method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
//         RO_FIELD(field_name, "Short documentation"); // read-only field
//         RW_FIELD(field_name, "Short documentation"); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         RECEIVER(inReceiver, "Short documentation"); // receiver in the solver
//         PROVIDER(outProvider, "Short documentation"); // provider in the solver
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
//     }

}

