#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../simple_optical.h"
using namespace plask::optical::simple_optical;

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
     METHOD(simpleVerticalSolver, simpleVerticalSolver, "This is method to calcurate field in 1D");
     METHOD(get_T_bb, get_T_bb, "This method return T bb");     
     METHOD(get_eField, get_eField, "This method return electric field");
     METHOD(get_bField, get_bField, "This method return magnetic field");
     METHOD(get_z, get_z, "This method return z axis points");
     METHOD(compute_electric_field_distribution, compute_electric_field_distribution, "This method compute electric field distribution");
    }

}