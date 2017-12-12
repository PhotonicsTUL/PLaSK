#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../simple_optical.h"
using namespace plask::optical::simple_optical;


BOOST_PYTHON_MODULE(simple_optical)
{
    if (!plask_import_array()) throw(py::error_already_set());
    {CLASS(SimpleOptical, "SimpleOpticalCyl", "Short solver description.")
     METHOD(simpleVerticalSolver, simpleVerticalSolver, "This is method to calcurate field in 1D");
     METHOD(get_T_bb, get_T_bb, "This method return T bb");   
     METHOD(computeField, computeField, "This method calcurate field");
     METHOD(getZ, getZ, "This method return z axis points");
     METHOD(getEz, getEz, "This method return Ez field");
     //PROVIDER(outLightMagnitude, "");
    }

}