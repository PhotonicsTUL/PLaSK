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
     METHOD(findMode, findMode, "This is method to find wavelength of mode", (arg("lam"), arg("m")=0));
     METHOD(get_vert_determinant, getVertDeterminant, "Get vertical modal determinant for debuging purposes", (arg("wavelength")) );
     PROVIDER(outLightMagnitude, "");
     PROVIDER(outRefractiveIndex, "");
     METHOD(getLightMagnitude, getLightMagnitude, "This method return electric field");
    }

}