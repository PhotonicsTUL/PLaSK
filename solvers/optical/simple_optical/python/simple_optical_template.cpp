#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../simple_optical_template.h"
using namespace plask::optical::simple_optical_template;


BOOST_PYTHON_MODULE(simple_optical_template)
{
    if (!plask_import_array()) throw(py::error_already_set());


    {CLASS(SimpleOpticalTemplate<Geometry2DCylindrical>, "SimpleOpticalCyl2D", "Solver performing calculations in 2D Cylindrical geometry by solve Helmholtz equation.")
      
    }
}