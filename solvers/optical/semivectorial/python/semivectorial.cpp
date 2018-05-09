#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../semivectorial.h"
using namespace plask::optical::semivectorial;


BOOST_PYTHON_MODULE(semivectorial)
{
    if (!plask_import_array()) throw(py::error_already_set());


    {CLASS(SemiVectorial<Geometry2DCylindrical>, "SemiVectorialCyl", "Solver semivectorial")
         METHOD(refractive_index, refractive_index, py::arg("lam"));
    }
    
    {CLASS(SemiVectorial<Geometry2DCartesian>, "SemiVectorial2D", "Solver semivectorial")
   
    }
    
  

}

