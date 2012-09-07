/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../trivial_gain.h"
using namespace plask::solvers::gain_trivial;

template <typename GeometryT>
void StepProfile_setElement(StepProfileGain<GeometryT>& self, shared_ptr<GeometryElementD<GeometryT::DIMS>> element, py::object path) {
    if (path == py::object())
        self.setElement(element);
    else
        self.setElement(element, py::extract<PathHints>(path));
}


BOOST_PYTHON_MODULE(trivial)
{
    {CLASS(StepProfileGain<Geometry2DCartesian>, "StepProfileGain2D", "Step-profile gain for 2D Cartesian geometry.")
        __solver__.def("setElement", StepProfile_setElement<Geometry2DCartesian>, "Set element on which there is a gain", (py::arg("element"), py::arg("path")=py::object()));
        RW_FIELD(gain, "Gain value [1/cm]");
        PROVIDER(outGain, "Gain distribution provider");
        py::scope().attr("StepProfile2D") = __solver__;
    }

}

