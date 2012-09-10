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
static void StepProfile_setElement(StepProfileGain<GeometryT>& self, const GeometryElementD<GeometryT::DIMS>* element, const PathHints& path) {
    auto shared = dynamic_pointer_cast<const GeometryElementD<GeometryT::DIMS>>(element->shared_from_this());
    self.setElement(shared, path);
}

template <typename GeometryT>
static void registerStepProfile(const std::string& variant, const std::string& full_variant) {
    CLASS(StepProfileGain<GeometryT>, ("StepProfileGain"+variant).c_str(),
        ("Step-profile gain for "+full_variant+" geometry.").c_str())
        __solver__.def("setElement", StepProfile_setElement<GeometryT>, "Set element on which there is a gain", (py::arg("element"), py::arg("path")=py::object()));
        RW_PROPERTY(gain, getGain, setGain, "Gain value [1/cm]");
        PROVIDER(outGain, "Gain distribution provider");
        py::scope().attr(("StepProfile"+variant).c_str()) = __solver__;
}

BOOST_PYTHON_MODULE(trivial)
{
    registerStepProfile<Geometry2DCartesian>("2D", "Cartesian 2D");
    registerStepProfile<Geometry2DCylindrical>("Cyl", "Cylindrical");
    registerStepProfile<Geometry3D>("3D", "3D");
}

