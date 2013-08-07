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
    {CLASS(FourierReflection2D, "FourierReflection2D",
        "Calculate optical modes and optical field distribution using Fourier modal method\n"
        " and reflection transfer in two-dimensional Cartesian space.")
        RW_FIELD(outdist, "Distance outside outer borders where material is sampled");
        //RO_FIELD(root, "Configuration of the global rootdigger");
        //RW_PROPERTY(emission, getEmission, setEmission, "Emission direction");
        METHOD(compute, computeMode, "Compute the mode near the specified effective index", "neff");
        //METHOD(find_modes, findModes, "Find the modes within the specified range using global method",
        //       arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        //METHOD(set_mode, setMode, "Set the current mode the specified effective index.\nneff can be a value returned e.g. by 'find_modes'.", "neff");
        //solver.def("get_determinant", &EffectiveIndex2DSolver_getDeterminant, "Get modal determinant", (py::arg("neff")));
        RECEIVER(inWavelength, "Wavelength of the light");
        RECEIVER(inTemperature, "Temperature distribution in the structure");
        RECEIVER(inGain, "Optical gain in the active region");
        PROVIDER(outNeff, "Effective index of the last computed mode");
        PROVIDER(outIntensity, "Light intensity of the last computed mode");
        RW_PROPERTY(interface, getInterface, setInterface, "Matching interface position");
        METHOD(set_interface, setInterfaceOn, "Set interface on object pointed by path", "object", py::arg("path")=py::object());
        METHOD(set_interface, setInterfaceAt, "Set interface around position pos", "pos");
        solver.add_property("stack", py::make_function<>(&FourierReflection2D::getStack, py::return_internal_reference<>()), "Stack of distinct layers");
        solver.add_property("layer_sets", py::make_function<>(&FourierReflection2D::getLayersPoints, py::return_internal_reference<>()), "Vertical positions of layers in each layer set");
        //BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
    }
}

