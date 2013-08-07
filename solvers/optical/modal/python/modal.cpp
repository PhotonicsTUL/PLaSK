/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../reflection_solver_2d.h"
#include "../reflection_solver_cyl.h"
using namespace plask::solvers::modal;

template <typename SolverT>
static const std::vector<std::size_t>& ModalSolver_getStack(const SolverT& self) { return self.getStack(); }

template <typename SolverT>
static const std::vector<RectilinearMesh1D>& ModalSolver_getLayerSets(const SolverT& self) { return self.getLayersPoints(); }

template <typename Class>
inline void export_base(Class solver) {
    typedef typename Class::wrapped_type Solver;
    solver.def_readwrite("outdist", &Solver::outdist, "Distance outside outer borders where material is sampled");
    solver.add_property("interface", &Solver::getInterface, &Solver::setInterface, "Matching interface position");
    solver.def("set_interface", &Solver::setInterfaceOn, "Set interface on object pointed by path", (py::arg("object"), py::arg("path")=py::object()));
    solver.def("set_interface", &Solver::setInterfaceAt, "Set interface around position pos", py::arg("pos"));
    solver.add_property("stack", py::make_function<>(&ModalSolver_getStack<Solver>, py::return_internal_reference<>()), "Stack of distinct layers");
    solver.add_property("layer_sets", py::make_function<>(&ModalSolver_getLayerSets<Solver>, py::return_internal_reference<>()), "Vertical positions of layers in each layer set");
    solver.add_receiver("inTemperature", &Solver::inTemperature, "Optical gain in the active region");
    solver.add_receiver("inGain", &Solver::inGain, "Optical gain in the active region");
    solver.add_provider("outIntensity", &Solver::outIntensity, "Light intensity of the last computed mode");
}



BOOST_PYTHON_MODULE(modal)
{
    {CLASS(FourierReflection2D, "FourierReflection2D",
        "Calculate optical modes and optical field distribution using Fourier modal method\n"
        " and reflection transfer in two-dimensional Cartesian space.")
        export_base(solver);
        RECEIVER(inWavelength, "Wavelength of the light");
        PROVIDER(outNeff, "Effective index of the last computed mode");
        METHOD(compute, computeMode, "Compute the mode near the specified effective index", "neff");
        //RO_FIELD(root, "Configuration of the global rootdigger");
        //RW_PROPERTY(emission, getEmission, setEmission, "Emission direction");
        //METHOD(find_modes, findModes, "Find the modes within the specified range using global method",
        //       arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6,1e-9));
        //METHOD(set_mode, setMode, "Set the current mode the specified effective index.\nneff can be a value returned e.g. by 'find_modes'.", "neff");
        //solver.def("get_determinant", &EffectiveIndex2DSolver_getDeterminant, "Get modal determinant", (py::arg("neff")));
        //BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
    }

    {CLASS(FourierReflectionCyl, "FourierReflectionCyl",
        "Calculate optical modes and optical field distribution using Fourier modal method\n"
        " and reflection transfer in two-dimensional cylindrical geometry.")
        export_base(solver);
        RECEIVER(inWavelength, "Wavelength of the light");
        PROVIDER(outNeff, "Effective index of the last computed mode");
        METHOD(compute, computeMode, "Compute the mode near the specified effective index", "neff");
    }

}

