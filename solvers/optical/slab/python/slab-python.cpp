/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../reflection_solver_2d.h"
#include "../reflection_solver_cyl.h"
using namespace plask::solvers::slab;

template <typename SolverT>
static const std::vector<std::size_t>& SlabSolver_getStack(const SolverT& self) { return self.getStack(); }

template <typename SolverT>
static const std::vector<RectilinearMesh1D>& SlabSolver_getLayerSets(const SolverT& self) { return self.getLayersPoints(); }

template <typename Class>
inline void export_base(Class solver) {
    typedef typename Class::wrapped_type Solver;
    solver.def_readwrite("outdist", &Solver::outdist, "Distance outside outer borders where material is sampled");
    solver.add_property("interface", &Solver::getInterface, &Solver::setInterface, "Matching interface position");
    solver.def("set_interface", &Solver::setInterfaceOn, "Set interface on object pointed by path", (py::arg("object"), py::arg("path")=py::object()));
    solver.def("set_interface", &Solver::setInterfaceAt, "Set interface around position pos", py::arg("pos"));
    solver.def_readwrite("smooth", &Solver::smooth, "Smoothing parameter");
    solver.add_property("stack", py::make_function<>(&SlabSolver_getStack<Solver>, py::return_internal_reference<>()), "Stack of distinct layers");
    solver.add_property("layer_sets", py::make_function<>(&SlabSolver_getLayerSets<Solver>, py::return_internal_reference<>()), "Vertical positions of layers in each layer set");
    solver.add_receiver("inTemperature", &Solver::inTemperature, "Optical gain in the active region");
    solver.add_receiver("inGain", &Solver::inGain, "Optical gain in the active region");
    solver.add_provider("outIntensity", &Solver::outIntensity, "Light intensity of the last computed mode");
}


DataVectorWrap<const Tensor3<dcomplex>,2> FourierReflection2D_getRefractiveIndexProfile(FourierReflection2D& self,
                const shared_ptr<RectilinearMesh2D>& dst_mesh, InterpolationMethod interp=INTERPOLATION_DEFAULT) {
    return DataVectorWrap<const Tensor3<dcomplex>,2>(self.getRefractiveIndexProfile(*dst_mesh, interp), dst_mesh);
}


BOOST_PYTHON_MODULE(slab)
{
    {CLASS(FourierReflection2D, "FourierReflection2D",
        "Calculate optical modes and optical field distribution using Fourier slab method\n"
        " and reflection transfer in two-dimensional Cartesian space.")
        export_base(solver);
        PROVIDER(outNeff, "Effective index of the last computed mode");
        METHOD(find_mode, findMode, "Compute the mode near the specified effective index", "neff");
        RW_PROPERTY(wavelength, getWavelength, setWavelength, "Wavelength of the light");
        RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size");
        solver.def("get_refractive_index_profile", &FourierReflection2D_getRefractiveIndexProfile,
                   "Get profile of the expanded refractive index", (py::arg("mesh"), py::arg("interp")=INTERPOLATION_DEFAULT));
    }

    {CLASS(FourierReflectionCyl, "FourierReflectionCyl",
        "Calculate optical modes and optical field distribution using Fourier slab method\n"
        " and reflection transfer in two-dimensional cylindrical geometry.")
        export_base(solver);
        METHOD(find_mode, findMode, "Compute the mode near the specified effective index", "neff");
        RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size");
    }
}

