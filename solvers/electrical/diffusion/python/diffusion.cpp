/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/common/fem/python.hpp>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../diffusion2d.hpp"
using namespace plask::electrical::diffusion;

template <typename SolverT> static double DiffusionSolver_compute(SolverT* solver, unsigned loops, const py::object& pact) {
    if (pact.is_none()) {
        return solver->compute(loops);
    } else {
        int act = py::extract<int>(pact);
        if (act < 0) act = solver->activeRegionsCount() + act;
        return solver->compute(loops, act);
    }
}

template <typename GeometryT> struct ExportedDiffusion2DSolverDefaultDefs {
    static void Solver_setMesh(Diffusion2DSolver<GeometryT>& self, py::object mesh) {
        if (mesh.is_none()) { self.setMesh(shared_ptr<RectangularMesh<2>>()); return; }

        py::extract<shared_ptr<RectangularMesh<2>>> mesh2d(mesh);
        if (mesh2d.check()) { self.setMesh(mesh2d()); return; }

        py::extract<shared_ptr<MeshGeneratorD<2>>> generator2d(mesh);
        if (generator2d.check()) { self.setMesh(generator2d()); return; }

        py::extract<shared_ptr<MeshD<1>>> mesh1d(mesh);
        if (mesh1d.check()) { self.setMesh(mesh1d()); return; }

        py::extract<shared_ptr<MeshGeneratorD<1>>> generator1d(mesh);
        if (generator1d.check()) { self.setMesh(generator1d()); return; }

        if (PySequence_Check(mesh.ptr())) {
            py::stl_input_iterator<double> begin(mesh), end;
            shared_ptr<OrderedAxis> ordered_axis(new OrderedAxis(std::vector<double>(begin, end)));
            self.setMesh(static_pointer_cast<MeshD<1>>(ordered_axis));
            return;
        }

        throw TypeError(u8"Cannot convert argument to proper mesh type");
    }

    template <typename PySolver> static auto init(PySolver& solver) -> PySolver& {
        solver.add_property("geometry", &Diffusion2DSolver<GeometryT>::getGeometry, &Diffusion2DSolver<GeometryT>::setGeometry,
                            u8"Geometry provided to the solver");
        solver.add_property("mesh", &Diffusion2DSolver<GeometryT>::getMesh, &Solver_setMesh, u8"Mesh provided to the solver");
        return solver;
    }
};

namespace plask { namespace python { namespace detail {

template <>
struct ExportedSolverDefaultDefs<Diffusion2DSolver<Geometry2DCylindrical>, void, void>
    : ExportedDiffusion2DSolverDefaultDefs<Geometry2DCylindrical> {};

template <>
struct ExportedSolverDefaultDefs<Diffusion2DSolver<Geometry2DCartesian>, void, void>
    : ExportedDiffusion2DSolverDefaultDefs<Geometry2DCartesian> {};

}}}  // namespace plask::python::detail

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(diffusion) {
    {
        CLASS(Diffusion2DSolver<Geometry2DCylindrical>, "DiffusionCyl",
              u8"Calculates carrier pairs concentration in active region using FEM in one-dimensional cylindrical space")
        solver.def("compute", &DiffusionSolver_compute<__Class__>, u8"Run diffusion calculations",
                   (py::arg("loops") = 0, py::arg("act") = py::object()));
        RW_FIELD(maxerr, u8"Limit for the potential updates");
        RECEIVER(inCurrentDensity, u8"");
        RECEIVER(inTemperature, u8"");
        RECEIVER(inGain, u8"");
        RECEIVER(inWavelength, u8"It is required only for the overthreshold computations.");
        RECEIVER(inLightE, u8"It is required only for the overthreshold computations.");
        PROVIDER(outCarriersConcentration, u8"");
        // METHOD(get_total_burning, burning_integral, u8"Compute total power burned over threshold [mW].");
        // solver.def_readonly("mode_burns", &__Class__::modesP, u8"Power burned over threshold by each mode [mW].");
        registerFemSolver(solver);
    }

    {
        CLASS(Diffusion2DSolver<Geometry2DCartesian>, "Diffusion2D",
              u8"Calculates carrier pairs concentration in active region using FEM in one-dimensional cartesian space")
        solver.def("compute", &DiffusionSolver_compute<__Class__>, u8"Run diffusion calculations",
                   (py::arg("loops") = 0, py::arg("act") = py::object()));
        RW_FIELD(maxerr, u8"Limit for the potential updates");
        RECEIVER(inCurrentDensity, u8"");
        RECEIVER(inTemperature, u8"");
        RECEIVER(inGain, u8"");
        RECEIVER(inWavelength, u8"It is required only for the overthreshold computations.");
        RECEIVER(inLightE, u8"It is required only for the overthreshold computations.");
        PROVIDER(outCarriersConcentration, u8"");
        // METHOD(get_total_burning, burning_integral, u8"Compute total power burned over threshold [mW].");
        // solver.def_readonly("mode_burns", &__Class__::modesP, u8"Power burned over threshold by each mode [mW].");
        registerFemSolver(solver);
    }
}
