/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/common/fem/python.hpp>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../diffusion2d.hpp"
#include "../diffusion3d.hpp"
using namespace plask::electrical::diffusion;

template <typename SolverT>
static double DiffusionSolver_compute(SolverT* solver, unsigned loops, bool shb, const py::object& pact) {
    if (pact.is_none()) {
        return solver->compute(loops, shb);
    } else {
        int act = py::extract<int>(pact);
        if (act < 0) act = solver->activeRegionsCount() + act;
        return solver->compute(loops, shb, act);
    }
}

static void DiffusionSolver2D_compute_threshold(Diffusion2DSolver<Geometry2DCartesian>* solver) {
    writelog(LOG_WARNING, u8"DiffusionSolver2D.compute_threshold() is deprecated. Use DiffusionSolver2D.compute() instead.");
    solver->compute(0, false);
}

static void DiffusionSolver2D_compute_overthreshold(Diffusion2DSolver<Geometry2DCartesian>* solver) {
    writelog(LOG_WARNING,
             u8"DiffusionSolver2D.compute_overthreshold() is deprecated. Use DiffusionSolver2D.compute(shb=True) instead.");
    solver->compute(0, true);
}

static void DiffusionSolverCyl_compute_threshold(Diffusion2DSolver<Geometry2DCylindrical>* solver) {
    writelog(LOG_WARNING, u8"DiffusionSolverCyl.compute_threshold() is deprecated. Use DiffusionSolverCyl.compute() instead.");
    solver->compute(0, false);
}

static void DiffusionSolverCyl_compute_overthreshold(Diffusion2DSolver<Geometry2DCylindrical>* solver) {
    writelog(LOG_WARNING,
             u8"DiffusionSolverCyl.compute_overthreshold() is deprecated. Use DiffusionSolverCyl.compute(shb=True) instead.");
    solver->compute(0, true);
}

template <typename SolverT> static double DiffusionSolver_get_burning_for_mode(SolverT* solver, int mode) {
    if (mode < 0) mode = solver->inLightE.size() + mode;
    return solver->get_burning_integral_for_mode(mode);
}

template <typename GeometryT> struct ExportedDiffusion2DSolverDefaultDefs {
    static void Solver_setMesh(Diffusion2DSolver<GeometryT>& self, py::object mesh) {
        if (mesh.is_none()) {
            self.setMesh(shared_ptr<RectangularMesh<2>>());
            return;
        }

        py::extract<shared_ptr<RectangularMesh<2>>> mesh2d(mesh);
        if (mesh2d.check()) {
            self.setMesh(mesh2d());
            return;
        }

        py::extract<shared_ptr<MeshGeneratorD<2>>> generator2d(mesh);
        if (generator2d.check()) {
            self.setMesh(generator2d());
            return;
        }

        py::extract<shared_ptr<MeshD<1>>> mesh1d(mesh);
        if (mesh1d.check()) {
            self.setMesh(mesh1d());
            return;
        }

        py::extract<shared_ptr<MeshGeneratorD<1>>> generator1d(mesh);
        if (generator1d.check()) {
            self.setMesh(generator1d());
            return;
        }

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
              u8"Calculates carrier pairs concentration in active region using FEM in two-dimensional cylindrical space")
        solver.def("compute", &DiffusionSolver_compute<__Class__>,
                   u8"Run diffusion calculations\n\n"
                   u8"Args:\n"
                   u8"    loops (int): Number of iterations to perform. If 0, the solver will run\n"
                   u8"                 until the convergence.\n"
                   u8"    shb (bool): If True, the solver will use take into account the spatial hole\n"
                   u8"                burning effect.\n"
                   u8"    reg (int or None): Index of the active region to compute. If None, perform\n"
                   u8"                       computations for all the active regions.",
                   (py::arg("loops") = 0, py::arg("shb") = false, py::arg("reg") = py::object()));
        RW_FIELD(maxerr, u8"Maximum relative residual error (%)");
        RECEIVER(inCurrentDensity, u8"");
        RECEIVER(inTemperature, u8"");
        RECEIVER(inGain, u8"It is required only for the SHB computations.");
        RECEIVER(inWavelength, u8"It is required only for the SHB computations.");
        RECEIVER(inLightE, u8"It is required only for the SHB computations.");
        PROVIDER(outCarriersConcentration, u8"");
        METHOD(get_total_burning, get_burning_integral, u8"Get total power burned over threshold [mW].");
        solver.def("get_burning_for_mode", DiffusionSolver_get_burning_for_mode<__Class__>,
                   u8"Get power burned over threshold by specified mode [mW].", py::arg("mode"));
        registerFemSolver(solver);

        // TODO remove some day
        solver.def("compute_threshold", &DiffusionSolverCyl_compute_threshold, u8"Deprecated method. Use compute() instead.");
        solver.def("compute_overthreshold", &DiffusionSolverCyl_compute_overthreshold,
                   u8"Deprecated method. Use compute(shb=True) instead.");
    }

    {
        CLASS(Diffusion2DSolver<Geometry2DCartesian>, "Diffusion2D",
              u8"Calculates carrier pairs concentration in active region using FEM in two-dimensional Cartesian space")
        solver.def("compute", &DiffusionSolver_compute<__Class__>,
                   u8"Run diffusion calculations\n\n"
                   u8"Args:\n"
                   u8"    loops (int): Number of iterations to perform. If 0, the solver will run\n"
                   u8"                 until the convergence.\n"
                   u8"    shb (bool): If True, the solver will use take into account the spatial hole\n"
                   u8"                burning effect.\n"
                   u8"    reg (int or None): Index of the active region to compute. If None, perform\n"
                   u8"                       computations for all the active regions.",
                   (py::arg("loops") = 0, py::arg("shb") = false, py::arg("reg") = py::object()));
        RW_FIELD(maxerr, u8"Maximum relative residual error (%)");
        RECEIVER(inCurrentDensity, u8"");
        RECEIVER(inTemperature, u8"");
        RECEIVER(inGain, u8"It is required only for the SHB computations.");
        RECEIVER(inWavelength, u8"It is required only for the SHB computations.");
        RECEIVER(inLightE, u8"It is required only for the SHB computations.");
        PROVIDER(outCarriersConcentration, u8"");
        METHOD(get_total_burning, get_burning_integral, u8"Get total power burned over threshold [mW].");
        solver.def("get_burning_for_mode", DiffusionSolver_get_burning_for_mode<__Class__>,
                   u8"Get power burned over threshold by specified mode [mW].", py::arg("mode"));
        registerFemSolver(solver);

        // TODO remove some day
        solver.def("compute_threshold", &DiffusionSolver2D_compute_threshold, u8"Deprecated method. Use compute() instead.");
        solver.def("compute_overthreshold", &DiffusionSolver2D_compute_overthreshold,
                   u8"Deprecated method. Use compute(shb=True) instead.");
    }

    {
        CLASS(Diffusion3DSolver, "Diffusion3D",
              u8"Calculates carrier pairs concentration in active region using FEM in three-dimensional space")
        solver.def("compute", &DiffusionSolver_compute<__Class__>,
                   u8"Run diffusion calculations\n\n"
                   u8"Args:\n"
                   u8"    loops (int): Number of iterations to perform. If 0, the solver will run\n"
                   u8"                 until the convergence.\n"
                   u8"    shb (bool): If True, the solver will use take into account the spatial hole\n"
                   u8"                burning effect.\n"
                   u8"    reg (int or None): Index of the active region to compute. If None, perform\n"
                   u8"                       computations for all the active regions.",
                   (py::arg("loops") = 0, py::arg("shb") = false, py::arg("reg") = py::object()));
        RW_FIELD(maxerr, u8"Maximum relative residual error (%)");
        RECEIVER(inCurrentDensity, u8"");
        RECEIVER(inTemperature, u8"");
        RECEIVER(inGain, u8"It is required only for the SHB computations.");
        RECEIVER(inWavelength, u8"It is required only for the SHB computations.");
        RECEIVER(inLightE, u8"It is required only for the SHB computations.");
        PROVIDER(outCarriersConcentration, u8"");
        METHOD(get_total_burning, get_burning_integral, u8"Get total power burned over threshold [mW].");
        solver.def("get_burning_for_mode", DiffusionSolver_get_burning_for_mode<__Class__>,
                   u8"Get power burned over threshold by specified mode [mW].", py::arg("mode"));
        registerFemSolver(solver);
    }
}
