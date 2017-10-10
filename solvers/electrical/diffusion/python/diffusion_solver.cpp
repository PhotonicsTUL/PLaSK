/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../diffusion_1D/diffusion_cylindrical.h"
using namespace plask::solvers::diffusion_cylindrical;

template <typename GeometryT>
shared_ptr<RegularMesh1D> DiffusionSolver_current_mesh(FiniteElementMethodDiffusion2DSolver<GeometryT>& self) {
    return plask::make_shared<RegularMesh1D>(self.current_mesh());
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(diffusion)
{
    {CLASS(FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>, "DiffusionCyl", u8"Calculates carrier pairs concentration in active region using FEM in one-dimensional cylindrical space")

        METHOD(compute_initial, compute_initial, u8"Perform the initial computation");
        METHOD(compute_threshold, compute_threshold, u8"Perform the threshold computation");
        METHOD(compute_overthreshold, compute_overthreshold, u8"Perform the overthreshold computation");
        solver.def_readwrite("initial", &__Class__::do_initial, u8"True if we start from initial computations");
        solver.def_readwrite("fem_method", &__Class__::fem_method, u8"Finite-element method (linear of parabolic)");
        solver.add_property("current_mesh", DiffusionSolver_current_mesh<Geometry2DCylindrical>, u8"Horizontal adaptive mesh)");
        solver.def_readwrite("accuracy", &__Class__::relative_accuracy, u8"Required relative accuracy");
        solver.def_readwrite("abs_accuracy", &__Class__::minor_concentration, u8"Required absolute minimal concentration accuracy");
        solver.def_readwrite("interpolation", &__Class__::interpolation_method, u8"Interpolation method used for injection current");
        solver.def_readwrite("maxrefines", &__Class__::max_mesh_changes, u8"Maximum number of allowed mesh refinements");
        solver.def_readwrite("maxiters", &__Class__::max_iterations, u8"Maximum number of allowed iterations before attempting to refine mesh");
        RECEIVER(inCurrentDensity, u8"");
        RECEIVER(inTemperature, u8"");
        RECEIVER(inGain, u8"");
        RECEIVER(inWavelength, u8"It is required only for the overthreshold computations.");
        RECEIVER(inLightE, u8"It is required only for the overthreshold computations.");
        PROVIDER(outCarriersConcentration, u8"");
        METHOD(get_total_burning, burning_integral, u8"Compute total power burned over threshold [mW].");
        solver.def_readonly("mode_burns", &__Class__::modesP, u8"Power burned over threshold by each mode [mW].");
//         RW_FIELD(global_QW_width, "Sum of all QWs' widths" ); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions

        py::scope scope = solver;

        py_enum<__Class__::FemMethod>()
            .value("LINEAR", __Class__::FEM_LINEAR)
            .value("PARABOLIC", __Class__::FEM_PARABOLIC)
        ;

        py_enum<__Class__::ComputationType>()
            .value("INITIAL", __Class__::COMPUTATION_INITIAL)
            .value("THRESHOLD", __Class__::COMPUTATION_THRESHOLD)
            .value("OVERTHRESHOLD", __Class__::COMPUTATION_OVERTHRESHOLD)
        ;

     }
     {CLASS(FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>, "Diffusion2D", u8"Calculates carrier pairs concentration in active region using FEM in one-dimensional cartesian space")

        METHOD(compute_initial, compute_initial, u8"Perform the initial computation");
        METHOD(compute_threshold, compute_threshold, u8"Perform the threshold computation");
        METHOD(compute_overthreshold, compute_overthreshold, u8"Perform the overthreshold computation");
        solver.def_readwrite("initial", &__Class__::do_initial, u8"True if we start from initial computations");
        solver.def_readwrite("fem_method", &__Class__::fem_method, u8"Finite-element method (linear of parabolic)");
        solver.add_property("current_mesh", DiffusionSolver_current_mesh<Geometry2DCartesian>, u8"Horizontal adaptive mesh)");
        solver.def_readwrite("accuracy", &__Class__::relative_accuracy, u8"Required relative accuracy");
        solver.def_readwrite("abs_accuracy", &__Class__::minor_concentration, u8"Required absolute minimal concentration accuracy");
        solver.def_readwrite("interpolation", &__Class__::interpolation_method, u8"Interpolation method used for injection current");
        solver.def_readwrite("maxrefines", &__Class__::max_mesh_changes, u8"Maximum number of allowed mesh refinements");
        solver.def_readwrite("maxiters", &__Class__::max_iterations, u8"Maximum number of allowed iterations before attempting to refine mesh");
        RECEIVER(inCurrentDensity, "");
        RECEIVER(inTemperature, "");
        RECEIVER(inGain, u8"It is required only for the overthreshold computations.");
        RECEIVER(inLightE, u8"It is required only for the overthreshold computations.");
        PROVIDER(outCarriersConcentration, u8"");
        METHOD(get_total_burning, burning_integral, u8"Compute total power burned over threshold [mW].");
        solver.def_readonly("mode_burns", &__Class__::modesP, u8"Power burned over threshold by each mode [mW].");
//         RW_FIELD(global_QW_width, "Sum of all QWs' widths" ); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions

        py::scope scope = solver;

        py_enum<__Class__::FemMethod>()
            .value("LINEAR", __Class__::FEM_LINEAR)
            .value("PARABOLIC", __Class__::FEM_PARABOLIC)
        ;

        py_enum<__Class__::ComputationType>()
            .value("INITIAL", __Class__::COMPUTATION_INITIAL)
            .value("THRESHOLD", __Class__::COMPUTATION_THRESHOLD)
            .value("OVERTHRESHOLD", __Class__::COMPUTATION_OVERTHRESHOLD)
        ;
     }

}

