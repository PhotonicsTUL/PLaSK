/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../diffusion_1D/diffusion_cylindrical.h"
using namespace plask::solvers::diffusion_cylindrical;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(diffusion)
{
    {CLASS(FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>, "DiffusionCyl", "Calculates carrier pairs concentration in active region using FEM in one-dimensional cylindrical space")

        py_enum<__Class__::FemMethod>("FemMethod")
            .value("LINEAR", __Class__::FEM_LINEAR)
            .value("PARABOLIC", __Class__::FEM_PARABOLIC);

        METHOD(compute, compute, "Perform the computation", arg("initial")=true, arg("threshold")=true);
        solver.def_readwrite("fem_method", &__Class__::fem_method, "Finite-element method (linear of parabolic)");
        RW_FIELD(mesh, "Horizontal adaptative mesh)");
        solver.def_readwrite("accuracy", &__Class__::relative_accuracy, "Required relative accuracy");
        solver.def_readwrite("absolute_accuracy", &__Class__::minor_concentration, "Required absolute minimal concentration accuracy");
        solver.def_readwrite("interpolation", &__Class__::interpolation_method, "Interpolation method used for injection current");
        solver.def_readwrite("maxrefines", &__Class__::max_mesh_changes, "Maximum number of allowed mesh refinements");
        solver.def_readwrite("maxiters", &__Class__::max_iterations, "Maximum number of allowed iterations before attempting to refine mesh");
        RECEIVER(inCurrentDensity, "Current density vector perpendicular to the active region"); // receiver in the solver
        RECEIVER(inTemperature, "Temperature distribution"); // receiver in the solver
        PROVIDER(outCarriersConcentration, "Carrier pairs concentration in the active region"); // provider in the solver
//         RW_FIELD(global_QW_width, "Sum of all QWs' widths" ); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
     }
     {CLASS(FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>, "Diffusion2D", "Calculates carrier pairs concentration in active region using FEM in one-dimensional cartesian space")

        py_enum<__Class__::FemMethod>("FemMethod")
            .value("LINEAR", __Class__::FEM_LINEAR)
            .value("PARABOLIC", __Class__::FEM_PARABOLIC);

        METHOD(compute, compute, "Perform the computation", arg("initial")=true, arg("threshold")=true);
        solver.def_readwrite("fem_method", &__Class__::fem_method, "Finite-element method (linear of parabolic)");
        RW_FIELD(mesh, "Horizontal adaptative mesh)");
        solver.def_readwrite("accuracy", &__Class__::relative_accuracy, "Required relative accuracy");
        solver.def_readwrite("absolute_accuracy", &__Class__::minor_concentration, "Required absolute minimal concentration accuracy");
        solver.def_readwrite("interpolation", &__Class__::interpolation_method, "Interpolation method used for injection current");
        solver.def_readwrite("maxrefines", &__Class__::max_mesh_changes, "Maximum number of allowed mesh refinements");
        solver.def_readwrite("maxiters", &__Class__::max_iterations, "Maximum number of allowed iterations before attempting to refine mesh");
        RECEIVER(inCurrentDensity, "Current density vector perpendicular to the active region"); // receiver in the solver
        RECEIVER(inTemperature, "Temperature distribution"); // receiver in the solver
        PROVIDER(outCarriersConcentration, "Carrier pairs concentration in the active region"); // provider in the solver
//         RW_FIELD(global_QW_width, "Sum of all QWs' widths" ); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
     }

}

