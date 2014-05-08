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
    return make_shared<RegularMesh1D>(self.current_mesh());
}

//TODO remove after 1.06.2014
py::object inLightIntensity_get(py::object self) {
    writelog(LOG_WARNING, "'inLightIntensity' is depreciated. Use 'inLightMagnitude' instead!");
    return self.attr("inLightMagnitude");
}
//TODO remove after 1.06.2014
void inLightIntensity_set(py::object self, py::object value) {
    writelog(LOG_WARNING, "'inLightIntensity' is depreciated. Use 'inLightMagnitude' instead!");
    self.attr("inLightMagnitude") = value;
}


/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(diffusion)
{
    {CLASS(FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>, "DiffusionCyl", "Calculates carrier pairs concentration in active region using FEM in one-dimensional cylindrical space")

        METHOD(compute_initial, compute_initial, "Perform the initial computation");
        METHOD(compute_threshold, compute_threshold, "Perform the threshold computation");
        METHOD(compute_overthreshold, compute_overthreshold, "Perform the overthreshold computation");
        solver.def_readwrite("initial", &__Class__::do_initial, "True if we start from initial computations");
        solver.def_readwrite("fem_method", &__Class__::fem_method, "Finite-element method (linear of parabolic)");
        solver.add_property("current_mesh", DiffusionSolver_current_mesh<Geometry2DCylindrical>, "Horizontal adaptive mesh)");
        solver.def_readwrite("accuracy", &__Class__::relative_accuracy, "Required relative accuracy");
        solver.def_readwrite("abs_accuracy", &__Class__::minor_concentration, "Required absolute minimal concentration accuracy");
        solver.def_readwrite("interpolation", &__Class__::interpolation_method, "Interpolation method used for injection current");
        solver.def_readwrite("maxrefines", &__Class__::max_mesh_changes, "Maximum number of allowed mesh refinements");
        solver.def_readwrite("maxiters", &__Class__::max_iterations, "Maximum number of allowed iterations before attempting to refine mesh");
        RECEIVER(inCurrentDensity, "");
        RECEIVER(inTemperature, "");
        RECEIVER(inGain, "");
        RECEIVER(inWavelength, "It is required only for the overthreshold computations.");
        RECEIVER(inGainOverCarriersConcentration, "It is required only for the overthreshold computations.");
        RECEIVER(inLightMagnitude, "It is required only for the overthreshold computations.");
        PROVIDER(outCarriersConcentration, "");
        METHOD(get_total_burning, burning_integral, "Compute power burned over threshold [mW].");
//         RW_FIELD(global_QW_width, "Sum of all QWs' widths" ); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions

        py::scope scope = solver;

        py_enum<__Class__::FemMethod>("Method")
            .value("LINEAR", __Class__::FEM_LINEAR)
            .value("PARABOLIC", __Class__::FEM_PARABOLIC)
        ;

        py_enum<__Class__::ComputationType>("Computation")
            .value("INITIAL", __Class__::COMPUTATION_INITIAL)
            .value("THRESHOLD", __Class__::COMPUTATION_THRESHOLD)
            .value("OVERTHRESHOLD", __Class__::COMPUTATION_OVERTHRESHOLD)
        ;

     }
     {CLASS(FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>, "Diffusion2D", "Calculates carrier pairs concentration in active region using FEM in one-dimensional cartesian space")

        METHOD(compute_initial, compute_initial, "Perform the initial computation");
        METHOD(compute_threshold, compute_threshold, "Perform the threshold computation");
        METHOD(compute_overthreshold, compute_overthreshold, "Perform the overthreshold computation");
        solver.def_readwrite("initial", &__Class__::do_initial, "True if we start from initial computations");
        solver.def_readwrite("fem_method", &__Class__::fem_method, "Finite-element method (linear of parabolic)");
        solver.add_property("current_mesh", DiffusionSolver_current_mesh<Geometry2DCartesian>, "Horizontal adaptive mesh)");
        solver.def_readwrite("accuracy", &__Class__::relative_accuracy, "Required relative accuracy");
        solver.def_readwrite("abs_accuracy", &__Class__::minor_concentration, "Required absolute minimal concentration accuracy");
        solver.def_readwrite("interpolation", &__Class__::interpolation_method, "Interpolation method used for injection current");
        solver.def_readwrite("maxrefines", &__Class__::max_mesh_changes, "Maximum number of allowed mesh refinements");
        solver.def_readwrite("maxiters", &__Class__::max_iterations, "Maximum number of allowed iterations before attempting to refine mesh");
        RECEIVER(inCurrentDensity, "");
        RECEIVER(inTemperature, "");
        RECEIVER(inGain, "It is required only for the overthreshold computations.");
        RECEIVER(inGainOverCarriersConcentration, "It is required only for the overthreshold computations.");
        RECEIVER(inLightMagnitude, "It is required only for the overthreshold computations.");
        PROVIDER(outCarriersConcentration, "");
        METHOD(get_total_burning, burning_integral, "Compute power burned over threshold [mW]");
//         RW_FIELD(global_QW_width, "Sum of all QWs' widths" ); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions

        //TODO remove after 1.06.2014
        solver.add_property("outLightIntensity", &inLightIntensity_get, &inLightIntensity_set, "DEPRECIATED");

        py::scope scope = solver;

        py_enum<__Class__::FemMethod>("Method")
            .value("LINEAR", __Class__::FEM_LINEAR)
            .value("PARABOLIC", __Class__::FEM_PARABOLIC)
        ;

        py_enum<__Class__::ComputationType>("Computation")
            .value("INITIAL", __Class__::COMPUTATION_INITIAL)
            .value("THRESHOLD", __Class__::COMPUTATION_THRESHOLD)
            .value("OVERTHRESHOLD", __Class__::COMPUTATION_OVERTHRESHOLD)
        ;
     }

}

