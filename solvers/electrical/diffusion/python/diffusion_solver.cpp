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
     {CLASS(DiffusionCylindricalSolver, "DiffusionCylindrical1D", "Calculates carrier pairs concentration in active region using FEM in one-dimensional cylindrical space")
//         METHOD(python_method_name, method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
//         RO_FIELD(field_name, "Short documentation"); // read-only field
//         RW_FIELD(field_name, "Short documentation"); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
         RECEIVER(inCurrentDensity, "Current density vector perpendicular to the active region"); // receiver in the solver
         RECEIVER(inTemperature, "Temperature distribution along 'r' direction in the active region"); // receiver in the solver
         PROVIDER(outCarriersConcentration, "Carrier pairs concentration along 'r' direction in the active region"); // provider in the solver
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
     }

}

