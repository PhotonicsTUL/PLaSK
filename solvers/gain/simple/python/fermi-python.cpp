/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../fermi.h"
using namespace plask::solvers::fermi;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(simple)
{
    {CLASS(FermiGainSolver<Geometry2DCartesian>, "Fermi2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        RECEIVER(inTemperature, "Temperature distribution along 'x' direction in the active region");
        RECEIVER(inCarriersConcentration, "Carrier pairs concentration along 'x' direction in the active region");
        PROVIDER(outGain, "Optical gain in the active region");
//         METHOD(python_method_name, method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
//         RO_FIELD(field_name, "Short documentation"); // read-only field
//         RW_FIELD(field_name, "Short documentation"); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
    }

}

