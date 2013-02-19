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
//        solver.def_readwrite("wavelegth", &__Class__::lambda, "Wavelength for which gain is calculated");
    }
    {CLASS(FermiGainSolver<Geometry2DCylindrical>, "FermiCyl", "Gain solver based on Fermi Golden Rule for Cylindrical 2D geometry.")
        RECEIVER(inTemperature, "Temperature distribution along 'x' direction in the active region");
        RECEIVER(inCarriersConcentration, "Carrier pairs concentration along 'x' direction in the active region");
        PROVIDER(outGain, "Optical gain in the active region");
//        solver.def_readwrite("wavelegth", &__Class__::lambda, "Wavelength for which gain is calculated");
    }

}

