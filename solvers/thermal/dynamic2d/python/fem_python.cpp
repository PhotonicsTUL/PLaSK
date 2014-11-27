#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femT.h"
using namespace plask::solvers::thermal;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(fem)
{
    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
    ;

    {CLASS(FiniteElementMethodDynamicThermal2DSolver<Geometry2DCartesian>, "Dynamic2D",
        "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("time"));
//        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
//        BOUNDARY_CONDITIONS(heatflux_boundary, "Boundary conditions for the constant heat flux");
//        BOUNDARY_CONDITIONS(convection_boundary, "Convective boundary conditions");
//        BOUNDARY_CONDITIONS(radiation_boundary, "Radiative boundary conditions");
//        RW_FIELD(inittemp, "Initial temperature");
        RW_FIELD(nstimestep, "Time step in ns");
//        RW_FIELD(maxerr, "Limit for the temperature updates");
        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
    }

//    {CLASS(FiniteElementMethodDynamicThermal2DSolver<Geometry2DCylindrical>, "StaticCyl",
//        "Finite element thermal solver for 2D Cylindrical Geometry.")
//        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
//        RO_PROPERTY(err, getErr, "Maximum estimated error");
//        RECEIVER(inHeat, "");
//        solver.setattr("inHeatDensity", solver.attr("inHeat"));
//        PROVIDER(outTemperature, "");
//        PROVIDER(outHeatFlux, "");
//        PROVIDER(outThermalConductivity, "");
//        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
//        BOUNDARY_CONDITIONS(heatflux_boundary, "Boundary conditions for the constant heat flux");
//        BOUNDARY_CONDITIONS(convection_boundary, "Convective boundary conditions");
//        BOUNDARY_CONDITIONS(radiation_boundary, "Radiative boundary conditions");
//        RW_FIELD(inittemp, "Initial temperature");
//        RW_FIELD(maxerr, "Limit for the temperature updates");
//        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
//        solver.def_readwrite("itererr", &__Class__::itererr, "Allowed residual iteration for iterative method");
//        solver.def_readwrite("iterlim", &__Class__::iterlim ,"Maximum number of iterations for iterative method");
//        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
//    }
}
