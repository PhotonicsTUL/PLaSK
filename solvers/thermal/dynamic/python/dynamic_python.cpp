#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femT.h"
#include "../femT3d.h"
using namespace plask::thermal::dynamic;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(dynamic)
{
    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
    ;

    {CLASS(FiniteElementMethodDynamicThermal2DSolver<Geometry2DCartesian>, "Dynamic2D",
        "Finite element thermal solver for 2D Cartesian geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("time"));
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
        RW_FIELD(inittemp, "Initial temperature [K]");
        RW_FIELD(timestep, "Time step [ns]");
        RW_FIELD(methodparam, "Initial parameter determining the calculation method: 0.5 - Crank-Nicolson method, 0 - explicit method, 1 - implicit method");
        RW_FIELD(lumping, "Chosen mass matrix type from lumped or non-lumped (consistent)");
        RW_FIELD(rebuildfreq, "Frequency of rebuild mass");
        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
        RO_PROPERTY(elapsed_time, getElapsTime, "Time of calculations performed so far since the last solver invalidation.");
    }

    {CLASS(FiniteElementMethodDynamicThermal2DSolver<Geometry2DCylindrical>, "DynamicCyl",
        "Finite element thermal solver for 2D cylindrical geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("time"));
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
        RW_FIELD(inittemp, "Initial temperature [K]");
        RW_FIELD(timestep, "Time step [ns]");
        RW_FIELD(methodparam, "Initial parameter determining the calculation method: 0.5 - Crank-Nicolson method, 0 - explicit method, 1 - implicit method");
        RW_FIELD(lumping, "Chosen mass matrix type from lumped or non-lumped (consistent)");
        RW_FIELD(rebuildfreq, "Frequency of rebuild mass");
        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
        RO_PROPERTY(elapsed_time, getElapsTime, "Time of calculations performed so far since the last solver invalidation.");
    }

    {CLASS(FiniteElementMethodDynamicThermal3DSolver, "Dynamic3D",
        "Finite element thermal solver for 3D Cartesian geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("time"));
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
        RW_FIELD(inittemp, "Initial temperature [K]");
        RW_FIELD(timestep, "Time step [ns]");
        RW_FIELD(methodparam, "Initial parameter determining the calculation method: 0.5 - Crank-Nicolson method, 0 - explicit method, 1 - implicit method");
        RW_FIELD(lumping, "Chosen mass matrix type from lumped or non-lumped (consistent)");
        RW_FIELD(rebuildfreq, "Frequency of rebuild mass");
        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
        RO_PROPERTY(elapsed_time, getElapsTime, "Time of calculations performed so far since the last solver invalidation.");
    }

}
